"""Quote fetching, ticker validation, and daily-bar helpers."""
import time
import threading
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import StringIO
import json
import urllib.parse
import urllib.request

import pandas as pd

from market import fh
from market.data_manager import get_daily as _managed_daily
from trading.config import DEFAULT_CONFIG
from utils.cache import (
    CACHE_MISS, cache_get, cache_get_stale, cache_set, record_api_failure, record_api_success,
    sanitize_provider_error, should_skip_api, api_cooldown_state, api_failure_snapshot,
)
from utils.deploy_config import FMP_KEY, PYTHONANYWHERE_MODE
from utils.storage import load_price_hist, append_price_snapshot
from utils.time_utils import completed_trading_days_since, is_market_open

FMP_DAILY_GLOBAL_ENDPOINT = "fmp_daily:global"
FMP_DAILY_RATE_LIMIT_COOLDOWN_SEC = int(
    DEFAULT_CONFIG.get("history", {}).get("fmp_daily_global_429_cooldown_sec", 30 * 60)
)
# Entry 027 — endpoint-global Finnhub daily circuit. A single 401/403 on the
# candle endpoint means the key has no candle access for ANY symbol, so we open
# one global circuit instead of retrying per symbol. Quotes are a separate
# endpoint (quote:{tk}) and are NOT affected.
FINNHUB_DAILY_GLOBAL_ENDPOINT = "finnhub_daily:global"
FINNHUB_DAILY_FORBIDDEN_COOLDOWN_SEC = int(
    DEFAULT_CONFIG.get("history", {}).get("finnhub_daily_forbidden_cooldown_sec", 6 * 3600)
)
# Entry 027 (B2) — completed daily bars do not change intraday (today's forming
# bar comes from the live-quote overlay), so a symbol fetched once can safely
# serve the whole session. A longer daily-history cache TTL keeps FMP's limited
# free-tier request budget from being burned on hourly re-fetches.
DAILY_HISTORY_CACHE_TTL_SEC = int(
    DEFAULT_CONFIG.get("history", {}).get("daily_history_cache_ttl_sec", 18 * 3600)
)
FMP_DAILY_CACHE_MARKET_SEC = DAILY_HISTORY_CACHE_TTL_SEC
FMP_DAILY_CACHE_CLOSED_SEC = 24 * 3600
STALE_DAILY_CACHE_MARKET_SEC = 72 * 3600
STALE_DAILY_CACHE_CLOSED_SEC = 72 * 3600
REGIME_STALE_CACHE_MAX_HOURS = 72
REGIME_STALE_CACHE_MAX_SEC = STALE_DAILY_CACHE_CLOSED_SEC
EXECUTION_CACHE_MAX_AGE_SEC = 300
_HISTORY_FETCH_BUDGET_LOCK = threading.Lock()
_HISTORY_FETCH_BUDGET_REMAINING = None


def _history_cfg():
    return DEFAULT_CONFIG.get("history", {})


def _norm_symbol(tk):
    return str(tk or "").upper().strip()


def set_history_fetch_budget(limit):
    global _HISTORY_FETCH_BUDGET_REMAINING
    with _HISTORY_FETCH_BUDGET_LOCK:
        _HISTORY_FETCH_BUDGET_REMAINING = None if limit is None else max(0, int(limit))


def _consume_history_fetch_budget():
    global _HISTORY_FETCH_BUDGET_REMAINING
    with _HISTORY_FETCH_BUDGET_LOCK:
        remaining = _HISTORY_FETCH_BUDGET_REMAINING
        if remaining is None:
            return True
        if remaining <= 0:
            return False
        _HISTORY_FETCH_BUDGET_REMAINING = remaining - 1
        return True


def _history_fetch_budget_remaining():
    with _HISTORY_FETCH_BUDGET_LOCK:
        remaining = _HISTORY_FETCH_BUDGET_REMAINING
    return None if remaining is None else int(remaining)


def _valid_daily_df(df):
    return (
        df is not None
        and isinstance(df, pd.DataFrame)
        and not df.empty
        and "Close" in df.columns
    )


def _daily_cache_keys(tk, full=False, include_stooq=False, stooq_prefix="stooq"):
    tk = _norm_symbol(tk)
    keys = []
    if include_stooq:
        keys.append((f"{stooq_prefix}_daily", f"{stooq_prefix}_full_{tk}" if full else f"{stooq_prefix}_{tk}"))
    keys.extend([
        ("finnhub_daily", f"fh_daily_full_{tk}" if full else f"fh_daily_{tk}"),
        ("fmp_daily", f"fmp_daily_full_{tk}" if full else f"fmp_daily_{tk}"),
    ])
    return keys


def _history_last_date(df):
    try:
        return pd.to_datetime(df.index[-1]).date()
    except Exception:
        return None


def _history_cache_invalid_reason(df, min_rows, max_completed_days):
    if not _valid_daily_df(df):
        return "not_parseable"
    try:
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    except Exception:
        return "invalid_close"
    if close.empty:
        return "invalid_close"
    if len(df) < int(min_rows or 0):
        return "insufficient_rows"
    last_date = _history_last_date(df)
    if last_date is None:
        return "missing_last_date"
    age_days = completed_trading_days_since(last_date)
    if age_days is None or age_days > int(max_completed_days):
        return "stale_completed_trading_day"
    return None


def _cached_daily_history(tk, full=False, include_stooq=False, stooq_prefix="stooq",
                          provider_chain=None):
    cfg = _history_cfg()
    if not cfg.get("history_cache_first", True):
        return None
    min_rows = int(cfg.get("history_cache_min_rows", 25))
    max_days = int(cfg.get("history_cache_max_completed_trading_day_age", 2))
    max_age = max(_stale_daily_cache_max_age(), (max_days + 5) * 86400)
    chain = list(provider_chain or [])
    for source, cache_key in _daily_cache_keys(tk, full=full, include_stooq=include_stooq, stooq_prefix=stooq_prefix):
        cached, age_sec = cache_get_stale(cache_key, max_age, default=CACHE_MISS)
        reason = _history_cache_invalid_reason(cached, min_rows, max_days)
        if reason:
            if cached is not CACHE_MISS:
                chain.append(_daily_chain_entry(
                    "history_cache",
                    df=cached if _valid_daily_df(cached) else None,
                    status="invalid_cache",
                    reason=reason,
                ))
            continue
        out = cached.copy(deep=False)
        age_days = completed_trading_days_since(_history_last_date(out))
        out.attrs.update({
            "source": f"{source}_cache",
            "status": "cache",
            "provider": source.replace("_daily", ""),
            "history_cache_used": True,
            "history_cache_valid": True,
            "history_cache_invalid_reason": None,
            "history_cache_age_completed_trading_days": age_days,
            "stale_daily_cache_age_sec": int(age_sec or 0),
            "stale_daily_cache_age_hours": round(float(age_sec or 0) / 3600.0, 2),
        })
        chain.append(_daily_chain_entry("history_cache", df=out, status="cache", reason=f"cache_hit:{source}"))
        out.attrs["provider_chain_debug"] = chain
        return out
    return None


def _blocked_or_forbidden_payload(raw):
    text = str(raw or "").lower()
    return (
        "403" in text
        or "forbidden" in text
        or "blocked" in text
        or "don't have access" in text
        or "do not have access" in text
        or "access to this resource" in text
    )


def _market_open_now():
    try:
        return bool(is_market_open())
    except Exception:
        return False


def _fmp_daily_cache_max_age():
    return FMP_DAILY_CACHE_MARKET_SEC if _market_open_now() else FMP_DAILY_CACHE_CLOSED_SEC


def _stale_daily_cache_max_age():
    return STALE_DAILY_CACHE_MARKET_SEC if _market_open_now() else STALE_DAILY_CACHE_CLOSED_SEC


def fmp_daily_global_circuit_state():
    state = api_cooldown_state(FMP_DAILY_GLOBAL_ENDPOINT, FMP_DAILY_RATE_LIMIT_COOLDOWN_SEC)
    active = bool(state.get("active"))
    return {
        "status": "rate_limited" if active else "ok",
        "active": active,
        "rate_limited": active,
        "cooldown_remaining_sec": int(state.get("cooldown_remaining_sec") or 0),
        "last_429_age_sec": state.get("last_429_age_sec"),
        "last_error": state.get("last_error"),
    }


def _finnhub_daily_cache_max_age():
    return DAILY_HISTORY_CACHE_TTL_SEC


def finnhub_daily_global_circuit_state():
    """Endpoint-global Finnhub daily-history circuit (Entry 027).

    Opened by a single 401/403 on the candle endpoint; applies to ALL symbols.
    Distinct from the Finnhub quote circuit (quote:{tk}); quotes stay enabled.
    """
    state = api_cooldown_state(
        FINNHUB_DAILY_GLOBAL_ENDPOINT,
        FINNHUB_DAILY_FORBIDDEN_COOLDOWN_SEC,
        active_statuses={"blocked_or_forbidden"},
    )
    active = bool(state.get("active"))
    return {
        "status": "blocked_or_forbidden" if active else "ok",
        "active": active,
        "forbidden": active,
        "cooldown_sec": FINNHUB_DAILY_FORBIDDEN_COOLDOWN_SEC,
        "cooldown_remaining_sec": int(state.get("cooldown_remaining_sec") or 0),
        "last_error": state.get("last_error"),
        "quote_endpoint_still_enabled": True,
    }


def _open_finnhub_daily_global_forbidden_circuit(error):
    """Record the endpoint-global Finnhub daily forbidden circuit (21600s)."""
    return record_api_failure(
        FINNHUB_DAILY_GLOBAL_ENDPOINT,
        error,
        status="blocked_or_forbidden",
        cooldown_sec=FINNHUB_DAILY_FORBIDDEN_COOLDOWN_SEC,
    )


def _classify_finnhub_daily_error(error):
    """Classify a Finnhub daily failure robustly.

    finnhub-python's FinnhubAPIException stores the HTTP code on `.status_code`
    (not `.code`/`.status`) and also embeds "status_code: 403" in its str(), so
    we check attributes first and fall back to text sniffing.
    """
    forbidden_codes = set(_history_cfg().get("finnhub_daily_forbidden_status_codes", [401, 403]))
    status_code = None
    for attr in ("status_code", "code", "status"):
        try:
            val = getattr(error, attr, None)
        except Exception:
            val = None
        if val is None:
            continue
        try:
            status_code = int(val)
            break
        except (TypeError, ValueError):
            continue
    if status_code in forbidden_codes:
        return "blocked_or_forbidden"
    if status_code == 429:
        return "rate_limited"
    text = str(error or "")
    low = text.lower()
    if _blocked_or_forbidden_payload(text) or "401" in low or "unauthorized" in low or "invalid api key" in low:
        return "blocked_or_forbidden"
    if "429" in low or "too many requests" in low or "rate limit" in low:
        return "rate_limited"
    return "provider_error"


def _finnhub_daily_cooldown_for_status(status):
    cfg = _history_cfg()
    status = str(status or "")
    if status == "blocked_or_forbidden":
        return int(cfg.get("finnhub_daily_forbidden_cooldown_sec", 21_600))
    if status == "rate_limited":
        return int(cfg.get("finnhub_daily_rate_limit_cooldown_sec", 1_800))
    return int(cfg.get("finnhub_daily_provider_error_cooldown_sec", 900))


def finnhub_daily_circuit_state(tk, full=False):
    endpoint = f"finnhub_daily:{_norm_symbol(tk)}:{int(bool(full))}"
    snap = api_failure_snapshot().get(endpoint, {})
    status = snap.get("status") or "ok"
    state = api_cooldown_state(
        endpoint,
        _finnhub_daily_cooldown_for_status(status),
        active_statuses={"blocked_or_forbidden", "rate_limited", "provider_error", "timeout", "parse_error"},
    )
    state["endpoint"] = endpoint
    state["quote_endpoint_still_enabled"] = True
    return state


def _retry_after_cooldown_sec(error):
    fallback = FMP_DAILY_RATE_LIMIT_COOLDOWN_SEC
    raw = None
    try:
        headers = getattr(error, "headers", None)
        if headers:
            raw = headers.get("Retry-After")
    except Exception:
        raw = None
    if raw is None:
        return fallback
    raw = str(raw).strip()
    if not raw:
        return fallback
    try:
        return max(1, int(float(raw)))
    except Exception:
        pass
    try:
        retry_at = parsedate_to_datetime(raw)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(1, int((retry_at - datetime.now(timezone.utc)).total_seconds()))
    except Exception:
        return fallback


def _fmp_daily_attempt_key(tk, full=False):
    return f"fmp_daily_attempts_{_norm_symbol(tk)}_{int(bool(full))}_{datetime.utcnow().strftime('%Y%m%d')}"


def _fmp_daily_attempt_allowed(tk, full=False):
    cfg = _history_cfg()
    max_attempts = int(cfg.get("fmp_daily_max_retries_per_symbol_per_day", 2))
    key = _fmp_daily_attempt_key(tk, full=full)
    raw_attempts = cache_get(key, max_age=86400, default=0)
    if raw_attempts is CACHE_MISS or raw_attempts is None:
        attempts = 0
    else:
        try:
            attempts = int(raw_attempts or 0)
        except Exception:
            attempts = 0
    if attempts >= max_attempts:
        return False
    cache_set(key, attempts + 1)
    return True


def _daily_chain_entry(provider, endpoint=None, df=None, status=None, reason=None):
    attrs = getattr(df, "attrs", {}) or {}
    snap = api_failure_snapshot().get(endpoint or "", {}) if endpoint else {}
    out = {
        "provider": provider,
        "source": attrs.get("source") or provider,
        "status": status or attrs.get("status") or snap.get("status") or "ok",
    }
    if endpoint:
        out["endpoint"] = endpoint
    if reason:
        out["reason"] = reason
    if _valid_daily_df(df):
        out["rows"] = int(len(df))
    if attrs.get("warnings"):
        out["warnings"] = list(attrs.get("warnings") or [])
    if attrs.get("stale_daily_cache_age_hours") is not None:
        out["stale_daily_cache_age_hours"] = attrs.get("stale_daily_cache_age_hours")
    if snap.get("rate_limited"):
        out["rate_limited"] = True
    if provider == "fmp_daily":
        state = fmp_daily_global_circuit_state()
        if state.get("active"):
            out["cooldown_remaining_sec"] = state.get("cooldown_remaining_sec")
            out["last_429_age_sec"] = state.get("last_429_age_sec")
    if provider == "finnhub_daily" and endpoint:
        state = finnhub_daily_circuit_state(endpoint.split(":")[1], endpoint.endswith(":1"))
        global_state = finnhub_daily_global_circuit_state()
        out["circuit_open"] = bool(state.get("active") or global_state.get("active"))
        out["circuit_cooldown_remaining_sec"] = (
            global_state.get("cooldown_remaining_sec")
            if global_state.get("active")
            else state.get("cooldown_remaining_sec")
        )
        out["quote_endpoint_still_enabled"] = True
        if global_state.get("active"):
            out["global_circuit_open"] = True
    if snap.get("last_error"):
        out["provider_error_type"] = snap.get("status")
    return out


def _daily_failure_entry(provider, endpoint, default_status="empty_response"):
    snap = api_failure_snapshot().get(endpoint or "", {})
    status = snap.get("status") or default_status
    reason = None
    if provider == "fmp_daily" and fmp_daily_global_circuit_state().get("active"):
        status = "skipped_by_global_rate_limit"
    elif provider == "fmp_daily" and should_skip_api(endpoint):
        status = "skipped_by_circuit"
    elif provider == "finnhub_daily" and finnhub_daily_global_circuit_state().get("active"):
        status = "blocked_or_forbidden"
        reason = "FINNHUB_DAILY_GLOBAL_FORBIDDEN"
    return _daily_chain_entry(provider, endpoint=endpoint, status=status, reason=reason)


def _attach_provider_chain(df, chain):
    if _valid_daily_df(df):
        df.attrs["provider_chain_debug"] = list(chain or [])
    return df


def _stale_daily_cache(tk, full=False, include_stooq=False, stooq_prefix="stooq", provider_chain=None):
    fmp_state = fmp_daily_global_circuit_state()
    fmp_key = ("fmp_daily", f"fmp_daily_full_{tk}" if full else f"fmp_daily_{tk}")
    finnhub_key = ("finnhub_daily", f"fh_daily_full_{tk}" if full else f"fh_daily_{tk}")
    stooq_key = (
        f"{stooq_prefix}_daily",
        f"{stooq_prefix}_full_{tk}" if full else f"{stooq_prefix}_{tk}",
    )
    if fmp_state.get("active"):
        keys = [fmp_key]
        if include_stooq:
            keys.append(stooq_key)
        keys.append(finnhub_key)
    else:
        keys = []
        if include_stooq:
            keys.append(stooq_key)
        keys.extend((
            finnhub_key,
            fmp_key,
        ))
    max_age = _stale_daily_cache_max_age()
    for source, cache_key in keys:
        cached, age_sec = cache_get_stale(
            cache_key,
            max_age,
            default=CACHE_MISS,
        )
        if _valid_daily_df(cached):
            stale = cached.copy(deep=False)
            warnings = ["STALE_DAILY_CACHE_USED"]
            if fmp_state.get("active"):
                warnings.extend(["FMP_DAILY_RATE_LIMITED", "FMP_DAILY_GLOBAL_COOLDOWN"])
            stale.attrs.update({
                "source": f"stale_cache:{source}",
                "status": "stale_cache",
                "warnings": list(dict.fromkeys(warnings)),
                "stale_daily_cache_age_sec": int(age_sec or 0),
                "stale_daily_cache_age_hours": round(float(age_sec or 0) / 3600.0, 2),
            })
            chain = list(provider_chain or [])
            chain.append(_daily_chain_entry(
                "stale_cache",
                df=stale,
                status="stale_cache",
                reason=f"using_stale_{source}",
            ))
            stale.attrs["provider_chain_debug"] = chain
            return stale
    return None


def record_price(tk, price):
    append_price_snapshot(tk, price, min_interval=60, limit=6000)


def _annotate_quote_for_execution(quote, *, cache_used=False, age_sec=None,
                                  stale_reason=None):
    q = dict(quote or {})
    now = int(time.time())
    source = q.get("source") or "unknown"
    existing_trust = q.get("execution_trusted")
    try:
        price = float(q.get("price", 0) or 0)
    except Exception:
        price = 0.0
    embedded_age_sec = None
    for age_key in ("stale_age_sec", "quote_age_seconds"):
        if q.get(age_key) is None:
            continue
        try:
            age_val = float(q.get(age_key))
            if age_val >= 0:
                embedded_age_sec = max(age_val, embedded_age_sec or 0.0)
        except Exception:
            pass
    if age_sec is None:
        age_sec = embedded_age_sec
    elif embedded_age_sec is not None:
        try:
            age_sec = max(float(age_sec), embedded_age_sec)
        except Exception:
            age_sec = embedded_age_sec
    if age_sec is None and q.get("quote_timestamp"):
        try:
            age_sec = max(0.0, time.time() - float(q.get("quote_timestamp")))
        except Exception:
            age_sec = None
    age_int = int(age_sec) if age_sec is not None and age_sec >= 0 else None
    q["quote_timestamp"] = q.get("quote_timestamp") or now
    q["execution_cache_max_age_sec"] = EXECUTION_CACHE_MAX_AGE_SEC
    q["cache_used"] = bool(cache_used)
    q["quote_age_seconds"] = age_int
    q["quote_source"] = f"cache:{source}" if cache_used else source
    if price <= 0:
        q["execution_trusted"] = False
        q["stale"] = True
        q["stale_reason"] = stale_reason or "invalid_price"
    elif cache_used or q.get("stale"):
        trusted = bool(
            existing_trust is not False
            and age_int is not None
            and age_int <= EXECUTION_CACHE_MAX_AGE_SEC
        )
        q["execution_trusted"] = trusted
        q["stale"] = not trusted
        q["stale_reason"] = None if trusted else (
            stale_reason or (
                "quote_cache_missing_timestamp"
                if age_int is None
                else "quote_cache_older_than_execution_limit"
            )
        )
    else:
        q["execution_trusted"] = True
        q["stale"] = False
        q["stale_reason"] = None
    return q


def _direct_stooq_daily(tk, full=False, cache_prefix="stooq"):
    """Direct Stooq CSV fetch. Do not route through data_manager."""
    endpoint = f"{cache_prefix}_daily:{tk}:{int(bool(full))}"
    if should_skip_api(endpoint):
        return None
    cache_key = f"{cache_prefix}_full_{tk}" if full else f"{cache_prefix}_{tk}"
    c = cache_get(cache_key, max_age=300, default=CACHE_MISS)
    if c is not CACHE_MISS:
        if isinstance(c, pd.DataFrame) and not c.empty:
            c.attrs.setdefault("source", f"{cache_prefix}_daily")
            c.attrs.setdefault("provider", "stooq")
            c.attrs.setdefault("status", "ok")
            c.attrs["cache_used"] = True
            return c
        return None
    try:
        import urllib.request

        s = tk.lower().lstrip("^") + ".us"
        url = f"https://stooq.com/q/d/l/?s={s}&i=d"
        req = urllib.request.Request(url, headers={"User-Agent": "stock-tracker/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        if not raw or raw.startswith("No data") or "," not in raw:
            cache_set(cache_key, None)
            return None
        df = pd.read_csv(StringIO(raw))
        if df.empty or "Close" not in df.columns or "Date" not in df.columns:
            cache_set(cache_key, None)
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"]).set_index("Date").sort_index()
        if df.empty:
            cache_set(cache_key, None)
            return None
        if not full:
            df = df.tail(400)
        df.attrs["source"] = f"{cache_prefix}_daily"
        df.attrs["provider"] = "stooq"
        df.attrs["status"] = "ok"
        df.attrs["cache_used"] = False
        cache_set(cache_key, df)
        record_api_success(endpoint)
        return df
    except Exception as e:
        try:
            print(f"[stooq] {tk} fetch failed: {type(e).__name__}: {e}")
        except Exception:
            pass
        cache_set(cache_key, None)
        record_api_failure(endpoint, e)
        return None


def _finnhub_daily(tk, full=False, use_cache=True):
    """Daily OHLCV from Finnhub candles. Primary PA free-tier history source."""
    tk = _norm_symbol(tk)
    endpoint = f"finnhub_daily:{tk}:{int(bool(full))}"
    cache_key = f"fh_daily_full_{tk}" if full else f"fh_daily_{tk}"
    # Serve valid cache first — the circuits only gate NEW network calls, not
    # already-cached data.
    if use_cache:
        c = cache_get(cache_key, max_age=_finnhub_daily_cache_max_age(), default=CACHE_MISS)
        if c is not CACHE_MISS:
            if isinstance(c, pd.DataFrame) and not c.empty:
                c.attrs.setdefault("source", "finnhub_daily")
                c.attrs.setdefault("provider", "finnhub")
                c.attrs.setdefault("status", "ok")
                c.attrs["cache_used"] = True
                c.attrs["cache_max_age_sec"] = _finnhub_daily_cache_max_age()
                return c
            return None
    # Entry 027 — endpoint-global forbidden circuit: a prior 401/403 means the
    # key has no candle access for ANY symbol, so skip the network call for all.
    if finnhub_daily_global_circuit_state().get("active"):
        return None
    circuit = finnhub_daily_circuit_state(tk, full=full)
    if circuit.get("active"):
        return None
    try:
        end = int(time.time())
        days = 3650 if full else 600
        start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        raw = fh.stock_candles(tk, "D", start, end) or {}
        if raw.get("s") != "ok" or not raw.get("c"):
            status = "blocked_or_forbidden" if _blocked_or_forbidden_payload(raw) else "empty_response"
            record_api_failure(
                endpoint,
                raw or "empty Finnhub daily candles",
                status=status,
                cooldown_sec=_finnhub_daily_cooldown_for_status(status),
            )
            if status == "blocked_or_forbidden":
                _open_finnhub_daily_global_forbidden_circuit(
                    raw or "Finnhub daily candles forbidden"
                )
            return None
        df = pd.DataFrame({
            "Open": raw.get("o", []),
            "High": raw.get("h", []),
            "Low": raw.get("l", []),
            "Close": raw.get("c", []),
            "Volume": raw.get("v", []),
        }, index=pd.to_datetime(raw.get("t", []), unit="s"))
        df.index.name = "Date"
        df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
        if df.empty:
            record_api_failure(
                endpoint,
                "empty Finnhub daily frame",
                status="empty_response",
                cooldown_sec=_finnhub_daily_cooldown_for_status("empty_response"),
            )
            return None
        if not full:
            df = df.tail(400)
        df.attrs["source"] = "finnhub_daily"
        df.attrs["provider"] = "finnhub"
        df.attrs["status"] = "ok"
        df.attrs["cache_used"] = False
        if use_cache:
            cache_set(cache_key, df)
        record_api_success(endpoint)
        return df
    except Exception as e:
        try:
            print(f"[finnhub-daily] {tk} failed: {type(e).__name__}: {e}")
        except Exception:
            pass
        status = _classify_finnhub_daily_error(e)
        record_api_failure(
            endpoint,
            e,
            status=status,
            cooldown_sec=_finnhub_daily_cooldown_for_status(status),
        )
        if status == "blocked_or_forbidden":
            _open_finnhub_daily_global_forbidden_circuit(e)
        return None


def _fmp_daily(tk, full=False, use_cache=True):
    """
    Daily OHLCV from Financial Modeling Prep.

    This is a daily/history fallback only; it is not a quote, news, or intraday
    provider for live trading ticks.
    """
    tk = _norm_symbol(tk)
    endpoint = f"fmp_daily:{tk}:{int(bool(full))}"
    if not FMP_KEY:
        record_api_failure(endpoint, "FMP_KEY/FMP_API_KEY is not configured", status="skipped_missing_key")
        return None
    cache_key = f"fmp_daily_full_{tk}" if full else f"fmp_daily_{tk}"
    if use_cache:
        c = cache_get(cache_key, max_age=_fmp_daily_cache_max_age(), default=CACHE_MISS)
        if c is not CACHE_MISS:
            if isinstance(c, pd.DataFrame) and not c.empty:
                c.attrs.setdefault("source", "fmp_daily")
                c.attrs.setdefault("provider", "fmp")
                c.attrs.setdefault("status", "ok")
                c.attrs["cache_used"] = True
                c.attrs["cache_max_age_sec"] = _fmp_daily_cache_max_age()
                return c
            return None
    if fmp_daily_global_circuit_state().get("active"):
        record_api_failure(endpoint, "FMP daily global rate-limit cooldown", status="skipped_by_global_rate_limit")
        return None
    if should_skip_api(endpoint):
        return None
    if not _fmp_daily_attempt_allowed(tk, full=full):
        record_api_failure(endpoint, "FMP daily symbol retry limit reached", status="skipped_by_circuit")
        return None
    try:
        params = urllib.parse.urlencode({"symbol": str(tk).upper(), "apikey": FMP_KEY})
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "stock-tracker/1.0"})
        timeout_sec = int(_history_cfg().get("provider_history_request_timeout_sec", 5))
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore") or "null")

        rows = payload
        if isinstance(payload, dict):
            rows = (
                payload.get("historical")
                or payload.get("data")
                or payload.get("results")
                or payload.get("historicalPriceFull")
                or []
            )
            if isinstance(rows, dict):
                rows = rows.get("historical") or rows.get("data") or []
        if not isinstance(rows, list) or not rows:
            record_api_failure(endpoint, "empty FMP daily response", status="empty_response")
            return None

        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized.append({
                "Date": row.get("date") or row.get("Date"),
                "Open": row.get("open") if row.get("open") is not None else row.get("Open"),
                "High": row.get("high") if row.get("high") is not None else row.get("High"),
                "Low": row.get("low") if row.get("low") is not None else row.get("Low"),
                "Close": row.get("close") if row.get("close") is not None else row.get("Close"),
                "Volume": row.get("volume") if row.get("volume") is not None else row.get("Volume"),
            })

        df = pd.DataFrame(normalized)
        if df.empty or "Date" not in df.columns:
            record_api_failure(endpoint, "empty FMP daily frame", status="empty_response")
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if df.empty:
            record_api_failure(endpoint, "empty FMP daily normalized frame", status="empty_response")
            return None
        if not full:
            df = df.tail(400)
        df.attrs["source"] = "fmp_daily"
        df.attrs["provider"] = "fmp"
        df.attrs["status"] = "ok"
        df.attrs["cache_used"] = False
        if use_cache:
            cache_set(cache_key, df)
        record_api_success(endpoint)
        record_api_success(FMP_DAILY_GLOBAL_ENDPOINT)
        return df
    except Exception as e:
        try:
            print(f"[fmp-daily] {tk} failed: {type(e).__name__}: {sanitize_provider_error(e)}")
        except Exception:
            pass
        cooldown_sec = None
        try:
            status_code = getattr(e, "code", None) or getattr(e, "status", None)
        except Exception:
            status_code = None
        if str(status_code) == "429":
            cooldown_sec = _retry_after_cooldown_sec(e)
        elif str(status_code) and str(status_code) != "None":
            cooldown_sec = int(_history_cfg().get("fmp_daily_symbol_error_cooldown_sec", 900))
        rec = record_api_failure(endpoint, e, cooldown_sec=cooldown_sec)
        if (rec or {}).get("status") == "rate_limited":
            record_api_failure(
                FMP_DAILY_GLOBAL_ENDPOINT,
                e,
                status="rate_limited",
                cooldown_sec=cooldown_sec or FMP_DAILY_RATE_LIMIT_COOLDOWN_SEC,
            )
        return None


def _raw_daily(tk, full=False):
    """Daily bars for normal ticker signal history."""
    tk = _norm_symbol(tk)
    finnhub_endpoint = f"finnhub_daily:{tk}:{int(bool(full))}"
    fmp_endpoint = f"fmp_daily:{tk}:{int(bool(full))}"
    chain = []
    if PYTHONANYWHERE_MODE:
        cached = _cached_daily_history(tk, full=full, provider_chain=chain)
        if _valid_daily_df(cached):
            return _attach_provider_chain(cached, cached.attrs.get("provider_chain_debug"))
        if not _consume_history_fetch_budget():
            chain.append(_daily_chain_entry("history_fetch_budget", status="skipped", reason="max_history_fetches_per_tick"))
            return _stale_daily_cache(tk, full=full, provider_chain=chain)
        df = _finnhub_daily(tk, full=full)
        if _valid_daily_df(df):
            chain.append(_daily_chain_entry("finnhub_daily", endpoint=finnhub_endpoint, df=df))
            return _attach_provider_chain(df, chain)
        chain.append(_daily_failure_entry("finnhub_daily", finnhub_endpoint))
        df = _fmp_daily(tk, full=full)
        if _valid_daily_df(df):
            chain.append(_daily_chain_entry("fmp_daily", endpoint=fmp_endpoint, df=df))
            return _attach_provider_chain(df, chain)
        chain.append(_daily_failure_entry("fmp_daily", fmp_endpoint))
        return _stale_daily_cache(tk, full=full, provider_chain=chain)

    stooq_endpoint = f"stooq_daily:{tk}:{int(bool(full))}"
    cached = _cached_daily_history(tk, full=full, include_stooq=True, provider_chain=chain)
    if _valid_daily_df(cached):
        return _attach_provider_chain(cached, cached.attrs.get("provider_chain_debug"))
    if not _consume_history_fetch_budget():
        chain.append(_daily_chain_entry("history_fetch_budget", status="skipped", reason="max_history_fetches_per_tick"))
        return _stale_daily_cache(tk, full=full, include_stooq=True, provider_chain=chain)
    df = _direct_stooq_daily(tk, full=full)
    if _valid_daily_df(df):
        chain.append(_daily_chain_entry("stooq_daily", endpoint=stooq_endpoint, df=df))
        return _attach_provider_chain(df, chain)
    chain.append(_daily_failure_entry("stooq_daily", stooq_endpoint))
    df = _finnhub_daily(tk, full=full)
    if _valid_daily_df(df):
        chain.append(_daily_chain_entry("finnhub_daily", endpoint=finnhub_endpoint, df=df))
        return _attach_provider_chain(df, chain)
    chain.append(_daily_failure_entry("finnhub_daily", finnhub_endpoint))
    df = _fmp_daily(tk, full=full)
    if _valid_daily_df(df):
        chain.append(_daily_chain_entry("fmp_daily", endpoint=fmp_endpoint, df=df))
        return _attach_provider_chain(df, chain)
    chain.append(_daily_failure_entry("fmp_daily", fmp_endpoint))
    return _stale_daily_cache(tk, full=full, include_stooq=True, provider_chain=chain)


def get_regime_daily(tk, full=False):
    """
    Return daily bars for regime/proxy symbols.

    PythonAnywhere free skips Stooq entirely, uses Finnhub first, then optional
    FMP daily fallback, then a visibly stale successful cache within 72 hours.
    Local/off-PA may use Stooq first with Finnhub/FMP fallback.
    """
    if PYTHONANYWHERE_MODE:
        finnhub_endpoint = f"finnhub_daily:{tk}:{int(bool(full))}"
        fmp_endpoint = f"fmp_daily:{tk}:{int(bool(full))}"
        chain = []
        df = _finnhub_daily(tk, full=full)
        if _valid_daily_df(df):
            chain.append(_daily_chain_entry("finnhub_daily", endpoint=finnhub_endpoint, df=df))
            return _attach_provider_chain(df, chain)
        chain.append(_daily_failure_entry("finnhub_daily", finnhub_endpoint))

        df = _fmp_daily(tk, full=full)
        if _valid_daily_df(df):
            chain.append(_daily_chain_entry("fmp_daily", endpoint=fmp_endpoint, df=df))
            return _attach_provider_chain(df, chain)
        chain.append(_daily_failure_entry("fmp_daily", fmp_endpoint))

        return _stale_daily_cache(tk, full=full, provider_chain=chain)

    stooq_endpoint = f"stooq_regime_daily:{tk}:{int(bool(full))}"
    finnhub_endpoint = f"finnhub_daily:{tk}:{int(bool(full))}"
    fmp_endpoint = f"fmp_daily:{tk}:{int(bool(full))}"
    chain = []
    df = _direct_stooq_daily(tk, full=full, cache_prefix="stooq_regime")
    if _valid_daily_df(df):
        chain.append(_daily_chain_entry("stooq_regime_daily", endpoint=stooq_endpoint, df=df))
        return _attach_provider_chain(df, chain)
    chain.append(_daily_failure_entry("stooq_regime_daily", stooq_endpoint))

    df = _finnhub_daily(tk, full=full)
    if _valid_daily_df(df):
        chain.append(_daily_chain_entry("finnhub_daily", endpoint=finnhub_endpoint, df=df))
        return _attach_provider_chain(df, chain)
    chain.append(_daily_failure_entry("finnhub_daily", finnhub_endpoint))

    df = _fmp_daily(tk, full=full)
    if _valid_daily_df(df):
        chain.append(_daily_chain_entry("fmp_daily", endpoint=fmp_endpoint, df=df))
        return _attach_provider_chain(df, chain)
    chain.append(_daily_failure_entry("fmp_daily", fmp_endpoint))

    return _stale_daily_cache(
        tk,
        full=full,
        include_stooq=True,
        stooq_prefix="stooq_regime",
        provider_chain=chain,
    )


def _stooq_daily(tk, full=False):
    return _managed_daily(tk, full=full)


def _append_live_bar(df, tk):
    """Apply today's fresh quote to daily bars without inventing volume."""
    if df is None or df.empty:
        return df
    try:
        from utils.time_utils import is_market_open

        cfg = _history_cfg()
        df.attrs["live_bar_applied"] = False
        df.attrs["live_bar_reason"] = "market_closed"
        df.attrs["live_quote_overlay_enabled"] = bool(cfg.get("live_quote_overlay_enabled", True))
        df.attrs["base_history_rows_before_live_overlay"] = int(len(df))
        df.attrs["history_rows_after_live_overlay"] = int(len(df))
        df.attrs["quote_fresh"] = None
        if not df.attrs["live_quote_overlay_enabled"]:
            df.attrs["live_bar_reason"] = "live_quote_overlay_disabled"
            return df
        if len(df) < int(cfg.get("live_quote_overlay_min_base_history_rows", 25)):
            df.attrs["live_bar_reason"] = "base_history_rows_below_overlay_min"
            return df
        if not is_market_open():
            return df
        today = pd.Timestamp(datetime.now().date())
        q = get_quote(tk) or {}
        live = q.get("price") or 0
        quote_fresh = bool(
            live > 0
            and (
                q.get("execution_trusted")
                if cfg.get("live_quote_overlay_source_must_be_execution_trusted", True)
                else not q.get("stale")
            )
        )
        df.attrs["quote_fresh"] = quote_fresh
        df.attrs["live_quote_overlay_source"] = q.get("quote_source") or q.get("source")
        if not quote_fresh:
            df.attrs["live_bar_reason"] = "quote_stale_or_missing"
            return df
        live = float(live)

        idx_norm = pd.to_datetime(df.index, errors="coerce").normalize()
        today_positions = [i for i, dt in enumerate(idx_norm) if dt == today]
        if today_positions:
            row_idx = df.index[today_positions[-1]]
            if "Close" in df.columns:
                df.loc[row_idx, "Close"] = live
            if "High" in df.columns:
                cur_high = pd.to_numeric(pd.Series([df.loc[row_idx, "High"]]), errors="coerce").iloc[0]
                df.loc[row_idx, "High"] = max(float(cur_high), live) if pd.notna(cur_high) and cur_high > 0 else live
            if "Low" in df.columns:
                cur_low = pd.to_numeric(pd.Series([df.loc[row_idx, "Low"]]), errors="coerce").iloc[0]
                df.loc[row_idx, "Low"] = min(float(cur_low), live) if pd.notna(cur_low) and cur_low > 0 else live
            df.attrs["live_bar_applied"] = True
            df.attrs["live_bar_reason"] = "updated_today_row"
            df.attrs["history_rows_after_live_overlay"] = int(len(df))
            return df

        o_hi = q.get("high") if q.get("high") and q.get("high") > 0 else live
        o_lo = q.get("low") if q.get("low") and q.get("low") > 0 else live
        o_op = q.get("open") if q.get("open") and q.get("open") > 0 else live
        o_vol = q.get("volume") if q.get("volume") and q.get("volume") > 0 else float("nan")
        row = {}
        for col in df.columns:
            if col == "Volume":
                row[col] = o_vol
            elif col == "High":
                row[col] = float(o_hi)
            elif col == "Low":
                row[col] = float(o_lo)
            elif col == "Open":
                row[col] = float(o_op)
            else:
                row[col] = float(live)
        df.loc[today] = row
        df.attrs["live_bar_applied"] = True
        df.attrs["live_bar_reason"] = "appended_today_row"
        df.attrs["history_rows_after_live_overlay"] = int(len(df))
    except Exception as e:
        df.attrs["live_bar_applied"] = False
        df.attrs["live_bar_reason"] = f"error:{type(e).__name__}"
        try:
            print(f"[live-bar] {tk}: {type(e).__name__}: {e}")
        except Exception:
            pass
    return df


def _fetch_quote_once(tk):
    """Single quote attempt: Finnhub primary, yfinance fallback off-PA."""
    endpoint = f"quote:{tk}"
    if should_skip_api(endpoint, cooldown_sec=120):
        return {"price": 0, "change": 0, "pct": 0, "high": 0, "low": 0,
                "open": 0, "prev": 0, "source": "finnhub_quote",
                "provider_attempts": [{
                    "provider": "finnhub",
                    "kind": "quote",
                    "status": "circuit_skipped",
                }],
                "provider_error_type": "circuit_skipped"}
    try:
        attempts = []
        q = fh.quote(tk)
        r = {"price": q.get("c", 0), "change": q.get("d", 0), "pct": q.get("dp", 0),
             "high": q.get("h", 0), "low": q.get("l", 0), "open": q.get("o", 0),
             "prev": q.get("pc", 0), "source": "finnhub_quote"}
        attempts.append({
            "provider": "finnhub",
            "kind": "quote",
            "status": "ok" if r["price"] > 0 else "zero_price",
        })
        if r["price"] == 0 and not PYTHONANYWHERE_MODE:
            import yfinance as yf
            h = yf.Ticker(tk).history(period="2d")
            if not h.empty:
                cur = round(float(h["Close"].iloc[-1]), 2)
                prev = round(float(h["Close"].iloc[-2]), 2) if len(h) >= 2 else cur
                r["price"] = cur
                r["prev"] = prev
                r["change"] = round(cur - prev, 2)
                r["pct"] = round((cur - prev) / prev * 100, 2) if prev else 0
                r["source"] = "yfinance_quote"
                attempts.append({
                    "provider": "yfinance",
                    "kind": "quote",
                    "status": "ok",
                })
            else:
                attempts.append({
                    "provider": "yfinance",
                    "kind": "quote",
                    "status": "empty",
                })
        r["provider_attempts"] = attempts
        r["provider_used"] = r.get("source")
        if r["price"] > 0:
            record_api_success(endpoint)
        return r
    except Exception as e:
        record_api_failure(endpoint, e)
        return {"price": 0, "change": 0, "pct": 0, "high": 0, "low": 0,
                "open": 0, "prev": 0, "source": "finnhub_quote",
                "provider_attempts": [{
                    "provider": "finnhub",
                    "kind": "quote",
                    "status": "error",
                    "error_type": type(e).__name__,
                }],
                "provider_error_type": type(e).__name__}


def get_quote(tk):
    cache_key = f"q_{tk}"
    c, c_age = cache_get_stale(cache_key, 60, default=CACHE_MISS)
    if c is not CACHE_MISS:
        return _annotate_quote_for_execution(c, cache_used=True, age_sec=c_age)
    r = _fetch_quote_once(tk)
    if r["price"] == 0:
        try:
            time.sleep(0.5)
        except Exception:
            pass
        r = _fetch_quote_once(tk)
    if r["price"] == 0:
        cached, cached_age = cache_get_stale(
            cache_key,
            EXECUTION_CACHE_MAX_AGE_SEC,
            default=CACHE_MISS,
        )
        if cached is not CACHE_MISS:
            annotated = _annotate_quote_for_execution(
                cached,
                cache_used=True,
                age_sec=cached_age,
            )
            if annotated.get("execution_trusted"):
                return annotated
        pts = load_price_hist().get(tk, [])
        if pts:
            last_ts, last_price = pts[-1][0], pts[-1][1]
            r["price"] = last_price
            r["prev"] = last_price
            r["source"] = "recorded_price"
            r["stale"] = True
            r["stale_age_sec"] = int(time.time()) - int(last_ts)
            try:
                print(f"[get_quote] {tk}: using last recorded ${last_price}")
            except Exception:
                pass
        else:
            r["stale"] = True
            r["stale_age_sec"] = -1
            r["source"] = "missing"
            try:
                print(f"[get_quote] {tk}: total failure, no recorded history")
            except Exception:
                pass
    annotated = _annotate_quote_for_execution(
        r,
        cache_used=bool(r.get("source") == "recorded_price"),
        age_sec=r.get("stale_age_sec") if r.get("source") == "recorded_price" else 0,
    )
    cache_set(cache_key, annotated)
    if annotated["price"] > 0 and annotated.get("execution_trusted"):
        record_price(tk, annotated["price"])
    return annotated


def is_valid_ticker(t):
    """Resolve ticker through Finnhub quote, daily bars, then yfinance off-PA."""
    try:
        q = fh.quote(t) or {}
        if (q.get("c") or 0) > 0:
            return True
    except Exception:
        pass
    try:
        df = _stooq_daily(t)
        if df is not None and not df.empty:
            return True
    except Exception:
        pass
    if PYTHONANYWHERE_MODE:
        return False
    try:
        import yfinance as yf
        h = yf.Ticker(t).history(period="5d")
        return not h.empty
    except Exception:
        return False
