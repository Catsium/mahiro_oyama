"""Quote fetching, ticker validation, and daily-bar helpers."""
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import StringIO
import json
import urllib.parse
import urllib.request

import pandas as pd

from market import fh
from market.data_manager import get_daily as _managed_daily
from utils.cache import (
    CACHE_MISS, cache_get, cache_get_stale, cache_set, record_api_failure, record_api_success,
    sanitize_provider_error, should_skip_api, api_cooldown_state, api_failure_snapshot,
)
from utils.deploy_config import FMP_KEY, PYTHONANYWHERE_MODE
from utils.storage import load_price_hist, append_price_snapshot
from utils.time_utils import is_market_open

FMP_DAILY_GLOBAL_ENDPOINT = "fmp_daily:global"
FMP_DAILY_RATE_LIMIT_COOLDOWN_SEC = 30 * 60
FMP_DAILY_CACHE_MARKET_SEC = 60 * 60
FMP_DAILY_CACHE_CLOSED_SEC = 24 * 3600
STALE_DAILY_CACHE_MARKET_SEC = 24 * 3600
STALE_DAILY_CACHE_CLOSED_SEC = 72 * 3600
REGIME_STALE_CACHE_MAX_HOURS = 72
REGIME_STALE_CACHE_MAX_SEC = STALE_DAILY_CACHE_CLOSED_SEC


def _valid_daily_df(df):
    return (
        df is not None
        and isinstance(df, pd.DataFrame)
        and not df.empty
        and "Close" in df.columns
    )


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
    return out


def _daily_failure_entry(provider, endpoint, default_status="empty_response"):
    snap = api_failure_snapshot().get(endpoint or "", {})
    status = snap.get("status") or default_status
    if provider == "fmp_daily" and fmp_daily_global_circuit_state().get("active"):
        status = "skipped_by_global_rate_limit"
    elif provider == "fmp_daily" and should_skip_api(endpoint):
        status = "skipped_by_circuit"
    return _daily_chain_entry(provider, endpoint=endpoint, status=status)


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
    endpoint = f"finnhub_daily:{tk}:{int(bool(full))}"
    if should_skip_api(endpoint):
        return None
    cache_key = f"fh_daily_full_{tk}" if full else f"fh_daily_{tk}"
    if use_cache:
        c = cache_get(cache_key, max_age=6 * 3600, default=CACHE_MISS)
        if c is not CACHE_MISS:
            if isinstance(c, pd.DataFrame) and not c.empty:
                c.attrs.setdefault("source", "finnhub_daily")
                c.attrs.setdefault("provider", "finnhub")
                c.attrs.setdefault("status", "ok")
                c.attrs["cache_used"] = True
                c.attrs["cache_max_age_sec"] = 6 * 3600
                return c
            return None
    try:
        end = int(time.time())
        days = 3650 if full else 600
        start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        raw = fh.stock_candles(tk, "D", start, end) or {}
        if raw.get("s") != "ok" or not raw.get("c"):
            status = "blocked_or_forbidden" if _blocked_or_forbidden_payload(raw) else "empty_response"
            record_api_failure(endpoint, raw or "empty Finnhub daily candles", status=status)
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
            record_api_failure(endpoint, "empty Finnhub daily frame", status="empty_response")
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
        record_api_failure(endpoint, e)
        return None


def _fmp_daily(tk, full=False, use_cache=True):
    """
    Daily OHLCV from Financial Modeling Prep.

    This is a daily/history fallback only; it is not a quote, news, or intraday
    provider for live trading ticks.
    """
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
    try:
        params = urllib.parse.urlencode({"symbol": str(tk).upper(), "apikey": FMP_KEY})
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "stock-tracker/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
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
    finnhub_endpoint = f"finnhub_daily:{tk}:{int(bool(full))}"
    fmp_endpoint = f"fmp_daily:{tk}:{int(bool(full))}"
    chain = []
    if PYTHONANYWHERE_MODE:
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

        df.attrs["live_bar_applied"] = False
        df.attrs["live_bar_reason"] = "market_closed"
        df.attrs["quote_fresh"] = None
        if not is_market_open():
            return df
        today = pd.Timestamp(datetime.now().date())
        q = get_quote(tk) or {}
        live = q.get("price") or 0
        quote_fresh = bool(live > 0 and not q.get("stale"))
        df.attrs["quote_fresh"] = quote_fresh
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
                "open": 0, "prev": 0, "source": "finnhub_quote"}
    try:
        q = fh.quote(tk)
        r = {"price": q.get("c", 0), "change": q.get("d", 0), "pct": q.get("dp", 0),
             "high": q.get("h", 0), "low": q.get("l", 0), "open": q.get("o", 0),
             "prev": q.get("pc", 0), "source": "finnhub_quote"}
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
        if r["price"] > 0:
            record_api_success(endpoint)
        return r
    except Exception as e:
        record_api_failure(endpoint, e)
        return {"price": 0, "change": 0, "pct": 0, "high": 0, "low": 0,
                "open": 0, "prev": 0, "source": "finnhub_quote"}


def get_quote(tk):
    c = cache_get(f"q_{tk}")
    if c:
        return c
    r = _fetch_quote_once(tk)
    if r["price"] == 0:
        try:
            time.sleep(0.5)
        except Exception:
            pass
        r = _fetch_quote_once(tk)
    if r["price"] == 0:
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
    cache_set(f"q_{tk}", r)
    if r["price"] > 0 and not r.get("stale"):
        record_price(tk, r["price"])
    return r


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
