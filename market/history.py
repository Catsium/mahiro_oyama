"""Daily history + intraday context fetchers.

`get_history` builds the technical-indicator ctx dict (used by signals/risk/bot).
Order: managed daily bars (provider chain with live-bar) then recorded
snapshots. Output ctx also gets enriched with weekly_posture (Round 1 #1.4)
and sector_relative_strength (Round 1 #1.3).

`get_intraday_context` returns 15-min ATR/RSI/VWMA-distance. Zero-filled when
yfinance is blocked (i.e. on PythonAnywhere) so consumers gracefully fall back
to the regime-static defaults.
"""
import pandas as pd

from market.data_manager import get_daily as _daily_bars
from market.quotes import _append_live_bar, set_history_fetch_budget
from trading.config import DEFAULT_CONFIG
from trading.indicators import (
    _ctx_from_series, _ctx_from_recorded, weekly_posture, sector_relative_strength, _safe
)
from utils.cache import api_failure_snapshot, cache_get, cache_set
from utils.deploy_config import PYTHONANYWHERE_MODE
from utils.time_utils import is_market_open


def _norm_symbol(tk):
    return str(tk or "").upper().strip()


def _recorded_last_date_str(ctx):
    """ET calendar date ('YYYY-MM-DD') of the newest recorded snapshot, or None."""
    ts = ctx.get("recorded_last_ts")
    if not ts:
        return None
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        return datetime.fromtimestamp(int(ts), ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")


def _recorded_ctx_trusted(ctx):
    """Audit P1-7 trust rule: ≥25 recorded closes AND newest point < 2
    completed trading days old → usable as synthetic daily history."""
    min_points = int(DEFAULT_CONFIG.get("history", {}).get("min_history_rows_for_buy", 25))
    if int(ctx.get("recorded_points_n") or 0) < min_points:
        return False
    date_str = _recorded_last_date_str(ctx)
    if not date_str:
        return False
    from datetime import datetime
    from utils.time_utils import completed_trading_days_since
    try:
        last_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        return completed_trading_days_since(last_date) < 2
    except Exception:
        return False


def _history_meta_from_df(df):
    attrs = getattr(df, "attrs", {}) or {}
    chain = list(attrs.get("provider_chain_debug") or [])
    providers = [row for row in chain if str(row.get("provider", "")).endswith("_daily")]
    failures = [row for row in providers if row.get("status") not in {"ok", "cache"}]
    out = {
        "history_source": attrs.get("source") or "daily",
        "history_status": attrs.get("status") or "ok",
        "history_provider": attrs.get("provider"),
        "history_rows": int(len(df)) if df is not None else 0,
        "history_warnings": list(attrs.get("warnings") or []),
        "history_cache_used": bool(attrs.get("history_cache_used") or attrs.get("cache_used")),
        "history_cache_age_completed_trading_days": attrs.get("history_cache_age_completed_trading_days"),
        "history_cache_valid": attrs.get("history_cache_valid"),
        "history_cache_invalid_reason": attrs.get("history_cache_invalid_reason"),
        "history_fetch_attempted": bool(providers and not attrs.get("history_cache_used")),
        "history_fetch_skipped_reason": None,
        "history_provider_used": attrs.get("provider"),
        "history_provider_error_type": failures[-1].get("provider_error_type") if failures else None,
        "live_bar_applied": bool(attrs.get("live_bar_applied", False)),
        "live_bar_reason": attrs.get("live_bar_reason"),
        "live_quote_overlay_enabled": attrs.get("live_quote_overlay_enabled"),
        "live_quote_overlay_source": attrs.get("live_quote_overlay_source"),
        "base_history_rows_before_live_overlay": attrs.get("base_history_rows_before_live_overlay"),
        "history_rows_after_live_overlay": attrs.get("history_rows_after_live_overlay"),
        "quote_fresh": attrs.get("quote_fresh"),
    }
    try:
        out["history_last_date"] = str(df.index[-1].date())
    except Exception:
        out["history_last_date"] = None
    if attrs.get("stale_daily_cache_age_hours") is not None:
        out["stale_daily_cache_age_hours"] = attrs.get("stale_daily_cache_age_hours")
        out["stale_daily_cache_age_sec"] = attrs.get("stale_daily_cache_age_sec")
    if attrs.get("provider_chain_debug"):
        out["provider_chain_debug"] = list(attrs.get("provider_chain_debug") or [])
        budget_rows = [
            row for row in out["provider_chain_debug"]
            if row.get("provider") == "history_fetch_budget"
        ]
        if budget_rows:
            out["history_fetch_skipped_reason"] = budget_rows[-1].get("reason")
    return out


def _missing_history(tk, skipped_reason=None):
    return {
        "history_source": "missing",
        "history_status": "missing",
        "history_rows": 0,
        "history_last_date": None,
        "history_cache_used": False,
        "history_cache_valid": False,
        "history_cache_invalid_reason": "missing",
        "history_fetch_attempted": skipped_reason is None,
        "history_fetch_skipped_reason": skipped_reason,
        "history_provider_used": None,
        "history_provider_error_type": None,
        "history_warnings": ["MISSING_HISTORY"],
    }


def get_history(tk):
    """Daily history + weekly posture. Cached 5 min."""
    tk = _norm_symbol(tk)
    c = cache_get(f"h_{tk}", max_age=300)
    if c is not None:
        try:
            cached_rows = int((c if isinstance(c, dict) else {}).get("history_rows") or 0)
        except Exception:
            cached_rows = 0
        if cached_rows <= 0:
            c = None
    if c is not None:
        return c
    r = {}
    used_df = None
    # Managed daily bars (copy so live-bar does not mutate cached frames).
    if not r:
        try:
            df = _daily_bars(tk)
            if df is not None and not df.empty:
                df = df.copy()
                df = _append_live_bar(df, tk)
                r = _ctx_from_series(df["Close"].round(2), df=df)
                if r:
                    r.update(_history_meta_from_df(df))
                used_df = df
        except Exception as e:
            r = _missing_history(tk)
            r["history_provider_error_type"] = type(e).__name__
    # Locally recorded snapshots (last resort; no weekly possible).
    if not r:
        r = _ctx_from_recorded(tk)
        if r and _recorded_ctx_trusted(r):
            # Audit P1-7: ≥25 recorded closes with a fresh newest point make a
            # decidable ctx — history_rows unlocks buys via the normal
            # _history_execution_status chokepoint; label keeps provenance.
            r["history_source"] = "synthetic_recorded"
            r["history_status"] = "synthetic_recorded"
            r["history_rows"] = int(r.get("recorded_points_n") or 0)
            r["history_last_date"] = _recorded_last_date_str(r)
            r.setdefault("history_warnings", []).append("SYNTHETIC_RECORDED_HISTORY")
        elif r:
            r.setdefault("history_source", "recorded")
            r.setdefault("history_status", "recorded_fallback")
            r.setdefault("history_rows", 0)
            r.setdefault("history_last_date", None)
    # #1.4: weekly posture
    if r and used_df is not None:
        wk = weekly_posture(used_df)
        if wk:
            r.update(wk)
    # #1.3: sector-relative strength (lazy import of get_sector to break cycle)
    if r and r.get("history_status") != "missing":
        from trading.risk import get_sector
        sec = get_sector(tk)
        rs = sector_relative_strength(tk, sec, lookback_days=20)
        if rs:
            r.update(rs)
    if r and r.get("history_status") != "missing" and int(r.get("history_rows") or 0) > 0:
        cache_set(f"h_{tk}", r)
        return r
    return r or _missing_history(tk)


def warm_history(symbols, max_symbols=None, max_fetches=None):
    cfg = DEFAULT_CONFIG.get("history", {})
    max_symbols = int(max_symbols or cfg.get("max_symbols_per_warm_call", 3))
    max_fetches = int(max_fetches or cfg.get("max_history_fetches_per_warm_call", 3))
    requested = [_norm_symbol(s) for s in (symbols or []) if _norm_symbol(s)]
    attempted = []
    warmed = []
    cache_hits = []
    failed = []
    skipped = []
    provider_used = {}
    rows_by_symbol = {}
    errors = {}
    set_history_fetch_budget(max_fetches)
    try:
        for sym in requested[:max_symbols]:
            attempted.append(sym)
            ctx = get_history(sym)
            rows = int(ctx.get("history_rows") or 0)
            rows_by_symbol[sym] = rows
            if ctx.get("history_cache_used"):
                cache_hits.append(sym)
            if rows >= int(cfg.get("min_history_rows_for_buy", 25)):
                warmed.append(sym)
                provider_used[sym] = ctx.get("history_source")
            else:
                failed.append(sym)
                errors[sym] = ctx.get("history_fetch_skipped_reason") or ctx.get("history_status") or "MISSING_HISTORY"
        for sym in requested[max_symbols:]:
            skipped.append({"symbol": sym, "reason": "max_symbols_per_warm_call"})
    finally:
        set_history_fetch_budget(None)
    return {
        "requested_symbols": requested,
        "attempted_symbols": attempted,
        "warmed_symbols": warmed,
        "skipped_symbols": skipped,
        "failed_symbols": failed,
        "cache_hit_symbols": cache_hits,
        "provider_used_by_symbol": provider_used,
        "rows_by_symbol": rows_by_symbol,
        "errors_by_symbol": errors,
        "provider_circuits": api_failure_snapshot(),
    }


def get_intraday_context(tk):
    """Returns {intra_atr_pct, intra_rsi, intra_vwma_dist}. Zero-filled when
    market is closed OR when yfinance fails (consumers naturally fall back)."""
    c = cache_get(f"intra_{tk}", max_age=300)
    if c is not None:
        return c
    r = {"intra_atr_pct": 0.0, "intra_rsi": 50.0, "intra_vwma_dist": 0.0}
    if PYTHONANYWHERE_MODE or not is_market_open():
        cache_set(f"intra_{tk}", r)
        return r
    try:
        import yfinance as yf
        h = yf.Ticker(tk).history(period="5d", interval="15m")
        if h.empty or len(h) < 15:
            cache_set(f"intra_{tk}", r)
            return r
        cl = h["Close"].round(2)
        cur = float(cl.iloc[-1])
        # ATR(14) as % of price
        tr = pd.concat([
            (h["High"] - h["Low"]),
            (h["High"] - h["Close"].shift()).abs(),
            (h["Low"]  - h["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_v = _safe(tr.rolling(14).mean().iloc[-1], 0.0)
        r["intra_atr_pct"] = round(atr_v / cur * 100, 2) if cur else 0.0
        # VWMA(14) distance
        if "Volume" in h.columns:
            tp = (h["High"] + h["Low"] + cl) / 3
            vol_sum = h["Volume"].rolling(14).sum().replace(0, 1e-9)
            vwma = (tp * h["Volume"]).rolling(14).sum() / vol_sum
            vwma_last = _safe(vwma.iloc[-1], cur)
            r["intra_vwma_dist"] = round((cur - vwma_last) / vwma_last * 100, 2) if vwma_last else 0.0
        # Intraday RSI(14)
        d = cl.diff()
        g = d.clip(lower=0).rolling(14).mean()
        ldn = (-d.clip(upper=0)).rolling(14).mean()
        rs = g / ldn.replace(0, 1e-9)
        r["intra_rsi"] = round(_safe((100 - 100 / (1 + rs)).iloc[-1], 50.0), 1)
    except Exception as e:
        try: print(f"[intraday] {tk} fetch failed: {type(e).__name__}: {e}")
        except Exception: pass
    cache_set(f"intra_{tk}", r)
    return r
