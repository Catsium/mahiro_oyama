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
from market.quotes import _append_live_bar
from trading.indicators import (
    _ctx_from_series, _ctx_from_recorded, weekly_posture, sector_relative_strength, _safe
)
from utils.cache import cache_get, cache_set
from utils.deploy_config import PYTHONANYWHERE_MODE
from utils.time_utils import is_market_open


def _history_meta_from_df(df):
    attrs = getattr(df, "attrs", {}) or {}
    out = {
        "history_source": attrs.get("source") or "daily",
        "history_status": attrs.get("status") or "ok",
        "history_provider": attrs.get("provider"),
        "history_rows": int(len(df)) if df is not None else 0,
        "history_warnings": list(attrs.get("warnings") or []),
        "live_bar_applied": bool(attrs.get("live_bar_applied", False)),
        "live_bar_reason": attrs.get("live_bar_reason"),
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
    return out


def get_history(tk):
    """Daily history + weekly posture. Cached 5 min."""
    c = cache_get(f"h_{tk}", max_age=300)
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
        except Exception:
            pass
    # Locally recorded snapshots (last resort; no weekly possible).
    if not r:
        r = _ctx_from_recorded(tk)
        if r:
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
    if r:
        from trading.risk import get_sector
        sec = get_sector(tk)
        rs = sector_relative_strength(tk, sec, lookback_days=20)
        if rs:
            r.update(rs)
    if r:
        cache_set(f"h_{tk}", r)
    return r


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
