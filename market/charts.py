"""Price-chart payload builder for /api/chart/<ticker>/<rng>.

Order: yfinance (often blocked on PA shared IPs) → Stooq daily + live-bar →
locally-recorded snapshots. Returns a dict the templates' Chart.js code
consumes directly (dates, prices, ma_short, ma_long, labels).
"""
import time
from datetime import datetime
import pandas as pd

from market.data_manager import get_daily as _daily_bars
from market.quotes import _append_live_bar
from utils.cache import cache_get, cache_set, CACHE_TTL
from utils.deploy_config import PYTHONANYWHERE_MODE
from utils.storage import load_price_hist


PERIOD_MAP = {
    "1d": ("1d",  "1m"),
    "1w": ("5d",  "15m"),
    "1m": ("1mo", "1d"),
    "3m": ("3mo", "1d"),
}

RANGE_SECS = {"1d": 86400, "1w": 7 * 86400, "1m": 30 * 86400, "3m": 90 * 86400}


def chart_from_recorded(tk, rng):
    """Fallback chart built from locally-recorded price snapshots."""
    d = load_price_hist()
    pts = d.get(tk, [])
    if not pts:
        return None
    cutoff = time.time() - RANGE_SECS.get(rng, 90 * 86400)
    pts = [p for p in pts if p[0] >= cutoff]
    if len(pts) < 2:
        return None
    prices = [p[1] for p in pts]
    if rng == "1d":   fmt = "%H:%M"
    elif rng == "1w": fmt = "%b %d %H:%M"
    else:             fmt = "%b %d %H:%M"
    labels = [datetime.fromtimestamp(p[0]).strftime(fmt) for p in pts]
    s = pd.Series(prices)
    ns = max(min(7,  len(s) // 4), 2)
    nl = max(min(30, len(s) // 2), 5)
    to_l = lambda x: [None if pd.isna(v) else round(float(v), 2) for v in x]
    return {"dates": labels, "prices": prices,
            "ma7_series":  to_l(s.rolling(ns).mean()),
            "ma30_series": to_l(s.rolling(nl).mean()),
            "range": rng, "source": "recorded",
            "ma_short_label": f"{ns}-pt MA", "ma_long_label": f"{nl}-pt MA"}


def _chart_from_df(h, rng):
    """Build chart payload from an OHLCV DataFrame (yfinance- or Stooq-shaped)."""
    cl = h["Close"].round(2)
    if   rng == "1d": labels = [d.strftime("%H:%M") for d in h.index]
    elif rng == "1w": labels = [d.strftime("%b %d %H:%M") for d in h.index]
    else:             labels = [d.strftime("%b %d") for d in h.index]
    ns = max(min(7,  len(cl) // 4), 2)
    nl = max(min(30, len(cl) // 2), 5)
    ma_s = cl.rolling(ns).mean().round(2)
    ma_l = cl.rolling(nl).mean().round(2)
    to_l = lambda s: [None if pd.isna(v) else float(v) for v in s]
    return {"dates": labels, "prices": to_l(cl), "ma7_series": to_l(ma_s),
            "ma30_series": to_l(ma_l), "range": rng,
            "ma_short_label": f"{ns}-pt MA", "ma_long_label": f"{nl}-pt MA"}


def get_chart(tk, rng="3m"):
    c = cache_get(f"c_{tk}_{rng}")
    if c is not None:
        return c
    period, interval = PERIOD_MAP.get(rng, ("3mo", "1d"))
    r = {}
    # 1) yfinance
    if not PYTHONANYWHERE_MODE:
        try:
            import yfinance as yf
            h = yf.Ticker(tk).history(period=period, interval=interval)
            if not h.empty:
                r = _chart_from_df(h, rng)
        except Exception as e:
            try: print(f"[chart] yf {tk} {rng} failed: {type(e).__name__}: {e}")
            except Exception: pass
    # 2) Stooq daily — coarse for 1d/1w but better than nothing.
    if not r:
        try:
            df = _daily_bars(tk)
            if df is not None and not df.empty:
                cutoff_days = {"1d": 14, "1w": 30, "1m": 35, "3m": 100}.get(rng, 100)
                df_window = df.tail(cutoff_days).copy()
                df_window = _append_live_bar(df_window, tk)
                if not df_window.empty:
                    r = _chart_from_df(df_window, rng)
                    r["source"] = "stooq-daily"
        except Exception as e:
            try: print(f"[chart] stooq {tk} {rng} failed: {type(e).__name__}: {e}")
            except Exception: pass
    # 3) locally recorded snapshots
    if not r:
        fb = chart_from_recorded(tk, rng)
        r = fb if fb else {"error": f"No data yet for {tk} ({rng}). The app collects price snapshots every {CACHE_TTL // 60} min — a chart will appear once enough points are gathered."}
    cache_set(f"c_{tk}_{rng}", r)
    return r
