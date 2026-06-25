"""Quote fetching, ticker validation, and daily-bar helpers."""
import time
from datetime import datetime, timedelta

import pandas as pd

from market import fh
from market.data_manager import get_daily as _managed_daily
from utils.cache import (
    CACHE_MISS, cache_get, cache_set, record_api_failure, record_api_success, should_skip_api,
)
from utils.deploy_config import PYTHONANYWHERE_MODE
from utils.storage import load_price_hist, append_price_snapshot


def record_price(tk, price):
    append_price_snapshot(tk, price, min_interval=60, limit=6000)


def _finnhub_daily(tk, full=False):
    """Daily OHLCV from Finnhub candles. Primary PA free-tier history source."""
    endpoint = f"finnhub_daily:{tk}:{int(bool(full))}"
    if should_skip_api(endpoint):
        return None
    cache_key = f"fh_daily_full_{tk}" if full else f"fh_daily_{tk}"
    c = cache_get(cache_key, max_age=6 * 3600, default=CACHE_MISS)
    if c is not CACHE_MISS:
        return c if isinstance(c, pd.DataFrame) and not c.empty else None
    try:
        end = int(time.time())
        days = 3650 if full else 600
        start = int((datetime.utcnow() - timedelta(days=days)).timestamp())
        raw = fh.stock_candles(tk, "D", start, end) or {}
        if raw.get("s") != "ok" or not raw.get("c"):
            cache_set(cache_key, None)
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
            cache_set(cache_key, None)
            return None
        if not full:
            df = df.tail(400)
        cache_set(cache_key, df)
        record_api_success(endpoint)
        return df
    except Exception as e:
        try:
            print(f"[finnhub-daily] {tk} failed: {type(e).__name__}: {e}")
        except Exception:
            pass
        cache_set(cache_key, None)
        record_api_failure(endpoint, e)
        return None


def _raw_daily(tk, full=False):
    """Daily bars. In PA mode this uses Finnhub; off-PA it uses Stooq CSV."""
    if PYTHONANYWHERE_MODE:
        return _finnhub_daily(tk, full=full)

    endpoint = f"stooq_daily:{tk}:{int(bool(full))}"
    if should_skip_api(endpoint):
        return None
    cache_key = f"stooq_full_{tk}" if full else f"stooq_{tk}"
    c = cache_get(cache_key, max_age=300, default=CACHE_MISS)
    if c is not CACHE_MISS:
        return c if isinstance(c, pd.DataFrame) and not c.empty else None
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
        from io import StringIO

        df = pd.read_csv(StringIO(raw))
        if df.empty or "Close" not in df.columns:
            cache_set(cache_key, None)
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        if not full:
            df = df.tail(400)
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


def _stooq_daily(tk, full=False):
    return _managed_daily(tk, full=full)


def _append_live_bar(df, tk):
    """Append today's live Finnhub quote as a synthetic daily row when absent."""
    if df is None or df.empty:
        return df
    try:
        last_dt = df.index[-1]
        today = pd.Timestamp(datetime.now().date())
        if pd.Timestamp(last_dt).normalize() >= today:
            return df
        q = get_quote(tk) or {}
        live = q.get("price") or 0
        if live <= 0:
            return df
        o_hi = q.get("high") or live
        o_lo = q.get("low") or live
        o_op = q.get("open") or live
        row = {}
        for col in df.columns:
            if col == "Volume":
                row[col] = 0.0
            elif col == "High":
                row[col] = float(o_hi)
            elif col == "Low":
                row[col] = float(o_lo)
            elif col == "Open":
                row[col] = float(o_op)
            else:
                row[col] = float(live)
        df.loc[today] = row
    except Exception as e:
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
                "open": 0, "prev": 0}
    try:
        q = fh.quote(tk)
        r = {"price": q.get("c", 0), "change": q.get("d", 0), "pct": q.get("dp", 0),
             "high": q.get("h", 0), "low": q.get("l", 0), "open": q.get("o", 0),
             "prev": q.get("pc", 0)}
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
        if r["price"] > 0:
            record_api_success(endpoint)
        return r
    except Exception as e:
        record_api_failure(endpoint, e)
        return {"price": 0, "change": 0, "pct": 0, "high": 0, "low": 0,
                "open": 0, "prev": 0}


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
            r["stale"] = True
            r["stale_age_sec"] = int(time.time()) - int(last_ts)
            try:
                print(f"[get_quote] {tk}: using last recorded ${last_price}")
            except Exception:
                pass
        else:
            r["stale"] = True
            r["stale_age_sec"] = -1
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
