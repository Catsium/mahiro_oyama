"""Technical indicator math + ctx builders.

Pure-math layer with no Flask, no Finnhub. Two helpers (sector_relative_strength
and median_atr_since) need _stooq_daily — we lazy-import it inside the function
body to avoid a cycle with market/history.py which itself imports this module.
"""
import pandas as pd

from utils.cache import cache_get, cache_set
from utils.storage import load_price_hist


# ── Sector ETF map (for #1.3 sector-relative strength) ───────────────────────
SECTOR_ETF_MAP = {
    "tech":          "XLK",
    "communication": "XLC",   # yfinance returns "Communication Services"
    "comms":         "XLC",
    "consumer":      "XLP",
    "auto":          "XLY",   # autos rolled into discretionary
    "discretionary": "XLY",
    "finance":       "XLF",
    "energy":        "XLE",
    "health":        "XLV",
    "industrial":    "XLI",
    "utilities":     "XLU",
    "materials":     "XLB",
    "real":          "XLRE",  # real-estate truncated by .split()[0]
}


def _safe(x, fallback=None):
    try:
        v = float(x)
        return v if not pd.isna(v) else fallback
    except Exception:
        return fallback


def _ctx_from_series(cl, df=None, max_bars: int = 80):
    """Build technicals from a Close series (and optional OHLCV df for vol/ATR)."""
    # Bound only per-ticker ctx work for live/suggestion cycles. Do not reuse this
    # cap for regime/breadth functions that need 200d+ market context.
    cl = cl.tail(max_bars)
    if df is not None:
        df = df.tail(max_bars).copy()
    if len(cl) < 2:
        return {}
    ma7  = cl.rolling(min(7,  max(2, len(cl) // 3))).mean()
    ma30 = cl.rolling(min(30, max(3, len(cl) // 2))).mean()
    cur  = float(cl.iloc[-1])
    ma7v  = _safe(ma7.iloc[-1],  cur)
    ma30v = _safe(ma30.iloc[-1], cur)

    # RSI(14)
    d = cl.diff()
    rw = min(14, max(2, len(cl) // 3))
    g = d.clip(lower=0).rolling(rw).mean()
    l = (-d.clip(upper=0)).rolling(rw).mean()
    rs = g / l.replace(0, 1e-9)
    rsi = round(_safe((100 - 100 / (1 + rs)).iloc[-1], 50.0), 1)

    # MACD (12, 26, 9)
    macd_h = macd_h_prev = macd_v = macd_sig = 0.0
    if len(cl) >= 26:
        ema12 = cl.ewm(span=12, adjust=False).mean()
        ema26 = cl.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        macd_v   = _safe(macd_line.iloc[-1], 0.0)
        macd_sig = _safe(signal_line.iloc[-1], 0.0)
        macd_h   = _safe(hist.iloc[-1], 0.0)
        macd_h_prev = _safe(hist.iloc[-2], 0.0) if len(hist) > 1 else 0.0

    # Bollinger Bands (20, 2σ)
    bb_pos = 0.5
    bb_width_pct = 0.0
    if len(cl) >= 20:
        bb_mid = cl.rolling(20).mean()
        bb_std = cl.rolling(20).std()
        bb_up = bb_mid.iloc[-1] + 2 * bb_std.iloc[-1]
        bb_lo = bb_mid.iloc[-1] - 2 * bb_std.iloc[-1]
        if not pd.isna(bb_up) and not pd.isna(bb_lo) and bb_up > bb_lo:
            bb_pos = (cur - float(bb_lo)) / (float(bb_up) - float(bb_lo))
            bb_pos = max(0.0, min(1.0, bb_pos))
            bb_width_pct = (float(bb_up) - float(bb_lo)) / cur * 100

    # ATR(14) — needs High/Low/Close
    atr_pct = 0.0
    if df is not None and "High" in df.columns and "Low" in df.columns and len(df) >= 14:
        tr = pd.concat([
            (df["High"] - df["Low"]),
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"]  - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        atr_v = _safe(tr.rolling(14).mean().iloc[-1], 0.0)
        atr_pct = round(atr_v / cur * 100, 2) if cur else 0.0

    # Volume trend (5d vs 20d avg) + average dollar volume (for liquidity filter)
    vol_ratio = 1.0
    avg_dollar_vol_20d = 0.0
    if df is not None and "Volume" in df.columns and len(df) >= 20:
        vol = df["Volume"]
        vol_recent = _safe(vol.iloc[-5:].mean(), 0.0)
        vol_avg    = _safe(vol.iloc[-20:].mean(), vol_recent)
        vol_ratio  = round(vol_recent / vol_avg, 2) if vol_avg else 1.0
        avg_close_20d = _safe(cl.iloc[-20:].mean(), cur) or cur
        avg_dollar_vol_20d = round(float(vol_avg) * float(avg_close_20d), 0)

    # Stochastic Oscillator %K and %D (14, 3)
    stoch_k = stoch_d = 50.0
    if df is not None and "High" in df.columns and "Low" in df.columns and len(df) >= 14:
        low14  = df["Low"].rolling(14).min()
        high14 = df["High"].rolling(14).max()
        rng = (high14 - low14).replace(0, 1e-9)
        k_full = (cl - low14) / rng * 100
        stoch_k = _safe(k_full.iloc[-1], 50.0)
        stoch_d = _safe(k_full.rolling(3).mean().iloc[-1], stoch_k)

    # Money Flow Index (MFI, 14) — volume-weighted RSI
    mfi = 50.0
    if df is not None and all(c in df.columns for c in ["High", "Low", "Volume"]) and len(df) >= 15:
        tp = (df["High"] + df["Low"] + cl) / 3
        mf = tp * df["Volume"]
        sign = tp.diff()
        pos_mf = mf.where(sign > 0, 0).rolling(14).sum()
        neg_mf = mf.where(sign < 0, 0).rolling(14).sum().abs().replace(0, 1e-9)
        ratio = pos_mf / neg_mf
        mfi = _safe((100 - 100 / (1 + ratio)).iloc[-1], 50.0)

    # Trend strength (ADX-lite)
    adx_lite = 0.0
    if df is not None and "High" in df.columns and "Low" in df.columns and len(df) >= 14:
        up_move = df["High"].diff()
        dn_move = -df["Low"].diff()
        plus_dm  = up_move.where((up_move > dn_move) & (up_move > 0), 0)
        minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)
        tr_high_low = df["High"] - df["Low"]
        atr_smooth  = tr_high_low.rolling(14).mean().replace(0, 1e-9)
        plus_di  = 100 * plus_dm.rolling(14).mean() / atr_smooth
        minus_di = 100 * minus_dm.rolling(14).mean() / atr_smooth
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)) * 100
        adx_lite = _safe(dx.rolling(14).mean().iloc[-1], 0.0)

    # VWAP-relative position (intraday only — useful when df is intraday)
    vwap_dist_pct = 0.0
    if df is not None and "Volume" in df.columns and "High" in df.columns and len(df) >= 20:
        tp = (df["High"] + df["Low"] + cl) / 3
        vwap = (tp * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum().replace(0, 1e-9)
        vwap_last = _safe(vwap.iloc[-1], cur)
        vwap_dist_pct = round((cur - vwap_last) / vwap_last * 100, 2) if vwap_last else 0.0

    # Week change
    lookback = min(5, len(cl) - 1)
    wk = float(cl.iloc[-1 - lookback]) if lookback > 0 else float(cl.iloc[0])
    wc = round((cur - wk) / wk * 100, 2) if wk else 0.0

    # 1-month change (for momentum)
    mom_lookback = min(21, len(cl) - 1)
    mom_base = float(cl.iloc[-1 - mom_lookback]) if mom_lookback > 0 else float(cl.iloc[0])
    mom_30d = round((cur - mom_base) / mom_base * 100, 2) if mom_base else 0.0

    # Dip detection — distance from recent 20-day high
    window = min(20, len(cl))
    recent_high = float(cl.iloc[-window:].max()) if window > 0 else cur
    dist_from_high_pct = round((cur - recent_high) / recent_high * 100, 2) if recent_high else 0.0
    is_dip = (dist_from_high_pct <= -5 and dist_from_high_pct >= -15
              and rsi < 45 and macd_h > macd_h_prev)

    # Round-4 algorithm #5: trend-exhaustion signal — count consecutive up days.
    # A vertical run of 5+ greens combined with extension above MA30 is a
    # late-trend indicator and should trim conviction on new buys.
    consec_up_days = 0
    if len(cl) >= 2:
        for i in range(1, min(9, len(cl))):
            if float(cl.iloc[-i]) > float(cl.iloc[-i - 1]):
                consec_up_days += 1
            else:
                break

    return {
        "current": cur, "ma7": ma7v, "ma30": ma30v, "above_ma30": cur > ma30v,
        "rsi": rsi, "week_chg_pct": wc, "mom_30d_pct": mom_30d,
        "macd": round(macd_v, 4), "macd_signal": round(macd_sig, 4),
        "macd_hist": round(macd_h, 4), "macd_hist_prev": round(macd_h_prev, 4),
        "bb_pos": round(bb_pos, 3), "bb_width_pct": round(bb_width_pct, 2),
        "atr_pct": atr_pct, "vol_ratio": vol_ratio,
        "avg_dollar_vol_20d": avg_dollar_vol_20d,
        "stoch_k": round(stoch_k, 1), "stoch_d": round(stoch_d, 1),
        "mfi": round(mfi, 1), "adx": round(adx_lite, 1),
        "vwap_dist_pct": vwap_dist_pct,
        "recent_high": round(recent_high, 2),
        "dist_from_high_pct": dist_from_high_pct,
        "is_dip": bool(is_dip),
        "consec_up_days": consec_up_days,
    }


def _ctx_from_recorded(tk):
    """Fallback: build technicals from locally-recorded price snapshots (no volume/ATR)."""
    pts = load_price_hist().get(tk, [])
    if len(pts) < 3:
        return {}
    cl = pd.Series([p[1] for p in pts])
    ctx = _ctx_from_series(cl)
    if ctx:
        ctx["source"] = "recorded"
        ctx["recorded_points_n"] = len(pts)
        ctx["recorded_last_ts"] = int(pts[-1][0])
    return ctx


def weekly_posture(daily_df):
    """#1.4 weekly higher-timeframe filter. Resample daily OHLCV → weekly bars,
    compute weekly RSI(14) + MACD posture + 10-week MA trend. Returns dict
    suitable for ctx.update(). Empty dict if df too short."""
    if daily_df is None or len(daily_df) < 70:
        return {}
    try:
        wk = daily_df.resample("W").agg({
            "Open":  "first",
            "High":  "max",
            "Low":   "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()
        cl = wk["Close"]
        if len(cl) < 14:
            return {}
        # Weekly RSI(14)
        d = cl.diff()
        g = d.clip(lower=0).rolling(14).mean()
        ldn = (-d.clip(upper=0)).rolling(14).mean()
        rs = g / ldn.replace(0, 1e-9)
        w_rsi = _safe((100 - 100 / (1 + rs)).iloc[-1], 50.0)
        # Weekly MACD(12,26,9)
        w_macd_bull = False
        if len(cl) >= 35:
            ema12 = cl.ewm(span=12, adjust=False).mean()
            ema26 = cl.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            sig_line  = macd_line.ewm(span=9, adjust=False).mean()
            w_macd_bull = bool(_safe(macd_line.iloc[-1], 0) > _safe(sig_line.iloc[-1], 0))
        ma10 = cl.rolling(10).mean()
        ma10_last = _safe(ma10.iloc[-1], _safe(cl.iloc[-1], 0))
        w_trend_up = bool(_safe(cl.iloc[-1], 0) > ma10_last)
        return {
            "weekly_rsi": round(float(w_rsi), 1),
            "weekly_macd_bullish": w_macd_bull,
            "weekly_trend_up": w_trend_up,
        }
    except Exception as e:
        try: print(f"[weekly_posture] failed: {type(e).__name__}: {e}")
        except Exception: pass
        return {}


def _daily_source_info(df):
    attrs = getattr(df, "attrs", {}) or {}
    return {
        "source": attrs.get("source"),
        "status": attrs.get("status"),
        "rows": int(len(df)) if df is not None and hasattr(df, "__len__") else 0,
    }


def sector_relative_strength(tk, sector, lookback_days=20):
    """#1.3 sector-relative strength. Returns dict {rel_str_pct, sector_etf}
    where rel_str_pct = stock return MINUS sector ETF return over `lookback_days`.
    Cached 5 min. Lazy-imports _stooq_daily to avoid market↔trading cycle."""
    if not sector:
        return {}
    etf = SECTOR_ETF_MAP.get(sector)
    if not etf:
        return {}
    cache_key = f"relstr_{tk}_{etf}_{lookback_days}"
    c = cache_get(cache_key, max_age=300)
    if c is not None:
        return c
    try:
        from market.data_manager import get_daily
        tk_df  = get_daily(tk)
        etf_df = get_daily(etf)
        benchmark = etf
        if ((etf_df is None or len(etf_df) < lookback_days + 1)
                and tk_df is not None and len(tk_df) >= lookback_days + 1):
            # Audit P1-12: sector ETF history unavailable (e.g. XLK/XLY 402 on
            # PA) — benchmark vs SPY so the REL_STR category stays alive.
            spy_df = get_daily("SPY")
            if spy_df is not None and len(spy_df) >= lookback_days + 1:
                etf_df = spy_df
                benchmark = "SPY"
        if (tk_df is None or etf_df is None or
            len(tk_df) < lookback_days + 1 or len(etf_df) < lookback_days + 1):
            tk_info = _daily_source_info(tk_df)
            etf_info = _daily_source_info(etf_df)
            out = {
                "sector_etf": etf,
                "relative_strength_source": "skipped",
                "relative_strength_skipped_reason": "missing_or_insufficient_history",
                "ticker_history_source": tk_info.get("source"),
                "ticker_history_status": tk_info.get("status"),
                "ticker_history_rows": tk_info.get("rows"),
                "sector_etf_history_source": etf_info.get("source"),
                "sector_etf_history_status": etf_info.get("status"),
                "sector_etf_history_rows": etf_info.get("rows"),
            }
            cache_set(cache_key, out)
            return out
        tk_ret  = float(tk_df["Close"].iloc[-1])  / float(tk_df["Close"].iloc[-(lookback_days + 1)]) - 1
        etf_ret = float(etf_df["Close"].iloc[-1]) / float(etf_df["Close"].iloc[-(lookback_days + 1)]) - 1
        rel_str_pct = round((tk_ret - etf_ret) * 100, 2)
        tk_info = _daily_source_info(tk_df)
        etf_info = _daily_source_info(etf_df)
        sources = [str(tk_info.get("source") or ""), str(etf_info.get("source") or "")]
        out = {
            "rel_str_pct": rel_str_pct,
            "sector_etf": etf,
            "rel_str_benchmark": benchmark,
            "relative_strength_source": (
                "spy_fallback" if benchmark == "SPY" and etf != "SPY"
                else "stale_cache" if any(s.startswith("stale_cache:") for s in sources)
                else "daily"
            ),
            "ticker_history_source": tk_info.get("source"),
            "ticker_history_status": tk_info.get("status"),
            "ticker_history_rows": tk_info.get("rows"),
            "sector_etf_history_source": etf_info.get("source"),
            "sector_etf_history_status": etf_info.get("status"),
            "sector_etf_history_rows": etf_info.get("rows"),
        }
        cache_set(cache_key, out)
        return out
    except Exception as e:
        try: print(f"[rel_str] {tk} vs {etf} failed: {type(e).__name__}: {e}")
        except Exception: pass
        out = {
            "sector_etf": etf,
            "relative_strength_source": "skipped",
            "relative_strength_skipped_reason": "error",
        }
        cache_set(cache_key, out)
        return out


def median_atr_since(tk, since_ts):
    """#4.2 median ATR% over daily bars since `since_ts` (unix seconds).
    Used as a stop-loss floor — long rallies compress current ATR, but the
    median over the trend reflects real noise we should tolerate."""
    if not since_ts:
        return 0.0
    from market.data_manager import get_daily
    df = get_daily(tk)
    if df is None or df.empty:
        return 0.0
    try:
        since_dt = pd.Timestamp(since_ts, unit="s")
        if df.index.tz is not None and since_dt.tz is None:
            since_dt = since_dt.tz_localize(df.index.tz)
        elif df.index.tz is None and since_dt.tz is not None:
            since_dt = since_dt.tz_localize(None)
        window = df[df.index >= since_dt]
        if len(window) < 5:
            return 0.0
        tr = pd.concat([
            window["High"] - window["Low"],
            (window["High"] - window["Close"].shift()).abs(),
            (window["Low"]  - window["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_n = min(14, max(3, len(window) // 2))
        atr_series = tr.rolling(atr_n).mean() / window["Close"] * 100
        med = atr_series.dropna().median()
        return float(med) if pd.notna(med) else 0.0
    except Exception as e:
        try: print(f"[median_atr_since] {tk}: {type(e).__name__}: {e}")
        except Exception: pass
        return 0.0


def classify_vol_regime(ctx):
    """Classify volatility regime from BB width as % of price.
       <5% = compressed (breakout setup, widen stops)
       >12% = explosive (reduce size, avoid mean-reversion entries)
       else = normal"""
    if not ctx or "bb_width_pct" not in ctx:
        return "normal"
    bw = ctx["bb_width_pct"]
    if   bw < 5:  return "compressed"
    elif bw > 12: return "explosive"
    return "normal"
