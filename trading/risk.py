"""Risk/regime lookups:
- Sector & correlation-group mapping
- Market regime (SPY trend + intraday tilt + credit-spread blend)
- Market-volatility gating (SPY realized vol — replaces dead yfinance ^VIX)
- Credit signal (HYG:IEI ratio)
- Earnings calendar
- Analyst recommendation aggregate
- Insider sentiment (MSPR)
"""
from datetime import datetime, timedelta
import pandas as pd

from market import fh
from market.data_manager import get_daily as _daily_bars
from market.quotes import _append_live_bar, get_regime_daily
from utils.cache import (
    cache_get, cache_set, record_api_failure, record_api_success, should_skip_api,
)
from utils.deploy_config import PYTHONANYWHERE_MODE
from utils.time_utils import is_market_open
from trading.config import DEFAULT_CONFIG, config_hash
from trading.regime_v3 import (
    SECTOR_ETFS, build_regime_v3, get_breadth_universe, load_live_close_history,
)


# ── Sector mapping ──────────────────────────────────────────────────────────
SPY_MOM_LOOKBACK_BARS = 22
SPY_MIN_REGIME_ROWS = 50
SPY_MOM_LABEL = "1M / 22 trading days"


def _regime_daily_bars(ticker, full=False):
    return get_regime_daily(ticker, full=full)


SECTOR_MAP = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOG": "tech", "GOOGL": "tech",
    "META": "tech", "AMZN": "tech", "ADBE": "tech", "CRM": "tech", "ORCL": "tech",
    "AVGO": "tech", "AMD": "tech", "INTC": "tech", "CSCO": "tech", "TXN": "tech",
    "QCOM": "tech", "IBM": "tech", "PLTR": "tech", "SHOP": "tech", "SNOW": "tech",
    "UBER": "tech", "DDOG": "tech", "ACN": "tech", "SPOT": "tech", "NFLX": "tech",
    "TSLA": "auto", "F": "auto", "GM": "auto",
    "JPM": "finance", "V": "finance", "MA": "finance", "BAC": "finance", "WFC": "finance",
    "PYPL": "finance", "COIN": "finance", "SQ": "finance", "SOFI": "finance",
    "XOM": "energy", "CVX": "energy",
    "WMT": "consumer", "COST": "consumer", "HD": "consumer", "PEP": "consumer", "KO": "consumer",
    "MCD": "consumer", "DIS": "consumer", "PG": "consumer",
    "JNJ": "health", "LLY": "health", "ABBV": "health", "TMO": "health", "ABT": "health",
    "BA": "industrial", "GE": "industrial",
    # Round-5 #6 additions
    "UNH": "health", "ISRG": "health",
    "CAT": "industrial", "RTX": "industrial",
    "LIN": "materials", "PM": "consumer",
}


def get_sector(t):
    """Return sector for ticker. Falls back to yfinance .info if not in SECTOR_MAP.
    Tickers returning no sector return None — the bot blocks purchase of these
    rather than dumping them into 'other' with no cap."""
    t = t.upper()
    if t in SECTOR_MAP:
        return SECTOR_MAP[t]
    c = cache_get(f"sec_{t}", max_age=24 * 3600)
    if c is not None:
        return c if c != "_unknown_" else None
    sec = None
    if not PYTHONANYWHERE_MODE:
        try:
            import yfinance as yf
            info = yf.Ticker(t).info or {}
            s = info.get("sector")
            if s:
                sec = s.lower().split()[0]
        except Exception:
            pass
    cache_set(f"sec_{t}", sec or "_unknown_")
    return sec


# ── Correlation groups (#22) ───────────────────────────────────────────────
CORR_GROUPS = {
    "ai_semis":      {"NVDA", "AMD", "AVGO", "TSM", "INTC", "SMCI", "ARM", "QCOM", "SOXL"},
    "megacap_tech":  {"AAPL", "MSFT", "GOOG", "GOOGL", "META", "AMZN"},
    "ev":            {"TSLA", "RIVN", "LCID", "NIO", "F", "GM"},
    "crypto_prox":   {"COIN", "MSTR", "SQ", "PYPL", "HOOD", "RIOT", "MARA"},
    "ai_software":   {"PLTR", "SNOW", "CRM", "DDOG", "NET", "CRWD", "S", "AI"},
    "big_banks":     {"JPM", "BAC", "WFC", "C", "GS", "MS"},
    "payments":      {"V", "MA", "PYPL", "SQ", "SOFI", "AXP"},
    "streaming":     {"NFLX", "DIS", "SPOT", "ROKU"},
}


def get_corr_group(t):
    t = t.upper()
    for grp, members in CORR_GROUPS.items():
        if t in members:
            return grp
    return None


# ── Market-volatility gating (#13) ──────────────────────────────────────────
def get_vix():
    """Market-volatility gate using SPY realized volatility.

    PythonAnywhere uses the regime daily-bar path: Finnhub first, optional FMP
    daily fallback, then bounded stale cache. The `vix` field carries realized
    volatility %, not the VIX index.
    """
    c = cache_get("vix", max_age=3600)
    if c is not None and "data_ok" in c and "volatility_window_days" in c:
        return c
    mode_cfg = DEFAULT_CONFIG.get("market_data_modes", {})
    window = int(mode_cfg.get("proxy_vol_window_days", 20))
    min_bars = int(mode_cfg.get("proxy_min_spy_bars", 60))
    r = {
        "vix": None,
        "volatility_value": None,
        "regime": "UNKNOWN",
        "mult": 1.0,
        "data_ok": False,
        "data_status": "missing_spy_history",
        "source": None,
        "volatility_source": None,
        "volatility_window_days": window,
        "vix_display": "unknown",
        "data_error": None,
    }
    try:
        df = _regime_daily_bars("SPY")
        if df is not None and not df.empty and len(df) >= min_bars:
            attrs = getattr(df, "attrs", {}) or {}
            warnings = list(attrs.get("warnings") or [])
            df = _append_live_bar(df.copy(), "SPY")
            rets = df["Close"].pct_change().dropna().tail(window)
            if len(rets) >= window:
                rv = float(rets.std() * (252 ** 0.5) * 100)
                r["vix"] = round(rv, 2)
                r["volatility_value"] = round(rv, 2)
                r["data_ok"] = True
                r["data_status"] = "ok"
                r["source"] = "spy_realized_vol_proxy"
                r["volatility_source"] = "spy_realized_vol_proxy"
                r["spy_data_source"] = attrs.get("source")
                if warnings:
                    r["data_health_warnings"] = warnings
                r["vix_display"] = "proxy"
                if   rv > 28: r["regime"] = "PANIC";     r["mult"] = 0.0
                elif rv > 18: r["regime"] = "HIGH_RISK"; r["mult"] = 0.5
                else:         r["regime"] = "NORMAL";    r["mult"] = 1.0
            else:
                r["data_status"] = "insufficient_spy_returns"
        else:
            r["data_status"] = "missing_or_insufficient_spy_history"
    except Exception as e:
        r["data_status"] = "spy_realized_vol_error"
        r["data_error"] = str(e)[:160]
        try: print(f"[get_vix] SPY realized-vol failed: {type(e).__name__}: {e}")
        except Exception: pass
    cache_set("vix", r)
    return r


# ── Credit-spread indicator (HYG:IEI ratio) #2.1 ───────────────────────────
def credit_signal():
    """Returns {credit_label, credit_pct} where credit_label is one of:
       'risk_on' (>70th pct), 'neutral' (30-70), 'risk_off' (<30th pct)."""
    c = cache_get("credit_signal", max_age=3600)
    if c is not None:
        return c
    r = {"credit_label": "neutral", "credit_pct": 50.0}
    try:
        hyg = _regime_daily_bars("HYG")
        iei = _regime_daily_bars("IEI")
        if hyg is None or iei is None or len(hyg) < 90 or len(iei) < 90:
            cache_set("credit_signal", r)
            return r
        hyg = _append_live_bar(hyg.copy(), "HYG")
        iei = _append_live_bar(iei.copy(), "IEI")
        df = pd.concat({"HYG": hyg["Close"], "IEI": iei["Close"]}, axis=1).dropna().tail(90)
        if len(df) < 30:
            cache_set("credit_signal", r)
            return r
        ratio = df["HYG"] / df["IEI"]
        cur, lo, hi = float(ratio.iloc[-1]), float(ratio.min()), float(ratio.max())
        pct = (cur - lo) / (hi - lo) * 100 if hi > lo else 50.0
        if   pct < 30: label = "risk_off"
        elif pct > 70: label = "risk_on"
        else:          label = "neutral"
        r = {"credit_label": label, "credit_pct": round(pct, 1)}
    except Exception as e:
        try: print(f"[credit_signal] failed: {type(e).__name__}: {e}")
        except Exception: pass
    cache_set("credit_signal", r)
    return r


# ── Market regime (SPY trend + intraday tilt + credit blend) ───────────────
def _date_str(index_value):
    try:
        return str(index_value.date())
    except Exception:
        try:
            return str(pd.to_datetime(index_value).date())
        except Exception:
            return None


def _fmt_optional_float(value, digits=1):
    try:
        return f"{float(value):.{int(digits)}f}"
    except Exception:
        return "n/a"


def _spy_macro_fallback(status, df=None, source=None, error=None):
    attrs = getattr(df, "attrs", {}) or {}
    source = attrs.get("source") or source
    rows = 0
    last_date = None
    last_close = None
    try:
        rows = int(len(df)) if df is not None else 0
        if rows > 0 and "Close" in df:
            last_date = _date_str(df.index[-1])
            last_close = float(df["Close"].iloc[-1])
    except Exception:
        rows = 0
    out = {
        "regime": "neutral",
        "spy_mom_30d": None,
        "above_ma50": None,
        "spy_data_ok": False,
        "spy_rows": rows,
        "spy_last_date": last_date,
        "spy_last_close": last_close,
        "spy_base_22_close": None,
        "spy_mom_lookback_bars": SPY_MOM_LOOKBACK_BARS,
        "spy_mom_label": SPY_MOM_LABEL,
        "regime_data_status": status,
        "regime_data_fallback": True,
        "regime_data_source": source,
        "spy_data_source": source,
        "spy_data_error": str(error)[:160] if error else None,
    }
    warnings = attrs.get("warnings") or []
    if warnings:
        out["regime_data_warnings"] = list(warnings)
        out["data_health_warnings"] = list(warnings)
    if attrs.get("stale_daily_cache_age_hours") is not None:
        out["stale_daily_cache_age_hours"] = attrs.get("stale_daily_cache_age_hours")
        out["stale_daily_cache_age_sec"] = attrs.get("stale_daily_cache_age_sec")
    if error:
        out["regime_data_error"] = str(error)[:160]
    return out


def _spy_macro_from_df(df, source, append_live=True):
    if df is None:
        return _spy_macro_fallback("missing_spy_history", source=source)
    try:
        attrs = getattr(df, "attrs", {}) or {}
        source = attrs.get("source") or source
        status = attrs.get("status") or "ok"
        warnings = list(attrs.get("warnings") or [])
        if getattr(df, "empty", False):
            return _spy_macro_fallback("missing_spy_history", df=df, source=source)
        df = df.copy()
        if append_live:
            try:
                df = _append_live_bar(df, "SPY")
            except Exception:
                pass
        if "Close" not in df or len(df) < SPY_MIN_REGIME_ROWS:
            return _spy_macro_fallback("insufficient_spy_history", df=df, source=source)
        cl = df["Close"].dropna()
        if len(cl) < SPY_MIN_REGIME_ROWS:
            return _spy_macro_fallback("insufficient_spy_history", df=df, source=source)
        from trading.indicators import _safe
        cur = float(cl.iloc[-1])
        ma50 = _safe(cl.rolling(50).mean().iloc[-1], cur)
        above_ma50 = bool(cur > ma50)
        mom_base = float(cl.iloc[-SPY_MOM_LOOKBACK_BARS])
        if mom_base <= 0:
            return _spy_macro_fallback("invalid_spy_history", df=df, source=source)
        mom = round((cur - mom_base) / mom_base * 100, 2)
        if above_ma50 and mom > 2:
            regime = "bull"
        elif not above_ma50 and mom < -2:
            regime = "bear"
        else:
            regime = "neutral"
        out = {
            "regime": regime,
            "spy_mom_30d": mom,
            "above_ma50": above_ma50,
            "spy_data_ok": True,
            "spy_rows": int(len(cl)),
            "spy_last_date": _date_str(cl.index[-1]),
            "spy_last_close": round(cur, 4),
            "spy_base_22_close": round(mom_base, 4),
            "spy_mom_lookback_bars": SPY_MOM_LOOKBACK_BARS,
            "spy_mom_label": SPY_MOM_LABEL,
            "regime_data_status": status,
            "regime_data_fallback": status != "ok",
            "regime_data_source": source,
            "spy_data_source": source,
            "spy_data_error": None,
        }
        if warnings:
            out["regime_data_warnings"] = warnings
            out["data_health_warnings"] = warnings
        if attrs.get("stale_daily_cache_age_hours") is not None:
            out["stale_daily_cache_age_hours"] = attrs.get("stale_daily_cache_age_hours")
            out["stale_daily_cache_age_sec"] = attrs.get("stale_daily_cache_age_sec")
        return out
    except Exception as e:
        return _spy_macro_fallback("spy_history_error", df=df, source=source, error=e)


def get_market_regime_pa_light():
    """PA free-tier regime from cheap legacy SPY, credit, and realized-vol gates."""
    try:
        macro = _spy_macro_from_df(_regime_daily_bars("SPY"), "regime_daily", append_live=True)
    except Exception as e:
        macro = _spy_macro_fallback("spy_history_error", source="regime_daily", error=e)

    macro_regime = macro["regime"]
    from market.history import get_intraday_context
    spy_intra = get_intraday_context("SPY") if is_market_open() else {}
    spy_intra_dist = spy_intra.get("intra_vwma_dist", 0) or 0
    intra_tilt = None
    regime_effective = macro_regime
    if macro.get("spy_data_ok"):
        if spy_intra_dist < -0.3:
            intra_tilt = "bearish"
            if macro_regime == "bull":
                regime_effective = "neutral"
            elif macro_regime == "neutral":
                regime_effective = "bear"
        elif spy_intra_dist > 0.4:
            intra_tilt = "bullish"
            if macro_regime == "bear":
                regime_effective = "neutral"
            elif macro_regime == "neutral":
                regime_effective = "bull"

    cs = credit_signal()
    credit_label = cs.get("credit_label", "neutral")
    if macro.get("spy_data_ok"):
        if credit_label == "risk_off":
            if regime_effective == "bull":
                regime_effective = "neutral"
            elif regime_effective == "neutral":
                regime_effective = "bear"
        elif credit_label == "risk_on":
            if regime_effective == "bear":
                regime_effective = "neutral"
            elif regime_effective == "neutral":
                regime_effective = "bull"

    vix = get_vix()
    v3 = "panic" if vix.get("regime") == "PANIC" else regime_effective
    out = dict(macro)
    out.update({
        "intra_tilt": intra_tilt,
        "spy_intra_vwma_dist": spy_intra_dist,
        "credit_label": credit_label,
        "credit_pct": cs.get("credit_pct", 50.0),
        "regime_effective": regime_effective,
        "regime_v3": v3,
        "regime_v3_effective": v3,
        "regime_v3_raw": v3,
        "regime_v3_source": "pa_light",
        "regime_v3_fallback": True,
        "regime_v3_reason": "",
        "actual_breadth_count": 0,
        "missing_breadth_count": 0,
        "min_effective_breadth_count": 0,
        "top_sectors": [],
    })
    if out.get("spy_data_ok"):
        out["regime_v3_reason"] = (
            f"PA-light: legacy={regime_effective} credit={credit_label} "
            f"rv={_fmt_optional_float(vix.get('vix'))}"
        )
    else:
        out["regime_v3_reason"] = (
            f"PA-light fallback neutral: {out.get('regime_data_status')} "
            f"credit={credit_label} rv={_fmt_optional_float(vix.get('vix'))}"
        )
    return out


def get_market_regime(config=None):
    cfg_hash = config_hash(config) if config else "default"
    cache_key = f"market_regime_{cfg_hash}"
    c = cache_get(cache_key, max_age=300)
    if c is not None and "spy_data_ok" in c:
        return c
    if PYTHONANYWHERE_MODE:
        r = get_market_regime_pa_light()
        cache_set(cache_key, r)
        return r
    macro = _spy_macro_fallback("missing_spy_history")
    if not PYTHONANYWHERE_MODE:
        try:
            import yfinance as yf
            h = yf.Ticker("SPY").history(period="3mo")
            macro = _spy_macro_from_df(h, "yfinance_3mo", append_live=False)
        except Exception as e:
            macro = _spy_macro_fallback("spy_history_error", source="yfinance_3mo", error=e)
    if not macro.get("spy_data_ok"):
        try:
            df = _regime_daily_bars("SPY")
            macro = _spy_macro_from_df(df, "regime_daily", append_live=True)
        except Exception as e:
            macro = _spy_macro_fallback("spy_history_error", source="regime_daily", error=e)

    macro_regime = macro["regime"]

    # Intraday tilt — degrades cleanly when yfinance intraday is blocked
    from market.history import get_intraday_context
    spy_intra = get_intraday_context("SPY") if is_market_open() else {}
    spy_intra_dist = spy_intra.get("intra_vwma_dist", 0) or 0
    intra_tilt = None
    regime_effective = macro_regime
    if macro.get("spy_data_ok"):
        if spy_intra_dist < -0.3:
            intra_tilt = "bearish"
            if   macro_regime == "bull":    regime_effective = "neutral"
            elif macro_regime == "neutral": regime_effective = "bear"
        elif spy_intra_dist > 0.4:
            intra_tilt = "bullish"
            if   macro_regime == "bear":    regime_effective = "neutral"
            elif macro_regime == "neutral": regime_effective = "bull"

    # #2.1 credit blend
    cs = credit_signal()
    credit_label = cs.get("credit_label", "neutral")
    if macro.get("spy_data_ok"):
        if credit_label == "risk_off":
            if   regime_effective == "bull":    regime_effective = "neutral"
            elif regime_effective == "neutral": regime_effective = "bear"
        elif credit_label == "risk_on":
            if   regime_effective == "bear":    regime_effective = "neutral"
            elif regime_effective == "neutral": regime_effective = "bull"

    r = dict(macro)
    r.update({"intra_tilt": intra_tilt, "spy_intra_vwma_dist": spy_intra_dist,
              "credit_label": credit_label, "credit_pct": cs.get("credit_pct", 50.0),
              "regime_effective": regime_effective})
    try:
        if not r.get("spy_data_ok"):
            raise RuntimeError(r.get("regime_data_status") or "missing_spy_history")
        if PYTHONANYWHERE_MODE:
            raise RuntimeError("V3 breadth skipped in PA free mode to preserve API budget")
        c_v3 = cache_get(f"market_regime_v3_raw_{cfg_hash}", max_age=6 * 3600)
        if c_v3 is None:
            aux = sorted(set(SECTOR_ETFS.values()) | {"SPY", "QQQ", "RSP", "HYG", "IEF", "XLU", "XLP"})
            breadth_universe = list(get_breadth_universe())
            hist = load_live_close_history(breadth_universe + aux)
            spy_series = hist.get("SPY")
            close_hist = {t: hist[t] for t in breadth_universe if t in hist}
            aux_hist = {t: hist[t] for t in aux if t in hist}
            c_v3 = build_regime_v3(close_hist, spy_series, aux_hist,
                                   fallback_legacy=regime_effective,
                                   config=config)
            cache_set(f"market_regime_v3_raw_{cfg_hash}", c_v3)
        r.update(c_v3 or {})
        r.setdefault("regime_v3_source", "full_v3")
    except Exception as e:
        reason = f"V3 regime unavailable: {type(e).__name__}"
        if not r.get("spy_data_ok"):
            reason = f"fallback neutral: {r.get('regime_data_status')}"
        r.update({
            "regime_v3": regime_effective,
            "regime_v3_effective": regime_effective,
            "regime_v3_raw": "fallback",
            "regime_v3_fallback": True,
            "regime_v3_reason": reason,
            "regime_v3_source": "fallback",
            "actual_breadth_count": 0,
            "missing_breadth_count": 0,
        })
    cache_set(cache_key, r)
    return r


# ── Earnings calendar ──────────────────────────────────────────────────────
def get_earnings_soon(tk):
    """Cached 1 hour — earnings dates don't move intraday."""
    c = cache_get(f"earn_{tk}", max_age=24 * 3600)
    if c is not None:
        return c
    r = {"soon": False, "date": None}
    endpoint = "finnhub_earnings"
    if should_skip_api(endpoint, cooldown_sec=600):
        return r
    try:
        from datetime import date
        start = date.today().strftime("%Y-%m-%d")
        end   = (date.today() + timedelta(days=5)).strftime("%Y-%m-%d")
        data = fh.earnings_calendar(_from=start, to=end, symbol=tk) or {}
        record_api_success(endpoint)
        cal = data.get("earningsCalendar") or []
        if cal:
            r = {"soon": True, "date": cal[0].get("date")}
    except Exception as e:
        record_api_failure(endpoint, e)
        pass
    cache_set(f"earn_{tk}", r)
    return r


# ── Analyst recommendations (Finnhub) ──────────────────────────────────────
def get_analyst_rec(tk):
    """Net analyst score (-1..+1), buy/hold/sell counts, age_hours.
    Cached 1 hour."""
    c = cache_get(f"ar_{tk}", max_age=24 * 3600)
    if c is not None:
        return c
    r = {"net": 0.0, "buy": 0, "hold": 0, "sell": 0, "total": 0, "age_hours": None}
    endpoint = "finnhub_analyst"
    if should_skip_api(endpoint, cooldown_sec=600):
        return r
    try:
        data = fh.recommendation_trends(tk) or []
        record_api_success(endpoint)
        if data:
            d = data[0]
            buy   = (d.get("buy", 0) or 0) + (d.get("strongBuy", 0) or 0)
            sell  = (d.get("sell", 0) or 0) + (d.get("strongSell", 0) or 0)
            hold  = d.get("hold", 0) or 0
            total = buy + sell + hold
            net   = (buy - sell) / total if total > 0 else 0
            age_hours = None
            period = d.get("period")
            if period:
                try:
                    pub = datetime.strptime(period[:10], "%Y-%m-%d")
                    age_hours = max(0.0, (datetime.utcnow() - pub).total_seconds() / 3600.0)
                except Exception:
                    pass
            r = {"net": round(net, 3), "buy": buy, "hold": hold, "sell": sell,
                 "total": total, "age_hours": age_hours}
    except Exception as e:
        record_api_failure(endpoint, e)
        pass
    cache_set(f"ar_{tk}", r)
    return r


# ── Insider sentiment (Finnhub) ────────────────────────────────────────────
def get_insider_sentiment(tk):
    """Aggregated MSPR over last 90 days, plus age_hours. Cached 1 hour."""
    c = cache_get(f"is_{tk}", max_age=24 * 3600)
    if c is not None:
        return c
    r = {"sentiment": 0.0, "samples": 0, "age_hours": None}
    endpoint = "finnhub_insider"
    if should_skip_api(endpoint, cooldown_sec=600):
        return r
    try:
        end_d   = datetime.now().date()
        start_d = end_d - timedelta(days=90)
        data = fh.stock_insider_sentiment(symbol=tk,
                                          _from=start_d.strftime("%Y-%m-%d"),
                                          to=end_d.strftime("%Y-%m-%d")) or {}
        record_api_success(endpoint)
        rows = data.get("data") or []
        if rows:
            msprs = [d.get("mspr") for d in rows if d.get("mspr") is not None]
            ages = []
            for d in rows:
                y = d.get("year"); m = d.get("month")
                if y and m:
                    try:
                        dt = datetime(int(y), int(m), 15)
                        ages.append((datetime.utcnow() - dt).total_seconds() / 3600.0)
                    except Exception:
                        pass
            age_hours = min(ages) if ages else None
            if msprs:
                avg = sum(msprs) / len(msprs) / 100.0
                r = {"sentiment": round(avg, 3), "samples": len(msprs), "age_hours": age_hours}
    except Exception as e:
        record_api_failure(endpoint, e)
        pass
    cache_set(f"is_{tk}", r)
    return r
