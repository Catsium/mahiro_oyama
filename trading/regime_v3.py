"""Market breadth / sector-rotation regime model."""

import time

import pandas as pd

from trading.config import REGIME_CONFIG
from utils.deploy_config import PYTHONANYWHERE_MODE


REGIME_LABELS = (
    "strong_bull",
    "narrow_bull",
    "weak_bull",
    "choppy_neutral",
    "risk_off_neutral",
    "bear",
    "panic",
)

SECTOR_ETFS = {
    "tech": "XLK",
    "semis": "SMH",
    "consumer_disc": "XLY",
    "financials": "XLF",
    "industrials": "XLI",
    "energy": "XLE",
    "healthcare": "XLV",
    "utilities": "XLU",
    "staples": "XLP",
    "real_estate": "XLRE",
}

RISK_ON_SYMBOLS = {
    "QQQ": "qqq_spy_rs_20d",
    "HYG": "hyg_ief_rs_20d",
    "RSP": "rsp_spy_rs_20d",
    "XLU": "xlu_spy_rs_20d",
    "XLP": "xlp_spy_rs_20d",
}

BREADTH_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "JPM", "V",
    "WMT", "LLY", "MA", "XOM", "ORCL", "PG", "JNJ", "HD", "COST", "BAC",
    "ABBV", "NFLX", "CVX", "KO", "AMD", "PEP", "TMO", "ADBE", "CSCO", "CRM",
    "MCD", "ACN", "WFC", "ABT", "DIS", "INTC", "QCOM", "TXN", "IBM", "BA",
    "UBER", "SHOP", "PYPL", "SPOT", "COIN", "SQ", "PLTR", "SNOW", "DDOG", "SOFI",
    "UNH", "ISRG", "CAT", "RTX", "LIN", "PM",
]

BREADTH_UNIVERSE_PA = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "JPM", "V",
    "WMT", "LLY", "MA", "XOM", "ORCL", "PG", "JNJ", "HD", "COST", "BAC",
    "ABBV", "NFLX", "CVX", "KO", "AMD", "PEP", "TMO", "ADBE", "CSCO", "CRM",
]


def get_breadth_universe(pa_mode=None):
    use_pa = PYTHONANYWHERE_MODE if pa_mode is None else bool(pa_mode)
    return BREADTH_UNIVERSE_PA if use_pa else BREADTH_UNIVERSE


def _cfg(config=None):
    if config and "regime" in config:
        return config["regime"]
    return REGIME_CONFIG


def _series(vals):
    if vals is None:
        return pd.Series(dtype="float64")
    s = vals.copy() if isinstance(vals, pd.Series) else pd.Series(vals)
    return pd.to_numeric(s, errors="coerce").dropna()


def _ret(series, lookback=20):
    s = _series(series)
    if len(s) <= lookback:
        return None
    base = float(s.iloc[-1 - lookback])
    cur = float(s.iloc[-1])
    return (cur - base) / base if base else None


def _above_ma(series, ma):
    s = _series(series)
    if len(s) < ma:
        return None
    return float(s.iloc[-1]) > float(s.tail(ma).mean())


def _realized_vol(series):
    s = _series(series)
    rets = s.pct_change().dropna().tail(20)
    if len(rets) < 10:
        return 0.0
    return float(rets.std() * (252 ** 0.5))


def breadth_metrics(close_history, spy_series, aux_history=None, config=None):
    cfg = _cfg(config)
    short_ma = int(cfg.get("breadth_short_ma", 20))
    med_ma = int(cfg.get("breadth_med_ma", 50))
    long_ma = int(cfg.get("breadth_long_ma", 200))
    min_count = int(cfg.get("min_effective_breadth_count", 30))
    usable = []
    missing = 0
    for ticker, series in (close_history or {}).items():
        s = _series(series)
        if len(s) < med_ma:
            missing += 1
            continue
        usable.append((ticker, s))
    n = len(usable)

    def pct(predicate):
        if not usable:
            return None
        hits = sum(1 for _, s in usable if predicate(s))
        return hits / len(usable)

    ret20 = [r for _, s in usable for r in [_ret(s, 20)] if r is not None]
    spy = _series(spy_series)
    spy_above_50 = _above_ma(spy, 50)
    spy_above_200 = _above_ma(spy, 200)
    spy_ret20 = _ret(spy, 20) or 0.0
    realized_vol = _realized_vol(spy)

    aux = aux_history or {}
    qqq_spy = (_ret(aux.get("QQQ"), 20) - spy_ret20) if _ret(aux.get("QQQ"), 20) is not None else 0.0
    rsp_spy = (_ret(aux.get("RSP"), 20) - spy_ret20) if _ret(aux.get("RSP"), 20) is not None else 0.0
    hyg = _ret(aux.get("HYG"), 20)
    ief = _ret(aux.get("IEF"), 20)
    hyg_ief = (hyg - ief) if hyg is not None and ief is not None else 0.0
    xlu_spy = (_ret(aux.get("XLU"), 20) - spy_ret20) if _ret(aux.get("XLU"), 20) is not None else 0.0
    xlp_spy = (_ret(aux.get("XLP"), 20) - spy_ret20) if _ret(aux.get("XLP"), 20) is not None else 0.0

    sector_rs = {}
    for sector, etf in SECTOR_ETFS.items():
        eret = _ret(aux.get(etf), int(cfg.get("sector_rs_lookback", 20)))
        if eret is not None:
            sector_rs[sector] = round(eret - spy_ret20, 4)
    top = [k for k, _ in sorted(sector_rs.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    weak = [k for k, _ in sorted(sector_rs.items(), key=lambda kv: kv[1])[:3]]

    metrics = {
        "actual_breadth_count": n,
        "missing_breadth_count": missing,
        "min_effective_breadth_count": min_count,
        "breadth_sufficient": n >= min_count,
        "pct_above_20ma": pct(lambda s: bool(_above_ma(s, short_ma))),
        "pct_above_50ma": pct(lambda s: bool(_above_ma(s, med_ma))),
        "pct_above_200ma": pct(lambda s: bool(_above_ma(s, long_ma))),
        "pct_new_20d_high": pct(lambda s: float(s.iloc[-1]) >= float(s.tail(20).max())),
        "pct_new_20d_low": pct(lambda s: float(s.iloc[-1]) <= float(s.tail(20).min())),
        "avg_20d_return": round(sum(ret20) / len(ret20), 4) if ret20 else None,
        "median_20d_return": round(float(pd.Series(ret20).median()), 4) if ret20 else None,
        "spy_above_50ma": bool(spy_above_50) if spy_above_50 is not None else False,
        "spy_above_200ma": bool(spy_above_200) if spy_above_200 is not None else False,
        "spy_20d_return": round(spy_ret20, 4),
        "realized_vol_20d": round(realized_vol, 4),
        "qqq_spy_rs_20d": round(qqq_spy, 4),
        "rsp_spy_rs_20d": round(rsp_spy, 4),
        "hyg_ief_rs_20d": round(hyg_ief, 4),
        "xlu_spy_rs_20d": round(xlu_spy, 4),
        "xlp_spy_rs_20d": round(xlp_spy, 4),
        "defensive_strength": bool(xlu_spy > 0.03 or xlp_spy > 0.03),
        "sector_rs": sector_rs,
        "top_sectors": top,
        "weak_sectors": weak,
    }
    return metrics


def classify_regime(metrics, previous=None, config=None):
    cfg = _cfg(config)
    if not metrics.get("breadth_sufficient"):
        return "fallback", "insufficient breadth data"
    h = cfg.get("hysteresis", {})
    bbuf = float(h.get("breadth_buffer", 0.03)) if previous else 0.0
    rbuf = float(h.get("rs_buffer", 0.005)) if previous else 0.0
    b50 = metrics.get("pct_above_50ma") or 0.0
    b200 = metrics.get("pct_above_200ma") or 0.0
    spy50 = metrics.get("spy_above_50ma")
    spy200 = metrics.get("spy_above_200ma")
    rv = metrics.get("realized_vol_20d", 0.0)
    qqq = metrics.get("qqq_spy_rs_20d", 0.0)
    rsp = metrics.get("rsp_spy_rs_20d", 0.0)
    hyg = metrics.get("hyg_ief_rs_20d", 0.0)
    defensive = metrics.get("defensive_strength")

    strong50 = float(cfg.get("breadth_50_strong", 0.65))
    strong200 = float(cfg.get("breadth_200_strong", 0.60))
    weak200 = float(cfg.get("breadth_200_weak", 0.40))
    narrow_max = float(cfg.get("narrow_bull_breadth_max", 0.55))

    if not spy200 and b50 < 0.30 - bbuf and rv > 0.18:
        return "panic", "SPY below 200d, breadth <30%, realized vol high"
    if not spy200 and b200 < weak200 - bbuf:
        return "bear", "SPY below 200d and long-term breadth weak"
    if spy50 and spy200 and b50 >= strong50 + bbuf and b200 >= strong200 + bbuf:
        return "strong_bull", "SPY uptrend with broad participation"
    if spy50 and qqq > 0.03 + rbuf and rsp < -0.02 - rbuf and b50 < narrow_max - bbuf:
        return "narrow_bull", "SPY strong but breadth weak and QQQ leading"
    if spy200 and 0.45 - bbuf <= b50 < strong50 + bbuf:
        return "weak_bull", "SPY above 200d with mixed breadth"
    if defensive or hyg < -0.02 - rbuf:
        return "risk_off_neutral", "defensive rotation or credit risk-off"
    return "choppy_neutral", "mixed market with no strong breadth edge"


def apply_confirmation(state, raw_regime, reason, config=None, ts=None):
    cfg = _cfg(config)
    ts = int(ts or time.time())
    state = state if isinstance(state, dict) else {}
    current = state.get("confirmed_regime") or raw_regime
    emergency = raw_regime in ("panic", "risk_off_neutral")
    need = int(cfg.get("panic_confirm_days" if emergency else "regime_confirm_days", 2))
    if raw_regime == current:
        state.update({"pending_regime": None, "pending_days": 0,
                      "confirmed_regime": current, "last_reason": reason,
                      "updated_ts": ts})
        return current
    pending = state.get("pending_regime")
    days = int(state.get("pending_days", 0) or 0)
    days = days + 1 if pending == raw_regime else 1
    if days >= need:
        current = raw_regime
        pending = None
        days = 0
    else:
        pending = raw_regime
    state.update({"pending_regime": pending, "pending_days": days,
                  "confirmed_regime": current, "last_reason": reason,
                  "updated_ts": ts})
    return current


def legacy_kind_for_v3(label, fallback="neutral"):
    if label in ("strong_bull", "narrow_bull", "weak_bull"):
        return "bull"
    if label in ("bear", "panic"):
        return "bear"
    if label in ("choppy_neutral", "risk_off_neutral"):
        return "neutral"
    return fallback or "neutral"


def regime_risk_mult(label, config=None):
    return float(_cfg(config).get("risk_mult", {}).get(label, 1.0))


def cluster_regime_mult(label, cluster, config=None):
    table = _cfg(config).get("cluster_mult", {}).get(label, {})
    return float(table.get(cluster, 1.0))


def build_regime_v3(close_history, spy_series, aux_history=None, previous_state=None,
                    update_state=False, fallback_legacy="neutral", config=None,
                    ts=None):
    metrics = breadth_metrics(close_history, spy_series, aux_history, config)
    raw, reason = classify_regime(metrics, previous=(previous_state or {}).get("confirmed_regime"),
                                  config=config)
    fallback = raw == "fallback"
    effective = legacy_kind_for_v3(None, fallback_legacy) if fallback else raw
    if not fallback and update_state and previous_state is not None:
        effective = apply_confirmation(previous_state, raw, reason, config=config, ts=ts)
    metrics.update({
        "regime_v3_raw": raw,
        "regime_v3": effective,
        "regime_v3_effective": effective,
        "regime_v3_reason": reason,
        "regime_v3_fallback": fallback,
        "legacy_kind": legacy_kind_for_v3(effective, fallback_legacy),
        "regime_risk_mult": regime_risk_mult(effective, config),
    })
    return metrics


def load_live_close_history(tickers, lookback_days=240):
    out = {}
    for ticker in tickers or []:
        if not PYTHONANYWHERE_MODE:
            try:
                import yfinance as yf
                h = yf.Ticker(ticker).history(period="1y")
                if h is not None and not h.empty and "Close" in h.columns:
                    out[ticker] = h["Close"].tail(lookback_days)
                    continue
            except Exception:
                pass
        try:
            from market.data_manager import get_daily
            from market.quotes import _append_live_bar
            df = get_daily(ticker)
            if df is not None and not df.empty and "Close" in df.columns:
                out[ticker] = _append_live_bar(df.copy(), ticker)["Close"].tail(lookback_days)
        except Exception:
            pass
    return out
