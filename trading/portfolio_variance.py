"""Rolling covariance portfolio-risk diagnostics for buy candidates."""

import math

import pandas as pd

from trading.config import CORRELATION_CONFIG
from trading.risk import get_corr_group, get_sector
from utils.deploy_config import PYTHONANYWHERE_MODE


CORR_LOOKBACK_DAYS = int(CORRELATION_CONFIG.get("lookback_days", 30))
CORR_MIN_DAYS = int(CORRELATION_CONFIG.get("min_days", 20))
ROLLING_CORR_WEIGHT = float(CORRELATION_CONFIG.get("rolling_weight", 0.70))
STATIC_CORR_WEIGHT = float(CORRELATION_CONFIG.get("static_weight", 0.30))
VAR_SOFT_CAP_PCT = float(CORRELATION_CONFIG.get("soft_var_cap", 0.05)) * 100.0
VAR_HARD_CAP_PCT = float(CORRELATION_CONFIG.get("hard_var_cap", 0.15)) * 100.0
VAR_ABSOLUTE_CAP_PCT = float(CORRELATION_CONFIG.get("absolute_var_cap", 0.25)) * 100.0
HIGH_CORR_WARN = 0.65
HIGH_CORR_DANGER = 0.80
VAR_SOFT_SIZE_MULT = float(CORRELATION_CONFIG.get("soft_size_mult", 0.65))
VAR_HARD_MIN_GROSS_EDGE = 2.0
VAR_HARD_MIN_NET_EDGE = 1.2
DEFAULT_DAILY_VOL = 0.025


def _round(value, digits=6):
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _regime_is_stress(regime):
    if isinstance(regime, dict):
        regime = regime.get("regime_effective") or regime.get("regime")
    text = str(regime or "").lower()
    return any(k in text for k in ("bear", "panic", "risk_off"))


def _series_for(ticker, close_history):
    vals = (close_history or {}).get(ticker)
    if vals is None:
        vals = (close_history or {}).get(str(ticker).upper())
    if vals is None:
        return pd.Series(dtype="float64")
    if isinstance(vals, pd.Series):
        s = vals.copy()
    else:
        s = pd.Series(vals)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.tail(CORR_LOOKBACK_DAYS + 1)


def daily_return_series(ticker, close_history):
    s = _series_for(ticker, close_history)
    if len(s) < CORR_MIN_DAYS + 1:
        return pd.Series(dtype="float64")
    return s.pct_change().dropna().tail(CORR_LOOKBACK_DAYS)


def static_corr_assumption(ticker_a, ticker_b, sector_lookup=None,
                           corr_group_lookup=None):
    sector_lookup = sector_lookup or get_sector
    corr_group_lookup = corr_group_lookup or get_corr_group
    a = str(ticker_a or "").upper()
    b = str(ticker_b or "").upper()
    if not a or not b:
        return 0.30
    if a == b:
        return 1.0
    ga = corr_group_lookup(a)
    gb = corr_group_lookup(b)
    if ga and gb and ga == gb:
        return 0.70
    sa = sector_lookup(a)
    sb = sector_lookup(b)
    if str(sa or "").lower() in ("unknown", "other", "none"):
        sa = None
    if str(sb or "").lower() in ("unknown", "other", "none"):
        sb = None
    if sa and sb and sa == sb:
        return 0.55
    if sa and sb:
        return 0.35
    return 0.30


def smoothed_corr(ticker_a, ticker_b, close_history, regime=None,
                  sector_lookup=None, corr_group_lookup=None):
    static = static_corr_assumption(ticker_a, ticker_b, sector_lookup,
                                    corr_group_lookup)
    rolling = None
    ra = daily_return_series(ticker_a, close_history)
    rb = daily_return_series(ticker_b, close_history)
    if len(ra) >= CORR_MIN_DAYS and len(rb) >= CORR_MIN_DAYS:
        joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(joined) >= CORR_MIN_DAYS:
            val = joined.iloc[:, 0].corr(joined.iloc[:, 1])
            if val == val:
                rolling = float(val)
    corr = static if rolling is None else (
        ROLLING_CORR_WEIGHT * rolling + STATIC_CORR_WEIGHT * static
    )
    if _regime_is_stress(regime):
        corr = max(corr, min(1.0, static + 0.10))
    return max(-0.95, min(1.0, float(corr)))


def daily_vol(ticker, close_history, ctx_by_ticker=None):
    r = daily_return_series(ticker, close_history)
    if len(r) >= CORR_MIN_DAYS:
        vol = float(r.std())
        if vol > 0:
            return vol
    ctx = (ctx_by_ticker or {}).get(ticker) or (ctx_by_ticker or {}).get(str(ticker).upper()) or {}
    atr_pct = float(ctx.get("atr_pct", 0) or 0)
    if atr_pct > 0:
        return max(0.005, atr_pct / 100.0 / 1.5)
    return DEFAULT_DAILY_VOL


def portfolio_variance(value_by_ticker, close_history, total_equity,
                       ctx_by_ticker=None, regime=None, sector_lookup=None,
                       corr_group_lookup=None):
    total = max(float(total_equity or 0.0), 1.0)
    items = [(str(t).upper(), float(v or 0.0))
             for t, v in (value_by_ticker or {}).items() if float(v or 0.0) > 0]
    if not items:
        return 0.0
    variance = 0.0
    for i, (ta, va) in enumerate(items):
        wa = va / total
        vola = daily_vol(ta, close_history, ctx_by_ticker)
        for j, (tb, vb) in enumerate(items):
            wb = vb / total
            volb = daily_vol(tb, close_history, ctx_by_ticker)
            corr = 1.0 if i == j else smoothed_corr(
                ta, tb, close_history, regime, sector_lookup, corr_group_lookup
            )
            variance += wa * wb * corr * vola * volb
    return max(0.0, float(variance))


def holdings_value_map(holdings, price_by_ticker):
    values = {}
    for ticker, holding in (holdings or {}).items():
        price = (price_by_ticker or {}).get(ticker) or (price_by_ticker or {}).get(str(ticker).upper())
        if not price:
            price = holding.get("avg_cost", 0) or 0
        val = float(holding.get("shares", 0) or 0) * float(price or 0)
        if val > 0:
            values[str(ticker).upper()] = val
    return values


def load_close_history(tickers, lookback_days=CORR_LOOKBACK_DAYS):
    """Live-only convenience loader. Backtests pass panel slices instead."""
    out = {}
    for ticker in sorted({str(t).upper() for t in (tickers or []) if t}):
        series = None
        if not PYTHONANYWHERE_MODE:
            try:
                import yfinance as yf
                h = yf.Ticker(ticker).history(period="3mo")
                if h is not None and not h.empty and "Close" in h.columns:
                    series = h["Close"].tail(lookback_days + 1)
            except Exception:
                series = None
        if series is None or len(series) < CORR_MIN_DAYS + 1:
            try:
                from market.data_manager import get_daily
                from market.quotes import _append_live_bar
                df = get_daily(ticker)
                if df is not None and not df.empty and "Close" in df.columns:
                    df = _append_live_bar(df.copy(), ticker)
                    series = df["Close"].tail(lookback_days + 1)
            except Exception:
                series = None
        if series is not None and len(series) > 0:
            out[ticker] = pd.to_numeric(series, errors="coerce").dropna()
    return out


def candidate_variance_check(holdings, price_by_ticker, candidate_ticker, spend,
                             total_equity, close_history, ctx_by_ticker=None,
                             regime=None, gross_edge_pct=0.0, net_edge_pct=0.0,
                             paper_debug_override=False, sector_lookup=None,
                             corr_group_lookup=None, config=None):
    ticker = str(candidate_ticker or "").upper()
    spend = max(0.0, float(spend or 0.0))
    values_before = holdings_value_map(holdings, price_by_ticker)
    holdings_n = len(values_before)
    before = portfolio_variance(values_before, close_history, total_equity,
                                ctx_by_ticker, regime, sector_lookup,
                                corr_group_lookup)
    values_after = dict(values_before)
    values_after[ticker] = values_after.get(ticker, 0.0) + spend
    after = portfolio_variance(values_after, close_history, total_equity,
                               ctx_by_ticker, regime, sector_lookup,
                               corr_group_lookup)
    if holdings_n == 0 or before <= 0:
        increase_pct = 0.0
    else:
        increase_pct = max(0.0, (after - before) / before * 100.0)
    contribution_pct = ((max(0.0, after - before) / after * 100.0)
                        if after > 0 else 0.0)

    max_pair_corr = 0.0
    highest_corr_ticker = None
    for held in values_before:
        corr = smoothed_corr(ticker, held, close_history, regime,
                             sector_lookup, corr_group_lookup)
        if highest_corr_ticker is None or corr > max_pair_corr:
            max_pair_corr = corr
            highest_corr_ticker = held

    gross = float(gross_edge_pct or 0.0)
    net = float(net_edge_pct or 0.0)
    corr_cfg = (config or {}).get("correlation", {}) if isinstance(config, dict) else {}
    soft_cap = float(corr_cfg.get("soft_var_cap", VAR_SOFT_CAP_PCT / 100.0)) * 100.0
    hard_cap = float(corr_cfg.get("hard_var_cap", VAR_HARD_CAP_PCT / 100.0)) * 100.0
    absolute_cap = float(corr_cfg.get("absolute_var_cap", VAR_ABSOLUTE_CAP_PCT / 100.0)) * 100.0
    soft_mult = float(corr_cfg.get("soft_size_mult", VAR_SOFT_SIZE_MULT))
    size_mult = 1.0
    risk_action = "allow"
    skip_reason = None
    if holdings_n > 0 and increase_pct > soft_cap:
        size_mult = soft_mult
        risk_action = "reduce"
    if holdings_n > 0 and increase_pct > hard_cap:
        if holdings_n == 1 and increase_pct <= absolute_cap:
            risk_action = "reduce"
        elif gross >= VAR_HARD_MIN_GROSS_EDGE and net >= VAR_HARD_MIN_NET_EDGE:
            risk_action = "reduce"
        else:
            risk_action = "skip"
            skip_reason = "portfolio_variance_too_high"
    if holdings_n > 0 and increase_pct > absolute_cap and not paper_debug_override:
        risk_action = "skip"
        skip_reason = "portfolio_variance_extreme"
    if risk_action == "skip":
        size_mult = 0.0

    return {
        "ticker": ticker,
        "holdings_n": holdings_n,
        "variance_before": _round(before),
        "variance_after": _round(after),
        "variance_increase_pct": _round(increase_pct, 2),
        "candidate_risk_contribution_pct": _round(contribution_pct, 2),
        "max_pair_corr": _round(max_pair_corr, 4),
        "highest_corr_ticker": highest_corr_ticker,
        "high_corr_warning": bool(max_pair_corr >= HIGH_CORR_WARN),
        "high_corr_danger": bool(max_pair_corr >= HIGH_CORR_DANGER),
        "risk_action": risk_action,
        "size_mult": _round(size_mult, 4),
        "skip_reason": skip_reason,
        "gross_edge_pct": _round(gross, 4),
        "net_edge_pct": _round(net, 4),
        "paper_debug_override": bool(paper_debug_override),
    }


def variance_reason(diag):
    if not diag:
        return "variance n/a"
    pair = ""
    if diag.get("highest_corr_ticker"):
        pair = (f", max corr {diag.get('max_pair_corr'):.2f} vs "
                f"{diag.get('highest_corr_ticker')}")
    return (
        f"variance +{diag.get('variance_increase_pct', 0):.1f}%"
        f", risk contribution {diag.get('candidate_risk_contribution_pct', 0):.1f}%"
        f"{pair}, action={diag.get('risk_action')}"
    )
