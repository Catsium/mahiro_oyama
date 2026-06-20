"""Walk-forward portfolio backtest — the measurement tool for the live decision rules.

Runs the SAME entry signal (`get_recommendation`, pure-technical), the SAME exit math
(`trading.exits`: cost-aware ratchet, ATR stop, ATR trail), the SAME position/sector/corr
caps, confidence-weighted sizing, Kelly + cold-streak downsizing, and the SAME $0.99/trade
commission as the live bot — over multiple years of Stooq daily bars — then compares against
buy-and-hold SPY over the identical test window.

Two rule-sets, so one run quantifies what the fixes changed:
  - mode="fixed"  : post-fix (cost-aware ratchet, $400 floor, daily-ATR trail, NET P&L)
  - mode="legacy" : pre-fix  (+1.5%→+0.1% breakeven, $50 floor, regime-static trail, GROSS P&L)

Walk-forward: V2 attribution buckets are learned on the train window from 5-day
forward returns, then reused for EV ranking and category multipliers in the held-out
test window. Exit buckets are tracked from completed train-window sells.

CAVEAT: pure-technical only. News / analyst / insider / intraday signals can't be
reconstructed historically, so this validates the technical + exit + sizing + cost layer —
exactly the layer the fixes touch — not the live news pipeline. Daily granularity also means
the 20-min min-hold and hour-based aging/time-decay thresholds are approximated by trading
days. Partial-take / degrade-trim are omitted (they apply identically to both modes, so they
don't affect the fixed-vs-legacy comparison).
"""
import time
from dataclasses import dataclass
import pandas as pd

from market.data_manager import get_daily
from trading.catalysts import classify_catalyst
from trading.config import DEFAULT_CONFIG, active_config, config_hash, merge_config
from trading.indicators import _ctx_from_series, classify_vol_regime
from trading.signals import get_recommendation
from trading.attribution import (
    attribution_signal_weights, ensure_attribution_state, exit_profile,
    record_entry_event, record_exit_event, update_entry_buckets,
)
from trading.risk import SECTOR_MAP, get_sector, get_corr_group
from trading.exits import (round_trip_cost_pct, breakeven_lock_pct,
                           dynamic_stop_pct, dynamic_trail_width)
from trading.exit_ladders import apply_regime_exit_tightening, compose_exit_profile
from trading.portfolio_variance import (
    CORR_LOOKBACK_DAYS, candidate_variance_check, variance_reason,
)
from trading.regime_v3 import build_regime_v3, cluster_regime_mult, regime_risk_mult
from trading.sizing import (
    COMMISSION_PER_TRADE, PARTIAL_TAKE_PCT, entry_cluster, rank_candidates,
)
from trading.bot import MIN_POSITION_USD, SCAN_UNIVERSE
from utils.cache import cache_get, cache_set
from utils.deploy_config import PYTHONANYWHERE_MODE

STARTING_CASH      = 10000.0
MAX_POSITIONS      = 10
BOT_MAX_BUYS       = 5
MAX_POS_PCT        = 0.35
MAX_SECTOR_PCT     = 0.55
MAX_CORR_GROUP_PCT = 0.45
MIN_CASH_RESERVE   = 0.02

ROLLING_MODES = (
    "technical_only",
    "technical_regime",
    "technical_regime_news",
    "technical_regime_news_analyst_insider",
    "full_current",
)
EXTERNAL_MODES = {
    "technical_regime_news",
    "technical_regime_news_analyst_insider",
    "full_current",
}
CAVEATS = [
    "non_price_historical_provider_empty",
    "daily_bar_approximation",
    "no_intraday_replay",
]


@dataclass
class BacktestContext:
    asof_date: object
    mode: str
    historical_provider: object
    allow_network: bool = False
    data_frequency: str = "daily"
    config: object = None


class HistoricalSignalProvider:
    name = "HistoricalSignalProvider"

    def news(self, ticker, asof_date):
        return [], 0.0

    def earnings(self, ticker, asof_date):
        return {"soon": False, "date": None}

    def analyst(self, ticker, asof_date):
        return {"net": 0.0, "buy": 0, "hold": 0, "sell": 0,
                "total": 0, "age_hours": None}

    def insider(self, ticker, asof_date):
        return {"sentiment": 0.0, "samples": 0, "age_hours": None}


class NullHistoricalSignalProvider(HistoricalSignalProvider):
    name = "NullHistoricalSignalProvider"


# ── Data panel ───────────────────────────────────────────────────────────────
def _default_universe():
    return list(SCAN_UNIVERSE)


def _history(tk, years):
    """Daily OHLCV with a normalized (tz-naive, midnight) DatetimeIndex.

    yfinance first — it works off-PythonAnywhere and returns long multi-year history in
    one call. Stooq is the fallback, but its CSV endpoint now requires an API key from
    many IPs (captcha wall), so it can't be relied on for backtests from a dev box. Both
    are existing dependencies — no new library."""
    if not PYTHONANYWHERE_MODE:
        try:
            import yfinance as yf
            period = "max" if years > 10 else f"{int(years) + 1}y"
            h = yf.Ticker(tk).history(period=period)
            if h is not None and not h.empty and "Close" in h.columns:
                h = h[["Open", "High", "Low", "Close", "Volume"]].copy()
                h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
                return h
        except Exception:
            pass
    df = get_daily(tk, full=True)
    if df is not None and not df.empty:
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def _load_panel(universe, years):
    """Fetch daily history for the universe + SPY, restricted to the last `years`.
    Returns (panel: {tk: df}, spy_df, dates list)."""
    spy = _history("SPY", years)
    if spy is None or spy.empty:
        raise RuntimeError("SPY history unavailable (yfinance + Stooq both failed). "
                           "Stooq now needs an API key on many IPs; run where yfinance works.")
    cutoff = spy.index[-1] - pd.Timedelta(days=int(years * 365.25))
    spy = spy[spy.index >= cutoff]
    dates = list(spy.index)
    panel = {}
    for tk in universe:
        if tk == "SPY":
            continue
        df = _history(tk, years)
        if df is None or df.empty:
            continue
        df = df[df.index >= cutoff]
        if len(df) >= 60:
            panel[tk] = df
    return panel, spy, dates


def _price(df, date):
    try:
        v = float(df.at[date, "Close"])
        return v if v > 0 else None
    except Exception:
        return None


def _ctx_at(df, date, window):
    """Build a pure-technical ctx as of `date` from a bounded trailing window (keeps the
    per-day cost O(window) instead of O(n))."""
    sub = df.loc[:date]
    if len(sub) < 30:
        return {}
    sub = sub.iloc[-window:]
    return _ctx_from_series(sub["Close"].round(2), df=sub)


def _mode_flags(mode):
    mode = mode or "full_current"
    return {
        "regime": mode != "technical_only",
        "news": mode in ("technical_regime_news",
                         "technical_regime_news_analyst_insider",
                         "full_current"),
        "analyst": mode in ("technical_regime_news_analyst_insider",
                            "full_current"),
        "insider": mode in ("technical_regime_news_analyst_insider",
                            "full_current"),
        "earnings": mode == "full_current",
    }


def _coverage_blank():
    return {
        "news_queries": 0, "news_events": 0,
        "earnings_queries": 0, "earnings_events": 0,
        "analyst_queries": 0, "analyst_events": 0,
        "insider_queries": 0, "insider_events": 0,
    }


def _coverage_add(dst, src):
    for key, val in (src or {}).items():
        dst[key] = dst.get(key, 0) + (val or 0)
    return dst


def _coverage_has_external(coverage):
    return any((coverage or {}).get(k, 0) > 0 for k in (
        "news_events", "earnings_events", "analyst_events", "insider_events"
    ))


def _historical_inputs(ticker, ctx, coverage):
    flags = _mode_flags(ctx.mode)
    provider = ctx.historical_provider
    out = {"sent": 0.0, "articles": [], "earnings": None,
           "analyst": None, "insider": None}
    if flags["news"]:
        coverage["news_queries"] += 1
        articles, sent = provider.news(ticker, ctx.asof_date)
        out["articles"] = articles or []
        out["sent"] = float(sent or 0.0)
        if out["articles"] or out["sent"]:
            coverage["news_events"] += 1
    if flags["earnings"]:
        coverage["earnings_queries"] += 1
        earn = provider.earnings(ticker, ctx.asof_date) or {}
        out["earnings"] = earn
        if earn.get("soon"):
            coverage["earnings_events"] += 1
    if flags["analyst"]:
        coverage["analyst_queries"] += 1
        analyst = provider.analyst(ticker, ctx.asof_date) or {}
        out["analyst"] = analyst
        if analyst.get("total", 0) > 0:
            coverage["analyst_events"] += 1
    if flags["insider"]:
        coverage["insider_queries"] += 1
        insider = provider.insider(ticker, ctx.asof_date) or {}
        out["insider"] = insider
        if insider.get("samples", 0) > 0:
            coverage["insider_events"] += 1
    return out


def _recommendation_for_backtest(ticker, ctx, tech_ctx, regime, weights, coverage):
    flags = _mode_flags(ctx.mode)
    ext = _historical_inputs(ticker, ctx, coverage)
    use_external = any(flags[k] for k in ("news", "analyst", "insider", "earnings"))
    rec = get_recommendation(
        ext["sent"], tech_ctx,
        regime=regime if flags["regime"] else None,
        earnings=ext["earnings"],
        analyst=ext["analyst"],
        insider=ext["insider"],
        news_articles=ext["articles"],
        pure_technical=not use_external,
        weights=weights,
        allow_live_risk=ctx.allow_network,
        config=ctx.config or DEFAULT_CONFIG,
    )
    catalyst = classify_catalyst(
        ext["articles"], ext["earnings"], ext["analyst"], ext["insider"],
        tech_ctx, config=ctx.config or DEFAULT_CONFIG,
    )
    rec = dict(rec)
    rec["catalyst"] = catalyst
    return rec


def _sector_for_backtest(ticker, ctx=None):
    if ctx is not None and not ctx.allow_network:
        return SECTOR_MAP.get(ticker.upper(), "unknown")
    return get_sector(ticker)


def _rolling_windows(dates, train_days=252, test_days=63, step_days=21):
    windows = []
    train_days = int(train_days)
    test_days = int(test_days)
    step_days = int(step_days)
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        return windows
    start = 0
    while start + train_days + test_days <= len(dates):
        train = dates[start:start + train_days]
        test = dates[start + train_days:start + train_days + test_days]
        windows.append({
            "index": len(windows),
            "train_dates": train,
            "test_dates": test,
            "train_period": [str(train[0].date()), str(train[-1].date())],
            "test_period": [str(test[0].date()), str(test[-1].date())],
        })
        start += step_days
    return windows


def _forward_at(panel, ticker, dates, di, days, price):
    j = di + days
    if j >= len(dates):
        return None
    fut = _price(panel.get(ticker), dates[j])
    if fut is None or not price:
        return None
    return round((fut - price) / price * 100.0, 4)


def _append_candidate_log(logs, entry, log_sample_size=500, include_full_logs=False):
    if logs is None:
        return
    if include_full_logs or len(logs) < int(log_sample_size or 0):
        logs.append(entry)


def _panel_close_history(panel, date, tickers, lookback_days=CORR_LOOKBACK_DAYS):
    out = {}
    for tk in tickers or []:
        df = panel.get(tk)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        sub = df.loc[:date, "Close"].tail(lookback_days + 1)
        if len(sub) > 0:
            out[tk] = sub
    return out


def _buy_hold_for(df, dates):
    if df is None or df.empty or not dates:
        return None
    return _buy_hold(df, dates)


# ── Regime (SPY-derived, no intraday/credit — not reconstructable historically) ──
def _spy_regime(spy_df, date, panel=None, config=None):
    sub = spy_df.loc[:date, "Close"]
    if len(sub) < 51:
        return {"regime": "neutral", "spy_mom_30d": 0.0}, 1.0
    cur  = float(sub.iloc[-1])
    ma50 = float(sub.tail(50).mean())
    base = float(sub.iloc[-22]) if len(sub) > 22 else float(sub.iloc[0])
    mom  = (cur - base) / base * 100 if base else 0.0
    if   cur > ma50 and mom >  2: regime = "bull"
    elif cur < ma50 and mom < -2: regime = "bear"
    else:                          regime = "neutral"
    # Realized-vol gate, mirroring trading.risk.get_vix thresholds.
    vix_mult = 1.0
    rets = sub.pct_change().dropna().tail(20)
    if len(rets) >= 10:
        rv = float(rets.std() * (252 ** 0.5) * 100)
        if   rv > 28: vix_mult = 0.0
        elif rv > 18: vix_mult = 0.5
    out = {"regime": regime, "spy_mom_30d": round(mom, 2)}
    if panel is not None:
        close_hist = {
            tk: df.loc[:date, "Close"].tail(240)
            for tk, df in (panel or {}).items()
            if df is not None and not df.empty and "Close" in df.columns
        }
        v3 = build_regime_v3(close_hist, spy_df.loc[:date, "Close"], {},
                             fallback_legacy=regime, config=config)
        out.update(v3)
    return out, vix_mult


def _regime_params(kind):
    """(stop_loss_pct, trail_static, regime_size_mult) — mirrors trading.bot."""
    if kind == "bull": return -8.0, 5.0, 1.0
    if kind == "bear": return -4.0, 3.0, 0.5
    return -6.0, 4.0, 0.8


def _kelly_mult(outcomes):
    """Half-Kelly from net outcomes, clamped 0.10..1.5 (mirrors trading.bot)."""
    if len(outcomes) < 20:
        return 1.0
    recent = outcomes[-50:]
    wins = [o for o in recent if o > 0]
    losses = [abs(o) for o in recent if o <= 0]
    if not wins or not losses:
        return 1.0
    wr = len(wins) / len(recent)
    b_ratio = (sum(wins) / len(wins)) / max(sum(losses) / len(losses), 0.01)
    f_full = wr - (1 - wr) / b_ratio
    return max(0.10, min(1.5, 0.5 * f_full))


# ── Core simulation ──────────────────────────────────────────────────────────
def _learn_forward_edges(panel, spy_df, dates, weights, window, mode="technical_regime",
                         provider=None, config=None):
    """Learn V2 entry attribution buckets from train-window forward returns."""
    state = {}
    ensure_attribution_state(state)
    provider = provider or NullHistoricalSignalProvider()
    cfg = config or DEFAULT_CONFIG
    coverage = _coverage_blank()
    for i, date in enumerate(dates):
        regime, _ = _spy_regime(spy_df, date, panel=panel, config=cfg)
        rk = regime["regime"]
        bt_ctx = BacktestContext(date, mode, provider, allow_network=False,
                                 config=cfg)
        for tk, df in panel.items():
            pr = _price(df, date)
            if pr is None or pr <= 0:
                continue
            ctx = _ctx_at(df, date, window)
            if not ctx:
                continue
            rec = _recommendation_for_backtest(tk, bt_ctx, ctx, regime, weights,
                                               coverage)
            if rec["cls"] not in ("buy", "strong-buy"):
                continue
            cluster = entry_cluster(rec, ctx)
            event = record_entry_event(
                state,
                {"ticker": tk, "source": "backtest", "rec": rec, "ctx": ctx,
                 "price": pr, "cluster": cluster, "friction": {"total_pct": 0.0}},
                "executed", "backtest train", ts=int(date.timestamp()), regime=rk,
            )
            j = i + 5
            if j >= len(dates):
                continue
            fut = _price(df, dates[j])
            if fut is None or fut <= 0:
                continue
            ret = (fut - pr) / pr * 100.0
            event.setdefault("forward_returns", {})["5d"] = round(ret, 4)
            event["mfe_pct"] = max(0.0, round(ret, 4))
            event["mae_pct"] = min(0.0, round(ret, 4))
            update_entry_buckets(state, event, "5d")
    state["coverage"] = coverage
    state["config_hash"] = config_hash(cfg)
    return state


def _simulate(panel, spy_df, dates, weights, mode, learn, window, edge_stats=None,
              bt_context=None, candidate_logs=None, log_sample_size=500,
              include_full_logs=False, friction_mult=1.0, slippage_bps_per_side=0.0):
    """One pass over dates using frozen weights and optional V2 attribution state."""
    legacy   = (mode == "legacy")
    min_pos  = 50.0 if legacy else float(MIN_POSITION_USD)
    cash     = STARTING_CASH
    holdings = {}                       # tk -> position dict
    trades   = []                       # exit records (with net pnl + reason)
    outcomes = []                       # net pnl_pct of closed trades (for Kelly)
    equity_curve = []
    bt = edge_stats or {}
    ensure_attribution_state(bt)
    wdict = dict(weights or {})
    provider = (bt_context.historical_provider if bt_context else
                NullHistoricalSignalProvider())
    allow_network = bool(bt_context.allow_network) if bt_context else True
    sim_mode = bt_context.mode if bt_context else mode
    coverage = _coverage_blank()
    cfg = bt_context.config if bt_context and bt_context.config else DEFAULT_CONFIG
    commission = COMMISSION_PER_TRADE * max(0.0, float(friction_mult or 1.0))
    slip_bps = max(0.0, float(slippage_bps_per_side or 0.0))
    slippage_pct_rt = slip_bps * 2.0 / 100.0
    slippage_cash_side = slip_bps / 10000.0

    for di, date in enumerate(dates):
        regime, vix_mult = _spy_regime(spy_df, date, panel=panel, config=cfg)
        rk = regime["regime"]
        stop_loss_pct, trail_static, regime_size_mult = _regime_params(rk)
        env = regime_size_mult * vix_mult

        # snapshot price + ctx + rec for every ticker trading today
        day = {}
        day_ctx = BacktestContext(date, sim_mode, provider,
                                  allow_network=allow_network,
                                  config=cfg)
        for tk, df in panel.items():
            pr = _price(df, date)
            if pr is None:
                continue
            ctx = _ctx_at(df, date, window)
            if not ctx:
                continue
            rec = _recommendation_for_backtest(tk, day_ctx, ctx, regime, wdict,
                                               coverage)
            day[tk] = {"price": pr, "ctx": ctx, "rec": rec}

        # ── SELL pass ────────────────────────────────────────────────────────
        for tk in list(holdings.keys()):
            snap = day.get(tk)
            if not snap:
                continue
            h = holdings[tk]; pr = snap["price"]; ctx = snap["ctx"]; rec = snap["rec"]
            learned = exit_profile(bt, rk, h.get("entry_cluster")) if not legacy else {}
            profile = compose_exit_profile(h.get("entry_cluster"), learned) if not legacy else None
            if profile:
                profile = apply_regime_exit_tightening(
                    profile, regime.get("regime_v3_effective") or regime.get("regime_v3")
                )
                rv = float(regime.get("realized_vol_20d", 0) or 0)
                if rv <= 1.0:
                    rv *= 100.0
                if rv > 25:
                    profile["failure_timeout_days"] = round(max(1.0, profile.get("failure_timeout_days", 1.0) * 0.50), 4)
                elif rv > 18:
                    profile["failure_timeout_days"] = round(max(1.0, profile.get("failure_timeout_days", 1.0) * 0.75), 4)
            sell, key = _exit_decision(h, pr, ctx, rec, rk, stop_loss_pct,
                                       trail_static, di, legacy, profile=profile)
            if not sell:
                continue
            sh = h["shares"]; avg = h["avg_cost"]
            cost_basis = sh * avg
            gross = (pr - avg) / avg * 100 if avg else 0
            net   = ((sh * pr - cost_basis - 2 * commission) / cost_basis * 100
                     if cost_basis else 0)
            net -= slippage_pct_rt
            pnl_for_learning = gross if legacy else net
            cash += sh * pr - commission - (sh * pr * slippage_cash_side)
            outcomes.append(pnl_for_learning)
            if profile:
                h["exit_ladder_profile"] = profile
            trades.append({"ticker": tk, "exit_reason": key,
                           "pnl_pct": round(net, 2), "gross_pnl_pct": round(gross, 2),
                           "days_held": di - h["entry_i"],
                           "entry_cluster": h.get("entry_cluster"),
                           "exit_ladder_profile": profile,
                           "shadow_old_exit": h.get("shadow_old_exit"),
                           "old_exit_shadow_pnl_pct": (h.get("shadow_old_exit") or {}).get("pnl_pct")})
            record_exit_event(bt, tk, h, key, net, pr, ts=int(date.timestamp()),
                              regime=h.get("entry_regime") or rk)
            del holdings[tk]

        # ── BUY pass ─────────────────────────────────────────────────────────
        regime_v3_label = regime.get("regime_v3_effective") or regime.get("regime_v3")
        if vix_mult <= 0 or (
                regime_v3_label == "panic"
                and not bool(bt.get("paper_debug_override", False))):
            equity_curve.append((date, _equity(cash, holdings, day)))
            continue

        pt = _equity(cash, holdings, day)
        cash_floor = pt * MIN_CASH_RESERVE
        spendable = max(0.0, cash - cash_floor)
        if spendable < min_pos:
            equity_curve.append((date, pt))
            continue

        # sector / corr exposure from current holdings
        sector_value, corr_value = {}, {}
        for tk, h in holdings.items():
            snap = day.get(tk)
            if not snap:
                continue
            val = h["shares"] * snap["price"]
            sec = _sector_for_backtest(tk, day_ctx)
            if sec:
                sector_value[sec] = sector_value.get(sec, 0) + val
            grp = get_corr_group(tk)
            if grp:
                corr_value[grp] = corr_value.get(grp, 0) + val

        cands = []
        for tk, snap in day.items():
            if tk in holdings:
                continue                                   # no pyramiding in backtest
            rec = snap["rec"]; ctx = snap["ctx"]
            if rec["cls"] not in ("buy", "strong-buy"):
                continue
            if _sector_for_backtest(tk, day_ctx) is None:
                continue
            if rk == "bear" and not (ctx.get("rsi", 100) < 35 or ctx.get("is_dip")):
                continue
            if rk == "neutral" and ctx.get("adx", 100) < 20 and not ctx.get("is_dip"):
                continue
            vr = classify_vol_regime(ctx)
            if vr == "explosive" and ctx.get("rsi", 50) < 35:
                continue
            cluster = entry_cluster(rec, ctx)
            regime_v3 = regime.get("regime_v3_effective") or regime.get("regime_v3")
            catalyst = rec.get("catalyst") or {}
            cands.append({"ticker": tk, "source": "watchlist", "rec": rec,
                          "ctx": ctx, "price": snap["price"],
                          "cluster": cluster,
                          "config_hash": config_hash(day_ctx.config or DEFAULT_CONFIG),
                          "regime_v3": regime_v3,
                          "regime_reason": regime.get("regime_v3_reason"),
                          "regime_risk_mult": regime_risk_mult(regime_v3, day_ctx.config),
                          "cluster_regime_mult": cluster_regime_mult(regime_v3, cluster, day_ctx.config),
                          "top_sectors": regime.get("top_sectors", []),
                          "catalyst_type": catalyst.get("type"),
                          "catalyst_score_shadow": catalyst.get("score_shadow"),
                          "catalyst_confirmed": catalyst.get("confirmed")})

        kelly = _kelly_mult(outcomes)
        last5 = outcomes[-5:]
        streak_losses = sum(1 for o in last5 if o <= 0)
        streak_mult = max(0.5, 1.0 - (streak_losses - 2) * 0.15) if streak_losses >= 3 else 1.0
        ranked = rank_candidates(cands, pt, stop_loss_pct, rk, vix_mult, streak_mult,
                                 kelly, bt,
                                 min_position_usd=min_pos,
                                 commission=commission,
                                 config=day_ctx.config or DEFAULT_CONFIG)
        close_history = _panel_close_history(
            panel, date, set(day.keys()) | set(holdings.keys())
        )
        ctx_by_ticker = {tk: snap["ctx"] for tk, snap in day.items()}
        price_by_ticker = {tk: snap["price"] for tk, snap in day.items()}
        if not legacy:
            for c in ranked:
                if not c.get("tradable"):
                    continue
                tk = c["ticker"]
                spend0 = min(c["risk"]["target_notional"], cash - cash_floor)
                spend0 = min(spend0, max(0, pt * MAX_POS_PCT))
                sec0 = _sector_for_backtest(tk, day_ctx)
                spend0 = min(spend0, max(0, pt * MAX_SECTOR_PCT - sector_value.get(sec0, 0)))
                grp0 = get_corr_group(tk)
                if grp0:
                    spend0 = min(spend0, max(0, pt * MAX_CORR_GROUP_PCT - corr_value.get(grp0, 0)))
                diag = candidate_variance_check(
                    holdings, price_by_ticker, tk, spend0, pt, close_history,
                    ctx_by_ticker=ctx_by_ticker, regime=rk,
                    gross_edge_pct=c.get("gross_edge_pct", 0),
                    net_edge_pct=c.get("net_edge_pct", 0),
                    paper_debug_override=bool(bt.get("paper_debug_override", False)),
                    sector_lookup=lambda sym: _sector_for_backtest(sym, day_ctx),
                    corr_group_lookup=get_corr_group,
                    config=day_ctx.config or DEFAULT_CONFIG,
                )
                c["portfolio_variance"] = diag
                if diag.get("risk_action") == "skip":
                    c["tradable"] = False
                    c["rank_reason"] = f"{diag.get('skip_reason')}: {variance_reason(diag)}"
        picks = [c for c in ranked if c.get("tradable")][:BOT_MAX_BUYS]
        pick_ids = {c["ticker"] for c in picks}
        for c in ranked:
            pr = c.get("price", 0)
            decision = "BUY" if c["ticker"] in pick_ids and c.get("tradable") else "SKIP"
            skip_reason = None if decision == "BUY" else (
                c.get("rank_reason") if not c.get("tradable") else "not_selected"
            )
            _append_candidate_log(candidate_logs, {
                "date": str(date.date()),
                "ticker": c["ticker"],
                "mode": sim_mode,
                "signal": c["rec"].get("signal"),
                "confidence": c["rec"].get("confidence"),
                "expected_edge_pct": c.get("gross_edge_pct"),
                "net_edge_pct": c.get("net_edge_pct"),
                "friction_pct": (c.get("friction") or {}).get("total_pct"),
                "decision": decision,
                "skip_reason": skip_reason,
                "position_size": ((c.get("risk") or {}).get("target_notional")
                                  if decision == "BUY" else 0),
                "portfolio_variance": c.get("portfolio_variance"),
                "config_hash": c.get("config_hash"),
                "regime_v3": c.get("regime_v3"),
                "regime_reason": c.get("regime_reason"),
                "regime_risk_mult": c.get("regime_risk_mult"),
                "cluster_regime_mult": c.get("cluster_regime_mult"),
                "top_sectors": c.get("top_sectors"),
                "catalyst_type": c.get("catalyst_type"),
                "catalyst_score_shadow": c.get("catalyst_score_shadow"),
                "catalyst_confirmed": c.get("catalyst_confirmed"),
                "forward_1d": _forward_at(panel, c["ticker"], dates, di, 1, pr),
                "forward_3d": _forward_at(panel, c["ticker"], dates, di, 3, pr),
                "forward_5d": _forward_at(panel, c["ticker"], dates, di, 5, pr),
            }, log_sample_size, include_full_logs)
        if not picks:
            equity_curve.append((date, pt))
            continue

        for cand in picks:
            if cash - cash_floor < min_pos:
                break
            if len(holdings) >= MAX_POSITIONS:
                break
            tk = cand["ticker"]; rec = cand["rec"]; ctx = cand["ctx"]; pr = cand["price"]
            total_conf = max(rec.get("confidence", 1), 1)
            env_eff = 1.0
            size_conf = max(rec.get("sizing_confidence", rec["confidence"]), 1)
            weight = size_conf / total_conf if total_conf else 1.0 / len(picks)
            # ticker tilt (dip / overbought / explosive) — mirrors trading.bot
            rsi = ctx.get("rsi", 50); is_dip = bool(ctx.get("is_dip"))
            mh = ctx.get("macd_hist", 0) or 0; mhp = ctx.get("macd_hist_prev", 0) or 0
            if   is_dip and rsi < 30:           tmult = 1.5
            elif is_dip:                         tmult = 1.25
            elif rsi >= 70 and mh <= mhp:        tmult = 0.5
            else:                                tmult = 1.0
            if classify_vol_regime(ctx) == "explosive" and not is_dip:
                tmult *= 0.7
            combined = weight * 0.7 * kelly * env_eff * tmult
            spend = spendable * combined
            spend = cand["risk"]["target_notional"]
            spend = min(spend, cash - cash_floor)
            # position / sector / corr caps
            spend = min(spend, max(0, pt * MAX_POS_PCT))
            sec = _sector_for_backtest(tk, day_ctx)
            spend = min(spend, max(0, pt * MAX_SECTOR_PCT - sector_value.get(sec, 0)))
            grp = get_corr_group(tk)
            if grp:
                spend = min(spend, max(0, pt * MAX_CORR_GROUP_PCT - corr_value.get(grp, 0)))
            if cand.get("portfolio_variance", {}).get("risk_action") == "skip":
                continue
            if cand.get("portfolio_variance", {}).get("size_mult", 1.0) < 1.0:
                spend *= cand["portfolio_variance"].get("size_mult", 1.0)
            if spend < min_pos:
                continue
            sh = spend / pr
            if sh <= 0:
                continue
            cost = sh * pr
            cash -= (cost + commission + cost * slippage_cash_side)
            holdings[tk] = {
                "shares": sh, "avg_cost": pr, "peak": pr, "trough": pr,
                "peak_pnl_pct": 0.0,
                "entry_i": di, "entry_categories": rec.get("categories", {}),
                "entry_ts": int(date.timestamp()),
                "entry_regime": rk, "entry_confidence": rec["confidence"],
                "entry_cluster": cand.get("cluster"),
                "entry_expected_edge_pct": cand.get("gross_edge_pct"),
                "entry_net_edge_pct": cand.get("net_edge_pct"),
                "entry_atr_pct": ctx.get("atr_pct"),
                "entry_reasons": rec.get("reasons", []),
                "portfolio_variance": cand.get("portfolio_variance"),
                "entry_snapshot": {"market_regime": rk, "entry_cluster": cand.get("cluster"),
                                   "portfolio_variance": cand.get("portfolio_variance"),
                                   "config_hash": cand.get("config_hash"),
                                   "regime_v3": cand.get("regime_v3"),
                                   "catalyst_type": cand.get("catalyst_type")},
                "config_hash": cand.get("config_hash"),
                "regime_v3": cand.get("regime_v3"),
                "catalyst_type": cand.get("catalyst_type"),
                "catalyst_score_shadow": cand.get("catalyst_score_shadow"),
                "catalyst_confirmed": cand.get("catalyst_confirmed"),
            }
            sector_value[sec] = sector_value.get(sec, 0) + cost
            if grp:
                corr_value[grp] = corr_value.get(grp, 0) + cost

        equity_curve.append((date, _equity(cash, holdings, day)))

    # mark-to-last using the final day's prices
    final_equity = equity_curve[-1][1] if equity_curve else cash
    return {"equity_curve": equity_curve, "trades": trades, "outcomes": outcomes,
            "final_equity": final_equity, "weights": dict(wdict),
            "edge_stats": bt, "coverage": coverage,
            "commission_per_trade": commission,
            "slippage_bps_per_side": slippage_bps_per_side,
            "config_hash": config_hash(cfg)}


def _equity(cash, holdings, day):
    total = cash
    for tk, h in holdings.items():
        snap = day.get(tk)
        px = snap["price"] if snap else h["avg_cost"]
        total += h["shares"] * px
    return total


def _exit_decision(h, pr, ctx, rec, rk, stop_loss_pct, trail_static, di, legacy,
                   profile=None):
    """Exit ladder mirroring trading.bot's SELL pass, using trading.exits for the
    numeric stop/trail/lock. Returns (sell: bool, exit_reason_key: str)."""
    avg = h["avg_cost"]
    peak = max(h.get("peak", avg), pr); h["peak"] = peak
    h["trough"] = min(h.get("trough", avg), pr)
    pnl_pct   = (pr - avg) / avg * 100 if avg else 0
    trail_pct = (pr - peak) / peak * 100 if peak else 0
    peak_pnl  = max(h.get("peak_pnl_pct", 0), pnl_pct); h["peak_pnl_pct"] = peak_pnl
    cls = rec["cls"]
    atr_pct = (ctx or {}).get("atr_pct", 0) or 0
    held_days  = di - h["entry_i"]
    held_hours = held_days * 6.5            # daily-bar proxy for the hour thresholds
    min_hold_ok = held_days >= 1
    profile = profile or {}
    atr_stop_mult = profile.get("atr_stop_mult", 1.0)
    trail_mult = profile.get("trail_mult", 1.0)
    trail_start = profile.get("trail_start_pct", 0.02) * 100.0
    partial_take = profile.get("partial_take_pct", PARTIAL_TAKE_PCT / 100.0) * 100.0
    failure_timeout = profile.get("failure_timeout_days")
    max_hold_days = profile.get("max_hold_days")

    def _decision(use_profile):
        stop_mult = atr_stop_mult if use_profile else 1.0
        trail_width_mult = trail_mult if use_profile else 1.0
        start_pct = trail_start if use_profile else 2.0
        dynamic_stop_local = dynamic_stop_pct(atr_pct * stop_mult,
                                              stop_loss_pct * stop_mult)
        if legacy:
            dyn_trail = trail_static
        else:
            dyn_trail, _ = dynamic_trail_width(0, atr_pct, trail_static)
            dyn_trail *= trail_width_mult
        trail_triggered_local = (pnl_pct > start_pct and trail_pct <= -dyn_trail)

        if legacy:
            if peak_pnl >= 1.5:
                effective_stop = max(dynamic_stop_local, 0.1)
            elif held_hours > 2 and pnl_pct < 0 and peak_pnl < 1.0:
                effective_stop = dynamic_stop_local * 0.5
            else:
                effective_stop = dynamic_stop_local
        else:
            rt = round_trip_cost_pct(h["shares"] * pr, COMMISSION_PER_TRADE)
            lock = breakeven_lock_pct(peak_pnl, rt)
            if lock is not None:
                effective_stop = max(dynamic_stop_local, lock)
            elif held_hours > 2 and pnl_pct < 0 and peak_pnl < 1.0:
                effective_stop = dynamic_stop_local * 0.5
            else:
                effective_stop = dynamic_stop_local

        if pnl_pct <= effective_stop:
            return True, ("breakeven" if legacy and effective_stop > 0
                          else "ratchet" if effective_stop > 0 else "loss")
        if not min_hold_ok:
            return False, ""
        if trail_triggered_local:
            return True, "trail"
        if use_profile and failure_timeout is not None:
            if held_days >= failure_timeout and peak_pnl < start_pct and pnl_pct <= 0.25:
                return True, f"{profile.get('cluster', 'mixed')}_failure_timeout"
            if max_hold_days is not None and held_days >= max_hold_days and peak_pnl < partial_take:
                return True, f"{profile.get('cluster', 'mixed')}_max_hold"
        if cls in ("sell", "strong-sell") and pnl_pct < 0:
            return True, "loss"
        if cls == "strong-sell" and pnl_pct >= 0:
            return True, "signal_flip_profit"
        cur_e = (ctx or {}).get("current", 0) or 0
        ma30  = (ctx or {}).get("ma30", 0) or 0
        if ma30 > 0 and cur_e > 0 and pnl_pct < 0 and cur_e < ma30 * 0.98:
            return True, "trend_failure"
        aging_base = {"bull": 18, "bear": 6, "neutral": 12}[rk]
        aging = min(aging_base * max((max_hold_days or 10) / 10.0, 0.25),
                    (max_hold_days or 10) * 6.5) if use_profile else aging_base
        if held_hours > aging and abs(pnl_pct) < 1.0 and h["shares"] * pr >= 200:
            return True, "aging"
        return False, ""

    old_sell, old_key = _decision(False)
    if old_sell and not h.get("shadow_old_exit"):
        h["shadow_old_exit"] = {
            "would_exit": True,
            "exit_reason": old_key,
            "pnl_pct": round(pnl_pct, 4),
            "held_days": held_days,
            "price": round(pr, 4),
        }
    return _decision(bool(profile))


# ── Metrics ──────────────────────────────────────────────────────────────────
def _metrics(run, dates):
    eq = run["equity_curve"]
    trades = run["trades"]
    if not eq:
        return {"error": "no equity points"}
    start_v, end_v = STARTING_CASH, run["final_equity"]
    total_ret = (end_v - start_v) / start_v * 100
    # max drawdown
    peak = start_v; max_dd = 0.0
    for _, v in eq:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak * 100)
    # daily returns → Sharpe (annualized, rf=0)
    vals = [v for _, v in eq]
    rets = pd.Series(vals).pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * (252 ** 0.5)) if len(rets) > 2 and rets.std() else 0.0
    n_days = len(eq)
    cagr = ((end_v / start_v) ** (252.0 / max(n_days, 1)) - 1) * 100 if end_v > 0 else -100.0
    # trade stats (net)
    n = len(trades)
    wins = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
    losses = [t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0]
    gross_win = sum(wins); gross_loss = abs(sum(losses))
    by_reason = {}
    profit_by_ticker = {}
    profit_by_sector = {}
    hold_days = []
    for t in trades:
        b = by_reason.setdefault(t["exit_reason"], {"n": 0, "sum": 0.0, "wins": 0})
        b["n"] += 1; b["sum"] += t["pnl_pct"]; b["wins"] += 1 if t["pnl_pct"] > 0 else 0
        if t.get("pnl_pct", 0) > 0:
            profit_by_ticker[t["ticker"]] = profit_by_ticker.get(t["ticker"], 0.0) + t["pnl_pct"]
            sec = SECTOR_MAP.get(t["ticker"]) or "unknown"
            profit_by_sector[sec] = profit_by_sector.get(sec, 0.0) + t["pnl_pct"]
        if t.get("days_held") is not None:
            hold_days.append(float(t.get("days_held") or 0))
    exit_breakdown = {k: {"n": v["n"], "avg_pnl_pct": round(v["sum"] / v["n"], 2),
                          "win_rate_pct": round(v["wins"] / v["n"] * 100, 0)}
                      for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]["n"])}
    total_positive_profit = sum(profit_by_ticker.values())
    top_ticker_contrib = (
        max(profit_by_ticker.values()) / total_positive_profit * 100.0
        if total_positive_profit > 0 else 0.0
    )
    top_sector_contrib = (
        max(profit_by_sector.values()) / total_positive_profit * 100.0
        if total_positive_profit > 0 and profit_by_sector else 0.0
    )
    return {
        "total_return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "trades": n,
        "net_win_rate_pct": round(len(wins) / n * 100, 1) if n else 0,
        "avg_win_pct": round(gross_win / len(wins), 2) if wins else 0,
        "avg_loss_pct": round(sum(losses) / len(losses), 2) if losses else 0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "commission_usd": round(n * 2 * float(run.get("commission_per_trade", COMMISSION_PER_TRADE)), 2),
        "trade_count": n,
        "avg_hold_days": round(sum(hold_days) / len(hold_days), 2) if hold_days else 0.0,
        "top_ticker_profit_contribution_pct": round(top_ticker_contrib, 2),
        "top_sector_profit_contribution_pct": round(top_sector_contrib, 2),
        "profit_concentration_rejected": bool(top_ticker_contrib > 50.0),
        "exit_breakdown": exit_breakdown,
        "final_equity": round(end_v, 2),
    }


def _buy_hold(spy_df, dates):
    if not dates:
        return 0.0
    first = _price(spy_df, dates[0]) or float(spy_df.loc[:dates[0], "Close"].iloc[-1])
    last  = _price(spy_df, dates[-1]) or float(spy_df.loc[:dates[-1], "Close"].iloc[-1])
    return round((last - first) / first * 100, 2) if first else 0.0


# ── Public entrypoint ────────────────────────────────────────────────────────
def _aggregate_mode_metrics(metrics_list):
    clean = [m for m in metrics_list if m and "total_return_pct" in m]
    if not clean:
        return {"windows": 0}
    n = len(clean)
    profit_factors = [m["profit_factor"] for m in clean
                      if m.get("profit_factor") is not None]
    return {
        "windows": n,
        "avg_total_return_pct": round(sum(m["total_return_pct"] for m in clean) / n, 2),
        "avg_max_drawdown_pct": round(sum(m["max_drawdown_pct"] for m in clean) / n, 2),
        "avg_sharpe": round(sum(m["sharpe"] for m in clean) / n, 2),
        "total_trades": sum(m["trades"] for m in clean),
        "avg_win_rate_pct": round(sum(m["net_win_rate_pct"] for m in clean) / n, 1),
        "max_top_ticker_profit_contribution_pct": round(
            max(m.get("top_ticker_profit_contribution_pct", 0.0) for m in clean), 2
        ),
        "max_top_sector_profit_contribution_pct": round(
            max(m.get("top_sector_profit_contribution_pct", 0.0) for m in clean), 2
        ),
        "avg_profit_factor": round(sum(profit_factors) / len(profit_factors), 2)
        if profit_factors else None,
    }


def _value_at_path(config, dotted_path):
    cur = config or {}
    for part in dotted_path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _window_metric_values(rolling_out, mode, key):
    vals = []
    for w in rolling_out.get("windows", []) or []:
        m = ((w.get("results") or {}).get(mode) or {})
        if key in m:
            vals.append(m.get(key))
    return vals


def _median(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _clamp01(value):
    return max(0.0, min(1.0, float(value or 0.0)))


def _robust_score(summary, turnover_penalty=0.0):
    ret = _clamp01((float(summary.get("avg_total_return_pct", 0.0)) + 20.0) / 40.0)
    sharpe = _clamp01((float(summary.get("avg_sharpe", 0.0)) + 1.0) / 3.0)
    pf_raw = summary.get("avg_profit_factor")
    pf = _clamp01(((float(pf_raw) if pf_raw is not None else 0.0) - 0.5) / 1.5)
    dd = _clamp01(float(summary.get("avg_max_drawdown_pct", 0.0)) / 30.0)
    turn = _clamp01(turnover_penalty)
    return round(0.30 * ret + 0.25 * sharpe + 0.20 * pf - 0.20 * dd - 0.05 * turn, 4)


def _profit_concentration(window_returns):
    positives = [float(v or 0.0) for v in window_returns if float(v or 0.0) > 0]
    if not positives:
        return 0.0
    return max(positives) / sum(positives)


def _sweep_row(config_name, sweep_type, params_changed, cfg, rolling_out,
               stress_out=None, mode="full_current"):
    summary = (rolling_out.get("summary_by_mode") or {}).get(mode) or {}
    returns = _window_metric_values(rolling_out, mode, "total_return_pct")
    total_windows = int(summary.get("windows", 0) or 0)
    positive_windows = sum(1 for v in returns if (v or 0) > 0)
    concentration = _profit_concentration(returns)
    ticker_concentration = float(
        summary.get("max_top_ticker_profit_contribution_pct", 0.0) or 0.0
    ) / 100.0
    trade_count = int(summary.get("total_trades", 0) or 0)
    stress_summary = ((stress_out or {}).get("summary_by_mode") or {}).get(mode) or {}
    stress_ret = float(stress_summary.get("avg_total_return_pct", -999.0) or 0.0)
    base_ret = float(summary.get("avg_total_return_pct", 0.0) or 0.0)
    friction_survives = bool(stress_summary and stress_ret >= 0.0)

    backtest_cfg = cfg.get("backtest", {}) if isinstance(cfg, dict) else {}
    rejections = []
    if trade_count < int(backtest_cfg.get("min_sweep_trades", 30)):
        rejections.append("trade_count_too_low")
    if (ticker_concentration > float(backtest_cfg.get("profit_concentration_limit", 0.50))
            and not bool(backtest_cfg.get("accept_profit_concentration", False))):
        rejections.append("profit_too_concentrated")
    if float(summary.get("avg_max_drawdown_pct", 0.0) or 0.0) > 30.0:
        rejections.append("max_drawdown_too_high")
    if not friction_survives:
        rejections.append("doubled_friction_failure")
    if total_windows and positive_windows <= max(1, total_windows // 4) and base_ret > 0:
        rejections.append("fragile_window_profile")

    turnover_penalty = min(1.0, trade_count / max(total_windows * 50.0, 1.0))
    return {
        "config_name": config_name,
        "sweep_type": sweep_type,
        "params_changed": params_changed,
        "config_hash": config_hash(cfg),
        "total_return_pct": summary.get("avg_total_return_pct"),
        "sharpe": summary.get("avg_sharpe"),
        "max_drawdown_pct": summary.get("avg_max_drawdown_pct"),
        "profit_factor": summary.get("avg_profit_factor"),
        "positive_windows": positive_windows,
        "total_windows": total_windows,
        "worst_window_return_pct": round(min(returns), 2) if returns else None,
        "median_window_return_pct": round(_median(returns), 2) if returns else None,
        "profit_concentration": round(concentration, 4),
        "top_ticker_profit_concentration": round(ticker_concentration, 4),
        "trade_count": trade_count,
        "friction_2x_survives": friction_survives,
        "friction_2x_return_pct": stress_summary.get("avg_total_return_pct"),
        "robust_score": _robust_score(summary, turnover_penalty),
        "rejections": rejections,
        "recommended": not rejections,
    }


def _run_sensitivity_sweep(panel, spy_df, dates, qqq_df=None, provider=None,
                           train_days=252, test_days=63, step_days=21, window=160,
                           base_config=None, sweep="one_at_a_time"):
    base_cfg = base_config or DEFAULT_CONFIG
    provider = provider or NullHistoricalSignalProvider()
    stress_mult = float(base_cfg.get("backtest", {}).get("friction_stress_mult", 2.0))
    variants = []

    variants.append(("current", "preset", {}, base_cfg))
    if sweep in ("one_at_a_time", "all", "true", "1"):
        for path, values in (base_cfg.get("sweep_params") or {}).items():
            current = _value_at_path(base_cfg, path)
            for val in values:
                if val == current:
                    continue
                variants.append((
                    f"{path}={val}",
                    "one_at_a_time",
                    {path: val},
                    merge_config({path: val}, base=base_cfg),
                ))
    if sweep in ("one_at_a_time", "preset_combo", "presets", "all", "true", "1"):
        for name, changes in (base_cfg.get("preset_sweeps") or {}).items():
            if name == "current":
                continue
            variants.append((name, "preset_combo", dict(changes),
                             merge_config(changes, base=base_cfg)))

    rows = []
    for name, sweep_type, changes, cfg in variants:
        rolling = _run_rolling_backtest(
            panel, spy_df, dates, qqq_df=qqq_df, modes=("full_current",),
            train_days=train_days, test_days=test_days, step_days=step_days,
            window=window, provider=provider, require_external_history=False,
            log_sample_size=0, include_full_logs=False, config=cfg,
            friction_mult=1.0,
        )
        if rolling.get("error"):
            rows.append({
                "config_name": name, "sweep_type": sweep_type,
                "params_changed": changes, "config_hash": config_hash(cfg),
                "error": rolling.get("error"), "recommended": False,
                "rejections": ["rolling_backtest_failed"],
            })
            continue
        stressed = _run_rolling_backtest(
            panel, spy_df, dates, qqq_df=qqq_df, modes=("full_current",),
            train_days=train_days, test_days=test_days, step_days=step_days,
            window=window, provider=provider, require_external_history=False,
            log_sample_size=0, include_full_logs=False, config=cfg,
            friction_mult=stress_mult,
        )
        rows.append(_sweep_row(name, sweep_type, changes, cfg, rolling, stressed))

    eligible = [r for r in rows if r.get("recommended")]
    ranked = sorted(eligible or rows, key=lambda r: (
        -len(r.get("rejections", [])),
        r.get("robust_score", -999),
    ), reverse=True)
    return {
        "type": sweep or "one_at_a_time",
        "advisory_only": True,
        "live_config_mutated": False,
        "mode": "full_current",
        "friction_stress_mult": stress_mult,
        "base_config_hash": config_hash(base_cfg),
        "results": rows,
        "recommendations": ranked[:5],
    }


def _run_rolling_backtest(panel, spy_df, dates, qqq_df=None, modes=None,
                          train_days=252, test_days=63, step_days=21,
                          window=160, provider=None,
                          require_external_history=False, log_sample_size=500,
                          include_full_logs=False, config=None,
                          friction_mult=1.0, slippage_bps_per_side=0.0):
    provider = provider or NullHistoricalSignalProvider()
    cfg = config or DEFAULT_CONFIG
    modes = tuple(modes or ROLLING_MODES)
    invalid = [m for m in modes if m not in ROLLING_MODES]
    if invalid:
        return {"error": f"Unknown rolling mode(s): {', '.join(invalid)}"}
    external_requested = any(m in EXTERNAL_MODES for m in modes)
    if require_external_history and external_requested and isinstance(provider, NullHistoricalSignalProvider):
        return {"error": "external_history_required",
                "modes": [m for m in modes if m in EXTERNAL_MODES],
                "provider_name": provider.name}

    window_defs = _rolling_windows(dates, train_days, test_days, step_days)
    if not window_defs:
        return {"error": "not_enough_data_for_rolling_windows"}

    windows = []
    by_mode = {m: [] for m in modes}
    coverage_by_mode = {m: _coverage_blank() for m in modes}
    candidate_logs = []
    train_edge_cache = {}
    for w in window_defs:
        win_out = {
            "index": w["index"],
            "train_period": w["train_period"],
            "test_period": w["test_period"],
            "results": {},
            "benchmarks": {
                "SPY": _buy_hold_for(spy_df, w["test_dates"]),
                "QQQ": _buy_hold_for(qqq_df, w["test_dates"]),
                "cash": 0.0,
            },
        }
        for mode in modes:
            edge_key = (w["train_period"][0], w["train_period"][1], mode, config_hash(cfg))
            train_edges = train_edge_cache.get(edge_key)
            if train_edges is None:
                train_edges = _learn_forward_edges(panel, spy_df, w["train_dates"], {},
                                                   window, mode=mode, provider=provider,
                                                   config=cfg)
                train_edge_cache[edge_key] = train_edges
            trained_weights = attribution_signal_weights(train_edges)
            ctx = BacktestContext(w["test_dates"][0], mode, provider,
                                  allow_network=False, config=cfg)
            run = _simulate(
                panel, spy_df, w["test_dates"], trained_weights, mode,
                learn=False, window=window, edge_stats=train_edges,
                bt_context=ctx, candidate_logs=candidate_logs,
                log_sample_size=log_sample_size,
                include_full_logs=include_full_logs,
                friction_mult=friction_mult,
                slippage_bps_per_side=slippage_bps_per_side,
            )
            _coverage_add(coverage_by_mode[mode], train_edges.get("coverage", {}))
            _coverage_add(coverage_by_mode[mode], run.get("coverage", {}))
            metrics = _metrics(run, w["test_dates"])
            win_out["results"][mode] = metrics
            by_mode[mode].append(metrics)
        windows.append(win_out)

    if require_external_history:
        missing = [m for m in modes if m in EXTERNAL_MODES
                   and not _coverage_has_external(coverage_by_mode[m])]
        if missing:
            return {"error": "external_history_required", "modes": missing,
                    "coverage_by_mode": coverage_by_mode,
                    "provider_name": provider.name}

    out = {
        "mode": "rolling",
        "windows": windows,
        "summary_by_mode": {
            mode: _aggregate_mode_metrics(metrics)
            for mode, metrics in by_mode.items()
        },
        "coverage_by_mode": coverage_by_mode,
        "benchmarks": {
            "SPY": _buy_hold_for(spy_df, dates),
            "QQQ": _buy_hold_for(qqq_df, dates),
            "cash": 0.0,
        },
        "candidate_logs_sample": candidate_logs[:int(log_sample_size or 0)],
        "config": {
            "version": cfg.get("version"),
            "hash": config_hash(cfg),
            "friction_mult": float(friction_mult or 1.0),
        },
        "caveats": list(CAVEATS),
        "data_source_audit": {
            "used_live_api": False,
            "used_historical_provider": True,
            "provider_name": provider.name,
            "network_calls_blocked": True,
            "config_hash": config_hash(cfg),
        },
    }
    if include_full_logs:
        out["candidate_logs"] = candidate_logs
    return out


def run_portfolio_backtest(universe=None, years=4, train_frac=0.5, window=160,
                           modes=("fixed", "legacy"), use_cache=True,
                           walkforward=None, train_days=252, test_days=63,
                           step_days=21, log_sample_size=500,
                           include_full_logs=False,
                           require_external_history=False, provider=None,
                           config_overrides=None, sweep=None,
                           slippage_bps=0.0):
    """Walk-forward portfolio backtest with V2 entry/exit attribution trained first."""
    universe = universe or _default_universe()
    cfg = merge_config(config_overrides or {}, base=active_config())
    effective_modes = (
        ROLLING_MODES if walkforward == "rolling" and modes == ("fixed", "legacy")
        else tuple(modes or ())
    )
    mode_key = "-".join(effective_modes)
    ck = (f"bt_{walkforward or 'split'}_{sweep or 'nosweep'}_{config_hash(cfg)}_"
          f"{years}_{train_frac}_{window}_"
          f"{train_days}_{test_days}_{step_days}_{mode_key}_"
          f"slip{float(slippage_bps or 0.0)}_"
          f"{'-'.join(sorted(universe))}")
    if use_cache:
        c = cache_get(ck, max_age=3600)
        if c is not None:
            return c

    t0 = time.time()
    panel, spy_df, dates = _load_panel(universe, years)
    if len(dates) < 120:
        return {"error": f"Only {len(dates)} trading days available — need ≥120."}
    if walkforward == "rolling":
        qqq_df = _history("QQQ", years)
        if qqq_df is not None and not qqq_df.empty:
            cutoff = spy_df.index[0]
            qqq_df = qqq_df[qqq_df.index >= cutoff]
        sweep_out = None
        if sweep:
            sweep_out = _run_sensitivity_sweep(
                panel, spy_df, dates, qqq_df=qqq_df, provider=provider,
                train_days=train_days, test_days=test_days, step_days=step_days,
                window=window, base_config=cfg, sweep=sweep,
            )
        rolling_modes = tuple(effective_modes or ROLLING_MODES)
        out = _run_rolling_backtest(
            panel, spy_df, dates, qqq_df=qqq_df, modes=rolling_modes,
            train_days=train_days, test_days=test_days, step_days=step_days,
            window=window, provider=provider,
            require_external_history=require_external_history,
            log_sample_size=log_sample_size,
            include_full_logs=include_full_logs,
            config=cfg,
            slippage_bps_per_side=slippage_bps,
        )
        if sweep_out is not None:
            out["sweep"] = sweep_out
        out.setdefault("params", {
            "universe_size": len(panel), "years": years,
            "train_days": train_days, "test_days": test_days,
            "step_days": step_days, "window": window,
            "commission_per_trade": COMMISSION_PER_TRADE,
            "min_position_usd": MIN_POSITION_USD,
            "log_sample_size": log_sample_size,
            "include_full_logs": bool(include_full_logs),
            "require_external_history": bool(require_external_history),
            "modes": list(rolling_modes),
            "sweep": sweep,
            "slippage_bps_per_side": slippage_bps,
            "config_version": cfg.get("version"),
            "config_hash": config_hash(cfg),
        })
        out["elapsed_sec"] = round(time.time() - t0, 1)
        if use_cache:
            cache_set(ck, out)
        return out
    split = int(len(dates) * train_frac)
    train_dates, test_dates = dates[:split], dates[split:]

    # 1) learn V2 attribution on the train window (entry forward returns + exit quality)
    split_provider = provider or NullHistoricalSignalProvider()
    train_ctx = BacktestContext(train_dates[0], "fixed", split_provider,
                                allow_network=False, config=cfg)
    train_run = _simulate(panel, spy_df, train_dates, {}, "fixed", learn=True,
                          window=window, bt_context=train_ctx,
                          slippage_bps_per_side=slippage_bps)
    trained_edges = _learn_forward_edges(panel, spy_df, train_dates, {}, window,
                                         provider=split_provider, config=cfg)
    train_state = train_run.get("edge_stats", {}) or {}
    for key in ("exit_attribution_events", "exit_attribution_buckets"):
        if train_state.get(key):
            trained_edges[key] = train_state[key]
    ensure_attribution_state(trained_edges)
    trained_weights = attribution_signal_weights(trained_edges)

    # 2) evaluate each mode on the test window with FROZEN trained weights
    results = {}
    for mode in modes:
        test_ctx = BacktestContext(test_dates[0], mode, split_provider,
                                   allow_network=False, config=cfg)
        run = _simulate(panel, spy_df, test_dates, trained_weights, mode,
                        learn=False, window=window, edge_stats=trained_edges,
                        bt_context=test_ctx,
                        slippage_bps_per_side=slippage_bps)
        results[mode] = _metrics(run, test_dates)
    # 3) default-weights (1.0) test run — did the learning actually help?
    run_def = _simulate(panel, spy_df, test_dates, {}, "fixed", learn=False,
                        window=window, edge_stats=trained_edges,
                        bt_context=BacktestContext(test_dates[0], "fixed",
                                                   split_provider,
                                                   allow_network=False,
                                                   config=cfg),
                        slippage_bps_per_side=slippage_bps)
    results["fixed_default_weights"] = _metrics(run_def, test_dates)

    out = {
        "params": {"universe_size": len(panel), "years": years, "train_frac": train_frac,
                   "window": window, "commission_per_trade": COMMISSION_PER_TRADE,
                   "slippage_bps_per_side": slippage_bps,
                   "min_position_usd": MIN_POSITION_USD,
                   "config_version": cfg.get("version"),
                   "config_hash": config_hash(cfg)},
        "train_period": [str(train_dates[0].date()), str(train_dates[-1].date())],
        "test_period":  [str(test_dates[0].date()),  str(test_dates[-1].date())],
        "spy_buy_hold_pct": _buy_hold(spy_df, test_dates),
        "results": results,
        "alpha_vs_spy": {m: round(results[m]["total_return_pct"] - _buy_hold(spy_df, test_dates), 2)
                         for m in results if "total_return_pct" in results[m]},
        "elapsed_sec": round(time.time() - t0, 1),
        "caveat": ("Pure-technical (no news/analyst/insider/intraday). Daily bars; hour-based "
                   "thresholds approximated by trading days; partial-take/degrade-trim omitted "
                   "(identical across modes). Compares post-fix 'fixed' vs pre-fix 'legacy' rules "
                   "on the same held-out window."),
    }
    if use_cache:
        cache_set(ck, out)
    return out


if __name__ == "__main__":
    # Local smoke run: python -m trading.backtest
    import json
    res = run_portfolio_backtest(years=4, train_frac=0.5, use_cache=False)
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))
    for mode, m in res.get("results", {}).items():
        print(f"\n=== {mode} ===")
        print(json.dumps(m, indent=2))
