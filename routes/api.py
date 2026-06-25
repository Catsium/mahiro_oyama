"""JSON API routes: /api/chart, /api/bot/equity, /api/attribution,
/api/backtest, /api/signal_validation."""
import re
import time
from flask import jsonify, request

from app import app
from market.charts import get_chart, PERIOD_MAP
from trading.bot import BOT_MAX_BUYS
from trading.attribution import summarize_attribution
from trading.config import active_config, config_hash
from trading.indicators import _ctx_from_series
from trading.signals import get_recommendation
from trading.sizing import slippage_bps, SLIPPAGE_BPS
from utils.auth import require_admin_token
from utils.cache import api_failure_snapshot, cache_get
from utils.deploy_config import PYTHONANYWHERE_MODE
from utils.storage import load_bot, load_tickers
from utils.threading_utils import _BOT_STATUS


@app.route("/api/chart/<ticker>/<rng>")
def api_chart(ticker, rng):
    ticker = ticker.upper().strip()
    if not re.match(r"^[A-Z][A-Z0-9.\-]{0,7}$", ticker):
        return jsonify({"error": "Invalid ticker"}), 400
    if rng not in PERIOD_MAP:
        return jsonify({"error": "Invalid range"}), 400
    return jsonify(get_chart(ticker, rng))


@app.route("/api/bot/equity")
def api_bot_equity():
    """Returns equity time-series. Buckets by range (1d/1w/1m/all)."""
    rng = (request.args.get("range") or "1m").lower()
    b = load_bot()
    eq = b.get("equity_history", [])
    now = time.time()

    range_cfg = {
        # Round-5 extra: 1d is a rolling COUNT window (last 390 1-min points ≈ one
        # session). No wall-clock cutoff → doesn't go empty / "reset" at each open;
        # oldest points drop off the back as new ones arrive. tail=390.
        "1d":  {"cutoff": None,           "bucket": 60,         "tail": 390},
        # Round-5 extra: 1W now 30-min buckets (was 60-min).
        "1w":  {"cutoff": 7 * 24 * 3600,  "bucket": 30 * 60},
        "1m":  {"cutoff": 30 * 24 * 3600, "bucket": 24 * 3600},
        "all": {"cutoff": None,           "bucket": 24 * 3600},
    }
    cfg = range_cfg.get(rng, range_cfg["1m"])
    cutoff = (now - cfg["cutoff"]) if cfg["cutoff"] is not None else 0
    bucket_secs = cfg["bucket"]

    buckets = {}
    for p in eq:
        if p[0] < cutoff:
            continue
        key = int(p[0] // bucket_secs)
        existing = buckets.get(key)
        if existing is None or p[0] > existing[0]:
            buckets[key] = p

    out = sorted(buckets.values(), key=lambda x: x[0])
    # Round-5 extra: count-tail for 1d (remove from the back, never daily-reset)
    tail = cfg.get("tail")
    if tail:
        out = out[-tail:]
    return jsonify({
        "points": out,
        "starting": b.get("starting", 10000),
        "range": rng,
        "bucket_secs": bucket_secs,
    })


@app.route("/api/bot/status")
def api_bot_status():
    b = load_bot()
    diag = b.get("last_no_buy_diagnostics") or {}
    api_breakers = diag.get("api_circuit_breakers", {}) or {}
    rate_limit_recent = bool(
        diag.get("rate_limit_recent")
        or any((snap or {}).get("rate_limit_recent") for snap in api_breakers.values() if isinstance(snap, dict))
    )
    provider_health_status = diag.get("provider_health_status")
    if not provider_health_status:
        if rate_limit_recent or any((snap or {}).get("rate_limited") for snap in api_breakers.values() if isinstance(snap, dict)):
            provider_health_status = "rate_limited"
        elif any((snap or {}).get("status") == "degraded" for snap in api_breakers.values() if isinstance(snap, dict)):
            provider_health_status = "degraded"
        else:
            provider_health_status = "healthy"
    return jsonify({
        "last_no_buy_diagnostics": {
            "main_blocker": diag.get("main_blocker"),
            "market_open": diag.get("market_open"),
            "tod_ok": diag.get("tod_ok"),
            "buy_window_open": diag.get("buy_window_open"),
            "regime_allow_buys": diag.get("regime_allow_buys"),
            "regime_kind": diag.get("regime_kind"),
            "regime_source": diag.get("regime_source"),
            "regime_v3": diag.get("regime_v3"),
            "trading_mode": diag.get("trading_mode"),
            "normal_mode_active": diag.get("normal_mode_active"),
            "proxy_mode_active": diag.get("proxy_mode_active"),
            "degraded_mode_active": diag.get("degraded_mode_active"),
            "degraded_mode_reason": diag.get("degraded_mode_reason"),
            "degraded_size_mult": diag.get("degraded_size_mult"),
            "degraded_min_confidence": diag.get("degraded_min_confidence"),
            "degraded_reject_counts": diag.get("degraded_reject_counts", {}),
            "degraded_buys_today": diag.get("degraded_buys_today", 0),
            "degraded_max_buys_today": diag.get("degraded_max_buys_today"),
            "degraded_gross_exposure_pct": diag.get("degraded_gross_exposure_pct"),
            "degraded_max_gross_exposure_pct": diag.get("degraded_max_gross_exposure_pct"),
            "signal_counts": diag.get("signal_counts", {}),
            "display_signal_counts": diag.get("display_signal_counts", {}),
            "raw_buy_count": diag.get("raw_buy_count", 0),
            "display_buy_candidate_count": diag.get("display_buy_candidate_count", 0),
            "stale_ticker_count": diag.get("stale_ticker_count", 0),
            "stale_tickers": diag.get("stale_tickers", []),
            "stale_positions": diag.get("stale_positions", []),
            "risk_unmanaged_positions": diag.get("risk_unmanaged_positions", []),
            "api_circuit_breakers": api_breakers,
            "provider_health_status": provider_health_status,
            "rate_limit_recent": rate_limit_recent,
            "candidate_pool_count": diag.get("candidate_pool_count", 0),
            "ranked_count": diag.get("ranked_count", 0),
            "tradable_count": diag.get("tradable_count", 0),
            "top_ranked": (diag.get("top_ranked") or [])[:5],
            "top_ranked_rejections": (diag.get("top_ranked_rejections") or [])[:5],
            "skip_reason_counts": diag.get("skip_reason_counts", {}),
            "buyable_reject_counts": diag.get("buyable_reject_counts", {}),
            "top_buyable_rejects": (diag.get("top_buyable_rejects") or [])[:5],
            "scan_payload_misses": diag.get("scan_payload_misses", 0),
            "scan_age_sec": diag.get("scan_age_sec"),
            "scan_rows_count": diag.get("scan_rows_count"),
            "scan_fresh_rows_count": diag.get("scan_fresh_rows_count"),
            "pa_stage_status": diag.get("pa_stage_status"),
            "data_health_ok": diag.get("data_health_ok"),
            "data_health_blocks": diag.get("data_health_blocks", []),
            "spy_data_ok": diag.get("spy_data_ok"),
            "spy_data_source": diag.get("spy_data_source"),
            "spy_data_error": diag.get("spy_data_error"),
            "regime_data_status": diag.get("regime_data_status"),
            "regime_data_fallback": diag.get("regime_data_fallback"),
            "regime_fallback_active": diag.get("regime_fallback_active", diag.get("regime_data_fallback")),
            "regime_data_source": diag.get("regime_data_source"),
            "regime_data_error": diag.get("regime_data_error"),
            "regime_data_size_mult": diag.get("regime_data_size_mult"),
            "vix_label": diag.get("vix_label"),
            "vix_value": diag.get("vix_value"),
            "vix_display": diag.get("vix_display"),
            "vix_data_ok": diag.get("vix_data_ok"),
            "vix_data_status": diag.get("vix_data_status"),
            "volatility_data_ok": diag.get("volatility_data_ok"),
            "volatility_source": diag.get("volatility_source"),
            "volatility_value": diag.get("volatility_value"),
            "volatility_data_error": diag.get("volatility_data_error"),
            "volatility_error": diag.get("volatility_error", diag.get("volatility_data_error")),
            "spy_rows": diag.get("spy_rows"),
            "spy_bar_count": diag.get("spy_bar_count", diag.get("spy_rows")),
            "spy_last_date": diag.get("spy_last_date"),
            "spy_mom_label": diag.get("spy_mom_label"),
            "cash": diag.get("cash"),
            "cash_floor": diag.get("cash_floor"),
            "cash_available_after_floor": diag.get("cash_available_after_floor"),
            "gross_exposure_pct": diag.get("gross_exposure_pct"),
            "buys_today": diag.get("buys_today"),
            "max_buys_today": diag.get("max_buys_today"),
            "paper_trading_locked": diag.get("paper_trading_locked"),
            "paper_lock_reason": diag.get("paper_lock_reason"),
            "tick_runtime_seconds": diag.get("tick_runtime_seconds"),
            "checked_tickers": diag.get("checked_tickers", []),
            "traded": diag.get("traded", False),
            "ts": diag.get("ts"),
        },
        "last_tick_status": _BOT_STATUS.get("last_tick_status"),
        "last_tick_time": _BOT_STATUS.get("last_tick_time"),
        "tick_runtime_seconds": _BOT_STATUS.get("tick_runtime_seconds"),
        "last_state_write_ts": b.get("last_state_write_ts"),
        "last_cache_prune": b.get("last_cache_prune"),
        "last_bot_error": _BOT_STATUS.get("last_error"),
        "last_bot_error_ts": _BOT_STATUS.get("last_error_ts"),
    })


@app.route("/api/data/health")
def api_data_health():
    auth = require_admin_token()
    if auth is not True:
        return auth
    live = str(request.args.get("live") or "").lower() in {"1", "true", "yes"}
    try:
        limit = max(1, min(50, int(request.args.get("limit", 20))))
    except Exception:
        limit = 20
    tickers = []
    for t in ["SPY"] + list(load_tickers()):
        tk = str(t or "").upper()
        if tk and tk not in tickers:
            tickers.append(tk)
    rows = []
    for tk in tickers[:limit]:
        q = cache_get(f"q_{tk}", max_age=3600) or {}
        daily = cache_get(f"dm_daily_{tk}_tail", max_age=6 * 3600)
        if live:
            try:
                from market.quotes import get_quote
                q = get_quote(tk) or q
            except Exception:
                pass
            try:
                from market.data_manager import get_daily
                daily = get_daily(tk)
            except Exception:
                pass
        rows.append({
            "ticker": tk,
            "quote_price": q.get("price"),
            "quote_stale": bool(q.get("stale")),
            "quote_stale_age_sec": q.get("stale_age_sec"),
            "daily_rows": int(len(daily)) if daily is not None else 0,
        })
    return jsonify({
        "pythonanywhere_mode": PYTHONANYWHERE_MODE,
        "live": live,
        "tickers_checked": rows,
        "api_circuit_breakers": api_failure_snapshot(),
    })


@app.route("/api/attribution")
def api_attribution():
    """V2 forward-return attribution diagnostics + archived legacy view."""
    b = load_bot()
    cfg = active_config()
    v2 = summarize_attribution(b)
    attr = b.get("signal_attribution", {}) or {}
    weights = b.get("signal_weights", {}) or {}
    out = {}
    for cat in sorted(set(list(attr.keys()) + list(weights.keys()))):
        bucket = attr.get(cat, {})
        wn = bucket.get("weighted_n", 0)
        out[cat] = {
            "current_weight": weights.get(cat, 1.0),
            "wins_weighted": round(bucket.get("wins", 0), 3),
            "losses_weighted": round(bucket.get("losses", 0), 3),
            "pnl_sum": round(bucket.get("pnl_sum", 0), 2),
            "weighted_n": round(wn, 3),
            "weighted_n_threshold": 30,
            "active": wn >= 30,
            "alpha":  round(bucket.get("alpha", 0), 3),
            "beta":   round(bucket.get("beta", 0), 3),
            "p_edge": round(bucket.get("p_edge", 0), 3),
        }
    # Round-6 T3: confidence calibration — bucket closed trades by entry
    # confidence, report win-rate + avg pnl. Makes the confidence score
    # meaningful (and is the data groundwork for future EV-based ranking).
    outcomes = b.get("trade_outcomes", [])
    cal_defs = [("<40", 0, 40), ("40-55", 40, 55), ("55-70", 55, 70),
                ("70-85", 70, 85), ("85+", 85, 1e9)]
    calibration = []
    for label, lo, hi in cal_defs:
        bucket_pls = [o["pnl_pct"] for o in outcomes
                      if o.get("entry_confidence") is not None
                      and lo <= o["entry_confidence"] < hi]
        n = len(bucket_pls)
        wins = sum(1 for x in bucket_pls if x > 0)
        calibration.append({
            "confidence": label, "n": n,
            "win_rate": round(wins / n * 100, 0) if n else None,
            "avg_pnl_pct": round(sum(bucket_pls) / n, 2) if n else None,
        })

    edge_summary = []
    for key, horizons in sorted((b.get("edge_stats", {}) or {}).items()):
        h5 = (horizons or {}).get("5d") or {}
        if not h5:
            continue
        edge_summary.append({
            "bucket": key,
            "n": h5.get("n", 0),
            "avg_return_pct": h5.get("avg_return_pct"),
            "hit_rate_pct": h5.get("hit_rate_pct"),
            "executed_n": h5.get("executed_n", 0),
            "skipped_n": h5.get("skipped_n", 0),
        })
    edge_summary.sort(key=lambda r: (r.get("n", 0), r.get("avg_return_pct") or 0),
                      reverse=True)

    obs_source = b.get("attribution_events") or b.get("candidate_observations", []) or []
    obs_summary = {}
    for obs in obs_source:
        decision = obs.get("decision") or "unknown"
        ret5 = (obs.get("forward_returns") or {}).get("5d")
        s = obs_summary.setdefault(
            decision,
            {"n": 0, "with_5d": 0, "sum_5d": 0.0, "wins_5d": 0,
             "sum_net_5d": 0.0},
        )
        s["n"] += 1
        if ret5 is not None:
            friction = float(obs.get("friction_pct") or 0.0)
            net5 = ret5 - friction
            s["with_5d"] += 1
            s["sum_5d"] += ret5
            s["sum_net_5d"] += net5
            s["wins_5d"] += 1 if net5 > 0 else 0
    candidate_summary = {}
    for decision, s in obs_summary.items():
        candidate_summary[decision] = {
            "n": s["n"],
            "with_5d": s["with_5d"],
            "avg_5d_return_pct": round(s["sum_5d"] / s["with_5d"], 3)
            if s["with_5d"] else None,
            "avg_net_5d_return_pct": round(s["sum_net_5d"] / s["with_5d"], 3)
            if s["with_5d"] else None,
            "win_rate_5d_pct": round(s["wins_5d"] / s["with_5d"] * 100.0, 1)
            if s["with_5d"] else None,
        }

    return jsonify({
        "config": {
            "version": cfg.get("version"),
            "hash": config_hash(cfg),
        },
        "regime_v3": b.get("last_regime_v3", {}),
        "entry_buckets": v2["entry_buckets"][:50],
        "exit_buckets": v2["exit_buckets"][:50],
        "best_clusters": v2.get("best_clusters", [])[:10],
        "worst_clusters": v2.get("worst_clusters", [])[:10],
        "skipped_winners": v2.get("skipped_winners", [])[:25],
        "active_exit_profiles": v2.get("active_exit_profiles", [])[:25],
        "recent_attribution_events": v2["recent_events"],
        "recent_exit_attribution_events": v2["recent_exit_events"],
        "attribution_status": v2["status"],
        "categories": v2["entry_buckets"][:50],
        "legacy": {
            "archived": True,
            "categories": out,
            "note": "Legacy closed-P&L signal_attribution/signal_weights are display-only.",
        },
        "total_closed_trades": len(outcomes),
        "kelly": b.get("last_kelly", {
            "diag": "kelly not yet computed (no bot run since restart)",
            "bot_max_buys": BOT_MAX_BUYS,
        }),
        "calibration": calibration,
        "edge_stats": edge_summary[:25],
        "candidate_summary": candidate_summary,
        "last_candidate_rankings": b.get("last_candidate_rankings", [])[:25],
        "last_portfolio_variance_checks": b.get("last_portfolio_variance_checks", [])[:25],
        "notes": ("V2 entry learning uses 5d forward returns after friction; "
                  "legacy closed-P&L attribution is archived. "
                  "Exit profiles go live per regime/cluster after 60 samples."),
    })


@app.route("/api/backtest/portfolio")
def api_backtest_portfolio():
    """Walk-forward PORTFOLIO backtest of the live decision rules vs buy-hold SPY.

    Query params: years (default 4), train_frac (0.5), window (160), universe
    ('scan' default | 'watchlist'). Compares post-fix 'fixed' vs pre-fix 'legacy'
    rules on the same held-out test window. Cached 1h; first run takes ~1 min."""
    auth = require_admin_token()
    if auth is not True:
        return auth
    from trading.backtest import run_portfolio_backtest
    from utils.storage import load_tickers
    def _bool_arg(name, default=False):
        v = request.args.get(name)
        if v is None:
            return default
        return str(v).lower() in ("1", "true", "yes", "on")
    try:
        years      = float(request.args.get("years", 4))
        train_frac = float(request.args.get("train_frac", 0.5))
        window     = int(request.args.get("window", 160))
        train_days = int(request.args.get("train_days", 252))
        test_days  = int(request.args.get("test_days", 63))
        step_days  = int(request.args.get("step_days", 21))
        log_sample_size = int(request.args.get("log_sample_size", 500))
        slippage_bps = float(request.args.get("slippage_bps", 0.0))
    except Exception:
        return jsonify({"error": "numeric backtest params are invalid"})
    years      = max(1.0, min(years, 10.0))
    train_frac = max(0.2, min(train_frac, 0.8))
    window     = max(60, min(window, 400))
    train_days = max(60, min(train_days, 1000))
    test_days  = max(20, min(test_days, 252))
    step_days  = max(5, min(step_days, 252))
    log_sample_size = max(0, min(log_sample_size, 5000))
    slippage_bps = max(0.0, min(slippage_bps, 50.0))
    universe = load_tickers() if request.args.get("universe") == "watchlist" else None
    walkforward = (request.args.get("walkforward") or "").lower() or None
    sweep = (request.args.get("sweep") or "").lower() or None
    modes_arg = request.args.get("modes")
    modes = tuple(m.strip() for m in modes_arg.split(",") if m.strip()) if modes_arg else None
    try:
        kwargs = {"universe": universe, "years": years, "train_frac": train_frac,
                  "window": window, "slippage_bps": slippage_bps}
        if walkforward == "rolling":
            kwargs.update({
                "walkforward": "rolling",
                "train_days": train_days,
                "test_days": test_days,
                "step_days": step_days,
                "log_sample_size": log_sample_size,
                "include_full_logs": _bool_arg("include_full_logs", False),
                "require_external_history": _bool_arg("require_external_history", False),
                "sweep": sweep,
            })
            if modes:
                kwargs["modes"] = modes
        return jsonify(run_portfolio_backtest(**kwargs))
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:300]}"})


@app.route("/api/backtest/<ticker>")
def api_backtest(ticker):
    """Technical-only walk-forward backtest over 1y daily bars."""
    auth = require_admin_token()
    if auth is not True:
        return auth
    ticker = ticker.upper()
    try:
        if PYTHONANYWHERE_MODE:
            from market.data_manager import get_daily
            h = get_daily(ticker)
        else:
            import yfinance as yf
            h = yf.Ticker(ticker).history(period="1y")
        if h is None or h.empty or len(h) < 100:
            return jsonify({"error": f"Not enough data for {ticker} (got {len(h)} days, need 100+ for warm-up + trades)."})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"})

    cl = h["Close"].round(2)
    cash = 10000.0; shares = 0; avg = 0.0; peak = 0.0
    trades = []
    equity_curve = []
    total_cost = 0.0
    LOOKBACK = 80
    for i in range(LOOKBACK, len(cl)):
        window = cl.iloc[:i]
        df_window = h.iloc[:i]
        ctx = _ctx_from_series(window, df=df_window)
        if not ctx: continue
        rec = get_recommendation(0.0, ctx, regime=None, earnings=None,
                                  analyst=None, insider=None, pure_technical=True)
        price = float(cl.iloc[i])
        date_str = h.index[i].strftime("%Y-%m-%d") if hasattr(h.index[i], "strftime") else str(i)
        if shares > 0:
            peak = max(peak, price)
            pnl_pct = (price - avg) / avg * 100
            trail_pct = (price - peak) / peak * 100
            sell = False; reason = ""
            if pnl_pct <= -5:
                sell = True; reason = f"stop-loss {pnl_pct:.1f}%"
            elif pnl_pct >= 12:
                sell = True; reason = f"take-profit {pnl_pct:.1f}%"
            elif pnl_pct > 3 and trail_pct <= -6:
                sell = True; reason = f"trail-stop ({trail_pct:.1f}% from peak)"
            elif rec["cls"] in ("sell", "strong-sell") and pnl_pct < 0:
                sell = True; reason = f"signal-flip + losing ({pnl_pct:.1f}%)"
            if sell:
                slip = slippage_bps(shares * price, ctx) / 10000.0
                effective_sell = price * (1 - slip)
                proceeds = shares * effective_sell
                cost_this = shares * price * slip
                total_cost += cost_this
                cash += proceeds
                pnl_pct_net = (effective_sell - avg) / avg * 100
                trades.append({"action": "SELL", "date": date_str, "price": round(price, 2),
                               "pnl_pct": round(pnl_pct_net, 2), "reason": reason,
                               "slippage_usd": round(cost_this, 2)})
                shares = 0; avg = 0; peak = 0
        else:
            if rec["cls"] in ("buy", "strong-buy"):
                spend = cash * 0.95
                slip = slippage_bps(spend, ctx) / 10000.0
                effective_buy = price * (1 + slip)
                shares = round(spend / effective_buy, 4)
                avg = effective_buy
                peak = price
                cost_this = shares * price * slip
                total_cost += cost_this
                cash -= shares * effective_buy
                trades.append({"action": "BUY", "date": date_str, "price": round(price, 2),
                               "confidence": rec["confidence"], "signal": rec["signal"],
                               "slippage_usd": round(cost_this, 2)})
        equity = cash + shares * price
        equity_curve.append([date_str, round(equity, 2)])

    if shares > 0:
        last = float(cl.iloc[-1])
        # Round-7 Bug 9: apply slippage to the forced close like every other sell,
        # so reported return isn't inflated. ctx is the final loop iteration's ctx.
        slip = slippage_bps(shares * last, ctx) / 10000.0
        effective_last = last * (1 - slip)
        cash += shares * effective_last
        total_cost += shares * last * slip
        pnl_pct = (effective_last - avg) / avg * 100
        trades.append({"action": "SELL-FINAL", "date": "final", "price": round(last, 2),
                       "pnl_pct": round(pnl_pct, 2), "reason": "end-of-backtest"})

    sells = [t for t in trades if t["action"].startswith("SELL")]
    wins = sum(1 for t in sells if t.get("pnl_pct", 0) > 0)
    losses = len(sells) - wins
    total_return = round((cash - 10000) / 10000 * 100, 2)
    buy_hold = round((float(cl.iloc[-1]) - float(cl.iloc[LOOKBACK])) / float(cl.iloc[LOOKBACK]) * 100, 2)
    return jsonify({
        "ticker": ticker,
        "model": "pure_technical",
        "caveat": ("Backtest uses pure_technical=True (no news/analyst/insider). "
                   "Live bot uses those signals too — only the technical layer is validated."),
        "period_days": len(cl),
        "trades_total": len(trades),
        "wins": wins, "losses": losses,
        "win_rate_pct": round(wins / len(sells) * 100, 1) if sells else 0,
        "total_return_pct": total_return,
        "buy_hold_return_pct": buy_hold,
        "alpha_pct": round(total_return - buy_hold, 2),
        "final_equity": round(cash, 2),
        "slippage_bps_per_side": SLIPPAGE_BPS,
        "total_slippage_usd": round(total_cost, 2),
        "slippage_drag_pct": round(total_cost / 100, 4),
        "trades": trades[-30:],
        "equity_curve": equity_curve[::max(1, len(equity_curve) // 100)],
    })


@app.route("/api/signal_validation/<ticker>")
def signal_validation(ticker):
    """Per-signal predictive-power test using 5-day forward returns."""
    auth = require_admin_token()
    if auth is not True:
        return auth
    ticker = ticker.upper().strip()
    try:
        if PYTHONANYWHERE_MODE:
            from market.data_manager import get_daily
            h = get_daily(ticker)
        else:
            import yfinance as yf
            h = yf.Ticker(ticker).history(period="1y", interval="1d")
        if h is None or h.empty or len(h) < 120:
            return jsonify({"error": f"Need ≥120 days for {ticker}, got {len(h)}."})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"})

    cl = h["Close"].round(2)
    LOOKBACK  = 80
    FORWARD   = 5
    buckets = {}
    for i in range(LOOKBACK, len(cl) - FORWARD):
        window = cl.iloc[:i]
        df_window = h.iloc[:i]
        ctx = _ctx_from_series(window, df=df_window)
        if not ctx: continue
        rec = get_recommendation(0.0, ctx, regime=None, earnings=None,
                                  analyst=None, insider=None, pure_technical=True)
        cats = rec.get("categories", {})
        fwd = (float(cl.iloc[i + FORWARD]) - float(cl.iloc[i])) / float(cl.iloc[i]) * 100
        for cat, vote in cats.items():
            if vote == 0: continue
            if   vote >=  1.5: key = f"{cat}:++"
            elif vote >   0:   key = f"{cat}:+"
            elif vote <= -1.5: key = f"{cat}:--"
            else:              key = f"{cat}:-"
            b = buckets.setdefault(key, {"n": 0, "wins": 0, "sum_fwd": 0.0})
            b["n"] += 1
            if fwd > 0: b["wins"] += 1
            b["sum_fwd"] += fwd

    results = []
    for key, b in sorted(buckets.items()):
        n = b["n"]
        results.append({
            "bucket": key,
            "n": n,
            "hit_rate_pct": round(b["wins"] / n * 100, 1) if n else 0,
            "avg_fwd_return_pct": round(b["sum_fwd"] / n, 3) if n else 0,
        })
    edge_summary = []
    for r in results:
        n = r["n"]
        if n < 20:
            verdict = "insufficient data"
        elif r["hit_rate_pct"] > 52 and r["avg_fwd_return_pct"] > 0.1:
            verdict = "possible edge (validate further)"
        elif r["hit_rate_pct"] < 48 or r["avg_fwd_return_pct"] < -0.1:
            verdict = "no edge / inverse"
        else:
            verdict = "no edge (noise)"
        edge_summary.append({**r, "verdict": verdict})

    return jsonify({
        "ticker": ticker,
        "forward_days": FORWARD,
        "lookback_bars": LOOKBACK,
        "total_bars_tested": len(cl) - LOOKBACK - FORWARD,
        "caveat": ("Daily bars, pure-technical signals, no costs/slippage."),
        "buckets": edge_summary,
    })
