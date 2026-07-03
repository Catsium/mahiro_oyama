"""Portfolio & bot routes:
# Ticker management
- Bot dashboard (/bot, /botcontrol, /bot/run, /bot/stop, /bot/start, /bot/reset)
- /health (lightweight keepalive ping)
"""
import re
import time
from datetime import datetime
from flask import render_template, request, redirect, url_for, flash, session, jsonify

from app import app
from market.history import warm_history
from market.quotes import get_quote, is_valid_ticker
from market.snapshots import signal_snapshot
from trading.bot import (
    bot_state, _render_bot_page, STARTING_CASH, run_bot, warm_scan_if_due,
    BOT_TICK_MAX_RUNTIME_SEC,
)
from trading.config import DEFAULT_CONFIG
from trading.risk import get_market_regime
from trading.suggestion_store import record_suggestion_feedback
from utils.auth import require_admin_token, require_machine_token
from utils.config import BOT_ENABLED
from utils.deploy_config import (
    FINNHUB_KEY, FMP_KEY, PERSISTENT_CACHE, PYTHONANYWHERE_MODE,
    PA_TICKERS_PER_BOT_RUN,
)
from utils.storage import (
    acquire_bot_file_lock, load_tickers, save_tickers, save_bot, load_bot,
    SUGGESTION_DB_FILE,
)
from utils.threading_utils import trigger_bot_if_due, _bot_run_lock
from utils.time_utils import is_market_open


# User session portfolio (paper trading simulator)
def init_pf():
    if "pf" not in session:
        session["pf"] = {"cash": STARTING_CASH, "holdings": {}, "history": []}


def pf_state():
    init_pf()
    pf = session["pf"]
    total = pf["cash"]
    rows = []
    for t, h in pf["holdings"].items():
        p = get_quote(t)["price"]
        val = h["shares"] * p
        cost = h["shares"] * h["avg_cost"]
        pnl = val - cost
        total += val
        rows.append({"ticker": t, "shares": h["shares"], "avg_cost": h["avg_cost"],
                     "price": p, "value": val, "pnl": pnl,
                     "pnl_pct": (pnl / cost * 100) if cost else 0})
    pnl_t = total - STARTING_CASH
    return pf, rows, total, pnl_t, (pnl_t / STARTING_CASH * 100)


# Ticker management
@app.route("/ticker/add", methods=["POST"])
def add_ticker():
    require_admin_token()
    t = (request.form.get("ticker") or "").upper().strip()
    if not re.match(r"^[A-Z][A-Z0-9.\-]{0,7}$", t):
        flash("Invalid ticker symbol.", "warning")
        return redirect(request.referrer or url_for("index"))
    tickers = load_tickers()
    if t in tickers:
        flash(f"{t} is already in your watchlist.", "info")
    elif not is_valid_ticker(t):
        flash(f"Could not find data for '{t}'. Check the symbol and try again.", "warning")
    else:
        tickers.append(t)
        save_tickers(tickers)
        flash(f"Added {t} to watchlist.", "success")
    return redirect(request.referrer or url_for("index"))


@app.route("/ticker/remove", methods=["POST"])
def remove_ticker():
    require_admin_token()
    t = (request.form.get("ticker") or "").upper().strip()
    tickers = load_tickers()
    if t in tickers and len(tickers) > 1:
        tickers.remove(t)
        save_tickers(tickers)
        flash(f"Removed {t} from watchlist.", "success")
    elif len(tickers) <= 1:
        flash("Can't remove the last ticker.", "warning")
    return redirect(request.referrer or url_for("index"))


# Simulator
@app.route("/simulator")
def simulator():
    pf, holdings, total, pnl, pnl_pct = pf_state()
    bot, b_hold, b_total, b_pnl, b_pnl_pct = bot_state()
    tickers = load_tickers()
    recs = {}
    regime = get_market_regime()
    for t in tickers:
        owned = pf["holdings"].get(t, {}).get("shares", 0)
        snap = signal_snapshot(t, regime=regime, live=not PYTHONANYWHERE_MODE,
                               owned=owned)
        recs[t] = {**snap["rec"], "price": snap["price"], "owned": owned}
    return render_template("simulator.html",
        pf=pf, holdings=holdings, total=total, pnl=pnl, pnl_pct=pnl_pct,
        bot=bot, bot_holdings=b_hold, bot_total=b_total, bot_pnl=b_pnl, bot_pnl_pct=b_pnl_pct,
        recs=recs, tickers=tickers, starting=STARTING_CASH,
        market_open=is_market_open(), now=datetime.now())


@app.route("/simulator/buy", methods=["POST"])
def sim_buy():
    init_pf()
    t = (request.form.get("ticker") or "").upper()
    try: sh = float(request.form.get("shares", 0))
    except Exception: sh = 0
    if t not in load_tickers() or sh <= 0:
        return redirect(url_for("simulator"))
    pf = session["pf"]; q = get_quote(t); p = q.get("price", 0)
    if p <= 0 or q.get("stale"):
        flash(f"No live price for {t}. Try again later.", "warning")
        return redirect(url_for("simulator"))
    cost = p * sh
    if pf["cash"] >= cost:
        pf["cash"] -= cost
        h = pf["holdings"].get(t, {"shares": 0, "avg_cost": 0})
        ns = h["shares"] + sh; na = (h["shares"] * h["avg_cost"] + cost) / ns
        pf["holdings"][t] = {"shares": round(ns, 4), "avg_cost": round(na, 4)}
        pf["history"] = [{"action": "BUY", "ticker": t, "shares": sh, "price": p, "total": cost,
                          "time": datetime.now().strftime("%m/%d %H:%M")}] + pf["history"][:19]
        session["pf"] = pf; session.modified = True
    else:
        flash(f"Not enough cash for {sh} shares of {t}.", "warning")
    return redirect(url_for("simulator"))


@app.route("/simulator/sell", methods=["POST"])
def sim_sell():
    init_pf()
    t = (request.form.get("ticker") or "").upper()
    try: sh = float(request.form.get("shares", 0))
    except Exception: sh = 0
    if t not in load_tickers() or sh <= 0:
        return redirect(url_for("simulator"))
    pf = session["pf"]; h = pf["holdings"].get(t)
    if not h or h["shares"] < sh:
        flash(f"You don't own enough {t} shares.", "warning")
        return redirect(url_for("simulator"))
    q = get_quote(t); p = q.get("price", 0)
    if p <= 0 or q.get("stale"):
        flash(f"No live price for {t}. Try again later.", "warning")
        return redirect(url_for("simulator"))
    pf["cash"] += p * sh
    h["shares"] = round(h["shares"] - sh, 4)
    if h["shares"] <= 0:
        del pf["holdings"][t]
    else:
        pf["holdings"][t] = h
    pf["history"] = [{"action": "SELL", "ticker": t, "shares": sh, "price": p, "total": p * sh,
                      "time": datetime.now().strftime("%m/%d %H:%M")}] + pf["history"][:19]
    session["pf"] = pf; session.modified = True
    return redirect(url_for("simulator"))


@app.route("/simulator/reset", methods=["POST"])
def sim_reset():
    session["pf"] = {"cash": STARTING_CASH, "holdings": {}, "history": []}
    session.modified = True
    flash("Your portfolio has been reset.", "success")
    return redirect(url_for("simulator"))


# Bot dashboard & controls
@app.route("/botcontrol")
def bot_dashboard():
    """Admin page with run/stop/start/reset controls."""
    auth = require_admin_token()
    if auth is not True:
        return auth
    return _render_bot_page(read_only=False)


@app.route("/bot")
def bot_view():
    """Public view-only; safe to share."""
    return _render_bot_page(read_only=True)


@app.route("/bot/view")
def bot_view_legacy():
    """Backwards-compat redirect; old /bot/view links keep working."""
    return redirect(url_for("bot_view"))


@app.route("/bot/reset", methods=["POST"])
def bot_reset():
    require_admin_token()
    try: starting = float(request.form.get("starting", 10000))
    except Exception: starting = 10000
    starting = max(100, min(starting, 1_000_000))
    with _bot_run_lock, acquire_bot_file_lock(timeout=10):
        b = load_bot()
        if b.get("_load_error"):
            flash("Bot state is corrupt; reset refused to avoid overwriting bot_state.json.", "danger")
            return redirect(url_for("bot_dashboard"))
        save_bot({"cash": starting, "starting": starting, "holdings": {}, "history": [],
                  "last_trade": time.time(), "recent_sells": {}, "stopped": False})
    flash(f"Bot reset with ${starting:.0f} starting balance.", "success")
    return redirect(url_for("bot_dashboard"))
# Simulator


@app.route("/bot/stop", methods=["POST"])
def bot_stop():
    require_admin_token()
    # Bug #2: hold _bot_run_lock for the read-modify-write so a user toggle can't
    # race (and clobber) an in-flight trading pass.
    with _bot_run_lock, acquire_bot_file_lock(timeout=10):
        b = load_bot()
        b["stopped"] = True
        b["pending_run"] = False
        save_bot(b)
    flash("Bot stopped. Existing positions are kept; click 'start bot' to resume.", "warning")
    return redirect(url_for("bot_dashboard"))


@app.route("/bot/start", methods=["POST"])
def bot_start():
    require_admin_token()
    with _bot_run_lock, acquire_bot_file_lock(timeout=10):
        b = load_bot()
        b["stopped"] = False
        save_bot(b)
    flash("Bot resumed; will run on the next health ping or manual run.", "success")
    return redirect(url_for("bot_dashboard"))


@app.route("/bot/run", methods=["POST"])
def bot_run():
    require_admin_token()
    if not is_market_open():
        flash("Market is closed; bot is already auto-queued for the next open.", "info")
        return redirect(url_for("bot_dashboard"))
    # A3: a human clicked "run now"; user_forced relaxes the TOD gate + buy-1 fallback.
    started = trigger_bot_if_due(force=True, user_forced=True)
    if started:
        flash("Bot run triggered; refresh in a few seconds to see results.", "info")
    elif not BOT_ENABLED:
        flash("Bot is disabled by configuration; no run started.", "warning")
    elif _bot_run_lock.locked():
        flash("Bot is already running; no duplicate run started.", "info")
    else:
        flash("Bot run was not started; check logs if this repeats.", "warning")
    return redirect(url_for("bot_dashboard"))


@app.route("/bot/tick", methods=["GET", "POST"])
def bot_tick():
    auth = require_machine_token()
    if auth is not True:
        return auth
    if _bot_run_lock.locked():
        return jsonify({"status": "already_running"}), 409
    scan_warm_started = False
    scan_warm_error = None
    if not PYTHONANYWHERE_MODE:
        try:
            scan_warm_started = bool(warm_scan_if_due())
        except Exception as e:
            scan_warm_started = False
            scan_warm_error = f"{type(e).__name__}: {e}"
    start = time.time()
    b, traded, last_action = run_bot(
        force=True,
        user_forced=True,
        max_runtime_sec=BOT_TICK_MAX_RUNTIME_SEC,
    )
    diag = b.get("last_no_buy_diagnostics") or {}
    runtime = round(time.time() - start, 3)
    diagnostics = {
        "main_blocker": diag.get("main_blocker"),
        "blocker_stage": diag.get("blocker_stage"),
        "blocker_code": diag.get("blocker_code"),
        "blocker_detail": diag.get("blocker_detail"),
        "market_open": diag.get("market_open"),
        "tod_ok": diag.get("tod_ok"),
        "buy_window_open": diag.get("buy_window_open"),
        "trading_mode": diag.get("trading_mode"),
        "normal_mode_active": diag.get("normal_mode_active"),
        "proxy_mode_active": diag.get("proxy_mode_active"),
        "degraded_mode_active": diag.get("degraded_mode_active"),
        "degraded_mode_reason": diag.get("degraded_mode_reason"),
        "min_buy_confidence": diag.get("min_buy_confidence"),
        "min_trade_size_effective": diag.get("min_trade_size_effective"),
        "degraded_size_mult": diag.get("degraded_size_mult"),
        "degraded_use_standard_gates_for_testing": diag.get("degraded_use_standard_gates_for_testing"),
        "degraded_standard_gates_active": diag.get("degraded_standard_gates_active"),
        "degraded_gate_policy": diag.get("degraded_gate_policy"),
        "effective_size_mult": diag.get("effective_size_mult"),
        "effective_min_buy_confidence": diag.get("effective_min_buy_confidence"),
        "normal_ev_gates_required": diag.get("normal_ev_gates_required"),
        "normal_risk_caps_required": diag.get("normal_risk_caps_required"),
        "fresh_quote_required": diag.get("fresh_quote_required"),
        "degraded_min_confidence": diag.get("degraded_min_confidence"),
        "degraded_reject_counts": diag.get("degraded_reject_counts", {}),
        "finnhub_key_configured": diag.get("finnhub_key_configured", bool(FINNHUB_KEY)),
        "fmp_key_configured": diag.get("fmp_key_configured", bool(FMP_KEY)),
        "stooq_status": diag.get("stooq_status"),
        "data_health_ok": diag.get("data_health_ok"),
        "data_health_blocks": diag.get("data_health_blocks", []),
        "data_health_warnings": diag.get("data_health_warnings", []),
        "spy_data_ok": diag.get("spy_data_ok"),
        "spy_data_source": diag.get("spy_data_source"),
        "spy_data_error": diag.get("spy_data_error"),
        "spy_rows": diag.get("spy_rows"),
        "spy_last_date": diag.get("spy_last_date"),
        "regime_data_status": diag.get("regime_data_status"),
        "regime_data_source": diag.get("regime_data_source"),
        "regime_data_error": diag.get("regime_data_error"),
        "regime_data_warnings": diag.get("regime_data_warnings", []),
        "stale_daily_cache_age_hours": diag.get("stale_daily_cache_age_hours"),
        "vix_label": diag.get("vix_label"),
        "vix_value": diag.get("vix_value"),
        "vix_data_ok": diag.get("vix_data_ok"),
        "volatility_data_ok": diag.get("volatility_data_ok"),
        "volatility_source": diag.get("volatility_source"),
        "volatility_value": diag.get("volatility_value"),
        "volatility_data_error": diag.get("volatility_data_error"),
        "signal_counts": diag.get("signal_counts", {}),
        "display_signal_counts": diag.get("display_signal_counts", {}),
        "raw_buy_count": diag.get("raw_buy_count", 0),
        "display_buy_candidate_count": diag.get("display_buy_candidate_count", 0),
        "history_source_counts": diag.get("history_source_counts", {}),
        "history_missing_count": diag.get("history_missing_count", 0),
        "top_missing_history_symbols": diag.get("top_missing_history_symbols", []),
        "history_fmp_fallback_count": diag.get("history_fmp_fallback_count", 0),
        "history_fmp_attempted_count": diag.get("history_fmp_attempted_count", 0),
        "history_fmp_skipped_count": diag.get("history_fmp_skipped_count", 0),
        "history_fmp_rate_limited_count": diag.get("history_fmp_rate_limited_count", 0),
        "history_fmp_global_circuit_skipped_count": diag.get("history_fmp_global_circuit_skipped_count", 0),
        "history_finnhub_daily_blocked_count": diag.get("history_finnhub_daily_blocked_count", 0),
        "history_stale_cache_count": diag.get("history_stale_cache_count", 0),
        "fmp_daily_global_circuit_status": diag.get("fmp_daily_global_circuit_status"),
        "fmp_daily_rate_limited": diag.get("fmp_daily_rate_limited", False),
        "fmp_daily_cooldown_remaining_sec": diag.get("fmp_daily_cooldown_remaining_sec", 0),
        "fmp_daily_last_429_age_sec": diag.get("fmp_daily_last_429_age_sec"),
        "finnhub_daily_global_circuit_status": diag.get("finnhub_daily_global_circuit_status"),
        "finnhub_daily_forbidden": diag.get("finnhub_daily_forbidden", False),
        "finnhub_daily_cooldown_remaining_sec": diag.get("finnhub_daily_cooldown_remaining_sec", 0),
        "max_history_fetches_per_tick": diag.get("max_history_fetches_per_tick"),
        "api_circuit_breakers": diag.get("api_circuit_breakers", {}),
        "provider_health_status": diag.get("provider_health_status"),
        "rate_limit_recent": diag.get("rate_limit_recent", False),
        "ticker_signal_debug": (diag.get("ticker_signal_debug") or [])[:PA_TICKERS_PER_BOT_RUN],
        "candidate_pool_count": diag.get("candidate_pool_count", 0),
        "ranked_count": diag.get("ranked_count", 0),
        "tradable_count": diag.get("tradable_count", 0),
        "top_ranked": (diag.get("top_ranked") or [])[:5],
        "top_ranked_rejections": (diag.get("top_ranked_rejections") or [])[:5],
        "skip_reason_counts": diag.get("skip_reason_counts", {}),
        "buyable_reject_counts": diag.get("buyable_reject_counts", {}),
        "top_buyable_rejects": (diag.get("top_buyable_rejects") or [])[:5],
        "scan_fresh": diag.get("scan_fresh"),
        "scan_age_sec": diag.get("scan_age_sec"),
        "scan_rows_count": diag.get("scan_rows_count"),
        "scan_fresh_rows_count": diag.get("scan_fresh_rows_count"),
        "cash": diag.get("cash"),
        "cash_floor": diag.get("cash_floor"),
        "cash_available_after_floor": diag.get("cash_available_after_floor"),
        "gross_exposure_pct": diag.get("gross_exposure_pct"),
        "buys_today": diag.get("buys_today"),
        "max_buys_today": diag.get("max_buys_today"),
        "partial_result": diag.get("partial_result"),
        "timeout_reason": diag.get("timeout_reason"),
        "timeout_stage": diag.get("timeout_stage"),
        "fetch_timeout_tickers": diag.get("fetch_timeout_tickers", []),
        "paper_trading_locked": diag.get("paper_trading_locked"),
        "paper_lock_reason": diag.get("paper_lock_reason"),
        "tick_runtime_seconds": diag.get("tick_runtime_seconds"),
        "runtime_seconds": diag.get("tick_runtime_seconds"),
    }
    return jsonify({
        "status": diag.get("main_blocker") or last_action or "complete",
        "traded": bool(traded),
        "last_action": last_action,
        "runtime_seconds": runtime,
        "max_runtime_seconds": BOT_TICK_MAX_RUNTIME_SEC,
        "history_source_counts": diag.get("history_source_counts", {}),
        "history_missing_count": diag.get("history_missing_count", 0),
        "top_missing_history_symbols": diag.get("top_missing_history_symbols", []),
        "history_fmp_fallback_count": diag.get("history_fmp_fallback_count", 0),
        "history_fmp_attempted_count": diag.get("history_fmp_attempted_count", 0),
        "history_fmp_skipped_count": diag.get("history_fmp_skipped_count", 0),
        "history_fmp_rate_limited_count": diag.get("history_fmp_rate_limited_count", 0),
        "history_fmp_global_circuit_skipped_count": diag.get("history_fmp_global_circuit_skipped_count", 0),
        "history_finnhub_daily_blocked_count": diag.get("history_finnhub_daily_blocked_count", 0),
        "history_stale_cache_count": diag.get("history_stale_cache_count", 0),
        "fmp_daily_global_circuit_status": diag.get("fmp_daily_global_circuit_status"),
        "fmp_daily_rate_limited": diag.get("fmp_daily_rate_limited", False),
        "fmp_daily_cooldown_remaining_sec": diag.get("fmp_daily_cooldown_remaining_sec", 0),
        "fmp_daily_last_429_age_sec": diag.get("fmp_daily_last_429_age_sec"),
        "finnhub_daily_global_circuit_status": diag.get("finnhub_daily_global_circuit_status"),
        "finnhub_daily_forbidden": diag.get("finnhub_daily_forbidden", False),
        "finnhub_daily_cooldown_remaining_sec": diag.get("finnhub_daily_cooldown_remaining_sec", 0),
        "ticker_signal_debug": (diag.get("ticker_signal_debug") or [])[:PA_TICKERS_PER_BOT_RUN],
        "scan_warm_started": scan_warm_started,
        "scan_warm_error": scan_warm_error,
        "environment": {
            "pythonanywhere_mode": bool(PYTHONANYWHERE_MODE),
            "persistent_cache": bool(PERSISTENT_CACHE),
            "finnhub_key_configured": bool(FINNHUB_KEY),
            "fmp_key_configured": bool(FMP_KEY),
        },
        "last_no_buy_diagnostics": diagnostics,
    })


def _warm_history_priority_symbols(limit):
    limit = max(0, int(limit or 0))
    if limit <= 0:
        return []
    symbols = []
    try:
        diag = (load_bot().get("last_no_buy_diagnostics") or {})
    except Exception:
        diag = {}
    symbols.extend(diag.get("top_missing_history_symbols") or [])
    for row in diag.get("top_buyable_rejects") or []:
        if str((row or {}).get("rejection_reason") or "").upper() == "MISSING_HISTORY":
            symbols.append((row or {}).get("ticker"))
    for row in diag.get("top_rejected_candidates") or []:
        code = str((row or {}).get("blocker_code") or (row or {}).get("blocker_detail") or "").upper()
        if code == "MISSING_HISTORY":
            symbols.append((row or {}).get("symbol") or (row or {}).get("ticker"))
    symbols.extend(load_tickers()[:PA_TICKERS_PER_BOT_RUN])
    out = []
    for sym in symbols:
        sym = str(sym or "").upper().strip()
        if not re.match(r"^[A-Z][A-Z0-9.\-]{0,7}$", sym):
            continue
        if sym not in out:
            out.append(sym)
        if len(out) >= limit:
            break
    return out


@app.route("/bot/warm-history", methods=["GET", "POST"])
def bot_warm_history():
    auth = require_machine_token()
    if auth is not True:
        return auth
    cfg = DEFAULT_CONFIG.get("history", {})
    max_symbols = int(cfg.get("warm_history_max_symbols_per_call", 3))
    max_fetches = int(cfg.get("warm_history_max_fetches_per_call", 3))
    raw_symbols = (request.values.get("symbols") or "").strip()
    if raw_symbols:
        requested = [
            s.strip().upper()
            for s in raw_symbols.split(",")
            if re.match(r"^[A-Z][A-Z0-9.\-]{0,7}$", s.strip().upper())
        ]
    else:
        requested = _warm_history_priority_symbols(max_symbols)
    requested = list(dict.fromkeys(requested))
    start = time.time()
    result = warm_history(requested, max_symbols=max_symbols, max_fetches=max_fetches)
    runtime = round(time.time() - start, 3)
    return jsonify({
        "ok": True,
        "requested_symbols": result.get("requested_symbols", requested),
        "attempted_symbols": result.get("attempted_symbols", []),
        "warmed_symbols": result.get("warmed_symbols", []),
        "skipped_symbols": result.get("skipped_symbols", []),
        "failed_symbols": result.get("failed_symbols", []),
        "cache_hit_symbols": result.get("cache_hit_symbols", []),
        "provider_used_by_symbol": result.get("provider_used_by_symbol", {}),
        "rows_by_symbol": result.get("rows_by_symbol", {}),
        "errors_by_symbol": result.get("errors_by_symbol", {}),
        "provider_circuits": result.get("provider_circuits", {}),
        "runtime_seconds": runtime,
        "max_symbols_per_call": max_symbols,
        "max_fetches_per_call": max_fetches,
    })


@app.route("/bot/suggestion-feedback", methods=["POST"])
def bot_suggestion_feedback():
    require_admin_token()
    action = (request.form.get("action") or "").strip().lower()
    ticker = (request.form.get("ticker") or "").strip().upper()
    run_id = (request.form.get("run_id") or "").strip()
    if action not in {"useful", "weak", "hide"} or not ticker:
        flash("Invalid suggestion feedback.", "warning")
        return redirect(url_for("bot_dashboard"))
    try:
        record_suggestion_feedback(SUGGESTION_DB_FILE, run_id, ticker, action, int(time.time()))
    except Exception as e:
        try: print(f"[suggestion_store] feedback failed: {e}")
        except Exception: pass
    if action == "hide":
        with _bot_run_lock, acquire_bot_file_lock(timeout=10):
            b = load_bot()
            b["extra_ticker_suggestions"] = [
                s for s in b.get("extra_ticker_suggestions", [])
                if s.get("ticker") != ticker or s.get("run_id") != run_id
            ]
            save_bot(b)
    flash(f"Saved {ticker} suggestion feedback.", "success")
    return redirect(url_for("bot_dashboard"))


@app.route("/health")
def health():
    """Passive keepalive endpoint. Trading is triggered by authenticated /bot/tick."""
    return "ok", 200, {"Content-Type": "text/plain"}
