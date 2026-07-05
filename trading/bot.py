"""Paper-trading bot: decision loop, position management, scan, snapshots.

Top-level entrypoints:
- run_bot(force=False)  → callable from scheduler, /bot/run, keepalive
- bot_state()           → snapshot for /bot dashboard
- _render_bot_page(read_only)  → builds the template context

State lives in mahiro_oyama/data/bot_state.json via utils.storage.
"""
import time
import threading
import gc
import uuid
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone

from flask import render_template

from market.charts import RANGE_SECS  # noqa: F401 — re-export sentinel
from market.history import get_history, get_intraday_context
from market.quotes import (
    EXECUTION_CACHE_MAX_AGE_SEC,
    FMP_DAILY_GLOBAL_ENDPOINT,
    get_quote,
    fmp_daily_global_circuit_state,
    finnhub_daily_global_circuit_state,
    set_history_fetch_budget,
)
from market.sentiment import get_news
from trading.indicators import classify_vol_regime, median_atr_since
from trading.risk import (
    get_sector, get_corr_group, get_market_regime, get_vix,
    get_earnings_soon, get_analyst_rec, get_insider_sentiment,
)
from trading.signals import classify_display_signal, get_recommendation
from trading.attribution import (
    ensure_attribution_state, exit_profile, record_entry_event,
    record_exit_event, update_exit_post_outcomes, update_forward_outcomes,
)
from trading.catalysts import classify_catalyst
from trading.config import DEFAULT_CONFIG, active_config, config_hash
from trading.exits import (
    round_trip_cost_pct, breakeven_lock_pct, dynamic_stop_pct, dynamic_trail_width,
)
from trading.exit_ladders import apply_regime_exit_tightening, compose_exit_profile
from trading.sizing import (
    slippage_bps, _partial_trim,
    PARTIAL_TAKE_ENABLED, PARTIAL_TAKE_PCT, PARTIAL_TAKE_FRACTION,
    SLIPPAGE_BPS, COMMISSION_PER_TRADE,
    entry_cluster, rank_candidates,
)
from trading.suggestions import _suggestion_cfg, rank_suggestion_candidates
from trading.suggestion_store import (
    load_feedback_stats,
    load_recent_suggestions,
    log_suggestion_run,
    prune_suggestion_store,
)
from trading.regime_v3 import (
    apply_confirmation, cluster_regime_mult, regime_risk_mult,
)
from utils.cache import api_failure_snapshot, cache_get, cache_set, prune_cache_dir   # noqa: F401
from utils.deploy_config import (
    FINNHUB_KEY,
    FMP_KEY,
    PA_SCAN_BATCH_SIZE,
    PA_TICKERS_PER_BOT_RUN,
    PYTHONANYWHERE_MODE,
)
from utils.config import BOT_ENABLED
from utils.storage import (
    acquire_bot_file_lock, load_bot, save_bot, load_tickers, SUGGESTION_DB_FILE,
    DATA_DIR, storage_debug_info,
)
from utils.threading_utils import _bot_run_lock, _BOT_STATUS, BOT_INTERVAL
from utils.time_utils import (
    NYSE_HOLIDAYS_FULL,
    is_market_open,
    _fmt_times,
    in_new_buy_window,
)

# ── Bot config ──────────────────────────────────────────────────────────────
STARTING_CASH      = 10_000.0
_RISK_DEFAULTS = DEFAULT_CONFIG.get("risk", {})
_SCAN_DEFAULTS = DEFAULT_CONFIG.get("scan", {})
_HISTORY_DEFAULTS = DEFAULT_CONFIG.get("history", {})
BOT_MAX_BUYS       = int(_RISK_DEFAULTS.get("max_new_buys_per_tick", 10))
BOT_SCAN_BUY       = int(_SCAN_DEFAULTS.get("max_scan_buys_per_tick", 10))
MAX_POSITIONS      = 24
SCAN_BUY_MIN_CONF  = int(_SCAN_DEFAULTS.get("scan_buy_min_confidence", 45))
SUGGESTION_MIN_SCAN_CONF = 68
SCAN_FRESHNESS_SEC = int(_SCAN_DEFAULTS.get("scan_data_max_age_sec", 300))
SUGGESTION_MAX_EXTRA_TICKERS = 2
SUGGESTION_DISCOVERY_TOP_CONF = 6
SUGGESTION_DISCOVERY_TOP_GAIN = 4
SUGGESTION_DISCOVERY_PREFILTER_MAX = 8
SUGGESTION_DISCOVERY_FULL_FETCH_MAX = 6
SUGGESTION_MIN_ADV_USD = 15_000_000
SUGGESTION_MIN_PRICE = 5.0
SUGGESTION_RECENT_TICKER_COOLDOWN_SEC = 6 * 3600
EQUITY_HISTORY_MAX = 400 if PYTHONANYWHERE_MODE else 2000
BENCHMARK_ONLY_TICKERS = {"SPY", "QQQ", "VOO"}
KNIFE_CATALYST_TYPES = {
    "earnings_miss",
    "guidance_cut",
    "regulatory_risk",
    "lawsuit_investigation",
}

# A2 (cost-aware sizing): minimum dollars for ANY position open or add. At $0.99/trade
# a $150 fill keeps the round-trip commission ≤ 1.32% (audit P0-3; $10 was 19.8%).
# Sizing and EV gates still decide whether a candidate is worth taking.
# Pyramid adds, new positions, and outside positions use the same minimum.
MIN_POSITION_USD   = float(_RISK_DEFAULTS.get("min_trade_size", 150.0))
BOT_TICK_MAX_RUNTIME_SEC = 25
MIN_SIGNAL_HISTORY_ROWS = int(_HISTORY_DEFAULTS.get("min_history_rows_for_buy", 25))
PREFERRED_SIGNAL_HISTORY_ROWS = int(_HISTORY_DEFAULTS.get("preferred_history_rows", 60))
STALE_HISTORY_MAX_COMPLETED_TRADING_DAYS = int(
    _HISTORY_DEFAULTS.get("stale_history_max_completed_trading_days", 2)
)
USE_CONFIDENCE_WEIGHTING = True
REQUIRED_TICK_LOG_FIELDS = (
    "timestamp",
    "trading_mode",
    "degraded_mode_active",
    "degraded_mode_reason",
    "degraded_use_standard_gates_for_testing",
    "degraded_standard_gates_active",
    "degraded_gate_policy",
    "effective_size_mult",
    "effective_min_buy_confidence",
    "normal_ev_gates_required",
    "normal_risk_caps_required",
    "fresh_quote_required",
    "market_open",
    "buy_window_open",
    "spy_data_ok",
    "spy_data_source",
    "spy_data_error",
    "volatility_data_ok",
    "volatility_source",
    "volatility_error",
    "data_health_blocks",
    "data_health_warnings",
    "raw_buy_count",
    "display_buy_candidate_count",
    "history_source_counts",
    "history_missing_count",
    "top_missing_history_symbols",
    "history_fmp_fallback_count",
    "history_fmp_attempted_count",
    "history_fmp_skipped_count",
    "history_fmp_rate_limited_count",
    "history_fmp_global_circuit_skipped_count",
    "history_finnhub_daily_blocked_count",
    "history_stale_cache_count",
    "fmp_daily_global_circuit_status",
    "fmp_daily_rate_limited",
    "fmp_daily_cooldown_remaining_sec",
    "fmp_daily_last_429_age_sec",
    "finnhub_daily_global_circuit_status",
    "finnhub_daily_forbidden",
    "finnhub_daily_cooldown_remaining_sec",
    "max_history_fetches_per_tick",
    "blocker_stage",
    "blocker_code",
    "blocker_detail",
    "candidate_pool_count",
    "ranked_count",
    "tradable_count",
    "top_rejection_reasons",
    "degraded_reject_counts",
    "buys_executed",
    "sells_executed",
    "runtime_seconds",
)

# Scheduler cooperation
_bot_running = False


def _bump(counts, reason):
    key = str(reason or "unknown")
    counts[key] = counts.get(key, 0) + 1


def _append_jsonl(filename, event):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    except Exception:
        try:
            _BOT_STATUS["last_log_error"] = f"{filename}:write_failed"
            _BOT_STATUS["last_log_error_ts"] = int(time.time())
        except Exception:
            pass


def _log_bot_event(event_type, **payload):
    event = {"ts": int(time.time()), "event": str(event_type or "BOT_EVENT")}
    event.update(payload)
    _append_jsonl("bot_events.jsonl", event)


def _tick_log_entry(diag):
    top_rejections = []
    for key in ("top_buyable_rejects", "top_ranked_rejections"):
        for row in diag.get(key) or []:
            reason = row.get("rejection_reason") or row.get("rank_reason")
            if reason:
                top_rejections.append(reason)
    event = dict(diag)
    event.update({
        "timestamp": diag.get("timestamp") or diag.get("ts") or int(time.time()),
        "volatility_error": diag.get("volatility_error") or diag.get("volatility_data_error"),
        "top_rejection_reasons": top_rejections[:10],
        "buys_executed": int(diag.get("buys_executed", 0) or 0),
        "sells_executed": int(diag.get("sells_executed", 0) or 0),
        "runtime_seconds": diag.get("tick_runtime_seconds"),
    })
    for key in REQUIRED_TICK_LOG_FIELDS:
        event.setdefault(key, [] if key in {"data_health_blocks", "top_rejection_reasons"} else None)
    return event


def _log_tick(diag):
    if isinstance(diag, dict):
        _append_jsonl("tick_log.jsonl", _tick_log_entry(diag))


def _annotate_signal(rec):
    if not isinstance(rec, dict):
        return rec
    raw = rec.get("signal") or rec.get("cls") or "hold"
    rec["raw_signal_label"] = raw
    rec["display_signal_label"] = classify_display_signal(
        rec.get("cls") or raw,
        rec.get("confidence", 0),
    )
    return rec


def _finalize_signal_confidence(payload, cfg):
    payload = payload or {}
    rec = payload.get("rec") or {}
    if not isinstance(rec, dict):
        return rec

    ctx = payload.get("ctx") or {}
    quote = payload.get("quote") or {}
    price = quote.get("price", payload.get("price", 0))
    quote_status = _quote_execution_status(payload)
    blockers = list(rec.get("confidence_floor_blockers") or [])
    if not quote_status["execution_trusted"]:
        blockers.append("stale_quote" if price and price > 0 else "invalid_price")
    history_status = _history_execution_status(ctx)
    if history_status["blocker"]:
        blockers.append(history_status["blocker"])
    elif history_status["warning"]:
        rec["data_quality_warning"] = history_status["warning"]
    if history_status.get("trusted"):
        blockers = [b for b in blockers if b != "low_data_quality"]
    ctx["history_warnings"] = history_status.get("history_warnings", ctx.get("history_warnings", []))
    payload["ctx"] = ctx
    payload["execution_trusted"] = quote_status["execution_trusted"]
    payload["quote_age_seconds"] = quote_status["quote_age_seconds"]
    payload["cache_used"] = quote_status["cache_used"]
    payload["stale"] = not quote_status["execution_trusted"]

    blockers = list(dict.fromkeys(blockers))
    rec["confidence_floor_blockers"] = blockers
    try:
        conf = float(
            rec.get("confidence_before_floor",
                    rec.get("confidence_after_penalties",
                            rec.get("confidence", 0))) or 0
        )
    except Exception:
        conf = 0.0
    rec.setdefault("confidence_before_floor", conf)
    rec["confidence"] = int(conf) if float(conf).is_integer() else conf
    if rec.get("cls") not in ("strong-buy", "strong-sell"):
        rec["sizing_confidence"] = rec["confidence"]
    min_conf = float(((cfg or {}).get("signal") or {}).get("min_buy_confidence", 40))
    should_floor = bool(rec.get("confidence_floor_candidate")) and not blockers and conf < min_conf
    if should_floor:
        floored = int(min_conf) if float(min_conf).is_integer() else min_conf
        rec["confidence"] = floored
        rec["confidence_final"] = floored
        rec["confidence_floor_applied"] = True
        rec["confidence_floor_reason"] = "valid_raw_buy_floor"
        if rec.get("cls") not in ("strong-buy", "strong-sell"):
            rec["sizing_confidence"] = floored
        reasons = rec.setdefault("reasons", [])
        floor_note = f"Confidence floor: valid raw BUY -> {floored}%"
        if floor_note not in reasons:
            reasons.append(floor_note)
    else:
        rec["confidence_final"] = rec.get("confidence")
        rec["confidence_floor_applied"] = False
        if rec.get("confidence_floor_candidate") and blockers:
            rec["confidence_floor_reason"] = blockers[0]
        elif not blockers:
            rec["confidence_floor_reason"] = None
    rec = _annotate_signal(rec)
    _apply_execution_profile(rec, cfg)
    return rec


def _history_source_bucket(source):
    source = str(source or "missing")
    if source.startswith("stale_cache"):
        return "stale_cache"
    return source or "missing"


def _degraded_standard_gates_enabled(mode_cfg):
    return bool((mode_cfg or {}).get("degraded_use_standard_gates_for_testing", True))


def _execution_min_confidence(cfg, degraded_mode_active=False):
    signal_cfg = (cfg or {}).get("signal", {})
    mode_cfg = (cfg or {}).get("market_data_modes", {})
    normal_min = float(signal_cfg.get("min_buy_confidence", 40))
    if degraded_mode_active and not _degraded_standard_gates_enabled(mode_cfg):
        return float(mode_cfg.get("degraded_min_confidence", normal_min))
    return normal_min


def _quote_execution_status(payload):
    payload = payload or {}
    quote = payload.get("quote") if isinstance(payload.get("quote"), dict) else payload
    quote = quote or {}
    try:
        price = float(quote.get("price", payload.get("price", 0)) or 0)
    except Exception:
        price = 0.0
    age = quote.get("quote_age_seconds")
    if age is None:
        age = quote.get("stale_age_sec")
    try:
        age = int(float(age)) if age is not None and float(age) >= 0 else None
    except Exception:
        age = None
    trusted = quote.get("execution_trusted")
    if trusted is None:
        stale = bool(payload.get("stale") or quote.get("stale"))
        trusted = bool(
            price > 0
            and (
                not stale
                or (age is not None and age <= EXECUTION_CACHE_MAX_AGE_SEC)
            )
        )
    trusted = bool(price > 0 and trusted)
    if price <= 0:
        blocker = "INVALID_PRICE"
        stale_reason = "invalid_price"
    elif not trusted:
        blocker = "STALE_CANDIDATE_QUOTE"
        stale_reason = quote.get("stale_reason") or (
            "quote_cache_missing_timestamp"
            if age is None
            else "quote_cache_older_than_execution_limit"
        )
    else:
        blocker = None
        stale_reason = quote.get("stale_reason")
    return {
        "price": price,
        "execution_trusted": trusted,
        "blocker": blocker,
        "quote_source": quote.get("quote_source") or quote.get("source"),
        "cache_used": bool(quote.get("cache_used")),
        "quote_age_seconds": age,
        "execution_cache_max_age_sec": EXECUTION_CACHE_MAX_AGE_SEC,
        "stale_reason": stale_reason,
    }


def _is_trading_day(d):
    return d.weekday() < 5 and d.strftime("%Y-%m-%d") not in NYSE_HOLIDAYS_FULL


def _previous_trading_day(d):
    cur = d - timedelta(days=1)
    while not _is_trading_day(cur):
        cur -= timedelta(days=1)
    return cur


def _latest_completed_trading_day():
    try:
        from zoneinfo import ZoneInfo

        et = datetime.now(ZoneInfo("America/New_York"))
        today = et.date()
        if _is_trading_day(today) and (et.hour * 60 + et.minute) >= 16 * 60:
            return today
        return _previous_trading_day(today)
    except Exception:
        today = datetime.utcnow().date()
        return today if _is_trading_day(today) else _previous_trading_day(today)


def _completed_trading_days_since(d):
    latest = _latest_completed_trading_day()
    if d >= latest:
        return 0
    cur = d + timedelta(days=1)
    days = 0
    while cur <= latest:
        if _is_trading_day(cur):
            days += 1
        cur += timedelta(days=1)
    return days


def _history_execution_status(ctx):
    ctx = ctx or {}
    try:
        rows = int(ctx.get("history_rows") or 0)
    except Exception:
        rows = 0
    source = str(ctx.get("history_source") or "")
    status = str(ctx.get("history_status") or "")
    warnings = list(ctx.get("history_warnings") or [])
    if rows <= 0:
        return {
            "trusted": False,
            "blocker": "missing_history",
            "warning": None,
            "history_rows": rows,
            "history_status": status or "missing",
            "history_source": source or "missing",
            "history_warnings": warnings,
        }
    if rows < MIN_SIGNAL_HISTORY_ROWS:
        return {
            "trusted": False,
            "blocker": "insufficient_history",
            "warning": None,
            "history_rows": rows,
            "history_status": status or "insufficient",
            "history_source": source or "daily",
            "history_warnings": warnings,
        }
    warning = None
    if rows < PREFERRED_SIGNAL_HISTORY_ROWS:
        warning = "INSUFFICIENT_HISTORY_WARNING"
        warnings.append(warning)
    stale = source.startswith("stale_cache") or status == "stale_cache"
    if stale:
        last_date = None
        try:
            last_date = datetime.strptime(str(ctx.get("history_last_date")), "%Y-%m-%d").date()
        except Exception:
            last_date = None
        stale_days = _completed_trading_days_since(last_date) if last_date else None
        if stale_days is None or stale_days > STALE_HISTORY_MAX_COMPLETED_TRADING_DAYS:
            return {
                "trusted": False,
                "blocker": "stale_history",
                "warning": None,
                "history_rows": rows,
                "history_status": status or "stale_cache",
                "history_source": source,
                "history_warnings": warnings,
                "stale_history_completed_trading_days": stale_days,
            }
        warning = warning or "STALE_HISTORY_CACHE_USED"
        warnings.append("STALE_HISTORY_CACHE_USED")
    return {
        "trusted": True,
        "blocker": None,
        "warning": warning,
        "history_rows": rows,
        "history_status": status or "ok",
        "history_source": source or "daily",
        "history_warnings": list(dict.fromkeys(warnings)),
    }


def _has_hard_bearish_signal(rec):
    rec = rec or {}
    raw_cls = str(rec.get("cls") or "").lower()
    display = str(rec.get("display_signal_label") or "").upper()
    if raw_cls in ("sell", "strong-sell") or display in ("SELL", "STRONG_SELL"):
        return True
    if bool(rec.get("force_hold")):
        return True
    try:
        if float(rec.get("cats_neg", 0) or 0) > 1:
            return True
    except Exception:
        pass
    return False


def _execution_profile(rec, cfg, *, degraded_mode_active=False):
    rec = _annotate_signal(rec or {})
    signal_cfg = (cfg or {}).get("signal", {})
    raw_cls = str(rec.get("cls") or "").lower()
    display = rec.get("display_signal_label")
    try:
        conf = float(rec.get("confidence", 0) or 0)
    except Exception:
        conf = 0.0
    min_conf = _execution_min_confidence(cfg, degraded_mode_active)
    hard_bearish = _has_hard_bearish_signal(rec)
    base = {
        "eligible": False,
        "candidate_type": None,
        "candidate_size_multiplier": 1.0,
        "reason": None,
    }
    if hard_bearish:
        base["reason"] = f"raw_class_{raw_cls or 'missing'}"
        return base
    if raw_cls in ("buy", "strong-buy"):
        if display in ("BUY_CANDIDATE", "STRONG_BUY_CANDIDATE") and conf >= min_conf:
            base.update({
                "eligible": True,
                "candidate_type": "standard_buy_candidate",
                "reason": "executable_buy_candidate",
            })
            return base
        if (
            display == "BULLISH_LEAN"
            and bool(signal_cfg.get("allow_bullish_lean_buy_attempts", False))
        ):
            base.update({
                "eligible": True,
                "candidate_type": "weak_paper_test_candidate",
                "candidate_size_multiplier": float(
                    signal_cfg.get("bullish_lean_candidate_size_multiplier", 0.85)
                ),
                "reason": "bullish_lean_paper_test_candidate",
            })
            return base
        if conf < min_conf:
            base["reason"] = "CONFIDENCE_BELOW_MIN_BUY"
        else:
            base["reason"] = f"display_signal_{display or 'missing'}"
        return base
    if (
        raw_cls == "hold"
        and display == "HOLD"
        and bool(signal_cfg.get("allow_high_confidence_hold_buy_attempts", False))
        and conf >= float(signal_cfg.get("high_confidence_hold_min_confidence", 55))
    ):
        base.update({
            "eligible": True,
            "candidate_type": "experimental_near_buy_candidate",
            "candidate_size_multiplier": float(
                signal_cfg.get("high_confidence_hold_candidate_size_multiplier", 0.65)
            ),
            "reason": "high_confidence_hold_paper_test_candidate",
        })
        return base
    if raw_cls == "hold":
        base["reason"] = "HOLD_BELOW_EXPERIMENTAL_THRESHOLD"
    else:
        base["reason"] = f"raw_class_{raw_cls or 'missing'}"
    return base


def _apply_execution_profile(rec, cfg, *, degraded_mode_active=False):
    profile = _execution_profile(rec, cfg, degraded_mode_active=degraded_mode_active)
    rec = rec or {}
    rec["candidate_type"] = profile.get("candidate_type")
    rec["candidate_size_multiplier"] = profile.get("candidate_size_multiplier", 1.0)
    rec["execution_candidate_reason"] = profile.get("reason")
    if profile.get("eligible") and profile.get("candidate_type") == "weak_paper_test_candidate":
        min_conf = _execution_min_confidence(cfg, degraded_mode_active)
        try:
            sizing_conf = float(rec.get("sizing_confidence", rec.get("confidence", 0)) or 0)
        except Exception:
            sizing_conf = 0.0
        if sizing_conf < min_conf:
            rec["sizing_confidence"] = min_conf
    return profile


def _why_not_execution_eligible(rec, cfg, degraded_mode_active=False):
    profile = _execution_profile(rec, cfg, degraded_mode_active=degraded_mode_active)
    return None if profile.get("eligible") else profile.get("reason")


def _ticker_debug_row(t, payload, cfg, degraded_mode_active=False):
    payload = payload or {}
    rec = _annotate_signal(payload.get("rec") or {})
    ctx = payload.get("ctx") or {}
    quote = payload.get("quote") or {}
    price = quote.get("price", payload.get("price", 0))
    quote_status = _quote_execution_status(payload)
    history_status = _history_execution_status(ctx)
    execution_eligible = is_execution_candidate(
        rec,
        cfg,
        degraded_mode_active=degraded_mode_active,
    )
    why_not_execution = None if execution_eligible else _why_not_execution_eligible(
        rec,
        cfg,
        degraded_mode_active=degraded_mode_active,
    )
    why_not_buy = None
    if not execution_eligible:
        why_not_buy = why_not_execution or "score_below_buy_threshold_or_not_enough_positive_categories"
    if quote_status["blocker"]:
        why_not_buy = quote_status["blocker"].lower()
    elif history_status["blocker"]:
        why_not_buy = history_status["blocker"]
    ctx_quote_fresh = ctx.get("quote_fresh")
    return {
        "ticker": t,
        "history_source": ctx.get("history_source") or ("recorded" if ctx.get("source") == "recorded" else "missing"),
        "history_status": ctx.get("history_status"),
        "history_warnings": history_status.get("history_warnings", ctx.get("history_warnings", [])),
        "history_rows": ctx.get("history_rows", 0),
        "history_status_for_execution": history_status.get("history_status"),
        "history_last_date": ctx.get("history_last_date"),
        "stale_daily_cache_age_hours": ctx.get("stale_daily_cache_age_hours"),
        "provider_chain_debug": ctx.get("provider_chain_debug", []),
        "relative_strength_source": ctx.get("relative_strength_source"),
        "relative_strength_skipped_reason": ctx.get("relative_strength_skipped_reason"),
        "sector_etf_history_source": ctx.get("sector_etf_history_source"),
        "quote_source": quote_status.get("quote_source") or quote.get("source"),
        "provider_attempts": quote.get("provider_attempts", []),
        "provider_used": quote.get("provider_used") or quote.get("source"),
        "provider_error_type": quote.get("provider_error_type"),
        "cache_used": quote_status.get("cache_used"),
        "quote_age_seconds": quote_status.get("quote_age_seconds"),
        "execution_trusted": quote_status.get("execution_trusted"),
        "execution_cache_max_age_sec": quote_status.get("execution_cache_max_age_sec"),
        "stale_reason": quote_status.get("stale_reason"),
        "quote_price": price,
        "quote_pct": quote.get("pct"),
        "raw_class": rec.get("cls") or "hold",
        "display_signal_label": rec.get("display_signal_label"),
        "confidence": rec.get("confidence"),
        "confidence_scale": rec.get("confidence_scale"),
        "confidence_before_floor": rec.get("confidence_before_floor"),
        "confidence_before_penalties": rec.get("confidence_before_penalties"),
        "confidence_after_penalties": rec.get("confidence_after_penalties"),
        "confidence_final": rec.get("confidence_final", rec.get("confidence")),
        "confidence_floor_applied": bool(rec.get("confidence_floor_applied", False)),
        "confidence_floor_reason": rec.get("confidence_floor_reason"),
        "confidence_penalties": rec.get("confidence_penalties", []),
        "score_total": rec.get("score_total", rec.get("score")),
        "buy_score_threshold": (rec.get("thresholds") or {}).get("buy_tot"),
        "sell_score_threshold": (rec.get("thresholds") or {}).get("sell_tot"),
        "cats_pos": rec.get("cats_pos"),
        "cats_neg": rec.get("cats_neg"),
        "cats_required_for_buy": (rec.get("thresholds") or {}).get("buy_cats"),
        "data_quality": rec.get("data_quality"),
        "data_quality_actual_n": rec.get("data_quality_actual_n", rec.get("n_raw")),
        "data_quality_expected_n": rec.get("data_quality_expected_n", rec.get("expected_n")),
        "data_quality_missing_fields": rec.get("data_quality_missing_fields", []),
        "data_quality_warning": rec.get("data_quality_warning"),
        "missing_optional_signals": rec.get("missing_optional_signals", []),
        "candidate_type": rec.get("candidate_type"),
        "candidate_size_multiplier": rec.get("candidate_size_multiplier"),
        "execution_eligible": bool(execution_eligible),
        "why_not_buy": why_not_buy,
        "why_not_execution_eligible": why_not_execution,
        "live_bar_applied": bool(ctx.get("live_bar_applied", False)),
        "live_bar_reason": ctx.get("live_bar_reason"),
        "quote_fresh": bool(quote_status["execution_trusted"] if ctx_quote_fresh is None else ctx_quote_fresh),
    }


def is_execution_candidate(rec, cfg, *, degraded_mode_active=False):
    """True only for signals allowed to reach execution sizing."""
    return bool(_execution_profile(
        rec,
        cfg,
        degraded_mode_active=degraded_mode_active,
    ).get("eligible"))


def _new_no_buy_diag(now, b, force=False, user_forced=False):
    diag = {
        "ts": int(now),
        "market_open": bool(is_market_open()),
        "force": bool(force),
        "user_forced": bool(user_forced),
        "tod_ok": None,
        "buy_window_open": None,
        "regime_allow_buys": None,
        "regime_kind": None,
        "regime_source": None,
        "regime_v3": None,
        "trading_mode": None,
        "normal_mode_active": False,
        "proxy_mode_active": False,
        "degraded_mode_active": False,
        "degraded_mode_reason": None,
        "degraded_size_mult": None,
        "degraded_use_standard_gates_for_testing": None,
        "degraded_standard_gates_active": False,
        "degraded_gate_policy": None,
        "effective_size_mult": None,
        "effective_min_buy_confidence": None,
        "normal_ev_gates_required": None,
        "normal_risk_caps_required": None,
        "fresh_quote_required": None,
        "degraded_min_confidence": None,
        "min_trade_size_effective": None,
        "degraded_reject_counts": {},
        "degraded_buys_today": 0,
        "degraded_max_buys_today": None,
        "degraded_gross_exposure_pct": None,
        "degraded_max_gross_exposure_pct": None,
        "spy_data_ok": None,
        "regime_data_status": None,
        "regime_data_fallback": None,
        "regime_data_source": None,
        "regime_data_error": None,
        "regime_data_warnings": [],
        "spy_data_source": None,
        "spy_data_error": None,
        "stale_daily_cache_age_hours": None,
        "regime_data_size_mult": None,
        "spy_rows": None,
        "spy_last_date": None,
        "spy_mom_label": None,
        "vix_label": None,
        "vix_value": None,
        "vix_data_ok": None,
        "vix_data_status": None,
        "vix_display": None,
        "volatility_data_ok": None,
        "volatility_source": None,
        "volatility_value": None,
        "volatility_data_error": None,
        "data_health_ok": None,
        "data_health_blocks": [],
        "data_health_warnings": [],
        "cash": round(float(b.get("cash", 0)), 2),
        "cash_floor": None,
        "cash_available_after_floor": None,
        "gross_exposure_pct": None,
        "paper_trading_locked": bool(b.get("stopped")),
        "paper_lock_reason": "user_stopped" if b.get("stopped") else None,
        "buys_today": 0,
        "max_buys_today": None,
        "pa_mode": bool(PYTHONANYWHERE_MODE),
        "finnhub_key_configured": bool(FINNHUB_KEY),
        "fmp_key_configured": bool(FMP_KEY),
        "stooq_status": "skipped_on_pythonanywhere" if PYTHONANYWHERE_MODE else None,
        "pa_stage_status": b.get("pa_stage_status"),
        "checked_tickers": [],
        "stale_ticker_count": 0,
        "stale_tickers": [],
        "stale_positions": [],
        "risk_unmanaged_positions": [],
        "api_circuit_breakers": {},
        "provider_health_status": "healthy",
        "rate_limit_recent": False,
        "buys_executed": 0,
        "sells_executed": 0,
        "signal_counts": {},
        "display_signal_counts": {},
        "raw_buy_count": 0,
        "display_buy_candidate_count": 0,
        "history_source_counts": {},
        "history_missing_count": 0,
        "top_missing_history_symbols": [],
        "history_fmp_fallback_count": 0,
        "history_fmp_attempted_count": 0,
        "history_fmp_skipped_count": 0,
        "history_fmp_rate_limited_count": 0,
        "history_fmp_global_circuit_skipped_count": 0,
        "history_finnhub_daily_blocked_count": 0,
        "history_stale_cache_count": 0,
        "fmp_daily_global_circuit_status": None,
        "fmp_daily_rate_limited": False,
        "fmp_daily_cooldown_remaining_sec": 0,
        "fmp_daily_last_429_age_sec": None,
        "finnhub_daily_global_circuit_status": None,
        "finnhub_daily_forbidden": False,
        "finnhub_daily_cooldown_remaining_sec": 0,
        "max_history_fetches_per_tick": int(
            DEFAULT_CONFIG.get("history", {}).get("max_history_fetches_per_tick", 1)
        ),
        "ticker_signal_debug": [],
        "buyable_reject_counts": {},
        "top_buyable_rejects": [],
        "top_ranked_rejections": [],
        "top_rejected_candidates": [],
        "candidate_pool_count": 0,
        "ranked_count": 0,
        "tradable_count": 0,
        "top_ranked": [],
        "skip_reason_counts": {},
        "scan_fresh": None,
        "scan_buys_enabled": None,
        "scan_data_max_age_sec": None,
        "scan_min_confidence": None,
        "scan_min_history_rows": None,
        "scan_age_sec": None,
        "scan_rows_count": None,
        "scan_fresh_rows_count": None,
        "scan_payload_misses": 0,
        "main_blocker": None,
        "blocker_stage": None,
        "blocker_code": None,
        "blocker_detail": None,
        "tick_runtime_seconds": None,
    }
    return diag


_CORE_DATA_BLOCKERS = ("MISSING_HISTORY", "INVALID_PRICE", "MISSING_QUOTE", "UNTRUSTED_QUOTE")


def _history_provider_status_suffix():
    """Short human summary of daily-history provider circuits, for the visible reason."""
    parts = []
    try:
        if finnhub_daily_global_circuit_state().get("active"):
            parts.append("Finnhub daily forbidden")
    except Exception:
        pass
    try:
        fmp = fmp_daily_global_circuit_state()
        if fmp.get("active"):
            rem = int(fmp.get("cooldown_remaining_sec") or 0)
            parts.append(
                f"FMP daily rate-limited ({rem // 60}m cooldown)" if rem else "FMP daily rate-limited"
            )
    except Exception:
        pass
    return ("History provider status: " + "; ".join(parts) + ".") if parts else ""


def _core_data_blocker_reason(diag):
    """Human-readable core-data blocker for the visible HOLD reason.

    Runs in the reasoning block, which executes BEFORE _set_main_blocker (that
    fires later inside _persist_no_buy_diag), so this derives its message from the
    raw counters that are already populated (history_missing_count,
    top_missing_history_symbols, buyable_reject_counts) rather than from
    main_blocker/blocker_detail. Returns "" when there is no core-data blocker.
    """
    if not diag:
        return ""
    missing = int(diag.get("history_missing_count", 0) or 0)
    reject_counts = diag.get("buyable_reject_counts") or {}
    quote_problem = any(
        int(reject_counts.get(code, 0) or 0) > 0
        for code in ("INVALID_PRICE", "MISSING_QUOTE", "UNTRUSTED_QUOTE")
    )
    if missing <= 0 and not quote_problem:
        return ""
    top = ", ".join([s for s in (diag.get("top_missing_history_symbols") or []) if s][:3])
    top_suffix = f" Top missing: {top}." if top else ""
    if missing > 0 and quote_problem:
        detail = (
            "No trade: daily history missing AND quotes unavailable/invalid for "
            "active candidates." + top_suffix
        )
    elif missing > 0:
        detail = (
            "No trade: daily history missing for active candidates. Quotes may be "
            "fresh, but signal history is unavailable." + top_suffix
        )
    else:
        detail = "No trade: candidate quotes are unavailable/invalid (price <= 0 or untrusted)."
    prov = _history_provider_status_suffix()
    return f"{detail} {prov}".strip() if prov else detail


def _set_main_blocker(diag):
    if not diag:
        return None
    min_position_usd = float(diag.get("min_trade_size_effective") or MIN_POSITION_USD)
    if not diag.get("market_open"):
        blocker = "market_closed"
    elif diag.get("tod_ok") is False:
        blocker = "outside_new_buy_window"
    elif diag.get("data_health_blocks") and (
            not diag.get("degraded_mode_active")
            or "STALE_HELD_QUOTE" in (diag.get("data_health_blocks") or [])):
        blocker = "data_health_block"
    elif diag.get("paper_trading_locked"):
        blocker = diag.get("paper_lock_reason") or "paper_trading_locked"
    elif diag.get("regime_allow_buys") is False:
        blocker = "regime_or_vix_blocks_buys"
    elif (diag.get("cash_available_after_floor") is not None
          and diag.get("cash_available_after_floor", 0) < min_position_usd):
        blocker = "cash_below_min_position"
    elif int(diag.get("history_missing_count", 0) or 0) > 0 and (
            int(diag.get("candidate_pool_count", 0) or 0) <= 0
            or int(diag.get("display_buy_candidate_count", 0) or 0) > 0
            or diag.get("top_missing_history_symbols")):
        blocker = "MISSING_HISTORY"
        diag["blocker_stage"] = "HISTORY_QUALITY"
        diag["blocker_code"] = "MISSING_HISTORY"
        top = ", ".join(diag.get("top_missing_history_symbols") or [])
        suffix = f" Top missing: {top}." if top else ""
        _rc = diag.get("buyable_reject_counts") or {}
        _quote_problem = any(
            int(_rc.get(code, 0) or 0) > 0
            for code in ("INVALID_PRICE", "MISSING_QUOTE", "UNTRUSTED_QUOTE")
        )
        if _quote_problem:
            diag["blocker_detail"] = (
                "No trade: daily history missing AND quotes unavailable/invalid "
                "for active candidates." + suffix
            )
        else:
            diag["blocker_detail"] = (
                "No trade: daily history missing for active candidates. "
                "Quotes may be fresh, but signal history is unavailable."
                + suffix
            )
    elif diag.get("candidate_pool_count", 0) <= 0:
        if int(diag.get("raw_buy_count", 0) or 0) > 0:
            if int(diag.get("display_buy_candidate_count", 0) or 0) <= 0:
                blocker = "weak_raw_buys_only"
            else:
                blocker = "raw_buys_rejected_pre_candidate"
            diag["main_blocker"] = blocker
            if not diag.get("blocker_code"):
                diag["blocker_code"] = blocker
            return blocker
        counts = diag.get("buyable_reject_counts") or diag.get("signal_counts") or {}
        blocker = max(counts, key=counts.get) if counts else "no_buy_candidates"
    elif diag.get("tradable_count", 0) <= 0:
        counts = diag.get("skip_reason_counts") or {}
        blocker = max(counts, key=counts.get) if counts else "ev_or_risk_gates"
    else:
        blocker = "no_order_selected"
    if blocker in ("INVALID_PRICE", "MISSING_QUOTE", "UNTRUSTED_QUOTE"):
        diag["blocker_stage"] = "QUOTE_QUALITY"
        diag["blocker_code"] = blocker
        _detail_map = {
            "INVALID_PRICE": "No trade: candidate quotes are invalid (price <= 0).",
            "MISSING_QUOTE": "No trade: no quote available for active candidates.",
            "UNTRUSTED_QUOTE": "No trade: quotes are stale/untrusted for execution.",
        }
        _affected = ", ".join(
            r.get("ticker") for r in (diag.get("top_buyable_rejects") or [])[:3] if r.get("ticker")
        )
        _asuffix = f" Affected: {_affected}." if _affected else ""
        if int(diag.get("history_missing_count", 0) or 0) > 0:
            diag["blocker_detail"] = (
                _detail_map.get(blocker, "No trade: quote-quality blocker.")
                + " Daily history is also missing for these candidates." + _asuffix
            )
        else:
            diag["blocker_detail"] = _detail_map.get(blocker, "No trade: quote-quality blocker.") + _asuffix
    diag["main_blocker"] = blocker
    if not diag.get("blocker_code"):
        diag["blocker_code"] = blocker
    return blocker


def _fmt_vix_value(vix_data):
    try:
        return f"{float((vix_data or {}).get('vix')):.1f}"
    except Exception:
        return "n/a"


def _market_data_mode(regime, vix_data, cfg):
    mode_cfg = (cfg or {}).get("market_data_modes", {})
    signal_cfg = (cfg or {}).get("signal", {})
    spy_ok = regime.get("spy_data_ok") is True
    vol_ok = vix_data.get("data_ok") is True
    vol_source = vix_data.get("volatility_source") or vix_data.get("source")
    is_proxy = vol_ok and vol_source == "spy_realized_vol_proxy"
    standard_gates_config = _degraded_standard_gates_enabled(mode_cfg)
    normal_min_confidence = float(signal_cfg.get("min_buy_confidence", 40))
    degraded_min_confidence = float(mode_cfg.get("degraded_min_confidence", normal_min_confidence))
    # DELIBERATE (pinned by test_missing_spy_is_neutral_warning_not_degraded and its
    # volatility variant): missing SPY/vol data is recorded as a *warning*, never a
    # *block*, so `blocks` stays empty and mode falls through to NORMAL_MODE rather than
    # DEGRADED_MODE — the buy-lean bot keeps trading through data gaps. Two known
    # consequences are left as-is ON PURPOSE (do not "fix" without updating those tests):
    #   1. DEGRADED_MODE is unreachable in the live path — its whole engine is dead code.
    #   2. On PA the healthy mode is PROXY_MODE (size x0.85); losing SPY/vol data flips to
    #      NORMAL_MODE (size x1.0), i.e. degraded data yields LARGER size + no vol gate.
    blocks = []
    warnings = []
    if not spy_ok:
        warnings.append("SPY_DATA_MISSING")
    if not vol_ok:
        warnings.append("VOLATILITY_DATA_MISSING")

    if spy_ok and vol_ok and is_proxy and mode_cfg.get("allow_proxy_mode", True):
        mode = "PROXY_MODE"
        size_mult = float(mode_cfg.get("proxy_size_mult", 0.85))
        reason = "real VIX unavailable; using SPY realized-vol proxy"
    elif not blocks:
        mode = "NORMAL_MODE"
        size_mult = float(mode_cfg.get("normal_size_mult", 1.0))
        reason = None
    elif blocks and mode_cfg.get("allow_degraded_paper_trading", True):
        mode = "DEGRADED_MODE"
        size_mult = 1.0 if standard_gates_config else float(mode_cfg.get("degraded_size_mult", 0.90))
        reason = ",".join(blocks)
    else:
        mode = "DEGRADED_MODE_DISABLED" if blocks else "DATA_HEALTH_BLOCKED"
        size_mult = 1.0
        reason = ",".join(blocks) if blocks else None

    degraded = mode == "DEGRADED_MODE"
    degraded_standard_gates_active = bool(degraded and standard_gates_config)
    degraded_gate_policy = None
    if degraded:
        degraded_gate_policy = (
            "standard_gates_for_testing"
            if degraded_standard_gates_active
            else "restricted_degraded_gates"
        )
    effective_min_buy_confidence = (
        normal_min_confidence
        if not degraded or degraded_standard_gates_active
        else degraded_min_confidence
    )
    normal_ev_gates_required = (
        True
        if not degraded or degraded_standard_gates_active
        else bool(mode_cfg.get("degraded_require_normal_ev_gates", True))
    )
    normal_risk_caps_required = (
        True
        if not degraded or degraded_standard_gates_active
        else bool(mode_cfg.get("degraded_require_normal_risk_caps", True))
    )
    fresh_quote_required = (
        True
        if not degraded or degraded_standard_gates_active
        else bool(mode_cfg.get("degraded_require_fresh_quote", True))
    )
    proxy = mode == "PROXY_MODE"
    normal = mode == "NORMAL_MODE"
    return {
        "trading_mode": mode,
        "normal_mode_active": normal,
        "proxy_mode_active": proxy,
        "degraded_mode_active": degraded,
        "degraded_mode_reason": reason if degraded else None,
        "degraded_use_standard_gates_for_testing": standard_gates_config,
        "degraded_standard_gates_active": degraded_standard_gates_active,
        "degraded_gate_policy": degraded_gate_policy,
        "effective_size_mult": size_mult,
        "effective_min_buy_confidence": effective_min_buy_confidence,
        "normal_ev_gates_required": normal_ev_gates_required,
        "normal_risk_caps_required": normal_risk_caps_required,
        "fresh_quote_required": fresh_quote_required,
        "data_health_blocks": blocks,
        "data_health_warnings": warnings,
        "allow_buys": normal or proxy or degraded,
        "mode_size_mult": size_mult,
        "mode_size_reason": mode if size_mult < 1.0 else None,
    }


def _provider_health_summary(snapshot):
    snap = snapshot or {}
    if any((v or {}).get("rate_limit_recent") for v in snap.values() if isinstance(v, dict)):
        return "rate_limited"
    healthy_statuses = {
        "ok",
        "healthy",
        "skipped_on_pythonanywhere",
        "skipped_missing_key",
        "skipped_by_global_rate_limit",
        "skipped_by_circuit",
    }
    if any((v or {}).get("status") not in healthy_statuses for v in snap.values() if isinstance(v, dict)):
        return "degraded"
    return "healthy"


def _apply_paper_loss_lockouts(b, diag, risk_cfg, equity, peak_equity, now):
    """Return True when new buys should be blocked by paper-only loss controls."""
    today_open = float(b.get("today_open_equity") or equity or 0.0)
    daily_return = ((float(equity or 0.0) - today_open) / today_open) if today_open else 0.0
    drawdown_return = ((float(equity or 0.0) - float(peak_equity or 0.0)) / float(peak_equity or 1.0)) if peak_equity else 0.0
    daily_warning = float(risk_cfg.get("daily_loss_warning_pct", -0.10))
    daily_limit = float(risk_cfg.get("daily_loss_hard_lockout_pct",
                                     risk_cfg.get("daily_loss_limit_pct", -0.15)))
    drawdown_warning = float(risk_cfg.get("drawdown_warning_pct", -0.20))
    drawdown_limit = float(risk_cfg.get("drawdown_hard_lockout_pct",
                                        risk_cfg.get("hard_drawdown_lockout_pct", -0.35)))
    hard_blocks = bool(risk_cfg.get("hard_lockouts_block_new_buys", True))
    diag["daily_pnl_pct"] = round(daily_return, 4)
    diag["drawdown_return_pct"] = round(drawdown_return, 4)
    diag["daily_loss_warning_pct"] = daily_warning
    diag["daily_loss_hard_lockout_pct"] = daily_limit
    diag["drawdown_warning_pct"] = drawdown_warning
    diag["drawdown_hard_lockout_pct"] = drawdown_limit
    warnings = diag.setdefault("loss_lockout_warnings", [])
    if daily_warning < 0 and daily_return <= daily_warning:
        warnings.append("DAILY_LOSS_WARNING")
    if drawdown_warning < 0 and drawdown_return <= drawdown_warning:
        warnings.append("DRAWDOWN_WARNING")
    diag["loss_lockout_warnings"] = list(dict.fromkeys(warnings))
    if hard_blocks and daily_limit < 0 and daily_return <= daily_limit:
        diag["paper_trading_locked"] = True
        diag["paper_lock_reason"] = "DAILY_LOSS_LIMIT"
        _log_bot_event(
            "DAILY_LOSS_LIMIT",
            equity=round(float(equity or 0.0), 2),
            today_open_equity=round(today_open, 2),
            daily_return=round(daily_return, 4),
            limit=daily_limit,
            ts=int(now),
        )
        return True
    if hard_blocks and drawdown_limit < 0 and drawdown_return <= drawdown_limit:
        diag["paper_trading_locked"] = True
        diag["paper_lock_reason"] = "DRAWDOWN_LOCKOUT"
        _log_bot_event(
            "DRAWDOWN_LOCKOUT",
            equity=round(float(equity or 0.0), 2),
            peak_equity=round(float(peak_equity or 0.0), 2),
            drawdown_return=round(drawdown_return, 4),
            limit=drawdown_limit,
            ts=int(now),
        )
        return True
    return False


def _raise_tick_timeout(b, diag):
    diag["partial_result"] = True
    diag["timeout_reason"] = "BOT_TICK_MAX_RUNTIME_SEC"
    diag["paper_trading_locked"] = True
    diag["paper_lock_reason"] = "BOT_TICK_TIMEOUT"
    _persist_no_buy_diag(b, diag, "partial_timeout")
    raise TimeoutError("partial_timeout")


def _raise_if_tick_deadline_exceeded(deadline_ts, b, diag):
    if deadline_ts is None or time.time() <= deadline_ts:
        return
    _raise_tick_timeout(b, diag)


def _call_with_deadline(fn, deadline_ts):
    if deadline_ts is None:
        return fn(), True
    remaining = deadline_ts - time.time()
    if remaining <= 0:
        return None, False
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        done, _pending = wait(
            {future},
            timeout=max(0.05, remaining),
            return_when=FIRST_COMPLETED,
        )
        if future not in done:
            return None, False
        return future.result(timeout=0), True
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def buyable_reason(t, s, recent_sells=None, regime_kind="neutral", holdings=None, cfg=None):
    recent_sells = recent_sells or {}
    holdings = holdings or {}
    signal_cfg = (cfg or {}).get("signal", {})
    risk_cfg = (cfg or {}).get("risk", {})
    catalyst_cfg = (cfg or {}).get("catalyst", {})
    hard_reject = float(signal_cfg.get(
        "near_zero_volume_ratio_hard_block",
        signal_cfg.get("volume_hard_reject_ratio", 0.10),
    ))
    adx_gate = float(signal_cfg.get("neutral_gate_adx_threshold", 20))
    quote_status = _quote_execution_status(s)
    if quote_status["blocker"]:
        return False, quote_status["blocker"]
    history_status = _history_execution_status(s.get("ctx") or {})
    if history_status["blocker"]:
        return False, history_status["blocker"].upper()
    if t in recent_sells:
        rec_l = s.get("rec") or {}
        display = rec_l.get("display_signal_label")
        conf = float(rec_l.get("confidence", 0) or 0)
        recent_reason = (recent_sells.get(t) or {}).get("reason", "")
        sell_ts = float((recent_sells.get(t) or {}).get("ts", 0) or 0)
        age_min = (time.time() - sell_ts) / 60.0 if sell_ts else 0.0
        reason_l = str(recent_reason or "").lower()
        if reason_l in {"loss", "stop_loss", "trend_failure"}:
            base_cd = float(risk_cfg.get("stop_loss_rebuy_cooldown_minutes", 30))
        elif reason_l in {"trail", "trailing_stop"}:
            base_cd = float(risk_cfg.get("trailing_stop_rebuy_cooldown_minutes", 15))
        elif reason_l in {"manual", "manual_exit"}:
            base_cd = float(risk_cfg.get("manual_sell_rebuy_cooldown_minutes", 10))
        else:
            base_cd = float(risk_cfg.get("take_profit_rebuy_cooldown_minutes", 5))
        bypass_after = None
        if rec_l.get("cls") == "strong-buy" or display == "STRONG_BUY_CANDIDATE":
            bypass_after = float(risk_cfg.get("strong_buy_cooldown_bypass_after_minutes", 5))
        elif display == "BUY_CANDIDATE":
            bypass_after = float(risk_cfg.get("buy_candidate_cooldown_bypass_after_minutes", 10))
        elif display == "BULLISH_LEAN":
            bypass_after = float(risk_cfg.get("bullish_lean_cooldown_bypass_after_minutes", 15))
        elif display == "HOLD" and risk_cfg.get("high_confidence_hold_cooldown_bypass_enabled", False):
            bypass_after = base_cd
        allow_bypass = bypass_after is not None and age_min >= bypass_after
        if conf <= 0:
            allow_bypass = False
        if not allow_bypass:
            if age_min < base_cd:
                return False, f"RECENT_SELL_COOLDOWN:{recent_reason or 'unknown'}"
    ctx_local = s.get("ctx") or {}
    adv = float(ctx_local.get("avg_dollar_vol_20d", 0) or 0)
    if 0 < adv < float(signal_cfg.get("min_avg_dollar_volume_hard_block", 250_000)):
        return False, "LOW_LIQUIDITY_HARD_BLOCK"
    atr_pct = float(ctx_local.get("atr_pct", 0) or 0)
    if (bool(signal_cfg.get("atr_hard_block_enabled", True))
            and atr_pct > float(signal_cfg.get("extreme_atr_hard_block_pct", 20.0))):
        return False, "ATR_TOO_HIGH"
    if regime_kind == "bear":
        if not (ctx_local.get("rsi", 100) < 35 or ctx_local.get("is_dip")):
            return False, "BEAR_GATE_REQUIRES_DIP_OR_RSI_LT_35"
    if (regime_kind in ("neutral", "degraded_neutral") and ctx_local.get("adx", 100) < adx_gate
            and not ctx_local.get("is_dip")
            and float((s.get("rec") or {}).get("confidence", 0) or 0) < 60):
        if t not in holdings:
            return False, "NEUTRAL_ADX_BELOW_GATE_NON_DIP"
    catalyst = (s.get("rec") or {}).get("catalyst") or {}
    catalyst_type = catalyst.get("type")
    week_chg = ctx_local.get("week_chg_pct", 0) or 0
    dist_high = ctx_local.get("dist_from_high_pct", 0) or 0
    falling_threshold = float(catalyst_cfg.get("negative_catalyst_falling_threshold_pct", -5.0))
    if (catalyst_type in KNIFE_CATALYST_TYPES
            and bool(catalyst_cfg.get("negative_catalyst_hard_block_enabled", True))
            and week_chg <= falling_threshold):
        return False, f"NEGATIVE_CATALYST_FALLING:{catalyst_type}"
    if week_chg <= -5.0 and dist_high <= -10.0:
        return False, "FALLING_KNIFE_WEEK_LT_-5_DIST_HIGH_LT_-10"
    vol_regime = classify_vol_regime(ctx_local)
    if vol_regime == "explosive" and ctx_local.get("rsi", 50) < 35:
        return False, "EXPLOSIVE_OVERSOLD_BLOCK"
    if not ctx_local.get("is_dip") and float(ctx_local.get("vol_ratio", 1.0) or 1.0) < hard_reject:
        return False, "VERY_LOW_VOLUME_CONFIRMATION"
    return True, "buyable"


def _pa_stage_tickers(tickers, holdings, b):
    """PA free-tier: always refresh holdings, rotate watchlist across pings."""
    if not PYTHONANYWHERE_MODE:
        return list(set(tickers) | set(holdings))
    holdings_ordered = [t for t in holdings if t]
    watch = [
        t for t in tickers
        if t not in holdings_ordered and t not in BENCHMARK_ONLY_TICKERS
    ]
    if not watch:
        return holdings_ordered
    room = max(0, PA_TICKERS_PER_BOT_RUN - len(holdings_ordered))
    room = max(1, room) if not holdings_ordered else room
    idx = int(b.get("pa_rotation_index", 0) or 0) % len(watch)
    rotated = watch[idx:] + watch[:idx]
    selected_watch = rotated[:room]
    b["pa_rotation_index"] = (idx + len(selected_watch)) % len(watch)
    b["pa_stage_status"] = {
        "mode": "pythonanywhere",
        "selected": holdings_ordered + selected_watch,
        "watchlist_size": len(tickers),
        "holdings_size": len(holdings_ordered),
        "batch_size": PA_TICKERS_PER_BOT_RUN,
        "next_index": b["pa_rotation_index"],
        "ts": int(time.time()),
    }
    return holdings_ordered + selected_watch

# ── Bot state runtime mirror ────────────────────────────────────────────────
def _set_running(flag):
    global _bot_running
    _bot_running = flag


def record_equity_snapshot(b, total_equity):
    """Append [ts, equity, change_pct, holdings] to bot's equity history.

    Round-3 Bug Fix #1: is_market_open guard so chart stops cleanly after close.
    Round-4 Bug Fix #1: if total_equity is suspiciously low (e.g. an API failure
    fed $0 prices in, dropping total below 50% of cash-only), skip the snapshot
    so the chart never draws a zero-spike. Caller is responsible for using
    last-known prices via bot_state(), but this is a belt-and-suspenders guard.
    """
    if not is_market_open():
        return
    eq_hist = b.setdefault("equity_history", [])
    ts = int(time.time())
    if eq_hist and ts - eq_hist[-1][0] < 60:
        return
    # Sanity guard: equity should be >= cash (positions are non-negative).
    # If equity < cash * 0.5, something's wrong (likely $0 prices). Skip.
    if total_equity < (b.get("cash", 0) * 0.5):
        try: print(f"[record_equity_snapshot] suspicious total ${total_equity:.2f} "
                   f"vs cash ${b.get('cash', 0):.2f} — skipping (likely API failure)")
        except Exception: pass
        return
    starting = b.get("starting", 10000) or 10000
    change_pct = round((total_equity - starting) / starting * 100, 3)
    holdings_snap = []
    any_stale = False
    for sym, h in b.get("holdings", {}).items():
        try:
            q = get_quote(sym) or {}
            pr = q.get("price") or 0
            # Round-4 Bug Fix #1: if quote is stale (no live data), use avg_cost
            # so the snapshot doesn't capture an artificial $0 spike.
            if pr <= 0 or q.get("stale"):
                pr = h.get("avg_cost", 0)
                any_stale = True   # this point is valued off avg_cost, not a live price
            holdings_snap.append([sym, round(h["shares"], 4), round(pr, 2)])
        except Exception:
            continue
    force_full = bool(b.pop("_force_full_snapshot", False))
    if PYTHONANYWHERE_MODE and holdings_snap and not force_full and (len(eq_hist) % 10 != 0):
        holdings_snap = []
    # Round-8 Bug #3: mark snapshots valued off any stale price. A stale point sits
    # at avg_cost (no movement), so the chart flatlines and HIDES real drawdown during
    # an outage. The 5th element lets the chart/stats distinguish honest points from
    # outage-frozen ones (consumers that index p[0..3] are unaffected).
    eq_hist.append([ts, round(total_equity, 2), change_pct, holdings_snap, bool(any_stale)])
    b["equity_history"] = eq_hist[-EQUITY_HISTORY_MAX:]


# ── Trade history records ──────────────────────────────────────────────────
def _record_trade(b, action, t, sh, pr, rec, arts, why, pnl_usd=None):
    """Append a trade (BUY/SELL) to history with reasoning + supporting article.
    Round-5 Bug #3: pnl_usd = realized profit on SELL (None for BUY) so the
    history table can show profit instead of raw cash flow."""
    sup = None
    if arts:
        want_pos = (action == "BUY")
        sup = next((a for a in arts if (a["score"] > 0) == want_pos and a["score"] != 0), arts[0])
    et, sgt = _fmt_times()
    display_signal = rec.get("display_signal_label") or rec.get("signal")
    raw_signal = rec.get("raw_signal_label") or rec.get("signal")
    b["history"].insert(0, {
        "action": action, "ticker": t, "shares": sh, "price": pr,
        "total": sh * pr,
        "pnl_usd": pnl_usd,   # realized profit (SELL only); None for BUY
        "time": datetime.now().strftime("%m/%d %H:%M"),
        "time_et": et, "time_sgt": sgt, "ts": int(time.time()),
        "signal": display_signal,
        "raw_signal": raw_signal,
        "display_signal": display_signal,
        "confidence": rec["confidence"],
        "reason": why or (rec["reasons"][0] if rec["reasons"] else ""),
        "news_title": sup["title"] if sup else "",
        "news_link":  sup.get("link", "") if sup else "",
    })
    b["total_trades"] = b.get("total_trades", 0) + 1


def _record_hold(b, reason, sigs):
    """HOLD entry — emitted only when truly idle (no trades, no skips)."""
    et, sgt = _fmt_times()

    def _display_rec(rec):
        rec = rec or {}
        return rec.get("display_signal_label", rec.get("signal"))

    sig_summary = ", ".join(
        f"{t}:{_display_rec(s.get('rec'))}({(s.get('rec') or {}).get('confidence')}%)"
        for t, s in sorted(sigs.items())
    )
    best = max(sigs.items(), key=lambda x: (x[1].get("rec") or {}).get("confidence", 0)) if sigs else None
    best_rec = (best[1].get("rec") or {}) if best else {}
    best_str = (f"Top signal: {best[0]} {_display_rec(best_rec)} @ {best_rec.get('confidence')}%"
                if best else "")
    b["history"].insert(0, {
        "action": "HOLD",
        "ticker": "—", "shares": 0, "price": 0, "total": 0,
        "time": datetime.now().strftime("%m/%d %H:%M"),
        "time_et": et, "time_sgt": sgt, "ts": int(time.time()),
        "signal": "HOLD", "confidence": 0,
        "reason": reason + (f" · {best_str}" if best_str else ""),
        "signals": sig_summary,
        "news_title": "", "news_link": "",
    })


def _record_skip(b, ticker, reason, signal, confidence, *,
                 display_signal=None, original_reason=None, skip_stage=None,
                 rank_reason_code=None, gross_edge_pct=None, net_edge_pct=None,
                 required_edge_pct=None, friction_pct=None, target_notional=None,
                 risk_pct=None):
    """SKIP entry — a buy attempt that was prepared but failed sizing checks."""
    et, sgt = _fmt_times()
    shown_signal = display_signal or signal or "UNKNOWN"
    b["history"].insert(0, {
        "action": "SKIP",
        "ticker": ticker, "shares": 0, "price": 0, "total": 0,
        "time": datetime.now().strftime("%m/%d %H:%M"),
        "time_et": et, "time_sgt": sgt, "ts": int(time.time()),
        "signal": shown_signal,
        "raw_signal": signal,
        "display_signal": display_signal,
        "confidence": confidence,
        "reason": reason,
        "original_reason": original_reason,
        "skip_stage": skip_stage,
        "rank_reason_code": rank_reason_code,
        "gross_edge_pct": gross_edge_pct,
        "net_edge_pct": net_edge_pct,
        "required_edge_pct": required_edge_pct,
        "friction_pct": friction_pct,
        "target_notional": target_notional,
        "risk_pct": risk_pct,
        "signals": "",
        "news_title": "", "news_link": "",
    })


# ── Main bot loop ───────────────────────────────────────────────────────────
def _suggestion_ui_row(c, run_id, ts):
    rec = c.get("rec") or {}
    reasons = c.get("suggestion_reasons") or rec.get("reasons", [])[:3]
    return {
        "run_id": run_id,
        "ts": int(ts),
        "ticker": c.get("ticker"),
        "source": c.get("source"),
        "price": round(float(c.get("price") or 0.0), 2),
        "signal": rec.get("signal"),
        "confidence": rec.get("confidence"),
        "cluster": c.get("cluster"),
        "sector": c.get("sector"),
        "corr_group": c.get("corr_group"),
        "gross_edge_pct": c.get("gross_edge_pct"),
        "net_edge_pct": c.get("net_edge_pct"),
        "suggestion_score": c.get("suggestion_score"),
        "suggestion_score_pct": int(round(float(c.get("suggestion_score") or 0.0) * 100)),
        "show_reason": c.get("show_reason"),
        "suggestion_reasons": reasons,
        "feedback_bucket": c.get("feedback_bucket"),
        "edge_source": c.get("edge_source"),
        "edge_samples": c.get("edge_samples"),
    }


def build_extra_ticker_suggestions(b: dict, sigs: dict, ranked_candidates: list[dict],
                                   regime: dict, now_ts: int,
                                   config: dict = DEFAULT_CONFIG,
                                   recent_sells: dict | None = None) -> list[dict]:
    """Build advisory extras from already-ranked candidates; never affects buys."""
    cfg = _suggestion_cfg(config or DEFAULT_CONFIG)
    if not cfg.get("enabled", True):
        b["extra_ticker_suggestions"] = []
        return []
    holdings = b.get("holdings", {})
    unheld = [
        c for c in (ranked_candidates or [])
        if c.get("ticker") and c.get("ticker") not in holdings
    ]
    if not unheld:
        b["extra_ticker_suggestions"] = []
        return []
    feedback_stats = {}
    recent_suggestions = {}
    try:
        cached_feedback = cache_get("suggestion_feedback_stats", max_age=300)
        if cached_feedback is None:
            feedback_stats = load_feedback_stats(
                SUGGESTION_DB_FILE,
                lookback_days=int(cfg.get("suggestion_log_retention_days", 90)),
            )
            cache_set("suggestion_feedback_stats", feedback_stats)
        else:
            feedback_stats = cached_feedback
        recent_suggestions = load_recent_suggestions(
            SUGGESTION_DB_FILE,
            lookback_sec=int(cfg.get("ticker_cooldown_sec", SUGGESTION_RECENT_TICKER_COOLDOWN_SEC)),
        )
    except Exception as e:
        try: print(f"[suggestion_store] load failed: {e}")
        except Exception: pass
    loss_cooldowns = {}
    loss_window = int(cfg.get("loss_cooldown_sec", 14 * 86400))
    for outcome in b.get("trade_outcomes", []):
        ts = int(outcome.get("ts", 0) or 0)
        if outcome.get("pnl_pct", 0) <= 0 and int(now_ts) - ts < loss_window:
            tk = str(outcome.get("ticker") or "").upper()
            if tk:
                loss_cooldowns[tk] = max(ts, loss_cooldowns.get(tk, 0))
    try:
        selected = rank_suggestion_candidates(
            unheld,
            holdings=holdings,
            top_sectors=regime.get("top_sectors", []),
            feedback_stats=feedback_stats,
            recent_suggestions=recent_suggestions,
            loss_cooldowns=loss_cooldowns,
            recent_sells=recent_sells or {},
            now_ts=int(now_ts),
            config=config,
        )[:int(cfg.get("max_extra_tickers", SUGGESTION_MAX_EXTRA_TICKERS))]
    except Exception as e:
        try: print(f"[suggestions] scoring failed: {type(e).__name__}: {e}")
        except Exception: pass
        selected = []
    run_id = f"{int(now_ts)}-{uuid.uuid4().hex[:8]}"
    rows = [_suggestion_ui_row(c, run_id, now_ts) for c in selected]
    b["extra_ticker_suggestions"] = rows
    b["last_suggestion_run_id"] = run_id
    try:
        log_suggestion_run(SUGGESTION_DB_FILE, run_id, int(now_ts), rows, rows)
        prune_suggestion_store(
            SUGGESTION_DB_FILE,
            log_retention_days=int(cfg.get("suggestion_log_retention_days", 90)),
            feedback_retention_days=int(cfg.get("feedback_retention_days", 180)),
        )
    except Exception as e:
        try: print(f"[suggestion_store] log failed: {e}")
        except Exception: pass
    return rows


def _ensure_v1_state(b):
    b.setdefault("candidate_observations", [])
    b.setdefault("edge_stats", {})
    b.setdefault("last_candidate_rankings", [])
    b.setdefault("extra_ticker_suggestions", [])
    ensure_attribution_state(b)


def _state_guard(b):
    if not isinstance(b, dict):
        return b
    pa = bool(PYTHONANYWHERE_MODE)
    caps = {
        "candidate_observations": 300 if pa else 600,
        "attribution_events": 500 if pa else 800,
        "exit_attribution_events": 500 if pa else 800,
        "last_portfolio_variance_checks": 50 if pa else 100,
    }
    for key, cap in caps.items():
        val = b.get(key)
        if isinstance(val, list) and len(val) > cap:
            b[key] = val[:cap]
    eq_cap = EQUITY_HISTORY_MAX
    if isinstance(b.get("equity_history"), list) and len(b["equity_history"]) > eq_cap:
        b["equity_history"] = b["equity_history"][-eq_cap:]
    return b


def _memory_guard(b):
    """No-dependency PA memory guard for large persisted state lists."""
    if not isinstance(b, dict) or not PYTHONANYWHERE_MODE:
        return b
    limits = {
        "equity_history": EQUITY_HISTORY_MAX,
        "candidate_observations": 300,
        "attribution_events": 500,
        "exit_attribution_events": 500,
        "last_candidate_rankings": 50,
        "last_portfolio_variance_checks": 50,
    }
    trimmed = False
    for key, cap in limits.items():
        val = b.get(key)
        if isinstance(val, list) and len(val) > cap:
            b[key] = val[-cap:] if key == "equity_history" else val[:cap]
            trimmed = True
    if trimmed:
        try:
            gc.collect()
        except Exception:
            pass
        b["last_memory_guard_ts"] = int(time.time())
    return b


def _persist_no_buy_diag(b, diag, blocker=None, traded=False):
    if diag.get("tick_runtime_seconds") is None:
        try:
            diag["tick_runtime_seconds"] = round(time.time() - float(diag.get("ts") or time.time()), 3)
        except Exception:
            diag["tick_runtime_seconds"] = None
    if blocker:
        diag["main_blocker"] = blocker
    elif traded:
        diag["main_blocker"] = "trade_executed"
    else:
        _set_main_blocker(diag)
    diag["traded"] = bool(traded)
    _BOT_STATUS.update({
        "last_tick_status": diag.get("main_blocker"),
        "last_tick_time": diag.get("ts"),
        "tick_runtime_seconds": diag.get("tick_runtime_seconds"),
    })
    b["last_no_buy_diagnostics"] = diag
    _log_tick(diag)
    try:
        b["last_cache_prune"] = prune_cache_dir(max_files=300, max_age_sec=7 * 86400)
    except Exception:
        b["last_cache_prune"] = {"removed": 0, "kept": 0}
    _memory_guard(b)
    _state_guard(b)
    save_bot(b)
    return b


def _record_candidate_observation(b, cand, decision, reason, now, regime_kind):
    """Persist one candidate for forward-return attribution."""
    if decision != "executed":
        return False
    event = record_entry_event(b, cand, decision, reason, ts=now, regime=regime_kind)
    return bool(event)


def _update_candidate_observations(b, sigs, now):
    """Fill entry/exit attribution forward outcomes using fresh prices only."""
    _ensure_v1_state(b)

    def price_lookup(ticker):
        snap = sigs.get(ticker) if isinstance(sigs, dict) else None
        if snap and snap.get("price", 0) > 0 and not snap.get("stale"):
            return snap["price"]
        q = get_quote(ticker) or {}
        if q.get("stale") or q.get("price", 0) <= 0:
            return None
        return q["price"]

    updated = update_forward_outcomes(b, price_lookup, ts=now,
                                      benchmark_lookup=price_lookup)
    updated += update_exit_post_outcomes(b, price_lookup, ts=now)
    return updated


def _generic_exit_shadow(h, pr, ctx, rec, stop_loss_pct, trail_stop_pct,
                         aging_hours, held_secs, min_holding_sec,
                         eff_atr_for_stop, intra_atr=0, vwma_dist=0):
    avg = h.get("avg_cost", 0) or 0
    peak = max(h.get("peak", avg), pr)
    pnl_pct = (pr - avg) / avg * 100 if avg else 0.0
    trail_pct = (pr - peak) / peak * 100 if peak else 0.0
    atr_pct = (ctx or {}).get("atr_pct", 0) or 0
    held_hours = held_secs / 3600.0
    peak_pnl_pct = max(h.get("peak_pnl_pct", 0), pnl_pct)
    dynamic_stop = dynamic_stop_pct(eff_atr_for_stop, stop_loss_pct)
    dyn_trail_pct, _ = dynamic_trail_width(intra_atr, atr_pct, trail_stop_pct)
    if pnl_pct >= 10 and held_hours > 24:
        dyn_trail_pct = max(1.5, dyn_trail_pct - 0.5 * ((held_hours - 24) / 24))
    trail_triggered = (pnl_pct > 2 and trail_pct <= -dyn_trail_pct)
    trail_active = trail_triggered and not ((vwma_dist > 1.0) and (trail_pct > -3.0))
    rt_cost_pct = round_trip_cost_pct(h.get("shares", 0) * pr, COMMISSION_PER_TRADE)
    ratchet_lock = breakeven_lock_pct(peak_pnl_pct, rt_cost_pct)
    if ratchet_lock is not None:
        effective_stop = max(dynamic_stop, ratchet_lock)
    elif held_hours > 2.0 and pnl_pct < 0 and peak_pnl_pct < 1.0:
        effective_stop = dynamic_stop * 0.5
    else:
        effective_stop = dynamic_stop
    cls = (rec or {}).get("cls")
    reason = ""
    if pnl_pct <= effective_stop:
        reason = "ratchet" if effective_stop > 0 else "loss"
    elif held_secs < min_holding_sec:
        reason = ""
    elif trail_active:
        reason = "trail"
    elif cls in ("sell", "strong-sell") and pnl_pct < 0:
        reason = "loss"
    elif cls == "strong-sell" and pnl_pct >= 0:
        reason = "signal_flip_profit"
    else:
        cur_e = (ctx or {}).get("current", 0) or 0
        ma30_e = (ctx or {}).get("ma30", 0) or 0
        if ma30_e > 0 and cur_e > 0 and pnl_pct < 0 and cur_e < ma30_e * 0.98:
            reason = "trend_failure"
        elif held_hours > aging_hours and abs(pnl_pct) < 1.0 and h.get("shares", 0) * pr >= 150:
            reason = "aging"
    return {
        "would_exit": bool(reason),
        "exit_reason": reason,
        "pnl_pct": round(pnl_pct, 4),
        "held_hours": round(held_hours, 2),
        "price": round(pr, 4),
    }


def _carried_peak_pnl_pct(prior_peak, price, new_avg_cost):
    """B5: initialize a holding's ratchet peak-P&L across pyramid adds.

    Carries the highest price seen (prior_peak vs this fill's price) but expresses it
    against the NEW blended avg cost, so a stale high peak from the old (lower) basis
    can't set a lock above the current P&L and trigger an immediate ratchet exit right
    after adding. New position (no prior peak): peak≈price≈new_avg_cost → ~0.
    """
    if not new_avg_cost:
        return 0.0
    peak_price = max(prior_peak if prior_peak else price, price)
    return round((peak_price - new_avg_cost) / new_avg_cost * 100, 3)


def run_bot(force=False, user_forced=False, max_runtime_sec=None):
    """Bot decision pass. Thread-safe via _bot_run_lock — concurrent triggers
    are dropped rather than queued. Returns (b, traded, last_action).

    A3: `force` = bypass the interval cooldown (used by the after-close auto-resume and
    the scheduler). `user_forced` = a human clicked "run now" on /bot/run — only THIS
    relaxes the 09:45–15:30 calm-window gate and the "buy at least one name" fallback.
    The auto-resume must NOT churn a marginal buy in the volatile first minutes of the
    session, which is what conflating the two flags used to cause."""
    if not BOT_ENABLED:
        b = load_bot()
        _BOT_STATUS.update({"last_run_ts": int(time.time()), "last_action": "bot_disabled",
                            "last_traded": False})
        return b, False, "bot_disabled"
    if not _bot_run_lock.acquire(blocking=False):
        b = load_bot()
        return b, False, "already_running"
    _set_running(True)
    try:
        try:
            with acquire_bot_file_lock(block=False):
                return _run_bot_locked(
                    force=force,
                    user_forced=user_forced,
                    max_runtime_sec=max_runtime_sec,
                )
        except TimeoutError as exc:
            b = load_bot()
            if str(exc) == "partial_timeout":
                return b, False, "partial_timeout"
            return b, False, "already_running"
    finally:
        _set_running(False)
        _bot_run_lock.release()


def _run_bot_locked(force=False, user_forced=False, max_runtime_sec=None):
    b = load_bot()
    _ensure_v1_state(b)
    _memory_guard(b)
    now = time.time()
    cfg = active_config()
    signal_cfg = (cfg or {}).get("signal", {})
    mode_cfg = (cfg or {}).get("market_data_modes", {})
    risk_cfg = (cfg or {}).get("risk", {}) if isinstance(cfg, dict) else {}
    scan_cfg = (cfg or {}).get("scan", {}) if isinstance(cfg, dict) else {}
    history_cfg = (cfg or {}).get("history", {}) if isinstance(cfg, dict) else {}
    min_position_usd = float(risk_cfg.get("min_trade_size", MIN_POSITION_USD))
    min_buy_confidence = float(signal_cfg.get("min_buy_confidence", 40))
    degraded_min_confidence = float(
        mode_cfg.get("degraded_min_confidence", min_buy_confidence)
    )
    degraded_standard_gates_config = _degraded_standard_gates_enabled(mode_cfg)
    cfg_hash = config_hash(cfg)
    no_buy_diag = _new_no_buy_diag(now, b, force=force, user_forced=user_forced)
    no_buy_diag["min_buy_confidence"] = min_buy_confidence
    no_buy_diag["degraded_min_confidence"] = degraded_min_confidence
    no_buy_diag["degraded_use_standard_gates_for_testing"] = degraded_standard_gates_config
    no_buy_diag["effective_min_buy_confidence"] = min_buy_confidence
    no_buy_diag["effective_size_mult"] = 1.0
    no_buy_diag["normal_ev_gates_required"] = True
    no_buy_diag["normal_risk_caps_required"] = True
    no_buy_diag["fresh_quote_required"] = True
    no_buy_diag["min_trade_size_effective"] = min_position_usd
    no_buy_diag["max_history_fetches_per_tick"] = int(
        history_cfg.get("max_history_fetches_per_tick", 1)
    )
    deadline_ts = (now + float(max_runtime_sec)) if max_runtime_sec else None
    collapse_diag = []
    stale_scan_skipped = False

    if b.get("stopped"):
        _persist_no_buy_diag(b, no_buy_diag, "user_stopped")
        _BOT_STATUS.update({"last_run_ts": int(now), "last_action": "user_stopped",
                            "last_traded": False})
        return b, False, "user_stopped"

    if not is_market_open():
        no_buy_diag["market_open"] = False
        if not b.get("pending_run"):
            b["pending_run"] = True
        _persist_no_buy_diag(b, no_buy_diag, "market_closed")
        _BOT_STATUS.update({"last_run_ts": int(now), "last_action": "market_closed_autoqueued",
                            "last_traded": False})
        return b, False, "market_closed_autoqueued"

    if b.get("pending_run"):
        b["pending_run"] = False
        _persist_no_buy_diag(b, no_buy_diag, "pending_run_resumed")
        force = True
    if not force and (now - b.get("last_trade", 0) < BOT_INTERVAL):
        _persist_no_buy_diag(b, no_buy_diag, "interval_cooldown")
        return b, False, "interval_cooldown"

    # Recently-sold tracking with reason-aware TTL
    raw_sells = b.setdefault("recent_sells", {})
    cleaned = {}
    for k, v in raw_sells.items():
        if isinstance(v, dict):
            ts, reason = v.get("ts", 0), v.get("reason", "")
        else:
            ts, reason = v, ""
        ttl = 2 * 3600 if reason == "loss" else 30 * 60
        if now - ts < ttl:
            cleaned[k] = {"ts": ts, "reason": reason}
    recent_sells = cleaned
    b["recent_sells"] = recent_sells

    def _mark_sold(tk, ts, reason):
        recent_sells[tk] = {"ts": ts, "reason": reason}

    tickers = load_tickers()
    all_to_check = _pa_stage_tickers(tickers, list(b["holdings"].keys()), b)
    no_buy_diag["checked_tickers"] = list(all_to_check)
    no_buy_diag["pa_stage_status"] = b.get("pa_stage_status")
    _raise_if_tick_deadline_exceeded(deadline_ts, b, no_buy_diag)
    regime, ok = _call_with_deadline(lambda: get_market_regime(cfg), deadline_ts)
    if not ok:
        no_buy_diag["timeout_stage"] = "market_regime"
        _raise_tick_timeout(b, no_buy_diag)
    if regime.get("regime_v3_raw") not in (None, "fallback"):
        regime_state = b.setdefault("regime_v3_state", {})
        confirmed_v3 = apply_confirmation(
            regime_state,
            regime.get("regime_v3_raw"),
            regime.get("regime_v3_reason", ""),
            ts=now,
        )
        regime["regime_v3"] = confirmed_v3
        regime["regime_v3_effective"] = confirmed_v3
        regime["legacy_kind"] = regime.get("legacy_kind") or regime.get("regime_effective")
    b["last_regime_v3"] = {
        "config_hash": cfg_hash,
        "regime_v3": regime.get("regime_v3"),
        "regime_v3_raw": regime.get("regime_v3_raw"),
        "reason": regime.get("regime_v3_reason"),
        "actual_breadth_count": regime.get("actual_breadth_count"),
        "missing_breadth_count": regime.get("missing_breadth_count"),
        "min_effective_breadth_count": regime.get("min_effective_breadth_count"),
        "top_sectors": regime.get("top_sectors", []),
        "fallback": regime.get("regime_v3_fallback"),
        "source": regime.get("regime_v3_source"),
        "spy_data_ok": regime.get("spy_data_ok"),
        "regime_data_status": regime.get("regime_data_status"),
        "regime_data_fallback": regime.get("regime_data_fallback"),
        "spy_rows": regime.get("spy_rows"),
        "spy_last_date": regime.get("spy_last_date"),
        "spy_last_close": regime.get("spy_last_close"),
        "spy_base_22_close": regime.get("spy_base_22_close"),
        "spy_mom_label": regime.get("spy_mom_label"),
        "regime_data_source": regime.get("regime_data_source"),
        "regime_data_warnings": regime.get("regime_data_warnings", []),
        "stale_daily_cache_age_hours": regime.get("stale_daily_cache_age_hours"),
    }
    macro_regime_kind = regime.get("regime", "neutral")
    regime_kind = regime.get("regime_effective", macro_regime_kind)
    no_buy_diag["spy_data_ok"] = regime.get("spy_data_ok")
    no_buy_diag["regime_data_status"] = regime.get("regime_data_status")
    no_buy_diag["regime_data_fallback"] = regime.get("regime_data_fallback")
    no_buy_diag["regime_data_source"] = regime.get("regime_data_source")
    no_buy_diag["regime_data_error"] = regime.get("regime_data_error")
    no_buy_diag["regime_data_warnings"] = regime.get("regime_data_warnings", [])
    no_buy_diag["data_health_warnings"] = regime.get("data_health_warnings", [])
    no_buy_diag["stale_daily_cache_age_hours"] = regime.get("stale_daily_cache_age_hours")
    no_buy_diag["stale_daily_cache_age_sec"] = regime.get("stale_daily_cache_age_sec")
    no_buy_diag["spy_data_source"] = regime.get("spy_data_source") or regime.get("regime_data_source")
    no_buy_diag["spy_data_error"] = regime.get("spy_data_error") or regime.get("regime_data_error")
    no_buy_diag["spy_rows"] = regime.get("spy_rows")
    no_buy_diag["spy_bar_count"] = regime.get("spy_rows")
    no_buy_diag["spy_last_date"] = regime.get("spy_last_date")
    no_buy_diag["spy_mom_label"] = regime.get("spy_mom_label")
    sigs = {}

    def _fetch_one(t):
        # Round-7 Bug 5: wrap in try/except — ThreadPoolExecutor.map re-raises a
        # worker exception on the calling thread, which would abort the whole tick
        # (no save, no buy/sell) if a single ticker errors (e.g. Finnhub 429).
        # Return a safe stale payload so the cycle continues without this ticker.
        try:
            arts, sent = get_news(t)
            ctx     = get_history(t)
            intra   = get_intraday_context(t)
            earn    = get_earnings_soon(t)
            analyst = get_analyst_rec(t)
            insider = get_insider_sentiment(t)
            rec = get_recommendation(sent, ctx, regime=regime, earnings=earn,
                                      analyst=analyst, insider=insider,
                                      news_articles=arts, config=cfg)
            catalyst = classify_catalyst(arts, earn, analyst, insider, ctx,
                                         config=cfg)
            rec = dict(rec)
            rec["catalyst"] = catalyst
            # Round-4 Bug Fix #1: capture stale flag so SELL/BUY halt on API failure.
            q = get_quote(t) or {}
            payload = {"rec": rec, "price": q.get("price", 0), "stale": bool(q.get("stale")),
                       "quote": q, "arts": arts, "ctx": ctx, "intra": intra,
                       "earn": earn, "analyst": analyst, "insider": insider}
            _finalize_signal_confidence(payload, cfg)
            return t, payload
        except Exception as e:
            try: print(f"[_fetch_one] {t}: {type(e).__name__}: {e}")
            except Exception: pass
            fallback_rec = dict(get_recommendation(0.0, {}, regime=regime, config=cfg))
            _annotate_signal(fallback_rec)
            return t, {"rec": fallback_rec,
                       "price": 0, "stale": True,
                       "quote": {"price": 0, "stale": True, "source": "missing"},
                       "arts": [], "ctx": {},
                       "intra": {}, "earn": {}, "analyst": {}, "insider": {}}

    fetch_workers = 2 if PYTHONANYWHERE_MODE else 8
    _raise_if_tick_deadline_exceeded(deadline_ts, b, no_buy_diag)
    fetch_list = list(all_to_check)
    set_history_fetch_budget(no_buy_diag.get("max_history_fetches_per_tick", 1))
    executor = ThreadPoolExecutor(max_workers=fetch_workers)
    futures = {executor.submit(_fetch_one, t): t for t in fetch_list}
    try:
        while futures:
            if deadline_ts is not None:
                remaining = deadline_ts - time.time()
                if remaining <= 0:
                    no_buy_diag["fetch_timeout_tickers"] = list(futures.values())[:10]
                    _raise_tick_timeout(b, no_buy_diag)
                timeout = max(0.05, min(1.0, remaining))
            else:
                timeout = 1.0
            done, _pending = wait(
                futures,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                if deadline_ts is not None and time.time() >= deadline_ts:
                    no_buy_diag["fetch_timeout_tickers"] = list(futures.values())[:10]
                    _raise_tick_timeout(b, no_buy_diag)
                continue
            for fut in done:
                expected_ticker = futures.pop(fut)
                try:
                    t, payload = fut.result(timeout=0)
                except Exception as e:
                    try:
                        print(f"[bot-fetch-timeout-safe] {expected_ticker}: {type(e).__name__}: {e}")
                    except Exception:
                        pass
                    fallback_rec = dict(get_recommendation(0.0, {}, regime=regime, config=cfg))
                    _annotate_signal(fallback_rec)
                    t, payload = expected_ticker, {
                        "rec": fallback_rec,
                        "price": 0,
                        "stale": True,
                        "quote": {"price": 0, "stale": True, "source": "missing"},
                        "arts": [],
                        "ctx": {},
                        "intra": {},
                        "earn": {},
                        "analyst": {},
                        "insider": {},
                    }
                sigs[t] = payload
                _raise_if_tick_deadline_exceeded(deadline_ts, b, no_buy_diag)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        set_history_fetch_budget(None)
    gc.collect()
    for payload in sigs.values():
        rec = _annotate_signal(payload.get("rec") or {})
        payload["rec"] = rec
        cls = (rec.get("cls") or "unknown")
        _bump(no_buy_diag["signal_counts"], cls)
        _bump(no_buy_diag["display_signal_counts"], rec.get("display_signal_label"))
        if cls in ("buy", "strong-buy"):
            no_buy_diag["raw_buy_count"] += 1
        if rec.get("display_signal_label") in ("BUY_CANDIDATE", "STRONG_BUY_CANDIDATE"):
            no_buy_diag["display_buy_candidate_count"] += 1
    for t in fetch_list:
        payload = sigs.get(t) or {}
        ctx = payload.get("ctx") or {}
        source = ctx.get("history_source") or ("recorded" if ctx.get("source") == "recorded" else "missing")
        bucket = _history_source_bucket(source if ctx else "missing")
        _bump(no_buy_diag["history_source_counts"], bucket)
        no_buy_diag["data_health_warnings"] = list(dict.fromkeys(
            list(no_buy_diag.get("data_health_warnings") or [])
            + list(ctx.get("history_warnings") or [])
        ))
        if (
            ctx.get("stale_daily_cache_age_hours") is not None
            and no_buy_diag.get("stale_daily_cache_age_hours") is None
        ):
            no_buy_diag["stale_daily_cache_age_hours"] = ctx.get("stale_daily_cache_age_hours")
            no_buy_diag["stale_daily_cache_age_sec"] = ctx.get("stale_daily_cache_age_sec")
        for entry in (ctx.get("provider_chain_debug") or []):
            if not isinstance(entry, dict):
                continue
            provider = str(entry.get("provider") or entry.get("source") or "")
            if provider != "fmp_daily":
                continue
            status = str(entry.get("status") or "")
            if status.startswith("skipped"):
                no_buy_diag["history_fmp_skipped_count"] += 1
                if status == "skipped_by_global_rate_limit":
                    no_buy_diag["history_fmp_global_circuit_skipped_count"] += 1
            else:
                no_buy_diag["history_fmp_attempted_count"] += 1
            if status == "rate_limited" or entry.get("rate_limited"):
                no_buy_diag["history_fmp_rate_limited_count"] += 1
        if bucket == "missing":
            no_buy_diag["history_missing_count"] += 1
            max_missing = int(
                history_cfg.get(
                    "top_missing_history_symbols_count",
                    DEFAULT_CONFIG.get("history", {}).get("top_missing_history_symbols_count", 3),
                )
            )
            if len(no_buy_diag.get("top_missing_history_symbols") or []) < max_missing:
                no_buy_diag.setdefault("top_missing_history_symbols", []).append(t)
        elif bucket == "fmp_daily":
            no_buy_diag["history_fmp_fallback_count"] += 1
        elif bucket == "stale_cache":
            no_buy_diag["history_stale_cache_count"] += 1
    stale_tickers = [tk for tk, payload in sigs.items() if payload.get("stale")]
    no_buy_diag["stale_ticker_count"] = len(stale_tickers)
    no_buy_diag["stale_tickers"] = stale_tickers[:10]
    no_buy_diag["api_circuit_breakers"] = api_failure_snapshot()
    fmp_state = fmp_daily_global_circuit_state()
    if not fmp_state.get("active"):
        global_snap = (no_buy_diag["api_circuit_breakers"] or {}).get(FMP_DAILY_GLOBAL_ENDPOINT, {})
        if (global_snap or {}).get("status") == "rate_limited":
            no_buy_diag["api_circuit_breakers"].pop(FMP_DAILY_GLOBAL_ENDPOINT, None)
    no_buy_diag["fmp_daily_global_circuit_status"] = fmp_state.get("status")
    no_buy_diag["fmp_daily_rate_limited"] = bool(fmp_state.get("active"))
    no_buy_diag["fmp_daily_cooldown_remaining_sec"] = int(fmp_state.get("cooldown_remaining_sec") or 0)
    no_buy_diag["fmp_daily_last_429_age_sec"] = fmp_state.get("last_429_age_sec")
    if fmp_state.get("active"):
        no_buy_diag["data_health_warnings"] = list(dict.fromkeys(
            list(no_buy_diag.get("data_health_warnings") or [])
            + ["FMP_DAILY_RATE_LIMITED", "FMP_DAILY_GLOBAL_COOLDOWN"]
        ))
    finnhub_daily_state = finnhub_daily_global_circuit_state()
    no_buy_diag["finnhub_daily_global_circuit_status"] = finnhub_daily_state.get("status")
    no_buy_diag["finnhub_daily_forbidden"] = bool(finnhub_daily_state.get("active"))
    no_buy_diag["finnhub_daily_cooldown_remaining_sec"] = int(
        finnhub_daily_state.get("cooldown_remaining_sec") or 0
    )
    if finnhub_daily_state.get("active"):
        no_buy_diag["data_health_warnings"] = list(dict.fromkeys(
            list(no_buy_diag.get("data_health_warnings") or [])
            + ["FINNHUB_DAILY_FORBIDDEN"]
        ))
    no_buy_diag["history_finnhub_daily_blocked_count"] = sum(
        1 for t in fetch_list
        if (
            no_buy_diag["api_circuit_breakers"].get(f"finnhub_daily:{t}:0", {})
            .get("status")
        ) == "blocked_or_forbidden"
    )
    no_buy_diag["provider_health_status"] = (
        "rate_limited"
        if fmp_state.get("active")
        else _provider_health_summary(no_buy_diag["api_circuit_breakers"])
    )
    no_buy_diag["rate_limit_recent"] = any(
        (snap or {}).get("rate_limit_recent")
        for snap in (no_buy_diag.get("api_circuit_breakers") or {}).values()
        if isinstance(snap, dict)
    ) or bool(fmp_state.get("active"))
    for endpoint, snap in (no_buy_diag.get("api_circuit_breakers") or {}).items():
        if snap.get("rate_limited") and (
            snap.get("rate_limit_recent")
            or (endpoint == "fmp_daily:global" and fmp_state.get("active"))
        ):
            _log_bot_event("RATE_LIMIT_HIT", endpoint=endpoint, provider_status=snap)
        elif snap.get("status") not in {
            "ok",
            "healthy",
            "skipped_on_pythonanywhere",
            "skipped_missing_key",
            "skipped_by_global_rate_limit",
            "skipped_by_circuit",
        }:
            _log_bot_event("PROVIDER_DEGRADED", endpoint=endpoint, provider_status=snap)

    obs_updates = _update_candidate_observations(b, sigs, now)
    if obs_updates:
        b["last_edge_update_ts"] = int(now)

    traded = False
    last_action = ""

    # VIX gating
    _raise_if_tick_deadline_exceeded(deadline_ts, b, no_buy_diag)
    vix_data, ok = _call_with_deadline(get_vix, deadline_ts)
    if not ok:
        no_buy_diag["timeout_stage"] = "volatility"
        _raise_tick_timeout(b, no_buy_diag)
    vix_label = vix_data["regime"]
    vix_mult  = vix_data["mult"]
    mode_info = _market_data_mode(regime, vix_data, cfg)
    data_health_blocks = mode_info["data_health_blocks"]
    regime_data_size_mult = mode_info["mode_size_mult"]
    sizing_vix_mult = vix_mult
    no_buy_diag["vix_label"] = vix_label
    no_buy_diag["vix_value"] = vix_data.get("vix")
    no_buy_diag["vix_data_ok"] = vix_data.get("data_ok")
    no_buy_diag["vix_data_status"] = vix_data.get("data_status")
    no_buy_diag["vix_display"] = (
        vix_data.get("vix_display")
        or ("SPY_REALIZED_VOL_PROXY" if vix_data.get("data_ok") else "unknown")
    )
    no_buy_diag["volatility_data_ok"] = vix_data.get("data_ok")
    no_buy_diag["volatility_source"] = vix_data.get("volatility_source") or vix_data.get("source")
    no_buy_diag["volatility_value"] = vix_data.get("volatility_value", vix_data.get("vix"))
    no_buy_diag["volatility_data_error"] = vix_data.get("data_error")
    no_buy_diag["volatility_error"] = vix_data.get("data_error")
    no_buy_diag["data_health_warnings"] = list(dict.fromkeys(
        list(no_buy_diag.get("data_health_warnings") or [])
        + list(mode_info.get("data_health_warnings") or [])
        + list(vix_data.get("data_health_warnings") or [])
    ))
    no_buy_diag["data_health_blocks"] = data_health_blocks
    no_buy_diag["data_health_ok"] = not bool(data_health_blocks)
    no_buy_diag["regime_data_size_mult"] = regime_data_size_mult
    no_buy_diag.update({
        "trading_mode": mode_info["trading_mode"],
        "normal_mode_active": mode_info["normal_mode_active"],
        "proxy_mode_active": mode_info["proxy_mode_active"],
        "degraded_mode_active": mode_info["degraded_mode_active"],
        "degraded_mode_reason": mode_info["degraded_mode_reason"],
        "degraded_use_standard_gates_for_testing": mode_info["degraded_use_standard_gates_for_testing"],
        "degraded_standard_gates_active": mode_info["degraded_standard_gates_active"],
        "degraded_gate_policy": mode_info["degraded_gate_policy"],
        "effective_size_mult": mode_info["effective_size_mult"],
        "effective_min_buy_confidence": mode_info["effective_min_buy_confidence"],
        "normal_ev_gates_required": mode_info["normal_ev_gates_required"],
        "normal_risk_caps_required": mode_info["normal_risk_caps_required"],
        "fresh_quote_required": mode_info["fresh_quote_required"],
    })
    no_buy_diag["ticker_signal_debug"] = [
        _ticker_debug_row(
            t,
            sigs.get(t) or {},
            cfg,
            degraded_mode_active=mode_info["degraded_mode_active"],
        )
        for t in fetch_list[:10]
    ]

    # Regime-conditional risk params
    if mode_info["degraded_mode_active"]:
        regime_kind = "degraded_neutral"
        regime["regime_effective"] = regime_kind
        regime["regime_v3_effective"] = regime_kind
        no_buy_diag["regime_data_source"] = "degraded_fallback"
        no_buy_diag["regime_source"] = "degraded_fallback"
    risk_regime_kind = "neutral" if regime_kind == "degraded_neutral" else regime_kind
    if risk_regime_kind == "bull":
        STOP_LOSS_PCT = -8.0;  TRAIL_STOP_PCT = 5.0
        regime_size_mult = 1.0; regime_label = "BULL"
        AGING_HOURS = 18
    elif risk_regime_kind == "bear":
        STOP_LOSS_PCT = -4.0;  TRAIL_STOP_PCT = 3.0
        regime_size_mult = 0.5; regime_label = "BEAR"
        AGING_HOURS = 6
    else:
        STOP_LOSS_PCT = -6.0;  TRAIL_STOP_PCT = 4.0
        regime_size_mult = 0.8; regime_label = "NEUTRAL"
        AGING_HOURS = 12

    if regime_kind != macro_regime_kind:
        intra_tilt = regime.get("intra_tilt") or "tilt"
        regime_label = f"{macro_regime_kind.upper()}→{regime_label} (intra {intra_tilt})"

    regime_allow_buys = (vix_mult > 0) and bool(mode_info["allow_buys"])
    regime_v3_label = regime.get("regime_v3_effective") or regime.get("regime_v3")
    if regime_v3_label == "panic" and not b.get("paper_debug_override", False):
        regime_allow_buys = False
    no_buy_diag["regime_allow_buys"] = bool(regime_allow_buys)
    no_buy_diag["regime_kind"] = regime_kind
    no_buy_diag["regime_v3"] = regime_v3_label
    no_buy_diag["regime_fallback_active"] = no_buy_diag.get("regime_data_fallback")
    if data_health_blocks:
        regime_label = f"{regime_label} (data health block: {','.join(data_health_blocks)})"
        _log_bot_event("DATA_HEALTH_BLOCK", blocks=data_health_blocks,
                       regime_data_status=regime.get("regime_data_status"),
                       vix_data_status=vix_data.get("data_status"))
    if mode_info["degraded_mode_active"]:
        regime_label = "DEGRADED NEUTRAL"
        _log_bot_event("DEGRADED_MODE_ACTIVE", blocks=data_health_blocks,
                       size_mult=mode_info["mode_size_mult"],
                       degraded_gate_policy=mode_info["degraded_gate_policy"],
                       degraded_use_standard_gates_for_testing=mode_info["degraded_use_standard_gates_for_testing"],
                       degraded_standard_gates_active=mode_info["degraded_standard_gates_active"],
                       effective_size_mult=mode_info["effective_size_mult"],
                       effective_min_buy_confidence=mode_info["effective_min_buy_confidence"],
                       normal_ev_gates_required=mode_info["normal_ev_gates_required"],
                       normal_risk_caps_required=mode_info["normal_risk_caps_required"],
                       fresh_quote_required=mode_info["fresh_quote_required"])
    elif mode_info["trading_mode"] == "DEGRADED_MODE_DISABLED":
        _log_bot_event("DEGRADED_MODE_DISABLED", blocks=data_health_blocks)

    max_positions = int(risk_cfg.get("max_positions", MAX_POSITIONS))
    MAX_POS_PCT        = float(risk_cfg.get("max_position_pct", 0.35))
    MAX_SECTOR_PCT     = float(risk_cfg.get("max_sector_pct", 1.00))
    MAX_CORR_GROUP_PCT = float(risk_cfg.get("max_corr_group_pct", 1.00))
    MIN_CASH_RESERVE   = float(risk_cfg.get("min_cash_reserve_pct", 0.00))
    MIN_HOLDING_SEC    = 20 * 60
    MAX_GROSS_EXPOSURE_PCT = float(risk_cfg.get("max_gross_exposure_pct", 1.00))

    outcomes = b.get("trade_outcomes", [])
    # Round-7 risk (bug 8 cleanup): the old `win_rate` var here was computed and
    # never read. Repurpose into a fast cold-streak circuit breaker — independent
    # of the 50-trade Kelly look-back. 3+ losses in the last 5 closes → downsize.
    last5 = outcomes[-5:]
    streak_losses = sum(1 for o in last5 if o["pnl_pct"] <= 0)
    streak_mult = max(0.5, 1.0 - (streak_losses - 2) * 0.15) if streak_losses >= 3 else 1.0

    # ── SELL pass ──────────────────────────────────────────────────────────
    for t in list(b["holdings"].keys()):
        if t not in sigs: continue
        s = sigs[t]; pr = s["price"]
        # Round-4 Bug Fix #1: halt trading on stale-quote tickers. We do NOT
        # trail-exit, signal-flip, or age-out a position based on a price that
        # may be hours/days old. Stop-loss is also disabled here — if the API
        # is down we don't have a valid current price to compare against.
        if s.get("stale"):
            try: print(f"[run_bot] {t}: stale quote (no live data), halting trade decisions")
            except Exception: pass
            continue
        if pr <= 0: continue
        h = b["holdings"][t]
        peak = max(h.get("peak", h["avg_cost"]), pr)
        h["peak"] = round(peak, 4)
        trough = min(h.get("trough", h["avg_cost"]), pr)
        h["trough"] = round(trough, 4)
        # Round-5 cost model: no bps slippage. pnl measured at raw price; the
        # flat $0.99 is applied to cash at execution, not baked into price.
        pnl_pct = (pr - h["avg_cost"]) / h["avg_cost"] * 100 if h["avg_cost"] else 0
        trail_pct = (pr - peak) / peak * 100 if peak else 0
        cls = s["rec"]["cls"]
        atr_pct = (s.get("ctx") or {}).get("atr_pct", 0)
        entry_ts = h.get("entry_ts", 0)
        held_secs = now - entry_ts if entry_ts else 999999
        held_hours = held_secs / 3600.0
        entry_cluster_h = (h.get("entry_cluster")
                           or (h.get("entry_snapshot") or {}).get("entry_cluster")
                           or "mixed")
        learned_exit = exit_profile(b, regime_kind, entry_cluster_h)
        exit_ladder = compose_exit_profile(entry_cluster_h, learned_exit)
        exit_ladder = apply_regime_exit_tightening(exit_ladder, regime_v3_label)
        rv = float(vix_data.get("vix", 0) or 0)
        if rv > 25:
            exit_ladder["failure_timeout_days"] = round(max(1.0, exit_ladder.get("failure_timeout_days", 1.0) * 0.50), 4)
        elif rv > 18:
            exit_ladder["failure_timeout_days"] = round(max(1.0, exit_ladder.get("failure_timeout_days", 1.0) * 0.75), 4)
        stop_loss_profiled = STOP_LOSS_PCT * exit_ladder.get("atr_stop_mult", 1.0)
        aging_hours_profiled = min(
            AGING_HOURS * max(exit_ladder.get("max_hold_days", 10) / 10.0, 0.25),
            exit_ladder.get("max_hold_days", 10) * 24.0,
        )
        exit_ladder_note = (
            f"exit-ladder {exit_ladder['cluster']}: stop x{exit_ladder['atr_stop_mult']}, "
            f"trail x{exit_ladder['trail_mult']}, partial {exit_ladder['partial_take_pct']*100:.1f}%, "
            f"timeout {exit_ladder['failure_timeout_days']:.1f}d, max {exit_ladder['max_hold_days']:.1f}d"
        )
        if exit_ladder.get("learned_live"):
            exit_ladder_note += f"; learned {exit_ladder.get('learned_notes')}"

        # #4.2 median-of-trend ATR floor → ATR-scaled hard stop (trading.exits)
        eff_atr_for_stop = atr_pct or 0
        trend_start_ts = h.get("trend_start_ts") or 0
        if trend_start_ts:
            med_atr = median_atr_since(t, trend_start_ts)
            if med_atr > eff_atr_for_stop:
                eff_atr_for_stop = med_atr
        dynamic_stop = dynamic_stop_pct(
            eff_atr_for_stop * exit_ladder.get("atr_stop_mult", 1.0),
            stop_loss_profiled,
        )

        # Dynamic trail width. A5: intra ATR is 0 on PA (yfinance intraday dead) → the
        # helper falls back to the daily ATR so the trail stays volatility-scaled
        # instead of collapsing to the flat regime-static width.
        intra_pack = s.get("intra", {}) or {}
        intra_atr = intra_pack.get("intra_atr_pct", 0) or 0
        vwma_dist = intra_pack.get("intra_vwma_dist", 0) or 0
        shadow_old_exit = _generic_exit_shadow(
            h, pr, s.get("ctx"), s["rec"], STOP_LOSS_PCT, TRAIL_STOP_PCT,
            AGING_HOURS, held_secs, MIN_HOLDING_SEC, eff_atr_for_stop,
            intra_atr=intra_atr, vwma_dist=vwma_dist,
        )
        if shadow_old_exit.get("would_exit") and not h.get("shadow_old_exit"):
            h["shadow_old_exit"] = shadow_old_exit
        dyn_trail_pct, trail_source = dynamic_trail_width(intra_atr, atr_pct, TRAIL_STOP_PCT)
        dyn_trail_pct = round(dyn_trail_pct * exit_ladder.get("trail_mult", 1.0), 3)
        trail_source += f"; {exit_ladder_note}"
        # #4.1 time-based trail tightening
        if pnl_pct >= 10 and held_hours > 24:
            days_held_extra = (held_hours - 24) / 24
            shrink = 0.5 * days_held_extra
            before = dyn_trail_pct
            dyn_trail_pct = max(1.5, dyn_trail_pct - shrink)
            if dyn_trail_pct < before:
                trail_source += f" (tightened −{(before - dyn_trail_pct):.2f}pp over {days_held_extra:.1f}d)"
        trail_start_pct = exit_ladder.get("trail_start_pct", 0.04) * 100.0
        trail_triggered = (pnl_pct > trail_start_pct and trail_pct <= -dyn_trail_pct)
        vwma_protects = (vwma_dist > 1.0) and (trail_pct > -3.0)
        trail_active = trail_triggered and not vwma_protects

        peak_pnl_pct = max(h.get("peak_pnl_pct", 0), pnl_pct)
        h["peak_pnl_pct"] = round(peak_pnl_pct, 3)

        # #4.3 signal-degradation partial exit
        DEGRADE_LADDER = {"strong-buy": 3, "buy": 2, "hold": 1, "sell": 0, "strong-sell": 0}
        DEGRADE_MIN_GAP = 2
        entry_cls = h.get("entry_signal_cls")
        if (entry_cls and held_secs >= MIN_HOLDING_SEC
                and not h.get("degrade_trimmed", False)):
            entry_rank = DEGRADE_LADDER.get(entry_cls, 1)
            cur_rank   = DEGRADE_LADDER.get(cls, 1)
            gap = entry_rank - cur_rank
            if gap >= DEGRADE_MIN_GAP and cur_rank <= 1:
                # Round-6 T2: 30% → 20%. Trail + breakeven already protect; smaller
                # trim avoids cutting long runners short.
                degrade_reason = (f"Signal degraded {entry_cls.upper()}→{cls.upper()} "
                                   f"(gap {gap}, left bullish zone) — trimming 20%")
                if _partial_trim(b, t, h, pr, s.get("ctx"), s["rec"], s["arts"],
                                  frac=0.20, reason=degrade_reason) > 0:
                    h["degrade_trimmed"] = True
                    b["holdings"][t] = h
                    traded = True

        # #2.4 partial take (opt-in)
        ladder_partial_take_pct = exit_ladder.get(
            "partial_take_pct", PARTIAL_TAKE_PCT / 100.0
        ) * 100.0
        if (PARTIAL_TAKE_ENABLED and peak_pnl_pct >= ladder_partial_take_pct
                and not h.get("partial_taken", False)
                and held_secs >= MIN_HOLDING_SEC):
            partial_reason = (f"Partial take: peak hit +{peak_pnl_pct:.1f}%, "
                              f"ladder target +{ladder_partial_take_pct:.1f}%, "
                              f"trimming {PARTIAL_TAKE_FRACTION*100:.0f}%. "
                              f"Remaining stays on trail.")
            if _partial_trim(b, t, h, pr, s.get("ctx"), s["rec"], s["arts"],
                              frac=PARTIAL_TAKE_FRACTION, reason=partial_reason) > 0:
                h["partial_taken"] = True
                b["holdings"][t] = h
                traded = True
                peak_pnl_pct = pnl_pct
                h["peak_pnl_pct"] = round(peak_pnl_pct, 3)

        # Effective stop — A1 cost-aware ratchet + time-decay. The ratchet arms only
        # after a real move (≥+3%, or higher for tiny positions) and then locks a floor
        # that ratchets up with the peak and is ALWAYS net-positive after commissions —
        # replacing the old +0.1% breakeven that sold winners at a net loss and capped
        # every runner.
        pos_val = h["shares"] * pr
        rt_cost_pct = round_trip_cost_pct(pos_val, COMMISSION_PER_TRADE)
        ratchet_lock = breakeven_lock_pct(peak_pnl_pct, rt_cost_pct)
        if ratchet_lock is not None:
            effective_stop = max(dynamic_stop, ratchet_lock)
            stop_label = (f"ratchet lock +{effective_stop:.1f}% (peak +{peak_pnl_pct:.1f}%, "
                          f"round-trip cost {rt_cost_pct:.2f}%)")
        elif held_hours > 2.0 and pnl_pct < 0 and peak_pnl_pct < 1.0:
            effective_stop = dynamic_stop * 0.5
            stop_label = f"time-decay {effective_stop:.1f}% (held {held_hours:.1f}h, no progress)"
        else:
            effective_stop = dynamic_stop
            atr_note = ""
            if eff_atr_for_stop > 0:
                atr_note = f", ATR={eff_atr_for_stop:.2f}% x{exit_ladder.get('atr_stop_mult', 1.0)}"
                if eff_atr_for_stop > (atr_pct or 0):
                    atr_note += " (median-of-trend, wider than current)"
            stop_label = f"{effective_stop:.1f}%, regime={regime_label}{atr_note}; {exit_ladder_note}"

        sell_reason = None; exit_reason_key = ""
        if pnl_pct <= effective_stop:
            if effective_stop > 0:
                # Cost-aware ratchet exit — locking in a NET-positive gain, not a loss.
                # Tag it distinctly so it gets the short (non-loss) cooldown and isn't
                # miscounted against the strategy.
                sell_reason = (f"Ratchet exit: pulled back to {pnl_pct:+.1f}% from peak "
                               f"+{peak_pnl_pct:.1f}% (locked {stop_label})")
                exit_reason_key = "ratchet"
            else:
                sell_reason = (f"Stop-loss: down {pnl_pct:.1f}% from avg cost (threshold {stop_label})")
                exit_reason_key = "loss"
        elif held_secs < MIN_HOLDING_SEC:
            sell_reason = None
        elif trail_active:
            sell_reason = (f"Dynamic trail: down {trail_pct:.1f}% from peak ${peak:.2f} "
                           f"(width = {trail_source}). Still +{pnl_pct:.1f}% from entry.")
            exit_reason_key = "trail"
        elif held_secs >= MIN_HOLDING_SEC and (
                held_hours / 24.0 >= exit_ladder.get("failure_timeout_days", 3)
                and peak_pnl_pct < trail_start_pct
                and pnl_pct <= 0.25):
            held_days = held_hours / 24.0
            sell_reason = (
                f"Cluster failure timeout: {exit_ladder['cluster']} held {held_days:.1f}d "
                f"without follow-through (peak +{peak_pnl_pct:.1f}%, now {pnl_pct:+.1f}%). "
                f"{exit_ladder_note}"
            )
            exit_reason_key = f"{exit_ladder['cluster']}_failure_timeout"
        elif held_secs >= MIN_HOLDING_SEC and (
                held_hours / 24.0 >= exit_ladder.get("max_hold_days", 10)
                and peak_pnl_pct < ladder_partial_take_pct):
            held_days = held_hours / 24.0
            sell_reason = (
                f"Cluster max hold: {exit_ladder['cluster']} held {held_days:.1f}d "
                f"without reaching +{ladder_partial_take_pct:.1f}% partial target "
                f"(now {pnl_pct:+.1f}%). {exit_ladder_note}"
            )
            exit_reason_key = f"{exit_ladder['cluster']}_max_hold"
        elif cls in ("sell", "strong-sell") and pnl_pct < 0:
            sell_reason = (f"Signal turned {s['rec']['signal']} AND position losing "
                           f"({pnl_pct:+.1f}%) — confirmed exit")
            exit_reason_key = "loss"
        elif cls == "strong-sell" and pnl_pct >= 0:
            # Round-7 risk: signal flipped STRONG SELL while still profitable. The
            # 20% degrade-trim alone leaves most of the position to erode waiting
            # for the trail. Take the full gain now.
            sell_reason = (f"Signal flipped STRONG SELL while still profitable "
                           f"({pnl_pct:+.1f}%) — exit before gains erode")
            exit_reason_key = "signal_flip_profit"
        else:
            # Round-4 algorithm #9: trend-failure exit. If price closes meaningfully
            # below the 30-day MA AND we're underwater, the daily-trend thesis
            # has broken — exit before it spirals into a stop-loss. 2% buffer
            # below MA30 to avoid noise-driven exits.
            ctx_e = s.get("ctx") or {}
            cur_e = ctx_e.get("current", 0) or 0
            ma30_e = ctx_e.get("ma30", 0) or 0
            if (ma30_e > 0 and cur_e > 0 and pnl_pct < 0
                    and cur_e < ma30_e * 0.98):
                sell_reason = (f"Trend failure: price ${cur_e:.2f} broke below 30-day MA "
                               f"${ma30_e:.2f} (−{(1-cur_e/ma30_e)*100:.1f}%), position "
                               f"underwater {pnl_pct:+.1f}%")
                exit_reason_key = "trend_failure"
            elif held_hours > aging_hours_profiled and abs(pnl_pct) < 1.0 and h["shares"] * pr >= 150:
                # Round-6: only age-sell positions ≥$150 (matches min_trade_size) —
                # don't pay $0.99 to dump a tiny stub. Stops/trails/trend-failure unaffected.
                sell_reason = (f"Position aging: held {held_hours:.1f}h, flat ({pnl_pct:+.1f}%), "
                               f"freeing capital (regime={regime_label} threshold={aging_hours_profiled:.1f}h; {exit_ladder_note})")
                exit_reason_key = "aging"

        if sell_reason:
            if "exit-ladder" not in sell_reason:
                sell_reason = f"{sell_reason} ({exit_ladder_note})"
            if "cfg=" not in sell_reason:
                sell_reason = f"{sell_reason} cfg={cfg_hash}, V3={regime_v3_label}."
            shadow = h.get("shadow_old_exit") or shadow_old_exit
            if shadow and shadow.get("would_exit"):
                sell_reason = (
                    f"{sell_reason} Shadow old exit: {shadow.get('exit_reason')} "
                    f"at {shadow.get('pnl_pct'):+.1f}%."
                )
            sh = h["shares"]
            # Round-5 cost model: flat $0.99, no bps markup.
            entry_commission = float(h.get("commission_invested", 0) or 0)
            realized = sh * pr - sh * h["avg_cost"] - entry_commission - COMMISSION_PER_TRADE
            b["total_costs_usd"] = round(b.get("total_costs_usd", 0) + COMMISSION_PER_TRADE, 2)
            b["cash"] += sh * pr - COMMISSION_PER_TRADE
            # A4: the learning loop (Kelly, cold-streak, win-rate, calibration) must see
            # NET P&L — the gross price move overstates edge by the full round-trip
            # commission. `pnl_pct` stays gross for the human-readable reason string.
            cost_basis = sh * h["avg_cost"] + entry_commission
            net_pnl_pct = (realized / cost_basis * 100) if cost_basis else 0
            _record_trade(b, "SELL", t, sh, pr, s["rec"], s["arts"], sell_reason,
                          pnl_usd=round(realized, 2))
            entry_snap = h.get("entry_snapshot") or {}
            entry_regime = entry_snap.get("market_regime")
            h["exit_ladder_profile"] = exit_ladder
            record_exit_event(b, t, h, exit_reason_key, net_pnl_pct, pr, ts=now,
                              regime=entry_regime or regime_kind)
            b.setdefault("trade_outcomes", []).append({
                "ticker": t, "pnl_pct": round(net_pnl_pct, 2),
                "gross_pnl_pct": round(pnl_pct, 2),
                "exit_reason": exit_reason_key, "ts": int(now),
                "entry_regime": entry_regime,
                "entry_cluster": entry_cluster_h,
                "exit_ladder_profile": exit_ladder,
                "shadow_old_exit": h.get("shadow_old_exit"),
                # Round-6 T3: log entry confidence for calibration (conf → win-rate)
                "entry_confidence": entry_snap.get("confidence"),
            })
            b["trade_outcomes"] = b["trade_outcomes"][-100:]
            # Round-5 extra: all-time win/loss counters (trade_outcomes is capped
            # at 100, so derive cumulative W/L from monotonic counters instead).
            if net_pnl_pct > 0:
                b["wins_total"] = b.get("wins_total", 0) + 1
            else:
                b["losses_total"] = b.get("losses_total", 0) + 1
            del b["holdings"][t]
            _mark_sold(t, now, exit_reason_key)
            traded = True
            no_buy_diag["sells_executed"] = int(no_buy_diag.get("sells_executed", 0) or 0) + 1
            last_action = f"SELL {t}"
        else:
            b["holdings"][t] = h

    # ── BUY pass ───────────────────────────────────────────────────────────
    starting_cash = b["cash"] + sum(
        h["shares"] * sigs[tt]["price"]
        for tt, h in b["holdings"].items()
        if tt in sigs and sigs[tt]["price"] > 0
    )
    cash_floor = starting_cash * MIN_CASH_RESERVE
    no_buy_diag["cash_floor"] = round(float(cash_floor), 2)
    no_buy_diag["cash_available_after_floor"] = round(float(b.get("cash", 0) - cash_floor), 2)
    exposure_value = 0.0
    stale_positions = []
    risk_unmanaged_positions = []
    for tt, h in b.get("holdings", {}).items():
        sig_t = sigs.get(tt) or {}
        pr = float(sig_t.get("price") or 0)
        if pr <= 0 or sig_t.get("stale"):
            stale_positions.append(tt)
            risk_unmanaged_positions.append(tt)
            pr = float(h.get("avg_cost", 0) or 0)
        exposure_value += float(h.get("shares", 0) or 0) * pr
    no_buy_diag["gross_exposure_pct"] = round(exposure_value / starting_cash, 4) if starting_cash else 0.0
    no_buy_diag["stale_positions"] = stale_positions[:10]
    no_buy_diag["risk_unmanaged_positions"] = risk_unmanaged_positions[:10]
    if risk_unmanaged_positions:
        no_buy_diag["data_health_blocks"] = list(dict.fromkeys(
            list(no_buy_diag.get("data_health_blocks") or []) + ["STALE_HELD_QUOTE"]
        ))
        no_buy_diag["data_health_ok"] = False
        regime_allow_buys = False
        no_buy_diag["regime_allow_buys"] = False
        _log_bot_event("STALE_HELD_QUOTE", tickers=risk_unmanaged_positions[:10])
    max_buys_today = int(risk_cfg.get("max_new_buys_per_day", 30))
    try:
        from zoneinfo import ZoneInfo
        et_zone = ZoneInfo("America/New_York")
        today_key = datetime.now(et_zone).strftime("%Y-%m-%d")
        buys_today = sum(
            1 for e in b.get("history", [])
            if e.get("action") == "BUY"
            and e.get("ts")
            and datetime.fromtimestamp(float(e.get("ts")), et_zone).strftime("%Y-%m-%d") == today_key
        )
    except Exception:
        buys_today = 0
        degraded_buys_today = 0
        scan_buys_today = 0
    else:
        degraded_buys_today = 0
        scan_buys_today = 0
        for e in b.get("history", []):
            try:
                if (e.get("action") == "BUY"
                        and e.get("trading_mode") == "DEGRADED_MODE"
                        and e.get("ts")
                        and datetime.fromtimestamp(float(e.get("ts")), et_zone).strftime("%Y-%m-%d") == today_key):
                    degraded_buys_today += 1
                if (e.get("action") == "BUY"
                        and e.get("source") == "scan"
                        and e.get("ts")
                        and datetime.fromtimestamp(float(e.get("ts")), et_zone).strftime("%Y-%m-%d") == today_key):
                    scan_buys_today += 1
            except Exception:
                pass
    degraded_max_buys_today = int(mode_cfg.get("degraded_max_new_buys_per_day", 30))
    degraded_max_buys_per_tick = int(mode_cfg.get("degraded_max_new_buys_per_tick", 10))
    max_scan_buys_today = int(scan_cfg.get("max_scan_buys_per_day", 30))
    max_scan_buys_per_tick = int(scan_cfg.get("max_scan_buys_per_tick", BOT_SCAN_BUY))
    degraded_max_position_pct = float(mode_cfg.get("degraded_max_position_pct", 0.05))
    degraded_max_gross_exposure_pct = float(mode_cfg.get("degraded_max_gross_exposure_pct", 0.35))
    no_buy_diag["buys_today"] = int(buys_today)
    no_buy_diag["max_buys_today"] = max_buys_today
    no_buy_diag["degraded_buys_today"] = int(degraded_buys_today)
    no_buy_diag["degraded_max_buys_today"] = degraded_max_buys_today
    no_buy_diag["scan_buys_today"] = int(scan_buys_today)
    no_buy_diag["max_scan_buys_today"] = max_scan_buys_today
    no_buy_diag["max_scan_buys_per_tick"] = max_scan_buys_per_tick
    no_buy_diag["degraded_size_mult"] = mode_cfg.get("degraded_size_mult", 0.90)
    no_buy_diag["effective_size_mult"] = mode_info["effective_size_mult"]
    no_buy_diag["effective_min_buy_confidence"] = mode_info["effective_min_buy_confidence"]
    no_buy_diag["degraded_use_standard_gates_for_testing"] = mode_info["degraded_use_standard_gates_for_testing"]
    no_buy_diag["degraded_standard_gates_active"] = mode_info["degraded_standard_gates_active"]
    no_buy_diag["degraded_gate_policy"] = mode_info["degraded_gate_policy"]
    no_buy_diag["normal_ev_gates_required"] = mode_info["normal_ev_gates_required"]
    no_buy_diag["normal_risk_caps_required"] = mode_info["normal_risk_caps_required"]
    no_buy_diag["fresh_quote_required"] = mode_info["fresh_quote_required"]
    no_buy_diag["degraded_gross_exposure_pct"] = no_buy_diag.get("gross_exposure_pct")
    no_buy_diag["degraded_max_gross_exposure_pct"] = degraded_max_gross_exposure_pct

    # Round-8 Bug #3: if ANY held position is priced off a stale quote, this
    # equity figure is partly valued at avg_cost (not a live price). Don't let such
    # an outage-valued total advance the all-time peak — a fake peak would inflate
    # every future drawdown number.
    valuation_has_stale = any(
        (sigs.get(tt) or {}).get("stale") for tt in b["holdings"] if tt in sigs
    )

    # Round-5 extra: peak is ALL-TIME, never resets. (Removed near_peak_streak
    # reset-to-current logic.) Running max of equity across the bot's whole life.
    if valuation_has_stale:
        peak_equity = b.get("peak_equity", starting_cash)   # hold prior peak; don't advance on stale
    else:
        peak_equity = max(b.get("peak_equity", starting_cash), starting_cash)
    b["peak_equity"] = round(peak_equity, 2)

    drawdown_pct = (peak_equity - starting_cash) / peak_equity * 100 if peak_equity else 0
    lockout_active = _apply_paper_loss_lockouts(b, no_buy_diag, risk_cfg, starting_cash, peak_equity, now)
    if lockout_active:
        regime_allow_buys = False
        no_buy_diag["regime_allow_buys"] = False
    buys_paused = bool(lockout_active)

    sector_value = {}
    corr_value = {}
    for tt, h2 in b["holdings"].items():
        if tt in sigs and sigs[tt]["price"] > 0:
            sec = get_sector(tt)
            if sec is None:
                continue
            sector_value[sec] = sector_value.get(sec, 0) + h2["shares"] * sigs[tt]["price"]
            grp = get_corr_group(tt)
            if grp:
                corr_value[grp] = corr_value.get(grp, 0) + h2["shares"] * sigs[tt]["price"]

    def buyable_reason_local(t, s):
        return buyable_reason(t, s, recent_sells, regime_kind, b.get("holdings", {}), cfg=cfg)

    def buyable(t, s):
        ok, reason = buyable_reason_local(t, s)
        if not ok:
            _bump(no_buy_diag["buyable_reject_counts"], reason)
            if len(no_buy_diag["top_buyable_rejects"]) < 10:
                rec = s.get("rec") or {}
                no_buy_diag["top_buyable_rejects"].append({
                    "ticker": t,
                    "raw_signal_label": rec.get("raw_signal_label") or rec.get("signal"),
                    "display_signal_label": rec.get("display_signal_label"),
                    "confidence": rec.get("confidence"),
                    "rejection_reason": reason,
                })
        return ok

    # #3 TOD gate: no NEW buys outside 09:45-15:30 ET unless a human forces it.
    tod_ok = user_forced or in_new_buy_window()
    no_buy_diag["tod_ok"] = bool(tod_ok)
    no_buy_diag["buy_window_open"] = bool(in_new_buy_window())

    b.pop("rotation_streak", None)
    rotation_actions = []
    bought = []

    def _portfolio_total():
        return b["cash"] + sum(h2["shares"] * sigs[tt]["price"]
                               for tt, h2 in b["holdings"].items()
                               if tt in sigs and sigs[tt]["price"] > 0)

    benchmark_prices = {}
    for bench in ("SPY", "QQQ"):
        try:
            q_b = get_quote(bench) or {}
            if q_b.get("price", 0) > 0 and not q_b.get("stale"):
                benchmark_prices[bench] = q_b["price"]
        except Exception:
            pass

    def _candidate(t, s, source):
        rec = s.get("rec") or {}
        _apply_execution_profile(
            rec,
            cfg,
            degraded_mode_active=mode_info["degraded_mode_active"],
        )
        ctx = s.get("ctx") or {}
        quote_status = _quote_execution_status(s)
        cluster = entry_cluster(rec, ctx)
        regime_v3 = regime.get("regime_v3_effective") or regime.get("regime_v3")
        r_mult = regime_risk_mult(regime_v3, cfg)
        c_mult = cluster_regime_mult(regime_v3, cluster, cfg)
        catalyst = rec.get("catalyst") or {}
        return {
            "ticker": t, "source": source, "rec": rec, "ctx": ctx,
            "price": s.get("price", 0), "arts": s.get("arts") or [],
            "earn": s.get("earn") or {},
            "cluster": cluster,
            "candidate_type": rec.get("candidate_type"),
            "candidate_size_multiplier": rec.get("candidate_size_multiplier", 1.0),
            "execution_trusted": quote_status.get("execution_trusted"),
            "quote_source": quote_status.get("quote_source"),
            "cache_used": quote_status.get("cache_used"),
            "quote_age_seconds": quote_status.get("quote_age_seconds"),
            "vol_regime": classify_vol_regime(ctx),
            "benchmark_prices": dict(benchmark_prices),
            "config_hash": cfg_hash,
            "regime_v3": regime_v3,
            "regime_reason": regime.get("regime_v3_reason"),
            "regime_risk_mult": r_mult,
            "cluster_regime_mult": c_mult,
            "top_sectors": regime.get("top_sectors", []),
            "catalyst_type": catalyst.get("type"),
            "catalyst_score_shadow": catalyst.get("score_shadow"),
            "catalyst_confirmed": catalyst.get("confirmed"),
        }

    # Kelly remains, but V1 applies it inside risk_pct instead of final spend.
    kelly_cfg = cfg.get("kelly", {}) if isinstance(cfg, dict) else {}
    kelly_enabled = bool(kelly_cfg.get("enabled", False))
    kelly_min_samples = int(kelly_cfg.get("min_samples", 100))
    if not kelly_enabled:
        kelly_mult = 1.0
        kelly_diag = "kelly disabled by config"
        kelly_raw = 0.0
    elif len(outcomes) < kelly_min_samples:
        kelly_mult = 1.0
        kelly_diag = f"warm-up <{kelly_min_samples} trades, kelly=1.0"
        kelly_raw = 0.0
    else:
        from trading.signals import _regime_key
        same_regime = [o for o in outcomes
                       if _regime_key(o.get("entry_regime")) == regime_kind]
        recent50 = (same_regime[-50:] if len(same_regime) >= 10 else outcomes[-50:])
        wins50   = [o["pnl_pct"] for o in recent50 if o["pnl_pct"] > 0]
        losses50 = [abs(o["pnl_pct"]) for o in recent50 if o["pnl_pct"] <= 0]
        if wins50 and losses50:
            wr = len(wins50) / len(recent50)
            avg_w = sum(wins50) / len(wins50)
            avg_l = sum(losses50) / len(losses50)
            b_ratio = avg_w / max(avg_l, 0.01)
            f_full = wr - (1 - wr) / b_ratio
            kelly_raw = 0.5 * f_full
            kelly_mult = max(0.10, min(1.5, kelly_raw))
            kelly_diag = (f"kelly: wr={wr:.0%}, W:L={b_ratio:.2f}, f={f_full:+.2f}, "
                          f"half-kelly={kelly_raw:.3f}, clamped={kelly_mult:.3f}")
        else:
            kelly_mult = 1.0
            kelly_diag = "insufficient W or L in last 50, kelly=1.0"
            kelly_raw = 0.0
    b["last_kelly"] = {
        "kelly_raw":  round(kelly_raw, 4),
        "kelly_mult": round(kelly_mult, 4),
        "diag":       kelly_diag,
        "ts":         int(now),
        "notes":      ("V1: Kelly modifies ATR risk budget, not final spend. "
                       "Disabled until ≥50 closed outcomes; then set kelly: enabled=True, "
                       "min_samples=50, fraction=0.5, min_mult=0.5, max_mult=1.25."),
    }

    candidate_pool = []
    scan_data_snapshot, scan_ts_snapshot = _scan_snapshot()
    scan_age = now - (scan_ts_snapshot or 0)
    scan_enabled = bool(scan_cfg.get("scan_buys_enabled", True))
    scan_freshness_sec = int(scan_cfg.get("scan_data_max_age_sec", SCAN_FRESHNESS_SEC))
    scan_min_conf = float(scan_cfg.get("scan_buy_min_confidence", SCAN_BUY_MIN_CONF))
    scan_min_history_rows = int(scan_cfg.get("scan_requires_min_history_rows", MIN_SIGNAL_HISTORY_ROWS) or 0)
    scan_min_dvol_hard = float(signal_cfg.get("min_avg_dollar_volume_hard_block", 250_000))
    scan_rows_all = scan_data_snapshot or []
    fresh_scan_rows = [
        r for r in scan_rows_all
        if now - float(r.get("ts") or scan_ts_snapshot or 0) <= scan_freshness_sec
    ]
    scan_fresh = bool(scan_enabled and fresh_scan_rows)
    no_buy_diag["scan_buys_enabled"] = bool(scan_enabled)
    no_buy_diag["scan_data_max_age_sec"] = scan_freshness_sec
    no_buy_diag["scan_min_confidence"] = scan_min_conf
    no_buy_diag["scan_min_history_rows"] = scan_min_history_rows
    no_buy_diag["scan_fresh"] = bool(scan_fresh)
    no_buy_diag["scan_age_sec"] = int(scan_age) if scan_ts_snapshot else None
    no_buy_diag["scan_rows_count"] = len(scan_rows_all)
    no_buy_diag["scan_fresh_rows_count"] = len(fresh_scan_rows)

    degraded_restricted_mode = (
        mode_info["degraded_mode_active"]
        and not mode_info["degraded_standard_gates_active"]
    )

    if degraded_restricted_mode:
        daily_buy_room = degraded_buys_today < degraded_max_buys_today
        exposure_room = (no_buy_diag.get("gross_exposure_pct") or 0) < degraded_max_gross_exposure_pct
    else:
        daily_buy_room = buys_today < max_buys_today
        exposure_room = (no_buy_diag.get("gross_exposure_pct") or 0) < MAX_GROSS_EXPOSURE_PCT
    if not daily_buy_room:
        _bump(no_buy_diag["skip_reason_counts"], "MAX_BUYS_PER_DAY")
    if not exposure_room:
        _bump(no_buy_diag["skip_reason_counts"], "MAX_GROSS_EXPOSURE")

    if not buys_paused and regime_allow_buys and tod_ok and daily_buy_room and exposure_room:
        for t, s in sigs.items():
            rec = s.get("rec") or {}
            _apply_execution_profile(
                rec,
                cfg,
                degraded_mode_active=mode_info["degraded_mode_active"],
            )
            s["rec"] = rec
            if (t in tickers
                    and is_execution_candidate(
                        rec,
                        cfg,
                        degraded_mode_active=mode_info["degraded_mode_active"],
                    )
                    and buyable(t, s)):
                candidate_pool.append(_candidate(t, s, "watchlist"))

        if scan_enabled and scan_fresh and b["cash"] - cash_floor > min_position_usd:
            bullish = [r for r in fresh_scan_rows if r["direction"] > 0 and r["price"] > 0]
            by_conf = sorted(bullish, key=lambda r: (-r["confidence"], -r["score"]))[:10]
            by_gain = sorted(bullish, key=lambda r: -r["pct"])[:10]
            seen = set(); pool = []
            for r in by_conf + by_gain:
                if r["ticker"] not in seen:
                    seen.add(r["ticker"]); pool.append(r)
            for r in pool:
                tk = r["ticker"]
                if tk in tickers or tk in b["holdings"]:
                    continue
                if float(r.get("confidence", 0) or 0) < scan_min_conf:
                    continue
                cached = cache_get(f"scan_payload_{tk}", max_age=scan_freshness_sec)
                if cached is None:
                    no_buy_diag["scan_payload_misses"] += 1
                    continue
                rec = dict(cached.get("rec") or {})
                _annotate_signal(rec)
                ctx_o = cached.get("ctx") or {}
                arts = cached.get("arts") or []
                q_o = cached.get("quote") or {}
                pr = cached.get("price") or q_o.get("price") or r.get("price", 0)
                payload = {
                    "rec": rec, "price": pr, "stale": bool(cached.get("stale") or q_o.get("stale")),
                    "arts": arts, "ctx": ctx_o, "intra": {},
                    "earn": cached.get("earn") or {},
                    "analyst": cached.get("analyst") or {},
                    "insider": cached.get("insider") or {},
                    "quote": q_o,
                }
                _finalize_signal_confidence(payload, cfg)
                rec = payload["rec"]
                ctx_o = payload.get("ctx") or ctx_o
                _apply_execution_profile(
                    rec,
                    cfg,
                    degraded_mode_active=mode_info["degraded_mode_active"],
                )
                quote_status = _quote_execution_status(payload)
                if pr <= 0:
                    continue
                if (
                    bool(scan_cfg.get("scan_requires_execution_trusted_quote", True))
                    and not quote_status["execution_trusted"]
                ):
                    continue
                if (
                    rec.get("display_signal_label") == "BULLISH_LEAN"
                    and not bool(scan_cfg.get("scan_allow_bullish_lean", True))
                ):
                    continue
                if (
                    rec.get("display_signal_label") == "HOLD"
                    and not bool(scan_cfg.get("scan_allow_high_confidence_hold", False))
                ):
                    continue
                if not is_execution_candidate(
                    rec,
                    cfg,
                    degraded_mode_active=mode_info["degraded_mode_active"],
                ):
                    continue
                history_status = _history_execution_status(ctx_o)
                if scan_min_history_rows and history_status["history_rows"] < scan_min_history_rows:
                    continue
                dvol = ctx_o.get("avg_dollar_vol_20d", 0) or 0
                if dvol > 0 and dvol < scan_min_dvol_hard:
                    continue
                sigs[tk] = payload
                if buyable(tk, payload):
                    candidate_pool.append(_candidate(tk, payload, "scan"))
        elif scan_enabled and b["cash"] - cash_floor > min_position_usd and not scan_fresh:
            stale_scan_skipped = True
    no_buy_diag["candidate_pool_count"] = len(candidate_pool)

    pt_for_rank = _portfolio_total()
    _raise_if_tick_deadline_exceeded(deadline_ts, b, no_buy_diag)
    ranked_candidates = rank_candidates(
        candidate_pool, pt_for_rank, STOP_LOSS_PCT, regime_kind,
        sizing_vix_mult, streak_mult, kelly_mult, b,
        min_position_usd=min_position_usd, commission=COMMISSION_PER_TRADE,
        config=cfg, mode_size_mult=mode_info["mode_size_mult"],
        mode_size_reason=mode_info["mode_size_reason"],
    ) if candidate_pool else []
    no_buy_diag["ranked_count"] = len(ranked_candidates)
    no_buy_diag["tradable_count"] = sum(1 for c in ranked_candidates if c.get("tradable"))
    variance_checks = []

    def _ranking_rows():
        return [{
            "ticker": c["ticker"], "source": c["source"], "cluster": c["cluster"],
            "signal": c["rec"].get("signal"), "confidence": c["rec"].get("confidence"),
            "confidence_scale": (c.get("risk") or {}).get("confidence_scale"),
            "candidate_type": c.get("candidate_type") or c["rec"].get("candidate_type"),
            "candidate_size_multiplier": c.get("candidate_size_multiplier"),
            "quote_source": c.get("quote_source"),
            "cache_used": c.get("cache_used"),
            "quote_age_seconds": c.get("quote_age_seconds"),
            "execution_trusted": c.get("execution_trusted"),
            "gross_edge_pct": c.get("gross_edge_pct"),
            "friction_pct": (c.get("friction") or {}).get("total_pct"),
            "net_edge_pct": c.get("net_edge_pct"), "ev_score": c.get("ev_score"),
            "rank_reason_code": c.get("rank_reason_code"),
            "friction_diagnostics": c.get("friction_diagnostics"),
            "edge_diagnostics": c.get("edge_diagnostics"),
            "ev_diagnostics": c.get("ev_diagnostics"),
            "expected_net_profit_usd": (c.get("ev_diagnostics") or {}).get("expected_net_profit_usd"),
            "edge_to_required_ratio": (c.get("ev_diagnostics") or {}).get("edge_to_required_ratio"),
            "ev_gate_passed": (c.get("ev_diagnostics") or {}).get("ev_gate_passed"),
            "ev_blocker_code": (c.get("ev_diagnostics") or {}).get("ev_blocker_code"),
            "risk_pct": (c.get("risk") or {}).get("risk_pct"),
            "target_notional": (c.get("risk") or {}).get("target_notional"),
            "tradable": c.get("tradable"), "reason": c.get("rank_reason"),
            "edge_source": c.get("edge_source"), "edge_samples": c.get("edge_samples"),
            "required_edge_pct": c.get("required_edge_pct"),
            "portfolio_variance": None,
            "config_hash": c.get("config_hash"),
            "regime_v3": c.get("regime_v3"),
            "regime_reason": c.get("regime_reason"),
            "regime_risk_mult": c.get("regime_risk_mult"),
            "cluster_regime_mult": c.get("cluster_regime_mult"),
            "top_sectors": c.get("top_sectors"),
            "catalyst_type": c.get("catalyst_type"),
            "catalyst_score_shadow": c.get("catalyst_score_shadow"),
            "catalyst_confirmed": c.get("catalyst_confirmed"),
        } for c in ranked_candidates[:25]]

    no_buy_diag["top_ranked"] = _ranking_rows()[:10]
    for c in ranked_candidates:
        if not c.get("tradable"):
            _bump(no_buy_diag["skip_reason_counts"], c.get("rank_reason_code") or c.get("rank_reason"))
            if len(no_buy_diag["top_ranked_rejections"]) < 10:
                rec = c.get("rec") or {}
                risk = c.get("risk") or {}
                no_buy_diag["top_ranked_rejections"].append({
                    "ticker": c.get("ticker"),
                    "raw_signal_label": rec.get("raw_signal_label") or rec.get("signal"),
                    "display_signal_label": rec.get("display_signal_label"),
                    "confidence": rec.get("confidence"),
                    "rank_reason_code": c.get("rank_reason_code"),
                    "rank_reason": c.get("rank_reason"),
                    "original_rank_reason": c.get("original_rank_reason") or c.get("rank_reason"),
                    "gross_edge_pct": c.get("gross_edge_pct"),
                    "friction_pct": (c.get("friction") or {}).get("total_pct"),
                    "friction_diagnostics": c.get("friction_diagnostics"),
                    "edge_diagnostics": c.get("edge_diagnostics"),
                    "ev_diagnostics": c.get("ev_diagnostics"),
                    "net_edge_pct": c.get("net_edge_pct"),
                    "required_edge_pct": c.get("required_edge_pct"),
                    "edge_source": c.get("edge_source"),
                    "edge_samples": c.get("edge_samples"),
                    "risk": risk,
                    "target_notional": risk.get("target_notional"),
                })

    scan_taken = 0
    total_taken = 0
    max_cycle_buys = int(risk_cfg.get("max_new_buys_per_tick", BOT_MAX_BUYS))
    max_cycle_buys = max(0, min(max_cycle_buys, max_buys_today - buys_today))
    if degraded_restricted_mode:
        degraded_day_room = max(0, degraded_max_buys_today - degraded_buys_today)
        max_cycle_buys = min(max_cycle_buys, degraded_max_buys_per_tick, degraded_day_room)

    def _collapse_skip(cand, reason, skip_stage, original_reason=None):
        rec = cand.get("rec") or {}
        risk = cand.get("risk") or {}
        collapse_diag.append({
            "ticker": cand.get("ticker"),
            "reason": reason,
            "rank_reason_code": cand.get("rank_reason_code"),
            "gross_edge_pct": cand.get("gross_edge_pct"),
            "net_edge_pct": cand.get("net_edge_pct"),
            "required_edge_pct": cand.get("required_edge_pct"),
            "friction_pct": (cand.get("friction") or {}).get("total_pct"),
            "target_notional": risk.get("target_notional"),
            "risk_pct": risk.get("risk_pct"),
            "raw_signal": rec.get("raw_signal_label") or rec.get("signal"),
            "display_signal": rec.get("display_signal_label") or rec.get("signal"),
            "confidence": rec.get("confidence"),
            "original_reason": original_reason,
            "skip_stage": skip_stage,
        })

    def _log_degraded_decision(cand, event, reason, final_size=None, gross_after=None):
        rec = cand.get("rec") or {}
        risk = cand.get("risk") or {}
        _log_bot_event(
            event,
            ticker=cand.get("ticker"),
            confidence=rec.get("confidence"),
            display_signal_label=rec.get("display_signal_label"),
            trade_bucket=cand.get("trade_bucket"),
            eligibility_result=event,
            rejection_reason=reason if event.endswith("REJECTED") else None,
            original_rank_reason=cand.get("original_rank_reason"),
            allowed_reason=reason if event.endswith("ALLOWED") else None,
            original_size=risk.get("pre_mode_target_notional", risk.get("target_notional")),
            degraded_size_mult=mode_cfg.get("degraded_size_mult", 0.90),
            degraded_use_standard_gates_for_testing=mode_info.get("degraded_use_standard_gates_for_testing"),
            degraded_standard_gates_active=mode_info.get("degraded_standard_gates_active"),
            degraded_gate_policy=mode_info.get("degraded_gate_policy"),
            effective_size_mult=mode_info.get("effective_size_mult"),
            effective_min_buy_confidence=mode_info.get("effective_min_buy_confidence"),
            normal_ev_gates_required=mode_info.get("normal_ev_gates_required"),
            normal_risk_caps_required=mode_info.get("normal_risk_caps_required"),
            fresh_quote_required=mode_info.get("fresh_quote_required"),
            final_size=final_size if final_size is not None else risk.get("target_notional"),
            gross_exposure_after_trade=gross_after,
        )

    def _degraded_reject_reason(cand, s, taken_this_tick):
        rec = cand.get("rec") or {}
        warnings = cand.get("warnings") or []
        risk_penalties = (cand.get("risk") or {}).get("size_penalties") or []
        mode_cfg_local = mode_cfg or {}
        if mode_info.get("degraded_standard_gates_active"):
            return None
        if (
            mode_cfg_local.get("degraded_require_fresh_quote", True)
            and not _quote_execution_status(s).get("execution_trusted")
        ):
            return "DEGRADED_STALE_QUOTE_BLOCKED"
        if mode_cfg_local.get("degraded_require_buy_candidate", True) and not is_execution_candidate(
            rec,
            cfg,
            degraded_mode_active=True,
        ):
            return "DEGRADED_NOT_BUY_CANDIDATE"
        if (
            rec.get("candidate_type") != "weak_paper_test_candidate"
            and float(rec.get("confidence", 0) or 0) < degraded_min_confidence
        ):
            return "DEGRADED_CONFIDENCE_BELOW_MIN_BUY"
        if mode_cfg_local.get("degraded_block_low_volume_penalty", False):
            if "LOW_VOLUME_PENALTY_ONLY" in warnings or "LOW_VOLUME_PENALTY_ONLY" in risk_penalties:
                return "DEGRADED_LOW_VOLUME_BLOCKED"
        if mode_cfg_local.get("degraded_block_confidence_prior", False):
            if cand.get("edge_source") == "confidence_prior":
                return "DEGRADED_CONFIDENCE_PRIOR_BLOCKED"
        if mode_cfg_local.get("degraded_block_pyramiding", False):
            if cand.get("ticker") in b.get("holdings", {}):
                return "DEGRADED_PYRAMIDING_BLOCKED"
        if mode_cfg_local.get("degraded_block_experimental", False):
            if cand.get("trade_bucket") == "experimental" or "EXPERIMENTAL_SIZE_REDUCED" in risk_penalties:
                return "DEGRADED_EXPERIMENTAL_BLOCKED"
        if taken_this_tick >= degraded_max_buys_per_tick:
            return "DEGRADED_MAX_BUYS_PER_TICK"
        if degraded_buys_today + taken_this_tick >= degraded_max_buys_today:
            return "DEGRADED_MAX_BUYS_PER_DAY"
        return None

    for cand in ranked_candidates:
        t = cand["ticker"]
        s = sigs.get(t) or {}
        rec = cand["rec"]
        conf = rec.get("confidence", 0)
        if total_taken >= max_cycle_buys:
            reason = "MAX_BUYS_PER_TICK"
            if degraded_restricted_mode:
                reason = ("DEGRADED_MAX_BUYS_PER_DAY" if degraded_buys_today >= degraded_max_buys_today
                          else "DEGRADED_MAX_BUYS_PER_TICK")
                _bump(no_buy_diag["degraded_reject_counts"], reason)
                _log_degraded_decision(cand, "DEGRADED_MODE_ENTRY_REJECTED", reason)
            _bump(no_buy_diag["skip_reason_counts"], reason)
            _record_candidate_observation(b, cand, "skipped", reason,
                                          now, regime_kind)
            continue
        if cand["source"] == "scan":
            if scan_taken >= max_scan_buys_per_tick:
                reason = "MAX_SCAN_BUYS_PER_TICK"
                _bump(no_buy_diag["skip_reason_counts"], reason)
                _record_candidate_observation(b, cand, "skipped", reason,
                                              now, regime_kind)
                _collapse_skip(cand, reason, "scan_frequency")
                continue
            if scan_buys_today + scan_taken >= max_scan_buys_today:
                reason = "MAX_SCAN_BUYS_PER_DAY"
                _bump(no_buy_diag["skip_reason_counts"], reason)
                _record_candidate_observation(b, cand, "skipped", reason,
                                              now, regime_kind)
                _collapse_skip(cand, reason, "scan_frequency")
                continue
        require_normal_ev = (
            not mode_info["degraded_mode_active"]
            or mode_info["degraded_standard_gates_active"]
            or mode_cfg.get("degraded_require_normal_ev_gates", True)
        )
        if not cand.get("tradable") and require_normal_ev:
            original_reason = cand.get("rank_reason")
            reason = cand.get("rank_reason_code") or original_reason
            if degraded_restricted_mode:
                reason = ("DEGRADED_FINAL_SIZE_TOO_SMALL"
                          if cand.get("rank_reason_code") == "FINAL_SIZE_TOO_SMALL"
                          else "DEGRADED_NORMAL_EV_GATE_FAILED")
                cand["original_rank_reason"] = original_reason
                _bump(no_buy_diag["degraded_reject_counts"], reason)
                _log_degraded_decision(cand, "DEGRADED_MODE_ENTRY_REJECTED", reason)
            _bump(no_buy_diag["skip_reason_counts"], reason)
            _record_candidate_observation(b, cand, "skipped", reason,
                                          now, regime_kind)
            _collapse_skip(cand, reason, "rank", original_reason=original_reason)
            continue
        if b["cash"] - cash_floor < min_position_usd:
            break
        existing = b["holdings"].get(t)
        if mode_info["degraded_mode_active"]:
            reject = _degraded_reject_reason(cand, s, total_taken)
            if reject:
                _bump(no_buy_diag["degraded_reject_counts"], reject)
                _bump(no_buy_diag["skip_reason_counts"], reject)
                _log_degraded_decision(cand, "DEGRADED_MODE_ENTRY_REJECTED", reject)
                _record_candidate_observation(b, cand, "skipped", reject, now, regime_kind)
                _collapse_skip(cand, reject, "degraded_eligibility")
                continue
        if existing:
            exist_pnl = ((cand["price"] - existing["avg_cost"]) / existing["avg_cost"] * 100
                         if existing.get("avg_cost") else 0)
            last_buy = existing.get("last_buy_ts", existing.get("entry_ts", 0))
            since_last_min = (now - last_buy) / 60 if last_buy else 1e9
            pyramid_enabled = bool(risk_cfg.get("pyramiding_enabled", True))
            pyramid_min_profit = float(risk_cfg.get("pyramiding_min_profit_pct", 1.0))
            pyramid_min_gap = float(risk_cfg.get("pyramiding_min_minutes_between_adds", 10))
            pyramid_max_loss = float(risk_cfg.get("pyramiding_max_loss_allowed_pct", -3.0))
            display_signal = rec.get("display_signal_label")
            strong_add = (
                rec.get("cls") == "strong-buy"
                or display_signal == "STRONG_BUY_CANDIDATE"
            )
            allow_strong_flat = bool(risk_cfg.get("pyramiding_allow_flat_add_for_strong_buy", True))
            allow_strong_losing = bool(risk_cfg.get("pyramiding_allow_losing_adds", True))
            pyramid_allowed = False
            pyramid_reason = None
            if not pyramid_enabled:
                pyramid_reason = "PYRAMIDING_DISABLED"
            elif since_last_min < pyramid_min_gap:
                pyramid_reason = (
                    f"PYRAMIDING_COOLDOWN:{since_last_min:.0f}m_LT_{pyramid_min_gap:.0f}m"
                )
            elif exist_pnl >= pyramid_min_profit:
                pyramid_allowed = True
            elif strong_add and allow_strong_flat and exist_pnl >= 0:
                pyramid_allowed = True
            elif strong_add and allow_strong_losing and exist_pnl >= pyramid_max_loss:
                pyramid_allowed = True
            elif exist_pnl < 0:
                pyramid_reason = (
                    f"PYRAMIDING_AVERAGING_DOWN_BLOCKED:{exist_pnl:+.1f}%"
                )
            else:
                pyramid_reason = (
                    f"PYRAMIDING_PROFIT_BELOW_MIN:{exist_pnl:+.1f}%_LT_{pyramid_min_profit:.1f}%"
                )
            if not pyramid_allowed:
                reason = pyramid_reason or "PYRAMIDING_BLOCKED"
                _bump(no_buy_diag["skip_reason_counts"], reason)
                _record_candidate_observation(b, cand, "skipped", reason, now, regime_kind)
                _collapse_skip(cand, reason, "pyramiding")
                continue
            sizing_note = f"pyramid {exist_pnl:+.1f}% ({since_last_min:.0f}m since add)"
        else:
            if len(b["holdings"]) >= max_positions:
                reason = "MAX_POSITIONS_REACHED"
                _bump(no_buy_diag["skip_reason_counts"], reason)
                _record_candidate_observation(b, cand, "skipped", reason, now, regime_kind)
                continue
            sizing_note = "new risk-budget"

        spend = min(cand["risk"]["target_notional"], b["cash"] - cash_floor)
        pt = _portfolio_total()
        existing_val = (b["holdings"].get(t, {}).get("shares", 0) *
                        cand["price"])
        spend = min(spend, max(0, pt * MAX_POS_PCT - existing_val))
        gross_room = max(0, pt * MAX_GROSS_EXPOSURE_PCT - exposure_value)
        spend = min(spend, gross_room)
        sec = get_sector(t)
        sec_current = sector_value.get(sec, 0) if sec else 0
        if sec:
            spend = min(spend, max(0, pt * MAX_SECTOR_PCT - sec_current))
        grp = get_corr_group(t)
        if grp:
            grp_current = corr_value.get(grp, 0)
            spend = min(spend, max(0, pt * MAX_CORR_GROUP_PCT - grp_current))
        if (degraded_restricted_mode
                and mode_cfg.get("degraded_require_normal_risk_caps", True)):
            degraded_pos_room = max(0, pt * degraded_max_position_pct - existing_val)
            if degraded_pos_room < min_position_usd:
                reason = "DEGRADED_MAX_POSITION_CAP"
                _bump(no_buy_diag["degraded_reject_counts"], reason)
                _bump(no_buy_diag["skip_reason_counts"], reason)
                _log_degraded_decision(cand, "DEGRADED_MODE_ENTRY_REJECTED", reason,
                                       final_size=0, gross_after=no_buy_diag.get("gross_exposure_pct"))
                _record_candidate_observation(b, cand, "skipped", reason, now, regime_kind)
                _collapse_skip(cand, reason, "degraded_position_cap")
                continue
            spend = min(spend, degraded_pos_room)
            degraded_gross_room = max(0, pt * degraded_max_gross_exposure_pct - exposure_value)
            if degraded_gross_room < min_position_usd:
                reason = "DEGRADED_GROSS_EXPOSURE_CAP"
                _bump(no_buy_diag["degraded_reject_counts"], reason)
                _bump(no_buy_diag["skip_reason_counts"], reason)
                _log_degraded_decision(cand, "DEGRADED_MODE_ENTRY_REJECTED", reason,
                                       final_size=0, gross_after=no_buy_diag.get("gross_exposure_pct"))
                _record_candidate_observation(b, cand, "skipped", reason, now, regime_kind)
                _collapse_skip(cand, reason, "degraded_gross_exposure_cap")
                continue
            spend = min(spend, degraded_gross_room)
        if spend < min_position_usd:
            original_reason = (f"Risk/cap spend ${spend:.0f} below ${min_position_usd:.0f} floor "
                               f"(target ${cand['risk']['target_notional']:.0f})")
            reason = original_reason
            cand["rank_reason_code"] = "FINAL_SIZE_TOO_SMALL"
            if degraded_restricted_mode:
                cand["original_rank_reason"] = original_reason
                reason = "DEGRADED_FINAL_SIZE_TOO_SMALL"
                _bump(no_buy_diag["degraded_reject_counts"], reason)
                _log_degraded_decision(cand, "DEGRADED_MODE_ENTRY_REJECTED", reason,
                                       final_size=spend,
                                       gross_after=no_buy_diag.get("gross_exposure_pct"))
            _bump(no_buy_diag["skip_reason_counts"], reason)
            _record_candidate_observation(b, cand, "skipped", reason, now, regime_kind)
            _collapse_skip(cand, reason, "sizing_floor", original_reason=original_reason)
            continue

        effective_buy = cand["price"]
        sh = round(spend / effective_buy, 4) if effective_buy else 0
        if sh <= 0:
            continue
        cost = sh * effective_buy
        b["total_costs_usd"] = round(b.get("total_costs_usd", 0) + COMMISSION_PER_TRADE, 2)
        b["cash"] -= (cost + COMMISSION_PER_TRADE)
        h = b["holdings"].get(t, {"shares": 0, "avg_cost": 0})
        ns = h["shares"] + sh
        na = (h["shares"] * h["avg_cost"] + cost) / ns
        ctx_at_entry = cand.get("ctx") or {}
        vol_regime = classify_vol_regime(ctx_at_entry)
        entry_snapshot = {
            "rsi": ctx_at_entry.get("rsi"),
            "atr_pct": ctx_at_entry.get("atr_pct"),
            "bb_width_pct": ctx_at_entry.get("bb_width_pct"),
            "macd_hist": ctx_at_entry.get("macd_hist"),
            "vol_ratio": ctx_at_entry.get("vol_ratio"),
            "adx": ctx_at_entry.get("adx"),
            "dist_from_high_pct": ctx_at_entry.get("dist_from_high_pct"),
            "is_dip": ctx_at_entry.get("is_dip"),
            "vol_regime": vol_regime,
            "vix": vix_data.get("vix"),
            "vix_regime": vix_label,
            "market_regime": regime_label,
            "trading_mode": mode_info["trading_mode"],
            "degraded_mode_active": mode_info["degraded_mode_active"],
            "degraded_mode_reason": mode_info["degraded_mode_reason"],
            "degraded_gate_policy": mode_info["degraded_gate_policy"],
            "effective_size_mult": mode_info["effective_size_mult"],
            "effective_min_buy_confidence": mode_info["effective_min_buy_confidence"],
            "sector": sec,
            "confidence": conf,
            "signal": rec.get("signal"),
            "entry_cluster": cand.get("cluster"),
            "gross_edge_pct": cand.get("gross_edge_pct"),
            "net_edge_pct": cand.get("net_edge_pct"),
            "friction_pct": (cand.get("friction") or {}).get("total_pct"),
            "risk_pct": (cand.get("risk") or {}).get("risk_pct"),
            "source": cand.get("source"),
            "portfolio_variance": None,
            "portfolio_variance_mode": "disabled_for_paper_execution",
            "config_hash": cfg_hash,
            "regime_v3": cand.get("regime_v3"),
            "regime_v3_reason": cand.get("regime_reason"),
            "regime_risk_mult": cand.get("regime_risk_mult"),
            "cluster_regime_mult": cand.get("cluster_regime_mult"),
            "top_sectors": cand.get("top_sectors"),
            "catalyst_type": cand.get("catalyst_type"),
            "catalyst_score_shadow": cand.get("catalyst_score_shadow"),
            "catalyst_confirmed": cand.get("catalyst_confirmed"),
        }
        prior = b["holdings"].get(t, {})
        existing_entry_cls = prior.get("entry_signal_cls")
        entry_cls_to_store = existing_entry_cls or rec.get("cls")
        trend_start = prior.get("trend_start_ts") or int(now)
        b["holdings"][t] = {
            "shares": round(ns, 4), "avg_cost": round(na, 4),
            "commission_invested": round(
                float(prior.get("commission_invested", 0) or 0) + COMMISSION_PER_TRADE,
                4,
            ),
            "peak": round(max(prior.get("peak", cand["price"]), cand["price"]), 4),
            "trough": round(min(prior.get("trough", cand["price"]), cand["price"]), 4),
            # B5: carry the ratchet peak across pyramid adds so the profit lock does not
            # reset to 0 on every add (see _carried_peak_pnl_pct).
            "peak_pnl_pct": _carried_peak_pnl_pct(prior.get("peak"), cand["price"], na),
            "entry_categories": prior.get("entry_categories") or rec.get("categories", {}),
            "entry_cluster": prior.get("entry_cluster") or cand.get("cluster"),
            "entry_ts": prior.get("entry_ts", int(now)),
            "last_buy_ts": int(now),
            "entry_snapshot": prior.get("entry_snapshot") or entry_snapshot,
            "entry_expected_edge_pct": cand.get("gross_edge_pct"),
            "entry_net_edge_pct": cand.get("net_edge_pct"),
            "entry_atr_pct": ctx_at_entry.get("atr_pct"),
            "entry_regime": regime_kind,
            "entry_reasons": rec.get("reasons", []),
            "entry_friction_pct": (cand.get("friction") or {}).get("total_pct"),
            "entry_risk_pct": (cand.get("risk") or {}).get("risk_pct"),
            "entry_source": cand.get("source"),
            "portfolio_variance": None,
            "portfolio_variance_mode": "disabled_for_paper_execution",
            "config_hash": cfg_hash,
            "regime_v3": cand.get("regime_v3"),
            "catalyst_type": cand.get("catalyst_type"),
            "catalyst_score_shadow": cand.get("catalyst_score_shadow"),
            "catalyst_confirmed": cand.get("catalyst_confirmed"),
            "entry_signal_cls": entry_cls_to_store,
            "trend_start_ts": trend_start,
            "partial_taken":   prior.get("partial_taken", False),
            "degrade_trimmed": prior.get("degrade_trimmed", False),
        }
        if sec:
            sector_value[sec] = sec_current + cost
        if grp:
            corr_value[grp] = corr_value.get(grp, 0) + cost
        buy_reason = (
            f"{'New position' if h['shares'] == 0 else 'Adding'} ({sizing_note}) - "
            f"{rec.get('signal')} @ {conf}% conf, EV {cand['net_edge_pct']:+.2f}% "
            f"after friction {(cand.get('friction') or {}).get('total_pct', 0):.2f}%, "
            f"risk {cand['risk']['risk_pct']:.2f}%/${cand['risk']['risk_dollars']:.0f}, "
            f"stop {cand['risk']['stop_distance_pct']:.1f}%, source={cand['source']}, "
            f"cluster={cand['cluster']}, regime={regime_label}, VIX={vix_label}, sector {sec or 'unknown'}. "
            f"V3={cand.get('regime_v3')} r×{cand.get('regime_risk_mult'):.2f}/c×{cand.get('cluster_regime_mult'):.2f}, "
            f"catalyst={cand.get('catalyst_type')} confirmed={cand.get('catalyst_confirmed')}, cfg={cfg_hash}. "
            "Portfolio variance disabled for paper execution."
        )
        if mode_info["proxy_mode_active"]:
            buy_reason = (
                f"PROXY MODE: real VIX unavailable; using SPY realized-vol proxy "
                f"{_fmt_vix_value(vix_data)}, size x{mode_info['mode_size_mult']:.2f}. "
                + buy_reason
            )
        elif mode_info["degraded_mode_active"]:
            blocks_text = ",".join(data_health_blocks) or "unknown"
            if mode_info["degraded_standard_gates_active"]:
                prefix = (
                    f"DEGRADED MODE ENTRY: standard-gates test active; "
                    f"data health: {blocks_text}; size x{mode_info['mode_size_mult']:.2f}. "
                )
            else:
                prefix = (
                    f"DEGRADED MODE ENTRY: SPY/VIX data missing; restricted paper-only "
                    f"fallback active, size x{mode_info['mode_size_mult']:.2f}. "
                )
            buy_reason = prefix + buy_reason
        if mode_info["degraded_mode_active"]:
            gross_after = round((exposure_value + cost) / pt, 4) if pt else None
            _log_degraded_decision(cand, "DEGRADED_MODE_ENTRY_ALLOWED",
                                   "DEGRADED_MODE_ENTRY_ALLOWED",
                                   final_size=round(cost, 2),
                                   gross_after=gross_after)
        _record_trade(b, "BUY", t, sh, cand["price"], rec, cand.get("arts"), buy_reason)
        if b.get("history"):
            b["history"][0].update({
                "trading_mode": mode_info["trading_mode"],
                "normal_mode_active": mode_info["normal_mode_active"],
                "proxy_mode_active": mode_info["proxy_mode_active"],
                "degraded_mode_active": mode_info["degraded_mode_active"],
                "degraded_mode_reason": mode_info["degraded_mode_reason"],
                "degraded_gate_policy": mode_info["degraded_gate_policy"],
                "effective_size_mult": mode_info["effective_size_mult"],
                "effective_min_buy_confidence": mode_info["effective_min_buy_confidence"],
                "source": cand.get("source"),
                "candidate_type": cand.get("candidate_type"),
                "final_spend": round(cost, 2),
                "final_size_blocker_stage": None,
            })
        _record_candidate_observation(b, cand, "executed", buy_reason, now, regime_kind)
        traded = True
        no_buy_diag["buys_executed"] = int(no_buy_diag.get("buys_executed", 0) or 0) + 1
        total_taken += 1
        exposure_value += cost
        if cand["source"] == "scan":
            scan_taken += 1
            bought.append(f"BUY {t} (scan)")
        else:
            bought.append(f"BUY {t}")

    build_extra_ticker_suggestions(
        b,
        sigs,
        ranked_candidates,
        regime,
        int(now),
        config=cfg,
        recent_sells=recent_sells,
    )
    b["last_portfolio_variance_checks"] = []
    b["last_candidate_rankings"] = _ranking_rows()
    top_rejected = []
    for row in no_buy_diag.get("top_buyable_rejects") or []:
        if len(top_rejected) >= 3:
            break
        top_rejected.append({
            "symbol": row.get("ticker"),
            "decision_stage": "pre_candidate",
            "blocker_code": row.get("rejection_reason"),
            "blocker_detail": row.get("rejection_reason"),
            "display_signal": row.get("display_signal_label"),
            "confidence": row.get("confidence"),
        })
    for row in no_buy_diag.get("top_ranked_rejections") or []:
        if len(top_rejected) >= 3:
            break
        top_rejected.append({
            "symbol": row.get("ticker"),
            "decision_stage": "ev_or_size",
            "blocker_code": row.get("rank_reason_code"),
            "blocker_detail": row.get("rank_reason"),
            "display_signal": row.get("display_signal_label"),
            "confidence": row.get("confidence"),
            "expected_net_profit_usd": (row.get("ev_diagnostics") or {}).get("expected_net_profit_usd"),
            "net_edge_pct": row.get("net_edge_pct"),
            "target_notional": row.get("target_notional"),
        })
    for row in collapse_diag:
        if len(top_rejected) >= 3:
            break
        top_rejected.append({
            "symbol": row.get("ticker"),
            "decision_stage": row.get("skip_stage") or "final_execution",
            "blocker_code": row.get("rank_reason_code") or row.get("reason"),
            "blocker_detail": row.get("reason"),
            "display_signal": row.get("display_signal"),
            "confidence": row.get("confidence"),
            "net_edge_pct": row.get("net_edge_pct"),
            "target_notional": row.get("target_notional"),
        })
    no_buy_diag["top_rejected_candidates"] = top_rejected[:3]

    if rotation_actions:
        bought = rotation_actions + bought
    if bought:
        last_action = ", ".join(bought) if not last_action else last_action + ", " + ", ".join(bought)

    # ── HOLD / SKIP records ───────────────────────────────────────────────
    def _tag_last_decision_mode():
        if b.get("history"):
            b["history"][0].update({
                "trading_mode": mode_info["trading_mode"],
                "normal_mode_active": mode_info["normal_mode_active"],
                "proxy_mode_active": mode_info["proxy_mode_active"],
                "degraded_mode_active": mode_info["degraded_mode_active"],
                "degraded_mode_reason": mode_info["degraded_mode_reason"],
                "degraded_gate_policy": mode_info["degraded_gate_policy"],
                "effective_size_mult": mode_info["effective_size_mult"],
                "effective_min_buy_confidence": mode_info["effective_min_buy_confidence"],
            })

    for cd in collapse_diag[:5]:
        if "reason" in cd:
            _record_skip(
                b,
                cd["ticker"],
                cd["reason"],
                cd.get("raw_signal"),
                cd.get("confidence"),
                display_signal=cd.get("display_signal"),
                original_reason=cd.get("original_reason"),
                skip_stage=cd.get("skip_stage"),
                rank_reason_code=cd.get("rank_reason_code"),
                gross_edge_pct=cd.get("gross_edge_pct"),
                net_edge_pct=cd.get("net_edge_pct"),
                required_edge_pct=cd.get("required_edge_pct"),
                friction_pct=cd.get("friction_pct"),
                target_notional=cd.get("target_notional"),
                risk_pct=cd.get("risk_pct"),
            )
            _tag_last_decision_mode()
            continue
        ms = cd["mults"]
        ec = ms.get("env_components", {})
        skip_reason = (
            f"Sizing collapsed to ${cd['spend']:.0f} (below ${cd['floor']:.0f} floor). "
            f"Stack (5-factor): weight={ms['weight']} × pos={ms['pos_size']} × "
            f"win-rate={ms['win_rate']} × env={ms['env']} × ticker={ms['ticker']} = "
            f"{cd['combined_mult']:.4f} of ${cd['spendable']:.0f} spendable "
            f"[env = regime {ec.get('regime')} × market-vol {ec.get('market_vol')} × "
            f"streak {ec.get('streak')}]"
        )
        _record_skip(
            b,
            cd["ticker"],
            skip_reason,
            cd.get("raw_signal") or cd.get("signal"),
            cd.get("confidence"),
            display_signal=cd.get("display_signal"),
            original_reason=cd.get("original_reason") or skip_reason,
            skip_stage=cd.get("skip_stage") or "sizing_floor",
            rank_reason_code=cd.get("rank_reason_code"),
            gross_edge_pct=cd.get("gross_edge_pct"),
            net_edge_pct=cd.get("net_edge_pct"),
            required_edge_pct=cd.get("required_edge_pct"),
            friction_pct=cd.get("friction_pct"),
            target_notional=cd.get("target_notional"),
            risk_pct=cd.get("risk_pct"),
        )
        _tag_last_decision_mode()
    if not traded and not collapse_diag:
        reasons = []
        has_buys = any(s["rec"]["cls"] in ("buy", "strong-buy") for s in sigs.values())
        # Entry 027 — surface core-data blockers (missing daily history / invalid
        # or missing quotes) in the visible reason instead of hiding them behind
        # the generic "no BUY signals across watchlist" line.
        core_blocker_reason = _core_data_blocker_reason(no_buy_diag)
        if mode_info["proxy_mode_active"]:
            reasons.append(
                f"PROXY MODE: real VIX unavailable; using SPY realized-vol proxy "
                f"{_fmt_vix_value(vix_data)}, size x{mode_info['mode_size_mult']:.2f}"
            )
        if no_buy_diag.get("paper_trading_locked") and no_buy_diag.get("paper_lock_reason"):
            reasons.append(f"paper trading locked: {no_buy_diag.get('paper_lock_reason')}")
        if no_buy_diag.get("data_health_blocks"):
            blocks_text = ",".join(no_buy_diag.get("data_health_blocks") or [])
            if mode_info["degraded_mode_active"]:
                if mode_info["degraded_standard_gates_active"]:
                    reasons.append(
                        f"DEGRADED MODE: standard-gates test active; "
                        f"data health: {blocks_text}; size x{mode_info['mode_size_mult']:.2f}"
                    )
                else:
                    reasons.append(
                        f"DEGRADED MODE: restricted paper-only fallback active, "
                        f"size x{mode_info['mode_size_mult']:.2f}; data health: {blocks_text}"
                    )
            else:
                reasons.append("buys blocked by data health: " + blocks_text)
        elif not regime_allow_buys:
            if regime_v3_label == "panic" and not b.get("paper_debug_override"):
                reasons.append("buys halted (V3 panic hard no-buy)")
            else:
                reasons.append(f"buys halted (VIX={vix_label} {_fmt_vix_value(vix_data)})")
        elif buys_paused:
            reasons.append(f"drawdown circuit-breaker active ({drawdown_pct:.1f}% from peak)")
        elif not daily_buy_room:
            if degraded_restricted_mode:
                reasons.append(f"degraded daily buy cap reached ({degraded_buys_today}/{degraded_max_buys_today})")
            else:
                reasons.append(f"daily buy cap reached ({buys_today}/{max_buys_today})")
        elif not exposure_room:
            if degraded_restricted_mode:
                reasons.append(
                    f"degraded gross exposure cap reached ({no_buy_diag.get('gross_exposure_pct'):.0%}/"
                    f"{degraded_max_gross_exposure_pct:.0%})"
                )
            else:
                reasons.append(f"gross exposure cap reached ({no_buy_diag.get('gross_exposure_pct'):.0%})")
        elif core_blocker_reason:
            reasons.append(core_blocker_reason)
        elif not has_buys:
            reasons.append("no BUY signals across watchlist")
        elif b["cash"] - cash_floor < min_position_usd:
            reasons.append(f"cash too low for a ${min_position_usd:.0f} min position "
                           f"(${b['cash']:.0f}, reserve floor ${cash_floor:.0f})")
        if stale_scan_skipped:
            reasons.append(f"scan data stale ({int(scan_age/60)} min old) — skipped outside-watchlist")
        reasons.append(f"regime={regime_label} · VIX={vix_label}({_fmt_vix_value(vix_data)})")
        reasons_str = " · ".join(reasons) if reasons else "signals not strong enough to act"
        _record_hold(b, reasons_str, sigs)
        _tag_last_decision_mode()

    # Round-8: cap history with SEPARATE budgets per action — 100 BUY + 100 SELL +
    # 20 HOLD/SKIP, ordered newest-first. History is stored newest-first
    # (history.insert(0, ...)), so keeping the first 100 of each keeps the newest and
    # drops the oldest "from the bottom". Feeds the split BUY/SELL dashboard windows.
    history = b.get("history", [])
    buys_kept, sells_kept, decisions_kept = [], [], []
    for entry in history:
        a = entry.get("action")
        if a == "BUY":
            if len(buys_kept) < 100:
                buys_kept.append(entry)
        elif a == "SELL":
            if len(sells_kept) < 100:
                sells_kept.append(entry)
        elif a in ("HOLD", "SKIP"):
            if len(decisions_kept) < 20:
                decisions_kept.append(entry)
    # Interleave by ts so the heartbeat (last_decision_ago_min) still works
    merged = buys_kept + sells_kept + decisions_kept
    merged.sort(key=lambda e: e.get("ts", 0), reverse=True)
    b["history"] = merged

    b["last_trade"] = now

    # Round-4 Bug Fix #4: track today's open-of-day equity so the dashboard
    # can show today's % change instead of all-time drawdown.
    # "Today" = market-open date in ET. First bot tick of a new market day
    # records the open equity; resets every day.
    try:
        from zoneinfo import ZoneInfo
        et_today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        et_today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # We compute end_total below; defer the today-open capture to after that

    # Round-2 fix #1: scheduler-end snapshot. With Round-3 #1 + Round-4 stale
    # guard, this becomes a no-op after-hours or when prices are bad.
    try:
        end_total = b["cash"] + sum(
            h2["shares"] * sigs[tt]["price"]
            for tt, h2 in b["holdings"].items()
            if tt in sigs and sigs[tt].get("price", 0) > 0 and not sigs[tt].get("stale")
        )
        # If any held ticker is stale or untracked, fall back to avg_cost for that one
        for tt, h2 in b["holdings"].items():
            sig_t = sigs.get(tt) or {}
            if not sig_t or sig_t.get("price", 0) <= 0 or sig_t.get("stale"):
                end_total += h2["shares"] * h2.get("avg_cost", 0)
        # Round-4 Bug #4: capture today-open equity on first tick of a new market day
        if is_market_open() and b.get("today_open_date") != et_today:
            b["today_open_date"] = et_today
            b["today_open_equity"] = round(end_total, 2)
        if traded:
            b["_force_full_snapshot"] = True
        record_equity_snapshot(b, end_total)
    except Exception as e:
        try: print(f"[run_bot] equity snapshot failed: {type(e).__name__}: {e}")
        except Exception: pass
    _persist_no_buy_diag(b, no_buy_diag, traded=traded)
    _BOT_STATUS.update({"last_run_ts": int(now), "last_action": last_action or "hold",
                        "last_traded": bool(traded)})
    return b, traded, last_action


def bot_state():
    """Returns (b, rows, total, pnl, pnl_pct). READ-ONLY valuation snapshot.

    Round-8 Bug #2: this used to mutate `b` (record_equity_snapshot) and save_bot(b)
    OUTSIDE _bot_run_lock. A dashboard poll overlapping a trading pass could read old
    state and write it back, silently erasing a buy/sell. It no longer writes — equity
    snapshots happen inside run_bot under the lock. /health and explicit admin run
    requests are the only request paths that should trigger bot work.

    Round-4 Bug Fix #1: if get_quote returns stale/zero, fall back to avg_cost
    for valuation so the chart and stats don't spike to zero on API failure."""
    b = load_bot()
    total = b["cash"]
    rows = []
    for t, h in b["holdings"].items():
        q = get_quote(t) or {}
        p = q.get("price") or 0
        is_stale = bool(q.get("stale"))
        # Round-4: avoid $0 spikes — use avg_cost as valuation fallback
        if p <= 0 or is_stale:
            p = h["avg_cost"] or 0
        val = h["shares"] * p
        cost = h["shares"] * h["avg_cost"]
        pnl = val - cost
        total += val
        rows.append({
            "ticker": t, "shares": h["shares"], "avg_cost": h["avg_cost"],
            "price": p, "value": val, "pnl": pnl,
            "pnl_pct": (pnl / cost * 100) if cost else 0,
            "stale": is_stale,
        })
    pnl_t = total - b["starting"]
    pnl_pct = (pnl_t / b["starting"] * 100) if b["starting"] else 0
    # NOTE: intentionally NO record_equity_snapshot / save_bot here — see docstring
    # (Bug #2). This function is read-only.
    return b, rows, total, pnl_t, pnl_pct


def _render_bot_page(read_only):
    """Template context builder for /bot and /botcontrol."""
    b, holdings, total, pnl, pnl_pct = bot_state()
    # Round-3 Bug Fix #4: cost_basis = qty × avg_cost. Market Value kept.
    # Round-4 Bug Fix #1: pass stale flag through so the UI can show a marker.
    positions = [{
        "symbol": h["ticker"], "qty": h["shares"], "avg_cost": h["avg_cost"],
        "price": h["price"], "value": h["value"],
        "cost_basis": round(h["shares"] * h["avg_cost"], 2),
        "pnl": h["pnl"], "pnl_pct": h["pnl_pct"],
        "sector": get_sector(h["ticker"]),
        "stale": h.get("stale", False),
    } for h in holdings]
    positions.sort(key=lambda p: p["value"], reverse=True)
    # Round-8: BUY and SELL go to separate side-by-side windows (cap 100 each);
    # decisions (HOLD/SKIP) keep their own window (cap 20).
    all_activity = list(b.get("history", []))
    buys      = [a for a in all_activity if a.get("action") == "BUY"]
    sells     = [a for a in all_activity if a.get("action") == "SELL"]
    decisions = [a for a in all_activity if a.get("action") in ("HOLD", "SKIP")]
    last_decision = b["history"][0] if b.get("history") else None
    last_decision_ago_min = None
    if last_decision and last_decision.get("ts"):
        last_decision_ago_min = max(0, int((time.time() - last_decision["ts"]) / 60))

    outcomes = b.get("trade_outcomes", [])
    # Round-5 extra: win/loss is ALL-TIME from monotonic counters (never resets).
    # Back-compat seeding now lives in storage.load_bot (Bug #2: this page no longer
    # persists state, so the seed has to come from the loader).
    wins = b.get("wins_total", 0)
    losses = b.get("losses_total", 0)
    win_rate = round(wins / (wins + losses) * 100, 1) if (wins + losses) else None
    # avg win/loss % from the recent window we still have (capped at 100)
    avg_win = round(sum(o["pnl_pct"] for o in outcomes if o["pnl_pct"] > 0) / max(1, sum(1 for o in outcomes if o["pnl_pct"] > 0)), 2) if any(o["pnl_pct"] > 0 for o in outcomes) else 0
    avg_loss = round(sum(o["pnl_pct"] for o in outcomes if o["pnl_pct"] <= 0) / max(1, sum(1 for o in outcomes if o["pnl_pct"] <= 0)), 2) if any(o["pnl_pct"] <= 0 for o in outcomes) else 0
    peak_equity = b.get("peak_equity", total)   # all-time (Round-5)
    drawdown_pct = round((peak_equity - total) / peak_equity * 100, 2) if peak_equity > 0 else 0

    # Round-5 #8: per-exit-reason P&L breakdown
    exit_breakdown = []
    by_reason = {}
    for o in outcomes:
        k = o.get("exit_reason") or "other"
        by_reason.setdefault(k, []).append(o["pnl_pct"])
    for k, pls in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        n = len(pls)
        w = sum(1 for x in pls if x > 0)
        exit_breakdown.append({
            "reason": k, "n": n,
            "win_rate": round(w / n * 100, 0) if n else 0,
            "avg_pnl": round(sum(pls) / n, 2) if n else 0,
        })

    sector_val = {}
    for p in positions:
        sector_val[p["sector"]] = sector_val.get(p["sector"], 0) + p["value"]
    sector_breakdown = sorted(
        [{"sector": s, "value": v, "pct": round(v / total * 100, 1) if total > 0 else 0}
         for s, v in sector_val.items()],
        key=lambda x: -x["value"]
    )

    # Round-4 Bug Fix #4: today's % change. Falls back to starting equity if
    # today's open hasn't been captured yet (first deploy / before today's tick).
    today_open = b.get("today_open_equity") or b.get("starting", 10000)
    today_pct = ((total - today_open) / today_open * 100) if today_open else 0
    no_buy = b.get("last_no_buy_diagnostics") or {}
    safe_no_buy = {
        "main_blocker": no_buy.get("main_blocker"),
        "market_open": no_buy.get("market_open"),
        "tod_ok": no_buy.get("tod_ok"),
        "buy_window_open": no_buy.get("buy_window_open"),
        "regime_allow_buys": no_buy.get("regime_allow_buys"),
        "regime_kind": no_buy.get("regime_kind"),
        "regime_source": no_buy.get("regime_source"),
        "regime_v3": no_buy.get("regime_v3"),
        "trading_mode": no_buy.get("trading_mode"),
        "normal_mode_active": no_buy.get("normal_mode_active"),
        "proxy_mode_active": no_buy.get("proxy_mode_active"),
        "degraded_mode_active": no_buy.get("degraded_mode_active"),
        "degraded_mode_reason": no_buy.get("degraded_mode_reason"),
        "min_buy_confidence": no_buy.get("min_buy_confidence"),
        "min_trade_size_effective": no_buy.get("min_trade_size_effective"),
        "degraded_size_mult": no_buy.get("degraded_size_mult"),
        "degraded_use_standard_gates_for_testing": no_buy.get("degraded_use_standard_gates_for_testing"),
        "degraded_standard_gates_active": no_buy.get("degraded_standard_gates_active"),
        "degraded_gate_policy": no_buy.get("degraded_gate_policy"),
        "effective_size_mult": no_buy.get("effective_size_mult"),
        "effective_min_buy_confidence": no_buy.get("effective_min_buy_confidence"),
        "normal_ev_gates_required": no_buy.get("normal_ev_gates_required"),
        "normal_risk_caps_required": no_buy.get("normal_risk_caps_required"),
        "fresh_quote_required": no_buy.get("fresh_quote_required"),
        "degraded_min_confidence": no_buy.get("degraded_min_confidence"),
        "degraded_reject_counts": no_buy.get("degraded_reject_counts", {}),
        "finnhub_key_configured": no_buy.get("finnhub_key_configured"),
        "fmp_key_configured": no_buy.get("fmp_key_configured"),
        "stooq_status": no_buy.get("stooq_status"),
        "degraded_buys_today": no_buy.get("degraded_buys_today", 0),
        "degraded_max_buys_today": no_buy.get("degraded_max_buys_today"),
        "degraded_gross_exposure_pct": no_buy.get("degraded_gross_exposure_pct"),
        "degraded_max_gross_exposure_pct": no_buy.get("degraded_max_gross_exposure_pct"),
        "signal_counts": no_buy.get("signal_counts", {}),
        "display_signal_counts": no_buy.get("display_signal_counts", {}),
        "raw_buy_count": no_buy.get("raw_buy_count", 0),
        "display_buy_candidate_count": no_buy.get("display_buy_candidate_count", 0),
        "history_source_counts": no_buy.get("history_source_counts", {}),
        "history_missing_count": no_buy.get("history_missing_count", 0),
        "history_fmp_fallback_count": no_buy.get("history_fmp_fallback_count", 0),
        "history_fmp_attempted_count": no_buy.get("history_fmp_attempted_count", 0),
        "history_fmp_skipped_count": no_buy.get("history_fmp_skipped_count", 0),
        "history_fmp_rate_limited_count": no_buy.get("history_fmp_rate_limited_count", 0),
        "history_fmp_global_circuit_skipped_count": no_buy.get("history_fmp_global_circuit_skipped_count", 0),
        "history_finnhub_daily_blocked_count": no_buy.get("history_finnhub_daily_blocked_count", 0),
        "history_stale_cache_count": no_buy.get("history_stale_cache_count", 0),
        "fmp_daily_global_circuit_status": no_buy.get("fmp_daily_global_circuit_status"),
        "fmp_daily_rate_limited": no_buy.get("fmp_daily_rate_limited", False),
        "fmp_daily_cooldown_remaining_sec": no_buy.get("fmp_daily_cooldown_remaining_sec", 0),
        "fmp_daily_last_429_age_sec": no_buy.get("fmp_daily_last_429_age_sec"),
        "finnhub_daily_global_circuit_status": no_buy.get("finnhub_daily_global_circuit_status"),
        "finnhub_daily_forbidden": no_buy.get("finnhub_daily_forbidden", False),
        "finnhub_daily_cooldown_remaining_sec": no_buy.get("finnhub_daily_cooldown_remaining_sec", 0),
        "ticker_signal_debug": (no_buy.get("ticker_signal_debug") or [])[:PA_TICKERS_PER_BOT_RUN],
        "stale_ticker_count": no_buy.get("stale_ticker_count", 0),
        "stale_tickers": no_buy.get("stale_tickers", []),
        "stale_positions": no_buy.get("stale_positions", []),
        "risk_unmanaged_positions": no_buy.get("risk_unmanaged_positions", []),
        "api_circuit_breakers": no_buy.get("api_circuit_breakers", {}),
        "provider_health_status": no_buy.get("provider_health_status"),
        "rate_limit_recent": no_buy.get("rate_limit_recent", False),
        "candidate_pool_count": no_buy.get("candidate_pool_count", 0),
        "ranked_count": no_buy.get("ranked_count", 0),
        "tradable_count": no_buy.get("tradable_count", 0),
        "top_ranked": (no_buy.get("top_ranked") or [])[:5],
        "top_ranked_rejections": (no_buy.get("top_ranked_rejections") or [])[:5],
        "top_rejected_candidates": (no_buy.get("top_rejected_candidates") or [])[:3],
        "skip_reason_counts": no_buy.get("skip_reason_counts", {}),
        "buyable_reject_counts": no_buy.get("buyable_reject_counts", {}),
        "top_buyable_rejects": (no_buy.get("top_buyable_rejects") or [])[:5],
        "scan_payload_misses": no_buy.get("scan_payload_misses", 0),
        "scan_buys_enabled": no_buy.get("scan_buys_enabled"),
        "scan_data_max_age_sec": no_buy.get("scan_data_max_age_sec"),
        "scan_min_confidence": no_buy.get("scan_min_confidence"),
        "scan_min_history_rows": no_buy.get("scan_min_history_rows"),
        "scan_buys_today": no_buy.get("scan_buys_today", 0),
        "max_scan_buys_today": no_buy.get("max_scan_buys_today"),
        "max_scan_buys_per_tick": no_buy.get("max_scan_buys_per_tick"),
        "scan_age_sec": no_buy.get("scan_age_sec"),
        "scan_rows_count": no_buy.get("scan_rows_count"),
        "scan_fresh_rows_count": no_buy.get("scan_fresh_rows_count"),
        "last_bot_error": _BOT_STATUS.get("last_error"),
        "last_bot_error_ts": _BOT_STATUS.get("last_error_ts"),
        "pa_stage_status": no_buy.get("pa_stage_status"),
        "data_health_ok": no_buy.get("data_health_ok"),
        "data_health_blocks": no_buy.get("data_health_blocks", []),
        "data_health_warnings": no_buy.get("data_health_warnings", []),
        "spy_data_ok": no_buy.get("spy_data_ok"),
        "spy_data_source": no_buy.get("spy_data_source"),
        "spy_data_error": no_buy.get("spy_data_error"),
        "regime_data_status": no_buy.get("regime_data_status"),
        "regime_data_fallback": no_buy.get("regime_data_fallback"),
        "regime_data_source": no_buy.get("regime_data_source"),
        "regime_data_error": no_buy.get("regime_data_error"),
        "regime_data_warnings": no_buy.get("regime_data_warnings", []),
        "stale_daily_cache_age_hours": no_buy.get("stale_daily_cache_age_hours"),
        "regime_data_size_mult": no_buy.get("regime_data_size_mult"),
        "vix_label": no_buy.get("vix_label"),
        "vix_value": no_buy.get("vix_value"),
        "vix_display": no_buy.get("vix_display"),
        "vix_data_ok": no_buy.get("vix_data_ok"),
        "vix_data_status": no_buy.get("vix_data_status"),
        "volatility_data_ok": no_buy.get("volatility_data_ok"),
        "volatility_source": no_buy.get("volatility_source"),
        "volatility_value": no_buy.get("volatility_value"),
        "volatility_data_error": no_buy.get("volatility_data_error"),
        "spy_rows": no_buy.get("spy_rows"),
        "spy_last_date": no_buy.get("spy_last_date"),
        "spy_mom_label": no_buy.get("spy_mom_label"),
        "cash": no_buy.get("cash"),
        "cash_floor": no_buy.get("cash_floor"),
        "cash_available_after_floor": no_buy.get("cash_available_after_floor"),
        "gross_exposure_pct": no_buy.get("gross_exposure_pct"),
        "buys_today": no_buy.get("buys_today"),
        "max_buys_today": no_buy.get("max_buys_today"),
        "paper_trading_locked": no_buy.get("paper_trading_locked"),
        "paper_lock_reason": no_buy.get("paper_lock_reason"),
        "tick_runtime_seconds": no_buy.get("tick_runtime_seconds"),
    }
    storage_debug = {}
    if not read_only:
        storage_debug = dict(storage_debug_info())
        storage_debug["last_state_write_ts"] = b.get("last_state_write_ts")

    stats = {
        "win_rate": win_rate, "wins": wins, "losses": losses,   # all-time (Round-5)
        "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
        "trades_total": b.get("total_trades", 0),
        "drawdown_pct": drawdown_pct, "peak_equity": round(peak_equity, 2),
        "today_pct": round(today_pct, 2),
        "today_open_equity": round(today_open, 2),
        "sectors": sector_breakdown,
        "exit_breakdown": exit_breakdown,   # Round-5 #8
        "total_costs_usd":    round(b.get("total_costs_usd", 0), 2),
        "commission_per_trade": COMMISSION_PER_TRADE,
    }

    return render_template("bot.html",
        bot=b, positions=positions,
        # Round-8: split BUY/SELL windows (100 each), decisions cap 20
        buys=buys[:100], sells=sells[:100], decisions=decisions[:20],
        total=total, pnl=pnl, pnl_pct=pnl_pct, starting=b.get("starting", 10000),
        stats=stats,
        no_buy=safe_no_buy,
        storage_debug=storage_debug,
        last_decision_ago_min=last_decision_ago_min,
        market_open=is_market_open(), now=datetime.now(),
        read_only=read_only)


# ── Market-wide scan (top confidence + top gainers) ─────────────────────────
SCAN_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "JPM", "V",
    "WMT", "LLY", "MA", "XOM", "ORCL", "PG", "JNJ", "HD", "COST", "BAC",
    "ABBV", "NFLX", "CVX", "KO", "AMD", "PEP", "TMO", "ADBE", "CSCO", "CRM",
    "MCD", "ACN", "WFC", "ABT", "DIS", "INTC", "QCOM", "TXN", "IBM", "BA",
    "UBER", "SHOP", "PYPL", "SPOT", "COIN", "SQ", "PLTR", "SNOW", "DDOG", "SOFI",
    # Round-5 #6: sector diversity (health/industrial/materials/consumer-staples)
    "UNH", "ISRG", "CAT", "RTX", "LIN", "PM",
]
SCAN_CACHE_TTL = 90

_scan_result = {"data": None, "ts": 0}
_scan_lock = threading.Lock()


def _scan_snapshot():
    """Return a thread-safe snapshot of (data, ts)."""
    with _scan_lock:
        return _scan_result["data"], _scan_result["ts"]


def run_scan():
    """Scan SCAN_UNIVERSE in parallel, build sorted leaderboard."""
    cfg = active_config()
    scan_deadline = time.time() + (20 if PYTHONANYWHERE_MODE else 45)
    universe = SCAN_UNIVERSE
    existing_rows = []
    if PYTHONANYWHERE_MODE:
        existing_rows, _old_ts = _scan_snapshot()
        start = int((_scan_result.get("pa_next_index") or 0) % len(SCAN_UNIVERSE))
        rotated = SCAN_UNIVERSE[start:] + SCAN_UNIVERSE[:start]
        universe = rotated[:PA_SCAN_BATCH_SIZE]
        _scan_result["pa_next_index"] = (start + len(universe)) % len(SCAN_UNIVERSE)
    regime, ok = _call_with_deadline(lambda: get_market_regime(cfg), scan_deadline)
    if not ok:
        rows = list(existing_rows) if PYTHONANYWHERE_MODE and existing_rows else []
        rows.sort(key=lambda r: (-r["direction"], -r["confidence"], -r["score"]))
        return rows

    def _scan_one(t):
        try:
            q = get_quote(t)
            if q["price"] <= 0: return None
            arts, sent = get_news(t)
            ctx = get_history(t)
            # Round-5 A: scan/buy parity — mirror _fetch_one's full inputs so the
            # leaderboard ranks tickers on the SAME rec the outside-buy path
            # re-checks. (All cached → no extra steady-state API cost.)
            earn = get_earnings_soon(t)
            analyst = get_analyst_rec(t)
            insider = get_insider_sentiment(t)
            rec = get_recommendation(sent, ctx, regime=regime, earnings=earn,
                                      analyst=analyst, insider=insider,
                                      news_articles=arts, config=cfg)
            catalyst = classify_catalyst(arts, earn, analyst, insider, ctx,
                                         config=cfg)
            rec = dict(rec)
            rec["catalyst"] = catalyst
            payload = {
                "arts": arts,
                "sent": sent,
                "ctx": ctx,
                "earn": earn,
                "analyst": analyst,
                "insider": insider,
                "rec": rec,
                "quote": q,
                "price": q.get("price", 0),
                "stale": bool(q.get("stale")),
            }
            _finalize_signal_confidence(payload, cfg)
            rec = payload["rec"]
            cache_set(f"scan_payload_{t}", payload)
            direction = 1 if rec["score"] > 0 else (-1 if rec["score"] < 0 else 0)
            return {
                "ticker": t, "price": q["price"], "change": q["change"], "pct": q["pct"],
                "signal": rec["signal"], "cls": rec["cls"],
                "confidence": rec["confidence"], "score": rec["score"],
                "sentiment": sent, "direction": direction,
                "rsi": ctx.get("rsi", 0) if ctx else 0,
                "ts": int(time.time()),
            }
        except Exception:
            return None

    rows = []
    workers = 2 if PYTHONANYWHERE_MODE else 8
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(_scan_one, t): t for t in universe}
    try:
        while futures:
            remaining = scan_deadline - time.time()
            if remaining <= 0:
                break
            done, _pending = wait(
                futures,
                timeout=max(0.05, min(1.0, remaining)),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            for fut in done:
                futures.pop(fut, None)
                try:
                    r = fut.result(timeout=0)
                except Exception:
                    r = None
                if r is not None:
                    rows.append(r)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    gc.collect()
    if PYTHONANYWHERE_MODE and existing_rows:
        merged = {r["ticker"]: r for r in existing_rows}
        merged.update({r["ticker"]: r for r in rows})
        rows = list(merged.values())
    rows.sort(key=lambda r: (-r["direction"], -r["confidence"], -r["score"]))
    return rows


_scan_refresh_in_flight = False
_scan_refresh_lock = threading.Lock()


def _refresh_scan_background():
    """Compute a fresh scan outside the request thread."""
    global _scan_refresh_in_flight
    with _scan_refresh_lock:
        if _scan_refresh_in_flight: return
        _scan_refresh_in_flight = True
    try:
        rows = run_scan()
        new_ts = time.time()
        with _scan_lock:
            _scan_result["data"] = rows
            _scan_result["ts"] = new_ts
    except Exception as e:
        print(f"[scan-refresh] error: {type(e).__name__}: {e}")
    finally:
        with _scan_refresh_lock:
            _scan_refresh_in_flight = False


def warm_scan_if_due():
    """Start one background scan refresh when /bot/tick keeps the app awake."""
    data, ts = _scan_snapshot()
    if not is_market_open() or (data and time.time() - ts < SCAN_CACHE_TTL * 0.8):
        return False
    try:
        threading.Thread(target=_refresh_scan_background, daemon=True).start()
        return True
    except Exception:
        return False


def get_scan():
    """Stale-while-revalidate: page loads return cached IMMEDIATELY, refresh in background."""
    data, ts = _scan_snapshot()
    is_stale = (not data) or (time.time() - ts >= SCAN_CACHE_TTL)
    if PYTHONANYWHERE_MODE:
        if is_stale:
            try:
                threading.Thread(target=_refresh_scan_background, daemon=True).start()
            except Exception:
                pass
        return data or [], ts
    if data and not is_stale:
        return data, ts
    if data:
        threading.Thread(target=_refresh_scan_background, daemon=True).start()
        return data, ts
    _refresh_scan_background()
    return _scan_snapshot()


def top_refresh_clear():
    """Clear the scan cache (called from /top/refresh route)."""
    with _scan_lock:
        _scan_result["data"] = None
        _scan_result["ts"] = 0
