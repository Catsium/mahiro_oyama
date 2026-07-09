"""Central active thresholds for V3 trading logic.

Only thresholds used by live/backtest paths live here. Sweeps pass merged copies;
live code reads defaults and never mutates them.
"""

import copy
import hashlib
import json


CONFIG_VERSION = "v3_intelligence_robustness_1"

RISK_CONFIG = {
    "base_risk_pct_by_regime": {
        "bull": 1.00,
        "neutral": 0.70,
        "bear": 0.40,
    },
    # A2 floor raised $10 → $150 (audit P0-3): $10 fill pays $1.98 round trip
    # = 19.8% commission; at $150 the round trip is ≤ 1.32%.
    "min_trade_size": 150.0,
    "max_position_pct": 0.35,
    "max_sector_pct": 1.00,
    "max_corr_group_pct": 1.00,
    "max_gross_exposure_pct": 1.00,
    "min_cash_reserve_pct": 0.00,
    "max_positions": 12,  # 24 slots at $10k equity = dust positions (audit P3-19)
    "max_new_buys_per_tick": 10,
    "max_new_buys_per_day": 30,
    "daily_loss_warning_pct": -0.10,
    "daily_loss_hard_lockout_pct": -0.15,
    "daily_loss_limit_pct": -0.15,
    "drawdown_warning_pct": -0.20,
    "drawdown_hard_lockout_pct": -0.35,
    "hard_drawdown_lockout_pct": -0.35,
    "warnings_block_new_buys": False,
    "hard_lockouts_block_new_buys": True,
    "take_profit_rebuy_cooldown_minutes": 5,
    "trailing_stop_rebuy_cooldown_minutes": 15,
    "stop_loss_rebuy_cooldown_minutes": 30,
    "manual_sell_rebuy_cooldown_minutes": 10,
    "strong_buy_cooldown_bypass_after_minutes": 5,
    "buy_candidate_cooldown_bypass_after_minutes": 10,
    "bullish_lean_cooldown_bypass_after_minutes": 15,
    "high_confidence_hold_cooldown_bypass_enabled": False,
    "pyramiding_enabled": True,
    "pyramiding_min_profit_pct": 1.0,
    "pyramiding_max_loss_allowed_pct": -3.0,
    "pyramiding_min_minutes_between_adds": 10,
    "pyramiding_allow_flat_add_for_strong_buy": True,
    "pyramiding_allow_losing_adds": True,
    "experimental_trade_size_mult": 0.25,
    "validated_trade_size_mult": 1.0,
}

SIGNAL_CONFIG = {
    "min_buy_confidence": 40,
    "min_strong_buy_confidence": 70,
    "min_expected_edge_pct": 0.25,
    "warmup_min_confidence": 48,
    "warmup_size_mult": 0.60,
    "min_net_edge_pct": 0.00,
    "require_net_edge_positive": True,
    "paper_ev_relaxation_enabled": True,
    "min_expected_net_profit_usd": 0.25,
    "default_min_expected_net_profit_usd": 0.25,
    "strong_buy_min_expected_net_profit_usd": 0.10,
    "buy_candidate_min_expected_net_profit_usd": 0.25,
    "bullish_lean_min_expected_net_profit_usd": 0.50,
    "high_confidence_hold_min_expected_net_profit_usd": 0.75,
    "strong_edge_pct": 2.00,
    "min_avg_dollar_volume_hard_block": 250_000,
    "low_avg_dollar_volume_warning": 1_000_000,
    "near_zero_volume_ratio_hard_block": 0.10,
    "very_low_volume_ratio_size_penalty": 0.30,
    "low_volume_warning_ratio": 0.70,
    "low_volume_size_multiplier": 0.90,
    "very_low_volume_size_multiplier": 0.75,
    "volume_hard_reject_ratio": 0.10,
    "volume_soft_penalty_ratio": 0.70,
    "low_volume_size_mult": 0.90,
    "atr_hard_block_enabled": True,
    "extreme_atr_hard_block_pct": 20.0,
    "high_atr_warning_pct": 10.0,
    "high_atr_size_multiplier": 0.85,
    "extreme_atr_size_multiplier": 0.65,
    "neutral_gate_adx_threshold": 15,
    "allow_bullish_lean_buy_attempts": True,
    "bullish_lean_candidate_size_multiplier": 0.85,
    "allow_high_confidence_hold_buy_attempts": True,
    "high_confidence_hold_min_confidence": 55,
    "high_confidence_hold_candidate_size_multiplier": 0.65,
    "hold_can_buy_by_default": False,
    "sell_can_buy": False,
}

# Kelly stays OFF until ≥50 closed trade outcomes exist (audit P3-20).
# Re-enable recipe once there: enabled=True, min_samples=50, fraction=0.5,
# min_mult=0.5, max_mult=1.25. Do not auto-flip.
KELLY_CONFIG = {
    "enabled": False,
    "min_samples": 100,
    "fraction": 0.50,
    "min_mult": 1.0,
    "max_mult": 1.0,
}

EXIT_CONFIG = {
    "generic_partial_take_pct": 0.06,
    "generic_trail_start_pct": 0.04,
    "generic_max_hold_days": 10,
    "min_hold_minutes": 20,
    "profile_min_mult": 0.70,
    "profile_max_mult": 1.50,
}

CORRELATION_CONFIG = {
    "lookback_days": 30,
    "min_days": 20,
    "rolling_weight": 0.70,
    "static_weight": 0.30,
    "soft_var_cap": 0.05,
    "hard_var_cap": 0.15,
    "absolute_var_cap": 0.25,
    "soft_size_mult": 0.65,
}

REGIME_CONFIG = {
    "breadth_short_ma": 20,
    "breadth_med_ma": 50,
    "breadth_long_ma": 200,
    "breadth_min_universe": 30,
    "min_effective_breadth_count": 30,
    "regime_confirm_days": 2,
    "panic_confirm_days": 1,
    "breadth_50_strong": 0.65,
    "breadth_50_weak": 0.45,
    "breadth_200_strong": 0.60,
    "breadth_200_weak": 0.40,
    "narrow_bull_breadth_max": 0.55,
    "sector_rs_lookback": 20,
    "sector_strong_rs": 0.03,
    "sector_weak_rs": -0.03,
    "hysteresis": {"breadth_buffer": 0.03, "rs_buffer": 0.005},
    "risk_mult": {
        "strong_bull": 1.00,
        "narrow_bull": 0.80,
        "weak_bull": 0.75,
        "choppy_neutral": 0.65,
        "risk_off_neutral": 0.45,
        "bear": 0.30,
        "panic": 0.00,
    },
    "cluster_mult": {
        "strong_bull": {
            "trend_continuation": 1.10, "breakout": 1.10,
            "momentum": 1.05, "dip": 1.00,
        },
        "narrow_bull": {
            "trend_continuation": 1.05, "breakout": 0.90,
            "momentum": 0.85, "dip": 1.00,
        },
        "weak_bull": {
            "trend_continuation": 0.95, "breakout": 0.90,
            "momentum": 0.90, "dip": 1.00,
        },
        "choppy_neutral": {"dip": 1.15, "momentum": 0.85, "breakout": 0.80},
        "risk_off_neutral": {"dip": 0.90, "breakout": 0.70, "momentum": 0.70},
        "bear": {"dip": 0.70, "breakout": 0.50, "momentum": 0.50},
        "panic": {"dip": 0.00, "breakout": 0.00, "momentum": 0.00},
    },
}

CATALYST_CONFIG = {
    "news_half_life_hours": 6,
    "news_max_age_hours": 48,
    "analyst_max_age_days": 14,
    "insider_max_age_days": 30,
    "min_rel_volume_confirm": 1.5,
    "gap_hold_minutes": 30,
    "require_price_confirmation_for_ambiguous": True,
    "max_score_contribution": 0.50,
    "max_size_mult": 1.05,
    "min_size_mult": 0.90,
    "shadow_first": True,
    "size_effect_enabled": False,
    "score_effect_enabled": False,
    "cluster_tagging_enabled": True,
    "earnings_hard_block_enabled": False,
    "earnings_same_day_size_multiplier": 0.90,
    "earnings_tomorrow_size_multiplier": 0.90,
    "earnings_within_3_days_size_multiplier": 1.00,
    "negative_catalyst_hard_block_enabled": True,
    "negative_catalyst_requires_price_falling": True,
    "negative_catalyst_falling_threshold_pct": -5.0,
    "negative_catalyst_size_multiplier": 0.90,
    "missing_earnings_policy": "neutral",
    "missing_news_policy": "neutral",
    "missing_catalyst_policy": "neutral",
    "missing_analyst_policy": "neutral",
    "missing_insider_policy": "neutral",
    "missing_politician_policy": "neutral",
}

CATALYST_SCORE = {
    "earnings_beat": 0.30,
    "earnings_miss": -0.40,
    "guidance_raise": 0.50,
    "guidance_cut": -0.60,
    "analyst_upgrade": 0.15,
    "analyst_downgrade": -0.20,
    "insider_buy": 0.15,
    "insider_sell": -0.10,
    "regulatory_risk": -0.50,
    "lawsuit_investigation": -0.40,
    "m_and_a": 0.00,
    "product_launch": 0.10,
    "macro_rates_inflation_jobs": 0.00,
    "layoffs_restructuring": 0.00,
    "mixed_or_unclear": 0.00,
}

BACKTEST_CONFIG = {
    "friction_stress_mult": 2.0,
    "min_sweep_trades": 30,
    "profit_concentration_limit": 0.50,
}

SUGGESTION_CONFIG = {
    "enabled": True,
    "max_extra_tickers": 2,
    "discovery_top_conf": 6,
    "discovery_top_gain": 4,
    "discovery_prefilter_max": 8,
    "discovery_full_fetch_max": 6,
    "min_scan_confidence": 68,
    "min_price": 5.0,
    "min_adv_usd": 15_000_000,
    "min_suggestion_score": 0.72,
    "min_net_edge_pct": 0.35,
    "min_gross_edge_live_pct": 0.60,
    "min_gross_edge_prior_pct": 0.90,
    "min_live_samples_soft": 20,
    "same_sector_limit": 1,
    "same_corr_group_limit": 1,
    "ticker_cooldown_sec": 21600,
    "loss_cooldown_sec": 14 * 86400,
    "non_dip_min_vol_ratio": 1.2,
    "max_holding_corr": 0.70,
    "freshness_max_age_hours": 24.0,
    "freshness_half_life_hours": 12.0,
    "feedback_min_samples": 15,
    "feedback_min_mult": 0.90,
    "feedback_max_mult": 1.10,
    "sqlite_wal_autocheckpoint_pages": 200,
    "sqlite_journal_size_limit_bytes": 16_777_216,
    "suggestion_log_retention_days": 90,
    "feedback_retention_days": 180,
}

SCAN_CONFIG = {
    "scan_buys_enabled": True,
    "scan_buy_mode": "aggressive_paper_test",
    "max_scan_buys_per_tick": 10,
    "max_scan_buys_per_day": 30,
    "scan_buy_min_confidence": 45,
    "scan_data_max_age_sec": 300,
    "scan_requires_execution_trusted_quote": True,
    "scan_requires_ev_pass": True,
    "scan_requires_min_history_rows": 25,
    "scan_requires_min_trade_size": True,
    "scan_allow_bullish_lean": True,
    "scan_allow_high_confidence_hold": False,
}

HISTORY_CONFIG = {
    "min_history_rows_for_buy": 25,
    "preferred_history_rows": 60,
    "stale_history_max_completed_trading_days": 3,
    "cached_history_allowed": True,
    "missing_core_history_policy": "block",
    "missing_optional_signal_policy": "neutral",
    "missing_core_quote_policy": "block",
    "max_history_fetches_per_tick": 2,
    "max_history_fetches_per_warm_call": 3,
    "max_symbols_per_warm_call": 3,
    "provider_history_request_timeout_sec": 5,
    "history_fetch_stop_after_success": True,
    "finnhub_quote_enabled": True,
    "finnhub_daily_history_enabled": True,
    "finnhub_daily_forbidden_cooldown_sec": 21_600,
    "finnhub_daily_forbidden_status_codes": [401, 403],
    "finnhub_daily_rate_limit_cooldown_sec": 1_800,
    "finnhub_daily_provider_error_cooldown_sec": 900,
    "finnhub_daily_circuit_scope": "endpoint_global",
    "finnhub_daily_global_forbidden_circuit_key": "finnhub_daily:global",
    "finnhub_daily_403_does_not_disable_quotes": True,
    "daily_history_cache_ttl_sec": 64_800,
    "fmp_daily_enabled": True,
    "fmp_daily_global_429_cooldown_sec": 1_800,
    "fmp_daily_symbol_error_cooldown_sec": 900,
    "fmp_daily_max_retries_per_symbol_per_day": 2,
    "fmp_daily_respect_global_circuit": True,
    "fmp_daily_cache_success_immediately": True,
    "history_cache_first": True,
    "history_cache_min_rows": 25,
    "history_cache_preferred_rows": 60,
    "history_cache_allow_previous_completed_trading_day": True,
    "history_cache_max_completed_trading_day_age": 2,
    "history_cache_use_if_provider_fails": True,
    "history_cache_write_on_success": True,
    "history_cache_persistent": True,
    "history_cache_symbol_key_normalized": True,
    "live_quote_overlay_enabled": True,
    "live_quote_overlay_requires_cached_history": True,
    "live_quote_overlay_min_base_history_rows": 25,
    "live_quote_overlay_does_not_replace_missing_history": True,
    "live_quote_overlay_source_must_be_execution_trusted": True,
    "warm_history_route_enabled": True,
    "warm_history_requires_admin_token": True,
    "warm_history_max_symbols_per_call": 3,
    "warm_history_max_fetches_per_call": 3,
    "warm_history_return_json": True,
    "warm_history_must_be_bounded": True,
    "warm_history_no_full_universe_fetch": True,
    "warm_history_not_called_from_health": True,
    "history_diagnostics_enabled": True,
    "provider_circuit_diagnostics_enabled": True,
    "missing_history_visible_in_reasoning": True,
    "top_missing_history_symbols_count": 3,
    "top_rejected_candidates_count": 3,
}

MARKET_DATA_MODES_CONFIG = {
    "allow_proxy_mode": True,
    "allow_degraded_paper_trading": True,
    "normal_size_mult": 1.0,
    "proxy_size_mult": 0.85,
    "proxy_vol_window_days": 20,
    "proxy_min_spy_bars": 60,
    "degraded_size_mult": 0.90,
    "degraded_use_standard_gates_for_testing": True,
    "degraded_min_confidence": 40,
    "degraded_max_new_buys_per_tick": 10,
    "degraded_max_new_buys_per_day": 30,
    "degraded_max_position_pct": 0.05,
    "degraded_max_gross_exposure_pct": 0.35,
    "degraded_block_pyramiding": False,
    "degraded_block_experimental": False,
    "degraded_block_confidence_prior": False,
    "degraded_block_low_volume_penalty": False,
    "degraded_require_buy_candidate": True,
    "degraded_require_fresh_quote": True,
    "degraded_require_normal_ev_gates": True,
    "degraded_require_normal_risk_caps": True,
}

SWEEP_PARAMS = {
    "signal.min_buy_confidence": [40, 50, 60],
    "signal.min_expected_edge_pct": [0.25, 0.40, 0.60],
    "risk.base_risk_pct_by_regime.neutral": [0.60, 0.70, 0.85],
    "correlation.soft_var_cap": [0.03, 0.05, 0.08],
    "correlation.hard_var_cap": [0.10, 0.15, 0.20],
    "exit.generic_partial_take_pct": [0.04, 0.06, 0.08],
    "regime.breadth_50_strong": [0.60, 0.65, 0.70],
}

PRESET_SWEEPS = {
    "current": {},
    "low_risk": {
        "signal.min_buy_confidence": 60,
        "signal.min_expected_edge_pct": 0.60,
        "risk.base_risk_pct_by_regime.neutral": 0.60,
        "correlation.soft_var_cap": 0.03,
        "correlation.hard_var_cap": 0.10,
    },
    "balanced": {
        "signal.min_buy_confidence": 40,
        "signal.min_expected_edge_pct": 0.40,
        "correlation.soft_var_cap": 0.05,
        "correlation.hard_var_cap": 0.15,
    },
    "aggressive": {
        "signal.min_buy_confidence": 40,
        "signal.min_expected_edge_pct": 0.25,
        "risk.base_risk_pct_by_regime.neutral": 0.85,
        "correlation.soft_var_cap": 0.08,
        "correlation.hard_var_cap": 0.20,
    },
}

DEFAULT_CONFIG = {
    "version": CONFIG_VERSION,
    "risk": RISK_CONFIG,
    "signal": SIGNAL_CONFIG,
    "kelly": KELLY_CONFIG,
    "exit": EXIT_CONFIG,
    "correlation": CORRELATION_CONFIG,
    "regime": REGIME_CONFIG,
    "catalyst": CATALYST_CONFIG,
    "catalyst_score": CATALYST_SCORE,
    "backtest": BACKTEST_CONFIG,
    "suggestion": SUGGESTION_CONFIG,
    "scan": SCAN_CONFIG,
    "history": HISTORY_CONFIG,
    "market_data_modes": MARKET_DATA_MODES_CONFIG,
    "sweep_params": SWEEP_PARAMS,
    "preset_sweeps": PRESET_SWEEPS,
}


def default_config():
    return copy.deepcopy(DEFAULT_CONFIG)


def active_config():
    cfg = default_config()
    try:
        from utils.deploy_config import PYTHONANYWHERE_MODE
    except Exception:
        PYTHONANYWHERE_MODE = False

    if PYTHONANYWHERE_MODE:
        cfg["suggestion"]["max_extra_tickers"] = 2
        cfg["suggestion"]["min_adv_usd"] = 15_000_000
        cfg["suggestion"]["min_suggestion_score"] = 0.72
        cfg["suggestion"]["min_net_edge_pct"] = 0.50
        cfg["suggestion"]["loss_cooldown_sec"] = 14 * 86400
        cfg["suggestion"]["non_dip_min_vol_ratio"] = 1.2
        cfg["suggestion"]["max_holding_corr"] = 0.70
        cfg["correlation"]["lookback_days"] = 20
        cfg["regime"]["breadth_min_universe"] = 30
        cfg["backtest"]["min_sweep_trades"] = 20

    return cfg


def set_path(config, dotted_path, value):
    cur = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _deep_merge(base, overrides):
    for key, val in (overrides or {}).items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = copy.deepcopy(val)
    return base


def merge_config(overrides=None, base=None):
    cfg = copy.deepcopy(base or DEFAULT_CONFIG)
    for key, val in (overrides or {}).items():
        if "." in key:
            set_path(cfg, key, val)
        elif isinstance(val, dict) and isinstance(cfg.get(key), dict):
            _deep_merge(cfg[key], val)
        else:
            cfg[key] = copy.deepcopy(val)
    return cfg


def config_hash(config=None):
    cfg = config or DEFAULT_CONFIG
    raw = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
