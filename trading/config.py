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
    "min_trade_size": 400.0,
    "max_position_pct": 0.08,
    "max_sector_pct": 0.25,
    "max_corr_group_pct": 0.25,
    "max_gross_exposure_pct": 0.70,
    "min_cash_reserve_pct": 0.30,
    "max_positions": 8,
    "max_new_buys_per_tick": 1,
    "max_new_buys_per_day": 2,
    "daily_loss_limit_pct": -0.02,
    "hard_drawdown_lockout_pct": -0.10,
    "experimental_trade_size_mult": 0.25,
    "validated_trade_size_mult": 1.0,
}

SIGNAL_CONFIG = {
    "min_buy_confidence": 55,
    "min_strong_buy_confidence": 65,
    "min_expected_edge_pct": 0.40,
    "min_net_edge_pct": 0.20,
    "strong_edge_pct": 2.00,
    "volume_hard_reject_ratio": 0.70,
    "volume_soft_penalty_ratio": 1.20,
    "low_volume_size_mult": 0.75,
    "neutral_gate_adx_threshold": 20,
}

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

MARKET_DATA_MODES_CONFIG = {
    "allow_proxy_mode": True,
    "allow_degraded_paper_trading": True,
    "normal_size_mult": 1.0,
    "proxy_size_mult": 0.85,
    "proxy_vol_window_days": 20,
    "proxy_min_spy_bars": 60,
    "degraded_size_mult": 0.70,
    "degraded_min_confidence": 55,
    "degraded_max_new_buys_per_tick": 1,
    "degraded_max_new_buys_per_day": 1,
    "degraded_max_position_pct": 0.05,
    "degraded_max_gross_exposure_pct": 0.35,
    "degraded_block_pyramiding": True,
    "degraded_block_experimental": True,
    "degraded_block_confidence_prior": True,
    "degraded_block_low_volume_penalty": True,
    "degraded_require_buy_candidate": True,
    "degraded_require_fresh_quote": True,
    "degraded_require_normal_ev_gates": True,
    "degraded_require_normal_risk_caps": True,
}

SWEEP_PARAMS = {
    "signal.min_buy_confidence": [50, 55, 60],
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
        "signal.min_buy_confidence": 55,
        "signal.min_expected_edge_pct": 0.40,
        "correlation.soft_var_cap": 0.05,
        "correlation.hard_var_cap": 0.15,
    },
    "aggressive": {
        "signal.min_buy_confidence": 50,
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
