"""Print paper-bot market-data mode diagnostics without placing trades."""
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading.bot import _market_data_mode, is_execution_candidate
from trading.config import active_config
from trading.risk import get_market_regime, get_vix
from utils.storage import load_bot


REQUIRED_STATUS_FIELDS = (
    "trading_mode",
    "normal_mode_active",
    "proxy_mode_active",
    "degraded_mode_active",
    "degraded_mode_reason",
    "min_buy_confidence",
    "degraded_size_mult",
    "degraded_use_standard_gates_for_testing",
    "degraded_standard_gates_active",
    "degraded_gate_policy",
    "effective_size_mult",
    "effective_min_buy_confidence",
    "normal_ev_gates_required",
    "normal_risk_caps_required",
    "fresh_quote_required",
    "degraded_min_confidence",
    "degraded_reject_counts",
    "degraded_buys_today",
    "degraded_max_buys_today",
    "degraded_gross_exposure_pct",
    "degraded_max_gross_exposure_pct",
    "data_health_blocks",
    "data_health_warnings",
    "spy_data_ok",
    "spy_data_source",
    "spy_data_error",
    "spy_bar_count",
    "spy_last_date",
    "regime_kind",
    "regime_source",
    "regime_fallback_active",
    "regime_data_status",
    "regime_data_warnings",
    "finnhub_key_configured",
    "fmp_key_configured",
    "stooq_status",
    "volatility_data_ok",
    "volatility_source",
    "volatility_value",
    "volatility_error",
    "vix_display",
    "raw_buy_count",
    "display_buy_candidate_count",
    "candidate_pool_count",
    "ranked_count",
    "tradable_count",
    "buyable_reject_counts",
    "skip_reason_counts",
    "top_buyable_rejects",
    "top_ranked_rejections",
    "main_blocker",
    "provider_health_status",
    "rate_limit_recent",
    "stale_positions",
    "risk_unmanaged_positions",
    "buys_today",
    "max_buys_today",
    "gross_exposure_pct",
    "paper_trading_locked",
    "paper_lock_reason",
)


def _load_status_diagnostics():
    try:
        from app import app as flask_app

        with flask_app.test_client() as client:
            response = client.get("/api/bot/status")
        if response.status_code == 200:
            payload = response.get_json(silent=True) or {}
            diag = payload.get("last_no_buy_diagnostics") or {}
            if isinstance(diag, dict):
                return diag, "api_bot_status"
    except Exception as exc:
        return load_bot().get("last_no_buy_diagnostics") or {}, f"bot_state_fallback:{type(exc).__name__}"
    return load_bot().get("last_no_buy_diagnostics") or {}, "bot_state_fallback"


def main():
    cfg = active_config()
    regime = get_market_regime(cfg)
    volatility = get_vix()
    mode = _market_data_mode(regime, volatility, cfg)
    diag, status_source = _load_status_diagnostics()
    missing = [field for field in REQUIRED_STATUS_FIELDS if field not in diag]

    sample_degraded = {
        "cls": "buy",
        "signal": "BUY",
        "ticker": "SAMPLE",
        "confidence": 40,
        "display_signal_label": "BUY_CANDIDATE",
    }
    sample_degraded["eligible"] = is_execution_candidate(
        sample_degraded,
        cfg,
        degraded_mode_active=True,
    )
    sample_degraded["reason"] = (
        "BUY_CANDIDATE >= min thresholds"
        if sample_degraded["eligible"]
        else "blocked by execution candidate policy"
    )

    out = {
        "spy_source": regime.get("spy_data_source") or regime.get("regime_data_source"),
        "spy_bar_count": regime.get("spy_rows"),
        "spy_latest_date": regime.get("spy_last_date"),
        "volatility_source": volatility.get("volatility_source") or volatility.get("source"),
        "volatility_value": volatility.get("volatility_value", volatility.get("vix")),
        "data_health_blocks": mode.get("data_health_blocks", []),
        "selected_trading_mode": mode.get("trading_mode"),
        "degraded_enabled": bool((cfg.get("market_data_modes") or {}).get("allow_degraded_paper_trading")),
        "degraded_use_standard_gates_for_testing": mode.get("degraded_use_standard_gates_for_testing"),
        "degraded_gate_policy": mode.get("degraded_gate_policy"),
        "effective_size_mult": mode.get("effective_size_mult"),
        "effective_min_buy_confidence": mode.get("effective_min_buy_confidence"),
        "normal_ev_gates_required": mode.get("normal_ev_gates_required"),
        "normal_risk_caps_required": mode.get("normal_risk_caps_required"),
        "fresh_quote_required": mode.get("fresh_quote_required"),
        "sample_degraded_eligibility_decision": sample_degraded,
        "status_source": status_source,
        "status_fields_missing": missing,
    }
    print(json.dumps(out, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
