import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _install_dependency_stubs():
    if "finnhub" not in sys.modules:
        sys.modules["finnhub"] = types.SimpleNamespace(
            Client=lambda api_key=None: types.SimpleNamespace()
        )
    if "yfinance" not in sys.modules:
        class _Ticker:
            def __init__(self, *_args, **_kwargs):
                pass

            def history(self, *_args, **_kwargs):
                return types.SimpleNamespace(empty=True)

            @property
            def info(self):
                return {}

        sys.modules["yfinance"] = types.SimpleNamespace(Ticker=_Ticker)
    if "flask" not in sys.modules:
        try:
            __import__("flask")
        except ImportError:
            pass
    if "flask" not in sys.modules:
        class _FakeFlask:
            def __init__(self, *_args, **_kwargs):
                self.secret_key = None

            def route(self, *_args, **_kwargs):
                return lambda fn: fn

            def context_processor(self, fn):
                return fn

        sys.modules["flask"] = types.SimpleNamespace(
            Flask=_FakeFlask,
            render_template=lambda *_args, **_kwargs: "",
            jsonify=lambda obj=None, **_kwargs: obj,
            request=types.SimpleNamespace(args={}, form={}, headers={}, referrer=None),
            redirect=lambda target=None, *_args, **_kwargs: target,
            url_for=lambda endpoint, **_kwargs: endpoint,
            flash=lambda *_args, **_kwargs: None,
            abort=lambda code=500: (_ for _ in ()).throw(RuntimeError(f"abort {code}")),
            session={},
        )


_install_dependency_stubs()

from trading.attribution import (  # noqa: E402
    ALPHA_PRIOR,
    BETA_PRIOR,
    ENTRY_LIVE_N,
    EXIT_LIVE_N,
    attribution_signal_weights,
    ensure_attribution_state,
    expected_edge_for_candidate,
    exit_profile,
    init_entry_bucket,
    record_entry_event,
    record_exit_event,
    sample_trust,
    update_entry_buckets,
    update_exit_post_outcomes,
    update_forward_outcomes,
)
from trading.catalysts import catalyst_cluster, classify_catalyst  # noqa: E402
from trading.config import DEFAULT_CONFIG, active_config, config_hash, merge_config  # noqa: E402
from trading.suggestion_store import (  # noqa: E402
    init_suggestion_db,
    load_feedback_stats,
    load_recent_suggestions,
    log_suggestion_run,
    record_suggestion_feedback,
)
from trading.suggestions import _suggestion_cfg, rank_suggestion_candidates  # noqa: E402
from trading.regime_v3 import (  # noqa: E402
    apply_confirmation,
    breadth_metrics,
    classify_regime,
    regime_risk_mult,
)
from trading.signals import _attribute_outcome, classify_display_signal, get_recommendation  # noqa: E402
from trading.exit_ladders import (  # noqa: E402
    apply_regime_exit_tightening,
    compose_exit_profile,
    normalize_cluster,
)
from trading.portfolio_variance import (  # noqa: E402
    candidate_variance_check,
    smoothed_corr,
    static_corr_assumption,
)
from trading.sizing import (  # noqa: E402
    build_friction_diagnostics,
    confidence_scale,
    entry_cluster,
    evaluate_candidate,
    rank_candidates,
)


def _entry_candidate(ticker, confidence=75, cluster="trend", atr_pct=2.0,
                     score=4.0, source="watchlist", friction_pct=0.0):
    return {
        "ticker": ticker,
        "source": source,
        "price": 100.0,
        "cluster": cluster,
        "ctx": {
            "atr_pct": atr_pct,
            "avg_dollar_vol_20d": 100_000_000,
            "rsi": 55,
        },
        "rec": {
            "signal": "BUY",
            "cls": "buy",
            "confidence": confidence,
            "score": score,
            "categories": {cluster: 2.0},
        },
        "friction": {"total_pct": friction_pct},
    }


def _add_entry_outcome(state, idx, ret_pct, cluster="trend", category=None,
                       friction_pct=0.0, decision="executed"):
    category = category or cluster
    cand = _entry_candidate(f"T{idx:03d}", cluster=cluster,
                            friction_pct=friction_pct)
    cand["rec"]["categories"] = {category: 2.0}
    event = record_entry_event(
        state, cand, decision, "test", ts=1_700_000_000 + idx * 86400,
        regime="neutral",
    )
    event.setdefault("forward_returns", {})["5d"] = ret_pct
    event["mfe_pct"] = max(0.0, ret_pct)
    event["mae_pct"] = min(0.0, ret_pct)
    update_entry_buckets(state, event, "5d")
    return event


class TradingV1SizingTests(unittest.TestCase):
    def _candidate(self, ticker, confidence, cluster, atr_pct=2.0, score=4.0):
        return _entry_candidate(ticker, confidence, cluster, atr_pct, score)

    def test_atr_risk_budget_sizes_high_atr_smaller(self):
        low_atr = evaluate_candidate(
            self._candidate("LOW", 75, "trend", atr_pct=2),
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="neutral",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats={},
            min_position_usd=100,
        )
        high_atr = evaluate_candidate(
            self._candidate("HIGH", 75, "trend", atr_pct=8),
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="neutral",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats={},
            min_position_usd=100,
        )
        self.assertGreater(low_atr["risk"]["target_notional"],
                           high_atr["risk"]["target_notional"])

    def test_ev_ranking_can_prefer_lower_confidence_higher_edge(self):
        edge_stats = {
            "neutral:trend": {"5d": {"n": 10, "avg_return_pct": 3.0}},
            "neutral:mixed": {"5d": {"n": 10, "avg_return_pct": 0.8}},
        }
        ranked = rank_candidates(
            [
                self._candidate("HIGHCONF", 90, "mixed", score=8),
                self._candidate("EDGE", 65, "trend", score=2),
            ],
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="neutral",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats=edge_stats,
            min_position_usd=100,
        )
        self.assertEqual(ranked[0]["ticker"], "EDGE")

    def test_friction_gate_blocks_negative_net_edge(self):
        edge_stats = {"neutral:trend": {"5d": {"n": 10, "avg_return_pct": 0.01}}}
        cand = evaluate_candidate(
            self._candidate("COSTLY", 80, "trend", score=8),
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="neutral",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats=edge_stats,
            min_position_usd=100,
        )
        self.assertFalse(cand["tradable"])
        self.assertIn("EV gate", cand["rank_reason"])

    def test_friction_diagnostics_explain_small_trade_cost_stack(self):
        diag = build_friction_diagnostics(
            158.40,
            {"avg_dollar_vol_20d": 60_000_000},
            commission=0.99,
            source="watchlist",
        )
        self.assertEqual(diag["notional_usd"], 158.40)
        self.assertEqual(diag["round_trip_commission_usd"], 1.98)
        self.assertAlmostEqual(diag["commission_pct"], 1.25, places=2)
        self.assertGreater(diag["model_slippage_pct"], 0.1)
        self.assertEqual(diag["spread_proxy_pct"], 0.03)
        self.assertEqual(diag["spread_liquidity_bucket"], "adv_ge_50m")
        self.assertEqual(diag["components_sum_check_pct"], diag["total_pct"])

    def test_ev_diagnostics_are_observational_and_canonical(self):
        cfg = merge_config({
            "signal": {
                "min_expected_edge_pct": 0.40,
                "min_net_edge_pct": 0.20,
            }
        })
        cand = evaluate_candidate(
            self._candidate("META", 40, "trend", score=2),
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="neutral",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats={},
            min_position_usd=100,
            config=cfg,
        )
        self.assertFalse(cand["tradable"])
        self.assertEqual(cand["rank_reason_code"], "EDGE_TOO_LOW")
        self.assertEqual(cand["edge_diagnostics"]["edge_source"], "confidence_prior")
        self.assertEqual(cand["edge_diagnostics"]["gross_edge_pct"], cand["gross_edge_pct"])
        self.assertEqual(cand["ev_diagnostics"]["min_expected_edge_pct"], 0.40)
        self.assertEqual(cand["ev_diagnostics"]["min_net_edge_pct"], 0.20)
        self.assertIn("friction_diagnostics", cand)
        self.assertGreater(cand["friction_diagnostics"]["total_pct"], cand["gross_edge_pct"])

    def test_relaxed_paper_ev_is_alternative_pass_path(self):
        cfg = merge_config({
            "signal": {
                "min_expected_edge_pct": 0.40,
                "min_net_edge_pct": 1.00,
                "paper_ev_relaxation_enabled": True,
                "default_min_expected_net_profit_usd": 1.00,
            }
        })
        cand = evaluate_candidate(
            self._candidate("RELAX", 80, "trend", score=8),
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="neutral",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats={"neutral:trend": {"5d": {"n": 10, "avg_return_pct": 1.0}}},
            min_position_usd=100,
            config=cfg,
        )
        ev = cand["ev_diagnostics"]
        self.assertFalse(ev["ev_pass"])
        self.assertTrue(ev["relaxed_ev_pass"])
        self.assertTrue(ev["ev_gate_passed"])
        self.assertTrue(cand["tradable"])
        self.assertGreaterEqual(ev["expected_net_profit_usd"], 1.00)
        self.assertGreater(ev["net_edge_pct"], 0)
        self.assertIn("edge_to_required_ratio", ev)
        self.assertNotIn("reward_to_risk", ev)

        untrusted = dict(self._candidate("NOQUOTE", 80, "trend", score=8))
        untrusted["execution_trusted"] = False
        blocked = evaluate_candidate(
            untrusted,
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="neutral",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats={"neutral:trend": {"5d": {"n": 10, "avg_return_pct": 1.0}}},
            min_position_usd=100,
            config=cfg,
        )
        self.assertFalse(blocked["ev_diagnostics"]["relaxed_ev_pass"])
        self.assertFalse(blocked["tradable"])

    def test_amzn_style_weak_edge_after_friction_stays_blocked(self):
        candidate = self._candidate("AMZN", 40, "trend", score=2)
        candidate["ctx"]["vol_ratio"] = 1.2
        cand = evaluate_candidate(
            candidate,
            total_equity=4114.285714285714,
            regime_stop_pct=-6,
            regime_kind="neutral",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats={"neutral:trend": {"5d": {"n": 8, "avg_return_pct": 0.08}}},
            min_position_usd=100,
        )
        # conf-40 bucket 0.70 → 0.80 (audit P2-17): 504 → 576
        self.assertEqual(cand["risk"]["target_notional"], 576.0)
        self.assertEqual(cand["gross_edge_pct"], 0.08)
        self.assertAlmostEqual(cand["friction"]["total_pct"], 0.49, places=2)
        self.assertAlmostEqual(cand["net_edge_pct"], -0.41, places=2)
        self.assertFalse(cand["tradable"])
        self.assertEqual(cand["rank_reason_code"], "EDGE_TOO_LOW")

    def test_market_data_mode_multiplier_applies_after_base_sizing(self):
        base = evaluate_candidate(
            self._candidate("BASE", 80, "trend", score=8),
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="neutral",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats={},
            min_position_usd=100,
        )
        proxy = evaluate_candidate(
            self._candidate("PROXY", 80, "trend", score=8),
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="neutral",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats={},
            min_position_usd=100,
            mode_size_mult=0.85,
            mode_size_reason="PROXY_MODE",
        )
        self.assertEqual(proxy["risk"]["pre_mode_target_notional"],
                         base["risk"]["target_notional"])
        self.assertAlmostEqual(proxy["risk"]["target_notional"],
                               round(base["risk"]["target_notional"] * 0.85, 2))
        self.assertIn("PROXY_MODE", proxy["risk"]["size_penalties"])

    def test_risk_and_kelly_defaults_match_paper_contract(self):
        risk = DEFAULT_CONFIG["risk"]
        kelly = DEFAULT_CONFIG["kelly"]
        signal = DEFAULT_CONFIG["signal"]
        modes = DEFAULT_CONFIG["market_data_modes"]
        scan = DEFAULT_CONFIG["scan"]
        self.assertEqual(signal["min_buy_confidence"], 40)
        self.assertEqual(risk["min_trade_size"], 150.0)
        self.assertEqual(risk["max_new_buys_per_tick"], 10)
        self.assertEqual(risk["max_new_buys_per_day"], 30)
        self.assertEqual(risk["max_positions"], 12)
        self.assertEqual(risk["max_position_pct"], 0.35)
        self.assertEqual(risk["max_sector_pct"], 1.00)
        self.assertEqual(risk["max_corr_group_pct"], 1.00)
        self.assertEqual(risk["max_gross_exposure_pct"], 1.00)
        self.assertEqual(risk["min_cash_reserve_pct"], 0.00)
        self.assertEqual(risk["daily_loss_warning_pct"], -0.10)
        self.assertEqual(risk["daily_loss_hard_lockout_pct"], -0.15)
        self.assertEqual(risk["daily_loss_limit_pct"], -0.15)
        self.assertEqual(risk["drawdown_warning_pct"], -0.20)
        self.assertEqual(risk["drawdown_hard_lockout_pct"], -0.35)
        self.assertEqual(risk["hard_drawdown_lockout_pct"], -0.35)
        self.assertFalse(kelly["enabled"])
        self.assertEqual(kelly["min_samples"], 100)
        self.assertEqual(kelly["max_mult"], 1.0)
        self.assertEqual(modes["proxy_size_mult"], 0.85)
        self.assertEqual(modes["degraded_size_mult"], 0.90)
        self.assertTrue(modes["degraded_use_standard_gates_for_testing"])
        self.assertEqual(modes["degraded_min_confidence"], 40)
        self.assertEqual(modes["degraded_max_new_buys_per_tick"], 10)
        self.assertEqual(modes["degraded_max_new_buys_per_day"], 30)
        self.assertTrue(scan["scan_buys_enabled"])
        self.assertEqual(scan["scan_buy_min_confidence"], 45)
        self.assertEqual(scan["scan_data_max_age_sec"], 300)
        self.assertEqual(scan["max_scan_buys_per_tick"], 10)
        self.assertEqual(scan["max_scan_buys_per_day"], 30)
        self.assertEqual(modes["degraded_max_position_pct"], 0.05)
        self.assertEqual(modes["degraded_max_gross_exposure_pct"], 0.35)
        self.assertTrue(modes["degraded_require_buy_candidate"])
        self.assertTrue(modes["degraded_require_fresh_quote"])
        self.assertTrue(modes["degraded_require_normal_ev_gates"])
        self.assertTrue(modes["degraded_require_normal_risk_caps"])
        self.assertFalse(modes["degraded_block_experimental"])
        self.assertFalse(modes["degraded_block_confidence_prior"])
        self.assertFalse(modes["degraded_block_low_volume_penalty"])
        self.assertFalse(modes["degraded_block_pyramiding"])

    def test_missing_spy_is_degraded_label_with_normal_gates(self):
        # Audit P1-6 reverses the warning-only pin: missing SPY now reports
        # DEGRADED_MODE truthfully, while standard-gates parity (Rule 1) keeps
        # every effective gate identical to NORMAL.
        import trading.bot as bot

        mode = bot._market_data_mode(
            {"spy_data_ok": False},
            {"data_ok": True, "source": "spy_realized_vol_proxy",
             "volatility_source": "spy_realized_vol_proxy"},
            DEFAULT_CONFIG,
        )
        self.assertEqual(mode["trading_mode"], "DEGRADED_MODE")
        self.assertTrue(mode["degraded_mode_active"])
        self.assertTrue(mode["degraded_standard_gates_active"])
        self.assertEqual(mode["degraded_gate_policy"], "standard_gates_for_testing")
        self.assertEqual(mode["mode_size_mult"], 1.0)
        self.assertIsNone(mode["mode_size_reason"])
        self.assertEqual(mode["effective_min_buy_confidence"], 40)
        self.assertTrue(mode["normal_ev_gates_required"])
        self.assertTrue(mode["normal_risk_caps_required"])
        self.assertTrue(mode["fresh_quote_required"])
        self.assertTrue(mode["allow_buys"])
        self.assertEqual(mode["data_health_blocks"], ["SPY_DATA_MISSING"])
        self.assertEqual(mode["data_health_warnings"], ["SPY_DATA_MISSING"])

    def test_missing_volatility_is_degraded_label_with_normal_gates(self):
        import trading.bot as bot

        mode = bot._market_data_mode(
            {"spy_data_ok": True},
            {"data_ok": False, "source": None, "volatility_source": None},
            DEFAULT_CONFIG,
        )
        self.assertEqual(mode["trading_mode"], "DEGRADED_MODE")
        self.assertTrue(mode["degraded_mode_active"])
        self.assertTrue(mode["degraded_standard_gates_active"])
        self.assertEqual(mode["mode_size_mult"], 1.0)
        self.assertIsNone(mode["mode_size_reason"])
        self.assertTrue(mode["allow_buys"])
        self.assertEqual(mode["data_health_blocks"], ["VOLATILITY_DATA_MISSING"])
        self.assertEqual(mode["data_health_warnings"], ["VOLATILITY_DATA_MISSING"])

    def test_degraded_parity_identical_decisions_and_sizes(self):
        # Rule 1 regression: the same candidate pool ranked under NORMAL and
        # DEGRADED(standard gates) mode_info must produce identical decisions.
        import trading.bot as bot

        normal = bot._market_data_mode(
            {"spy_data_ok": True},
            {"data_ok": True, "source": "vix", "volatility_source": "vix"},
            DEFAULT_CONFIG,
        )
        degraded = bot._market_data_mode(
            {"spy_data_ok": False},
            {"data_ok": False, "source": None, "volatility_source": None},
            DEFAULT_CONFIG,
        )
        self.assertEqual(normal["trading_mode"], "NORMAL_MODE")
        self.assertEqual(degraded["trading_mode"], "DEGRADED_MODE")
        pool = [
            _entry_candidate("PAR1", confidence=55, cluster="trend"),
            _entry_candidate("PAR2", confidence=42, cluster="mixed", score=0),
            _entry_candidate("PAR3", confidence=75, cluster="dip", score=6),
        ]
        results = {}
        for label, mode in (("normal", normal), ("degraded", degraded)):
            ranked = rank_candidates(
                [dict(c, ctx=dict(c["ctx"]), rec=dict(c["rec"])) for c in pool],
                10_000, -6, "neutral", 1, 1, 1, {},
                min_position_usd=150,
                mode_size_mult=mode["mode_size_mult"],
                mode_size_reason=mode["mode_size_reason"],
            )
            results[label] = [
                (c["ticker"], c["tradable"], c["rank_reason_code"],
                 c["risk"]["target_notional"])
                for c in ranked
            ]
        self.assertEqual(results["normal"], results["degraded"])

    def test_confidence_scale_uses_40_percent_entry_buckets(self):
        self.assertEqual(confidence_scale(39), 0.0)
        self.assertEqual(confidence_scale(40), 0.80)
        self.assertEqual(confidence_scale(44), 0.80)
        self.assertEqual(confidence_scale(45), 0.95)
        self.assertEqual(confidence_scale(50), 1.00)
        self.assertEqual(confidence_scale(55), 1.10)
        self.assertEqual(confidence_scale(65), 1.25)
        self.assertEqual(confidence_scale(70), 1.60)

    def test_scan_source_gets_smaller_risk_budget(self):
        base = self._candidate("BASE", 75, "trend")
        scan = dict(base)
        scan["source"] = "scan"
        watch_eval = evaluate_candidate(base, 10_000, -6, "neutral", 1, 1, 1, {},
                                        min_position_usd=100)
        scan_eval = evaluate_candidate(scan, 10_000, -6, "neutral", 1, 1, 1, {},
                                       min_position_usd=100)
        self.assertLess(scan_eval["risk"]["risk_pct"], watch_eval["risk"]["risk_pct"])

    def test_non_dip_low_volume_soft_size_penalty(self):
        hi = self._candidate("HI", 80, "trend")
        lo = self._candidate("LO", 80, "trend")
        hi["ctx"]["vol_ratio"] = 1.5
        lo["ctx"]["vol_ratio"] = 0.5
        hi_eval = evaluate_candidate(hi, 10_000, -6, "neutral", 1, 1, 1, {},
                                     min_position_usd=100)
        lo_eval = evaluate_candidate(lo, 10_000, -6, "neutral", 1, 1, 1, {},
                                     min_position_usd=100)
        self.assertLess(lo_eval["risk"]["target_notional"],
                        hi_eval["risk"]["target_notional"])
        self.assertIn("LOW_VOLUME_PENALTY_ONLY", lo_eval["risk"]["size_penalties"])
        self.assertEqual(lo_eval["warnings"], ["LOW_VOLUME_PENALTY_ONLY"])

    def test_weak_raw_buys_do_not_display_as_buy(self):
        self.assertEqual(classify_display_signal("buy", 28), "BULLISH_LEAN")
        self.assertEqual(classify_display_signal("buy", 34), "BULLISH_LEAN")
        self.assertEqual(classify_display_signal("buy", 39), "BULLISH_LEAN")
        self.assertEqual(classify_display_signal("buy", 40), "BUY_CANDIDATE")
        self.assertEqual(classify_display_signal("buy", 54), "BUY_CANDIDATE")
        self.assertEqual(classify_display_signal("buy", 69), "BUY_CANDIDATE")
        self.assertEqual(classify_display_signal("buy", 70), "STRONG_BUY_CANDIDATE")

    def test_execution_candidate_helper_uses_display_label_and_shared_threshold(self):
        import trading.bot as bot

        cfg = DEFAULT_CONFIG
        self.assertTrue(bot.is_execution_candidate({
            "cls": "hold", "signal": "HOLD",
            "display_signal_label": "BUY_CANDIDATE", "confidence": 80,
        }, cfg))
        self.assertTrue(bot.is_execution_candidate({
            "cls": "hold", "signal": "HOLD",
            "display_signal_label": "HOLD", "confidence": 64,
        }, cfg))
        self.assertTrue(bot.is_execution_candidate({
            "cls": "hold", "signal": "HOLD",
            "display_signal_label": "HOLD", "confidence": 74,
        }, cfg))
        self.assertFalse(bot.is_execution_candidate({
            "cls": "sell", "signal": "SELL",
            "display_signal_label": "SELL", "confidence": 50,
        }, cfg))
        self.assertTrue(bot.is_execution_candidate({
            "cls": "buy", "signal": "BUY",
            "display_signal_label": "BUY_CANDIDATE", "confidence": 70,
            "catalyst": {"type": "guidance_cut"},
        }, cfg))
        self.assertTrue(bot.is_execution_candidate({
            "cls": "buy", "signal": "BUY",
            "display_signal_label": "BULLISH_LEAN", "confidence": 39,
        }, cfg))
        self.assertTrue(bot.is_execution_candidate({
            "cls": "buy", "signal": "BUY",
            "display_signal_label": "WATCH_OR_LEAN", "confidence": 54,
        }, cfg))
        self.assertTrue(bot.is_execution_candidate({
            "cls": "buy", "signal": "BUY",
            "display_signal_label": "BUY_CANDIDATE", "confidence": 39,
        }, cfg))
        self.assertFalse(bot.is_execution_candidate({
            "cls": "hold", "signal": "HOLD",
            "display_signal_label": "HOLD", "confidence": 54,
        }, cfg))
        weak = {"cls": "buy", "signal": "BUY",
                "display_signal_label": "BULLISH_LEAN", "confidence": 39}
        weak_profile = bot._execution_profile(weak, cfg)
        self.assertEqual(weak_profile["candidate_type"], "weak_paper_test_candidate")
        self.assertEqual(weak_profile["candidate_size_multiplier"], 0.85)
        hold_profile = bot._execution_profile({
            "cls": "hold", "signal": "HOLD",
            "display_signal_label": "HOLD", "confidence": 55,
        }, cfg)
        self.assertEqual(hold_profile["candidate_type"], "experimental_near_buy_candidate")
        self.assertEqual(hold_profile["candidate_size_multiplier"], 0.65)
        self.assertTrue(bot.is_execution_candidate({
            "cls": "buy", "signal": "BUY",
            "display_signal_label": "BUY_CANDIDATE", "confidence": 40,
        }, cfg))
        self.assertTrue(bot.is_execution_candidate({
            "cls": "buy", "signal": "BUY",
            "display_signal_label": "BUY_CANDIDATE", "confidence": 40,
        }, cfg, degraded_mode_active=True))
        self.assertTrue(bot.is_execution_candidate({
            "cls": "buy", "signal": "BUY",
            "display_signal_label": "BUY_CANDIDATE", "confidence": 54,
        }, cfg))
        self.assertTrue(bot.is_execution_candidate({
            "cls": "buy", "signal": "BUY",
            "display_signal_label": "BUY_CANDIDATE", "confidence": 54,
        }, cfg, degraded_mode_active=True))
        self.assertTrue(bot.is_execution_candidate({
            "cls": "strong-buy", "signal": "STRONG_BUY",
            "display_signal_label": "STRONG_BUY_CANDIDATE", "confidence": 70,
        }, cfg, degraded_mode_active=True))
        standard_degraded_cfg = merge_config({
            "market_data_modes": {
                "degraded_min_confidence": 65,
                "degraded_use_standard_gates_for_testing": True,
            }
        })
        self.assertTrue(bot.is_execution_candidate({
            "cls": "buy", "signal": "BUY",
            "display_signal_label": "BUY_CANDIDATE", "confidence": 40,
        }, standard_degraded_cfg, degraded_mode_active=True))
        restricted_degraded_cfg = merge_config({
            "market_data_modes": {
                "degraded_min_confidence": 65,
                "degraded_use_standard_gates_for_testing": False,
            }
        })
        self.assertFalse(bot.is_execution_candidate({
            "cls": "buy", "signal": "BUY",
            "display_signal_label": "BUY_CANDIDATE", "confidence": 40,
        }, restricted_degraded_cfg, degraded_mode_active=True))

    def test_confidence_prior_watchlist_warmup_requires_min_confidence(self):
        cand = self._candidate("WARM", 35, "trend", atr_pct=2.0, score=4.0)
        cand["rec"]["cls"] = "strong-buy"
        cand["rec"]["signal"] = "STRONG BUY"
        cand["rec"]["sizing_confidence"] = 70
        out = evaluate_candidate(
            cand,
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="bull",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats={},
            min_position_usd=100,
        )
        self.assertFalse(out["tradable"])
        self.assertIn("EV gate", out["rank_reason"])
        self.assertGreaterEqual(out["risk"]["target_notional"], 100)
        self.assertEqual(out["sizing_confidence"], 70)

    def test_ev_dead_zone_fixed_conf50_tradable_conf42_not(self):
        # Audit P0-1 sanity targets: conf-50 liquid watchlist BUY tradable,
        # conf-42 never (prior = 0 → no gross edge).
        liquid = evaluate_candidate(
            self._candidate("LIQ", 50, "trend", score=0),
            total_equity=10_000, regime_stop_pct=-6, regime_kind="neutral",
            vix_mult=1, streak_mult=1, kelly_mult=1, edge_stats={},
            min_position_usd=100,
        )
        self.assertTrue(liquid["tradable"])

        dead = evaluate_candidate(
            self._candidate("DEAD", 42, "trend", score=0),
            total_equity=10_000, regime_stop_pct=-6, regime_kind="neutral",
            vix_mult=1, streak_mult=1, kelly_mult=1, edge_stats={},
            min_position_usd=100,
        )
        self.assertFalse(dead["tradable"])
        self.assertEqual(dead["gross_edge_pct"], 0.0)

    def test_warmup_path_gross_edge_gate_and_size_haircut(self):
        # Small notional → friction eats the prior edge; only the warmup
        # path (conf ≥ 48, gross > 0) may allow it, at 0.60 size.
        warm_cand = self._candidate("WARM", 50, "trend", score=0)
        warm_cand["ctx"]["avg_dollar_vol_20d"] = 2_000_000
        warm = evaluate_candidate(
            warm_cand,
            total_equity=3_000, regime_stop_pct=-6, regime_kind="neutral",
            vix_mult=1, streak_mult=1, kelly_mult=1, edge_stats={},
            min_position_usd=100,
        )
        self.assertLessEqual(warm["gross_edge_pct"],
                             warm["ev_diagnostics"]["required_edge_pct"])
        self.assertFalse(warm["ev_diagnostics"]["ev_pass"])
        self.assertTrue(warm["ev_diagnostics"]["warmup_watchlist"])
        self.assertTrue(warm["tradable"])
        self.assertEqual(warm["rank_reason_code"], "WARMUP_CONFIDENCE_PRIOR_ALLOWED")
        self.assertIn("WARMUP_CONFIDENCE_PRIOR_SIZE", warm["risk"]["size_penalties"])

        below_cand = self._candidate("BELOW", 47, "trend", score=0)
        below_cand["ctx"]["avg_dollar_vol_20d"] = 2_000_000
        below = evaluate_candidate(
            below_cand,
            total_equity=3_000, regime_stop_pct=-6, regime_kind="neutral",
            vix_mult=1, streak_mult=1, kelly_mult=1, edge_stats={},
            min_position_usd=100,
        )
        self.assertFalse(below["ev_diagnostics"]["warmup_watchlist"])
        self.assertFalse(below["tradable"])

    def test_exit_quality_size_penalty(self):
        edge_stats = {
            "exit_attribution_buckets": {
                "neutral:trend": {"n": 50, "too_late_rate_pct": 46}
            }
        }
        late = self._candidate("LATE", 80, "trend")
        late["ctx"]["vol_ratio"] = 1.5
        cand = evaluate_candidate(
            late,
            10_000, -6, "neutral", 1, 1, 1, edge_stats,
            min_position_usd=100,
        )
        self.assertIn("exit quality too-late penalty", cand["risk"]["size_penalties"])
        self.assertEqual(cand["risk"]["size_mult"], 0.75)


class TradingClusterExitLadderTests(unittest.TestCase):
    def test_cluster_priority_confirmed_news_then_technicals(self):
        rec = {
            "categories": {"news": 2.0, "trend": 1.0},
            "reasons": ["Bullish news catalyst"],
        }
        ctx = {
            "vol_ratio": 1.6,
            "mom_30d_pct": 3.0,
            "macd_hist": 0.4,
            "macd_hist_prev": 0.1,
            "rsi": 60,
            "current": 105,
            "ma30": 100,
        }
        self.assertEqual(entry_cluster(rec, ctx), "news_catalyst_confirmed")

        weak = dict(rec)
        weak["categories"] = {"news": 2.0}
        self.assertEqual(entry_cluster(weak, {"vol_ratio": 0.8, "rsi": 55}), "mixed")

        dip = {"categories": {"news": 1.0}, "reasons": ["news"]}
        self.assertEqual(entry_cluster(dip, {"is_dip": True, "rsi": 35}), "dip")

    def test_cluster_ladder_composes_and_clamps(self):
        profile = compose_exit_profile(
            "dip",
            {"live": True, "stop_mult": 2.0, "trail_mult": 2.0,
             "aging_mult": 3.0, "notes": "wide"},
        )
        self.assertEqual(profile["cluster"], "dip")
        self.assertEqual(profile["atr_stop_mult"], 1.8)
        self.assertEqual(profile["trail_mult"], 1.5)
        self.assertEqual(profile["max_hold_days"], 25.0)
        self.assertEqual(profile["failure_timeout_days"], 8.0)
        self.assertEqual(normalize_cluster("trend"), "trend_continuation")
        self.assertEqual(normalize_cluster("news"), "mixed")

    def test_learned_exit_quality_adjusts_partial_take(self):
        late = compose_exit_profile(
            "mixed",
            {"live": True, "too_late_rate_pct": 40.0},
        )
        early = compose_exit_profile(
            "mixed",
            {"live": True, "too_early_rate_pct": 30.0},
        )
        self.assertAlmostEqual(late["partial_take_pct"], 0.051, places=3)
        self.assertAlmostEqual(early["partial_take_pct"], 0.069, places=3)

    def test_regime_exit_tightening_is_clamped(self):
        profile = compose_exit_profile("trend_continuation", {})
        tightened = apply_regime_exit_tightening(profile, "panic")
        self.assertEqual(tightened["regime_v3_exit_tightening"], "panic")
        self.assertLessEqual(tightened["trail_mult"], profile["trail_mult"])
        self.assertGreaterEqual(tightened["atr_stop_mult"], 0.80)


class TradingPortfolioVarianceTests(unittest.TestCase):
    def _history(self):
        import pandas as pd

        idx = pd.date_range("2025-01-01", periods=31, freq="B")
        base = pd.Series([100 + i for i in range(31)], index=idx)
        return {
            "AAA": base,
            "BBB": base * 2,
            "CCC": pd.Series([80 + i * 0.1 for i in range(31)], index=idx),
        }

    def test_static_corr_and_stress_override(self):
        sector = {"AAA": "tech", "BBB": "tech", "CCC": "health"}.__getitem__
        group = {"AAA": "ai", "BBB": "ai", "CCC": None}.get
        self.assertEqual(static_corr_assumption("AAA", "BBB", sector, group), 0.70)
        corr_normal = smoothed_corr("AAA", "CCC", self._history(), "neutral", sector, group)
        corr_bear = smoothed_corr("AAA", "CCC", self._history(), "bear", sector, group)
        self.assertGreaterEqual(corr_bear, corr_normal)
        self.assertGreaterEqual(corr_bear, 0.45)

    def test_variance_gate_empty_soft_hard_and_debug(self):
        hist = self._history()
        sector = {"AAA": "tech", "BBB": "tech", "CCC": "health"}.get
        group = {"AAA": "ai", "BBB": "ai", "CCC": None}.get
        empty = candidate_variance_check(
            {}, {}, "AAA", 1000, 10_000, hist, sector_lookup=sector,
            corr_group_lookup=group,
        )
        self.assertEqual(empty["risk_action"], "allow")
        self.assertEqual(empty["variance_before"], 0.0)

        holdings = {"AAA": {"shares": 90, "avg_cost": 100}}
        prices = {"AAA": 100, "BBB": 200}
        hard = candidate_variance_check(
            holdings, prices, "BBB", 9000, 10_000, hist,
            gross_edge_pct=1.0, net_edge_pct=0.5,
            sector_lookup=sector, corr_group_lookup=group,
        )
        self.assertEqual(hard["risk_action"], "skip")
        self.assertIn(hard["skip_reason"], ("portfolio_variance_too_high",
                                             "portfolio_variance_extreme"))
        self.assertIn("candidate_risk_contribution_pct", hard)

        override = candidate_variance_check(
            holdings, prices, "BBB", 9000, 10_000, hist,
            gross_edge_pct=3.0, net_edge_pct=2.0, paper_debug_override=True,
            sector_lookup=sector, corr_group_lookup=group,
        )
        self.assertNotEqual(override["risk_action"], "skip")
        self.assertLess(override["size_mult"], 1.0)


class TradingV3ConfigRegimeCatalystTests(unittest.TestCase):
    def test_config_merge_hash_stable_and_no_live_mutation(self):
        base_hash = config_hash(DEFAULT_CONFIG)
        merged = merge_config({"signal.min_buy_confidence": 60})
        self.assertEqual(config_hash(DEFAULT_CONFIG), base_hash)
        self.assertNotEqual(config_hash(merged), base_hash)
        self.assertEqual(DEFAULT_CONFIG["signal"]["min_buy_confidence"], 40)
        self.assertEqual(merged["signal"]["min_buy_confidence"], 60)

    def test_config_merge_deep_merges_nested_blocks(self):
        merged = merge_config({"suggestion": {"enabled": False}})
        self.assertFalse(merged["suggestion"]["enabled"])
        self.assertEqual(merged["suggestion"]["max_extra_tickers"], 2)
        self.assertTrue(DEFAULT_CONFIG["suggestion"]["enabled"])

    def test_pa_light_regime_is_source_labeled(self):
        import pandas as pd
        import trading.risk as risk

        old_daily = risk._regime_daily_bars
        old_append = risk._append_live_bar
        old_credit = risk.credit_signal
        old_vix = risk.get_vix
        old_open = risk.is_market_open
        try:
            idx = pd.date_range("2025-01-01", periods=70, freq="B")
            risk._regime_daily_bars = lambda _tk: pd.DataFrame({
                "Close": list(range(100, 170)),
            }, index=idx)
            risk._append_live_bar = lambda df, _tk: df
            risk.credit_signal = lambda: {"credit_label": "neutral", "credit_pct": 50.0}
            risk.get_vix = lambda: {"vix": 12.0, "regime": "NORMAL", "mult": 1.0}
            risk.is_market_open = lambda: False
            regime = risk.get_market_regime_pa_light()
        finally:
            risk._regime_daily_bars = old_daily
            risk._append_live_bar = old_append
            risk.credit_signal = old_credit
            risk.get_vix = old_vix
            risk.is_market_open = old_open
        self.assertEqual(regime["regime_v3_source"], "pa_light")
        self.assertIn("regime_v3_effective", regime)
        self.assertTrue(regime["spy_data_ok"])
        self.assertGreater(regime["spy_rows"], 50)

    def test_pa_light_missing_spy_history_is_explicit_fallback(self):
        import trading.risk as risk

        old_daily = risk._regime_daily_bars
        old_credit = risk.credit_signal
        old_vix = risk.get_vix
        old_open = risk.is_market_open
        try:
            risk._regime_daily_bars = lambda _tk: None
            risk.credit_signal = lambda: {"credit_label": "neutral", "credit_pct": 50.0}
            risk.get_vix = lambda: {"vix": 12.0, "regime": "NORMAL", "mult": 1.0}
            risk.is_market_open = lambda: False
            regime = risk.get_market_regime_pa_light()
        finally:
            risk._regime_daily_bars = old_daily
            risk.credit_signal = old_credit
            risk.get_vix = old_vix
            risk.is_market_open = old_open
        self.assertEqual(regime["regime"], "neutral")
        self.assertIsNone(regime["spy_mom_30d"])
        self.assertIsNone(regime["above_ma50"])
        self.assertFalse(regime["spy_data_ok"])
        self.assertTrue(regime["regime_data_fallback"])
        self.assertEqual(regime["regime_data_status"], "missing_spy_history")

    def test_market_regime_missing_spy_history_is_not_fake_zero(self):
        import types
        import trading.risk as risk

        old_pa = risk.PYTHONANYWHERE_MODE
        old_daily = risk._regime_daily_bars
        old_credit = risk.credit_signal
        old_open = risk.is_market_open
        old_cache_get = risk.cache_get
        old_cache_set = risk.cache_set
        old_load = risk.load_live_close_history
        try:
            risk.PYTHONANYWHERE_MODE = False
            fake_yf = types.SimpleNamespace(
                Ticker=lambda *_args, **_kwargs: types.SimpleNamespace(
                    history=lambda *_a, **_k: types.SimpleNamespace(empty=True)
                )
            )
            risk._regime_daily_bars = lambda _tk: None
            risk.credit_signal = lambda: {"credit_label": "neutral", "credit_pct": 50.0}
            risk.is_market_open = lambda: False
            risk.cache_get = lambda *_args, **_kwargs: None
            risk.cache_set = lambda *_args, **_kwargs: None
            risk.load_live_close_history = lambda *_args, **_kwargs: {}
            with patch.dict(sys.modules, {"yfinance": fake_yf}):
                regime = risk.get_market_regime()
        finally:
            risk.PYTHONANYWHERE_MODE = old_pa
            risk._regime_daily_bars = old_daily
            risk.credit_signal = old_credit
            risk.is_market_open = old_open
            risk.cache_get = old_cache_get
            risk.cache_set = old_cache_set
            risk.load_live_close_history = old_load
        self.assertEqual(regime["regime"], "neutral")
        self.assertIsNone(regime["spy_mom_30d"])
        self.assertFalse(regime["spy_data_ok"])
        self.assertEqual(regime["regime_data_status"], "missing_spy_history")

    def test_vix_missing_spy_history_is_explicit_unknown(self):
        import trading.risk as risk

        old_daily = risk._regime_daily_bars
        old_cache_get = risk.cache_get
        old_cache_set = risk.cache_set
        try:
            risk._regime_daily_bars = lambda _tk: None
            risk.cache_get = lambda *_args, **_kwargs: None
            risk.cache_set = lambda *_args, **_kwargs: None
            out = risk.get_vix()
        finally:
            risk._regime_daily_bars = old_daily
            risk.cache_get = old_cache_get
            risk.cache_set = old_cache_set
        self.assertIsNone(out["vix"])
        self.assertEqual(out["regime"], "UNKNOWN")
        self.assertEqual(out["mult"], 1.0)
        self.assertFalse(out["data_ok"])
        self.assertEqual(out["data_status"], "missing_or_insufficient_spy_history")

    def test_pa_breadth_universe_is_trimmed(self):
        from trading.regime_v3 import BREADTH_UNIVERSE, BREADTH_UNIVERSE_PA, get_breadth_universe

        self.assertEqual(len(BREADTH_UNIVERSE_PA), 30)
        self.assertGreater(len(BREADTH_UNIVERSE), len(BREADTH_UNIVERSE_PA))
        self.assertEqual(get_breadth_universe(pa_mode=True), BREADTH_UNIVERSE_PA)
        self.assertEqual(get_breadth_universe(pa_mode=False), BREADTH_UNIVERSE)

    def test_breadth_quality_falls_back_when_count_low(self):
        import pandas as pd

        idx = pd.date_range("2025-01-01", periods=60, freq="B")
        close_history = {
            f"T{i:02d}": pd.Series(range(100, 160), index=idx)
            for i in range(29)
        }
        metrics = breadth_metrics(close_history, pd.Series(range(400, 460), index=idx))
        self.assertFalse(metrics["breadth_sufficient"])
        self.assertEqual(metrics["actual_breadth_count"], 29)
        self.assertEqual(metrics["min_effective_breadth_count"], 30)
        label, reason = classify_regime(metrics)
        self.assertEqual(label, "fallback")
        self.assertIn("insufficient", reason)

    def test_panic_rule_and_confirmation(self):
        metrics = {
            "breadth_sufficient": True,
            "pct_above_50ma": 0.20,
            "pct_above_200ma": 0.20,
            "spy_above_50ma": False,
            "spy_above_200ma": False,
            "realized_vol_20d": 0.25,
            "qqq_spy_rs_20d": 0.0,
            "rsp_spy_rs_20d": 0.0,
            "hyg_ief_rs_20d": 0.0,
            "defensive_strength": False,
        }
        label, _ = classify_regime(metrics)
        self.assertEqual(label, "panic")
        state = {"confirmed_regime": "weak_bull"}
        self.assertEqual(apply_confirmation(state, "panic", "test", ts=1), "panic")
        self.assertEqual(regime_risk_mult("panic"), 0.0)

    def test_hysteresis_buffer_avoids_tiny_strong_bull_flip(self):
        metrics = {
            "breadth_sufficient": True,
            "pct_above_50ma": 0.66,
            "pct_above_200ma": 0.61,
            "spy_above_50ma": True,
            "spy_above_200ma": True,
            "realized_vol_20d": 0.10,
            "qqq_spy_rs_20d": 0.0,
            "rsp_spy_rs_20d": 0.0,
            "hyg_ief_rs_20d": 0.0,
            "defensive_strength": False,
        }
        label, _ = classify_regime(metrics, previous="weak_bull")
        self.assertNotEqual(label, "strong_bull")

    def test_catalyst_shadow_confirmed_and_stale(self):
        now = 1_700_000_000
        ctx = {
            "vol_ratio": 1.8,
            "mom_30d_pct": 3.0,
            "current": 105.0,
            "ma30": 100.0,
            "macd_hist": 0.3,
            "macd_hist_prev": 0.1,
        }
        catalyst = classify_catalyst(
            articles=[{"title": "Company raises outlook after earnings beat",
                       "pub_ts": now - 3600}],
            ctx=ctx,
            now_ts=now,
        )
        self.assertEqual(catalyst["type"], "guidance_raise")
        self.assertTrue(catalyst["confirmed"])
        self.assertFalse(catalyst["size_effect_enabled"])
        self.assertFalse(catalyst["score_effect_enabled"])
        self.assertEqual(catalyst_cluster(catalyst), "news_catalyst_confirmed")

        stale = classify_catalyst(
            articles=[{"title": "Company raises outlook", "pub_ts": now - 49 * 3600}],
            ctx=ctx,
            now_ts=now,
        )
        self.assertEqual(stale["type"], "mixed_or_unclear")
        self.assertFalse(stale["confirmed"])

    def test_weak_catalyst_does_not_override_technical_cluster(self):
        catalyst = classify_catalyst(
            analyst={"net": 1.0, "total": 5, "age_hours": 10},
            ctx={"vol_ratio": 0.7, "rsi": 55},
        )
        self.assertEqual(catalyst["type"], "analyst_upgrade")
        self.assertFalse(catalyst["confirmed"])
        rec = {"catalyst": catalyst, "categories": {"news": 2.0}}
        self.assertEqual(entry_cluster(rec, {"vol_ratio": 0.7, "rsi": 55}), "mixed")


class TradingV2AttributionTests(unittest.TestCase):
    def test_bucket_init_uses_prior(self):
        bucket = init_entry_bucket({})
        self.assertEqual(bucket["alpha"], ALPHA_PRIOR)
        self.assertEqual(bucket["beta"], BETA_PRIOR)
        self.assertEqual(bucket["n"], 0)

    def test_success_failure_increment_by_one_not_vote_magnitude(self):
        state = {}
        _add_entry_outcome(state, 1, 3.0)
        bucket = state["attribution_buckets"]["neutral:trend:trend"]
        self.assertEqual(bucket["alpha"], ALPHA_PRIOR + 1)
        self.assertEqual(bucket["beta"], BETA_PRIOR)
        _add_entry_outcome(state, 2, -2.0)
        bucket = state["attribution_buckets"]["neutral:trend:trend"]
        self.assertEqual(bucket["alpha"], ALPHA_PRIOR + 1)
        self.assertEqual(bucket["beta"], BETA_PRIOR + 1)
        self.assertEqual(bucket["n"], 2)

    def test_neutral_outcome_updates_ema_not_alpha_beta(self):
        state = {}
        _add_entry_outcome(state, 1, 0.2)
        bucket = state["attribution_buckets"]["neutral:trend:trend"]
        self.assertEqual(bucket["alpha"], ALPHA_PRIOR)
        self.assertEqual(bucket["beta"], BETA_PRIOR)
        self.assertEqual(bucket["neutral"], 1)
        self.assertEqual(bucket["ema_net_ret_pct"], 0.2)

    def test_skipped_candidates_fill_edge_buckets_with_per_tick_cap(self):
        # Audit P0-2: top skipped candidates are recorded (decision="skipped")
        # so forward-return buckets fill without trading; cap 5 per tick.
        import trading.bot as bot
        state = {}
        bot._SKIPS_RECORDED_THIS_TICK = 0
        recorded = 0
        for i in range(7):
            cand = _entry_candidate(f"SKP{i:02d}", confidence=55, cluster="trend")
            ok = bot._record_candidate_observation(
                state, cand, "skipped", "EDGE_TOO_LOW", 1_700_000_000, "neutral")
            recorded += 1 if ok else 0
        self.assertEqual(recorded, 5)  # TRACK_TOP_SKIPS_PER_CYCLE
        events = state["attribution_events"]
        self.assertEqual(len(events), 5)
        self.assertTrue(all(e["decision"] == "skipped" for e in events))
        self.assertTrue(all(e["skip_reason"] == "EDGE_TOO_LOW" for e in events))

        updated = update_forward_outcomes(
            state, lambda tk: 105.0, ts=1_700_000_000 + 6 * 86400)
        self.assertGreater(updated, 0)
        self.assertTrue(all(e["forward_returns"].get("5d") == 5.0
                            for e in state["attribution_events"]))
        bucket = state["attribution_buckets"]["neutral:trend:trend"]
        self.assertEqual(bucket["skipped_n"], 5)
        self.assertEqual(bucket["executed_n"], 0)

        # executed events are never blocked by the skip cap
        bot._SKIPS_RECORDED_THIS_TICK = bot.TRACK_TOP_SKIPS_PER_CYCLE
        execd = bot._record_candidate_observation(
            state, _entry_candidate("EXEC1"), "executed", "buy",
            1_700_100_000, "neutral")
        self.assertTrue(execd)

    def test_negative_votes_fall_back_to_global_bucket(self):
        state = {}
        cand = _entry_candidate("NEG", cluster="mixed")
        cand["rec"]["categories"] = {"overbought": -2.0}
        event = record_entry_event(state, cand, "executed", "warning", ts=123,
                                   regime="neutral")
        event["forward_returns"]["5d"] = 1.5
        update_entry_buckets(state, event, "5d")
        self.assertIn("all:all:all", state["attribution_buckets"])
        self.assertNotIn("neutral:mixed:overbought", state["attribution_buckets"])

    def test_fallback_blend_no_live_edge_before_sixty_samples(self):
        state = {}
        for i in range(ENTRY_LIVE_N - 1):
            _add_entry_outcome(state, i, 2.0)
        edge = expected_edge_for_candidate(
            state, "neutral", "trend", {"trend": 2.0}, friction_pct=0.1
        )
        self.assertIsNone(edge)
        _add_entry_outcome(state, 99, 2.0)
        edge = expected_edge_for_candidate(
            state, "neutral", "trend", {"trend": 2.0}, friction_pct=0.1
        )
        self.assertIsNotNone(edge)
        self.assertTrue(edge["live"])
        self.assertEqual(edge["edge_source"], "attribution_v2")

    def test_expected_edge_uses_payoff_and_friction_safety(self):
        state = {}
        for i in range(45):
            _add_entry_outcome(state, i, 2.5)
        for i in range(45, 60):
            _add_entry_outcome(state, i, -1.2)
        edge = expected_edge_for_candidate(
            state, "neutral", "trend", {"trend": 2.0}, friction_pct=0.2
        )
        self.assertGreater(edge["gross_edge_pct"], 0.5)
        self.assertAlmostEqual(edge["net_edge_pct"],
                               edge["gross_edge_pct"] - 0.2, places=4)
        self.assertEqual(edge["required_edge_pct"], 0.5)
        self.assertTrue(edge["live"])

    def test_legacy_attribute_outcome_does_not_drive_live_weights(self):
        state = {}
        _attribute_outcome(state, "AAA", 25.0, {"trend": 2.0}, entry_regime="bull")
        self.assertIn("signal_weights", state)
        self.assertEqual(attribution_signal_weights(state), {})
        self.assertIsNone(expected_edge_for_candidate(
            state, "bull", "trend", {"trend": 2.0}, friction_pct=0.1
        ))

    def test_exit_profile_activates_after_sixty_samples(self):
        state = {}
        holding = {
            "shares": 1,
            "avg_cost": 100.0,
            "peak": 110.0,
            "trough": 99.0,
            "entry_cluster": "momentum",
            "entry_ts": 1_700_000_000,
        }
        for i in range(EXIT_LIVE_N - 1):
            record_exit_event(state, f"M{i}", holding, "trail", 4.0, 104.0,
                              ts=1_700_100_000 + i, regime="neutral")
        self.assertFalse(exit_profile(state, "neutral", "momentum")["live"])
        record_exit_event(state, "M60", holding, "trail", 4.0, 104.0,
                          ts=1_700_200_000, regime="neutral")
        profile = exit_profile(state, "neutral", "momentum")
        self.assertTrue(profile["live"])
        self.assertLess(profile["trail_mult"], 1.0)
        self.assertLess(profile["aging_mult"], 1.0)

    def test_attribution_trust_decays_and_caps_skipped_heavy_bucket(self):
        fresh = {"n": ENTRY_LIVE_N, "last_updated": int(time.time()),
                 "executed_n": 20, "skipped_n": 0}
        stale = {"n": ENTRY_LIVE_N, "last_updated": int(time.time()) - 120 * 86400,
                 "executed_n": 20, "skipped_n": 0}
        skipped_heavy = {"n": ENTRY_LIVE_N, "last_updated": int(time.time()),
                         "executed_n": 2, "skipped_n": 80}
        self.assertLess(sample_trust(bucket=stale), sample_trust(bucket=fresh))
        self.assertLessEqual(sample_trust(bucket=skipped_heavy), 0.50)

    def test_attribution_entry_events_cap_per_bucket_not_cluster(self):
        state = {}
        now = int(time.time())
        for i in range(180):
            cand = _entry_candidate(f"A{i:03d}", cluster="trend")
            cand["rec"]["categories"] = {"trend": 2.0}
            record_entry_event(state, cand, "executed", "test", ts=now + i,
                               regime="neutral")
        self.assertEqual(len(state["attribution_events"]), 150)

        for i in range(10):
            cand = _entry_candidate(f"B{i:03d}", cluster="trend")
            cand["rec"]["categories"] = {"dip": 2.0}
            record_entry_event(state, cand, "executed", "test", ts=now + 500 + i,
                               regime="neutral")
        self.assertEqual(len(state["attribution_events"]), 160)

    def test_attribution_pa_bucket_cap_is_100(self):
        import utils.deploy_config as dc

        old_pa = dc.PYTHONANYWHERE_MODE
        try:
            dc.PYTHONANYWHERE_MODE = True
            state = {}
            now = int(time.time())
            for i in range(130):
                cand = _entry_candidate(f"P{i:03d}", cluster="trend")
                cand["rec"]["categories"] = {"trend": 2.0}
                record_entry_event(state, cand, "executed", "test", ts=now + i,
                                   regime="neutral")
            self.assertEqual(len(state["attribution_events"]), 100)
        finally:
            dc.PYTHONANYWHERE_MODE = old_pa

    def test_record_entry_event_accepts_skipped_rejects_others(self):
        # Reversed by audit P0-2: skipped candidates ARE recorded now so edge
        # buckets can fill without trading. Other decisions stay ignored.
        state = {}
        event = record_entry_event(
            state,
            _entry_candidate("SKP", cluster="trend"),
            "skipped",
            "EV gate",
            ts=123,
            regime="neutral",
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["decision"], "skipped")
        self.assertEqual(event["skip_reason"], "EV gate")
        other = record_entry_event(
            state,
            _entry_candidate("IGN", cluster="trend"),
            "blocked",
            "whatever",
            ts=123,
            regime="neutral",
        )
        self.assertIsNone(other)
        self.assertEqual(len(state["attribution_events"]), 1)


class TradingV2IntegrationTests(unittest.TestCase):
    def test_forward_update_fills_due_horizons_and_bucket(self):
        state = {}
        now = int(time.time())
        cand = _entry_candidate("AAA", cluster="trend", friction_pct=0.1)
        cand["benchmark_prices"] = {"SPY": 100.0, "QQQ": 200.0}
        event = record_entry_event(
            state,
            cand,
            "executed",
            "EV gate",
            ts=now - 5 * 86400 - 60,
            regime="neutral",
        )
        updated = update_forward_outcomes(
            state,
            lambda ticker: {"AAA": 110.0, "SPY": 102.0, "QQQ": 198.0}.get(ticker),
            ts=now,
            benchmark_lookup=lambda ticker: {"SPY": 102.0, "QQQ": 198.0}.get(ticker),
        )
        self.assertEqual(updated, 3)
        self.assertEqual(event["forward_returns"]["5d"], 10.0)
        self.assertEqual(event["relative_returns"]["SPY_5d"], 8.0)
        self.assertEqual(event["relative_returns"]["QQQ_5d"], 11.0)
        self.assertEqual(event["mfe_pct"], 10.0)
        self.assertEqual(state["attribution_buckets"]["neutral:trend:trend"]["n"], 1)

    def test_bot_update_candidate_observations_uses_v2_state(self):
        import trading.bot as bot

        now = time.time()
        state = {}
        record_entry_event(
            state,
            _entry_candidate("AAA", cluster="trend"),
            "executed",
            "EV gate",
            ts=int(now - 5 * 86400 - 60),
            regime="neutral",
        )
        updated = bot._update_candidate_observations(
            state, {"AAA": {"price": 110.0, "stale": False}}, now
        )
        self.assertEqual(updated, 3)
        self.assertEqual(state["attribution_events"][0]["forward_returns"]["5d"], 10.0)
        self.assertEqual(state["attribution_buckets"]["neutral:trend:trend"]["n"], 1)

    def test_exit_post_update_marks_too_early(self):
        state = {}
        event = record_exit_event(
            state,
            "AAA",
            {"shares": 1, "avg_cost": 100.0, "peak": 103.0,
             "trough": 98.0, "entry_cluster": "trend", "entry_ts": 100},
            "signal_flip_profit",
            1.0,
            101.0,
            ts=1_000,
            regime="neutral",
        )
        updated = update_exit_post_outcomes(
            state, lambda ticker: 106.0 if ticker == "AAA" else None,
            ts=1_000 + 3 * 86400 + 1,
        )
        self.assertEqual(updated, 1)
        self.assertTrue(event["too_early"])
        self.assertGreater(
            state["exit_attribution_buckets"]["neutral:trend"]["too_early_n"], 0
        )
        self.assertGreater(
            state["exit_attribution_buckets"]["neutral:trend:signal_flip_profit"]["too_early_n"], 0
        )
        self.assertEqual(
            state["exit_attribution_buckets"]["neutral:trend:signal_flip_profit"]["post_3d_n"], 1
        )

    def test_exit_capture_average_uses_capture_sample_count(self):
        state = {}
        record_exit_event(
            state,
            "WIN",
            {"shares": 1, "avg_cost": 100.0, "peak": 105.0,
             "trough": 100.0, "entry_cluster": "trend", "entry_ts": 100},
            "trail",
            2.0,
            102.0,
            ts=1_000,
            regime="neutral",
        )
        record_exit_event(
            state,
            "FLAT",
            {"shares": 1, "avg_cost": 100.0, "peak": 100.0,
             "trough": 98.0, "entry_cluster": "trend", "entry_ts": 100},
            "loss",
            -1.0,
            99.0,
            ts=2_000,
            regime="neutral",
        )
        bucket = state["exit_attribution_buckets"]["neutral:trend"]
        self.assertEqual(bucket["n"], 2)
        self.assertEqual(bucket["capture_n"], 1)
        self.assertEqual(bucket["avg_capture_ratio"], 0.4)


class TradingV1SignalTests(unittest.TestCase):
    def _raw_buy_payload(self, *, confidence=20, score=2.0, cats_pos=3,
                         cats_neg=1, data_quality=0.70, floor_candidate=True,
                         blockers=None, quote_stale=False, history_rows=400,
                         history_source="fmp_daily", history_status="ok"):
        return {
            "rec": {
                "cls": "buy",
                "signal": "BUY",
                "confidence": confidence,
                "score": score,
                "score_total": score,
                "thresholds": {"buy_tot": 1.5, "buy_cats": 2, "sell_tot": -1.5},
                "cats_pos": cats_pos,
                "cats_neg": cats_neg,
                "data_quality": data_quality,
                "data_quality_actual_n": 7,
                "data_quality_expected_n": 10,
                "confidence_before_penalties": confidence,
                "confidence_after_penalties": confidence,
                "confidence_before_floor": confidence,
                "confidence_final": confidence,
                "confidence_floor_candidate": floor_candidate,
                "confidence_floor_blockers": list(blockers or []),
                "confidence_floor_applied": False,
                "confidence_floor_reason": (
                    "signal_floor_candidate" if floor_candidate
                    else ((blockers or [None])[0])
                ),
                "confidence_penalties": list(blockers or []),
            },
            "ctx": {
                "history_source": history_source,
                "history_status": history_status,
                "history_rows": history_rows,
                "history_last_date": "2026-06-30",
            },
            "quote": {"price": 100.0, "pct": 1.7864, "stale": quote_stale},
            "price": 100.0,
            "stale": quote_stale,
        }

    def test_tsm_like_raw_buy_gets_buy_candidate_floor(self):
        import trading.bot as bot

        payload = self._raw_buy_payload()
        rec = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)

        self.assertEqual(rec["confidence"], 40)
        self.assertEqual(rec["confidence_final"], 40)
        self.assertTrue(rec["confidence_floor_applied"])
        self.assertEqual(rec["confidence_floor_reason"], "valid_raw_buy_floor")
        self.assertEqual(rec["display_signal_label"], "BUY_CANDIDATE")
        self.assertTrue(bot.is_execution_candidate(rec, DEFAULT_CONFIG))

    def test_raw_buy_barely_passes_threshold_gets_floor(self):
        import trading.bot as bot

        payload = self._raw_buy_payload(score=1.5, cats_pos=2)
        rec = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)

        self.assertGreaterEqual(rec["confidence_final"], 40)
        self.assertEqual(rec["display_signal_label"], "BUY_CANDIDATE")

    def test_raw_buy_with_severe_blocker_can_stay_bullish_lean(self):
        import trading.bot as bot

        payload = self._raw_buy_payload(
            confidence=20,
            data_quality=0.40,
            floor_candidate=False,
            blockers=["low_data_quality"],
        )
        rec = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)

        self.assertEqual(rec["confidence_final"], 20)
        self.assertFalse(rec["confidence_floor_applied"])
        self.assertIsNone(rec.get("confidence_floor_reason"))
        self.assertNotIn("low_data_quality", rec["confidence_floor_blockers"])
        self.assertEqual(rec["display_signal_label"], "BULLISH_LEAN")
        self.assertTrue(bot.is_execution_candidate(rec, DEFAULT_CONFIG))
        self.assertEqual(rec["candidate_type"], "weak_paper_test_candidate")

    def test_raw_buy_stale_quote_blocks_floor(self):
        import trading.bot as bot

        payload = self._raw_buy_payload(quote_stale=True)
        rec = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)

        self.assertEqual(rec["confidence_final"], 20)
        self.assertFalse(rec["confidence_floor_applied"])
        self.assertEqual(rec["confidence_floor_reason"], "stale_quote")
        self.assertEqual(rec["display_signal_label"], "BULLISH_LEAN")

    def test_cached_quote_within_execution_window_allows_floor(self):
        import trading.bot as bot

        payload = self._raw_buy_payload(quote_stale=True)
        payload["quote"]["cache_used"] = True
        payload["quote"]["quote_age_seconds"] = 120
        payload["quote"]["execution_trusted"] = True
        rec = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)

        self.assertEqual(rec["confidence"], 40)
        self.assertTrue(rec["confidence_floor_applied"])
        self.assertEqual(rec["confidence_floor_reason"], "valid_raw_buy_floor")
        self.assertFalse(payload["stale"])
        self.assertTrue(payload["execution_trusted"])

    def test_cached_quote_beyond_execution_window_blocks_floor(self):
        import trading.bot as bot

        payload = self._raw_buy_payload(quote_stale=True)
        payload["quote"]["cache_used"] = True
        payload["quote"]["quote_age_seconds"] = 301
        rec = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)

        self.assertEqual(rec["confidence_final"], 20)
        self.assertFalse(rec["confidence_floor_applied"])
        self.assertEqual(rec["confidence_floor_reason"], "stale_quote")

    def test_raw_buy_floor_is_idempotent_for_cached_scan_payload(self):
        import trading.bot as bot

        payload = self._raw_buy_payload()
        first = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)
        second = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)

        self.assertEqual(first["confidence"], 40)
        self.assertEqual(second["confidence"], 40)
        self.assertTrue(second["confidence_floor_applied"])
        self.assertEqual(second["confidence_floor_reason"], "valid_raw_buy_floor")
        self.assertEqual(second["display_signal_label"], "BUY_CANDIDATE")
        self.assertEqual(
            second["reasons"].count("Confidence floor: valid raw BUY -> 40%"),
            1,
        )

    def test_stale_history_blocks_raw_buy_floor(self):
        import trading.bot as bot

        payload = self._raw_buy_payload(
            history_source="stale_cache:fmp_daily",
            history_status="stale_cache",
        )
        payload["ctx"]["history_last_date"] = "2020-01-01"
        rec = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)

        self.assertEqual(rec["confidence_final"], 20)
        self.assertFalse(rec["confidence_floor_applied"])
        self.assertEqual(rec["confidence_floor_reason"], "stale_history")
        self.assertEqual(rec["display_signal_label"], "BULLISH_LEAN")

    def test_recent_stale_history_and_25_rows_are_allowed_with_warning(self):
        import trading.bot as bot

        payload = self._raw_buy_payload(
            history_source="stale_cache:fmp_daily",
            history_status="stale_cache",
            history_rows=25,
        )
        payload["ctx"]["history_last_date"] = str(bot._latest_completed_trading_day())
        rec = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)

        self.assertEqual(rec["confidence"], 40)
        self.assertTrue(rec["confidence_floor_applied"])
        self.assertEqual(rec["confidence_floor_reason"], "valid_raw_buy_floor")
        self.assertIn("INSUFFICIENT_HISTORY_WARNING", payload["ctx"]["history_warnings"])

    def test_fewer_than_25_history_rows_blocks_buy_floor(self):
        import trading.bot as bot

        payload = self._raw_buy_payload(history_rows=24)
        rec = bot._finalize_signal_confidence(payload, DEFAULT_CONFIG)

        self.assertEqual(rec["confidence_final"], 20)
        self.assertFalse(rec["confidence_floor_applied"])
        self.assertEqual(rec["confidence_floor_reason"], "insufficient_history")

    def test_earnings_warning_has_confidence_diagnostics_without_forced_hold(self):
        ctx = {
            "current": 100, "ma30": 95, "ma7": 99, "mom_30d_pct": 3,
            "rsi": 55, "macd_hist": 0.2, "macd_hist_prev": 0.1,
            "bb_pos": 0.55, "stoch_k": 60, "vol_ratio": 1.3,
            "avg_dollar_vol_20d": 100_000_000, "atr_pct": 2,
            "consec_up_days": 1, "recent_high": 105,
            "dist_from_high_pct": -2, "week_chg_pct": 1,
        }
        rec = get_recommendation(
            1.0,
            ctx,
            earnings={"soon": True, "date": "2026-07-01"},
            allow_live_risk=False,
        )

        self.assertFalse(rec.get("force_hold", False))
        self.assertTrue(rec["earnings_risk"]["soon"])
        self.assertIn("warning", rec["earnings_risk"]["policy"])
        self.assertIn("confidence_before_penalties", rec)
        self.assertIn("confidence_after_penalties", rec)
        self.assertIn("confidence_before_floor", rec)
        self.assertIn("confidence_final", rec)
        self.assertIn("confidence_floor_blockers", rec)

    def test_signal_marks_low_data_quality_raw_buy_as_floor_blocked(self):
        ctx = {
            "current": 100, "ma30": 95, "ma7": 99, "mom_30d_pct": 3,
            "rsi": 55, "macd_hist": 0.2, "macd_hist_prev": 0.1,
            "bb_pos": 0.55, "stoch_k": 60, "vol_ratio": 1.3,
            "avg_dollar_vol_20d": 100_000_000, "atr_pct": 2,
            "consec_up_days": 1, "recent_high": 105,
            "dist_from_high_pct": -2, "week_chg_pct": 1,
        }
        rec = get_recommendation(
            0.0,
            ctx,
            regime={"regime": "neutral"},
            earnings=None,
            analyst=None,
            insider=None,
            allow_live_risk=False,
        )

        # Audit P1-11 reverses the old pin: core daily history is present and
        # only optional categories (mfi/weekly/rel_str) are missing, so the
        # 0.70 floor applies and low_data_quality no longer blocks the floor.
        self.assertEqual(rec["cls"], "buy")
        self.assertLess(rec["data_quality_raw"], 0.60)
        self.assertGreaterEqual(rec["data_quality"], 0.70)
        self.assertTrue(rec["data_quality_floor_applied"])
        self.assertNotIn("low_data_quality", rec["confidence_floor_blockers"])

    def test_vote_positive_buy_labels_survive_confidence_penalties(self):
        ctx = {
            "current": 100, "ma30": 90, "ma7": 100, "mom_30d_pct": 8,
            "week_chg_pct": 4,
            "adx": 40, "macd_hist": 1, "macd_hist_prev": 0.5,
            "bb_pos": 0.5, "stoch_k": 35, "vol_ratio": 2, "mfi": 65,
            "weekly_trend_up": False, "rel_str_pct": 5, "atr_pct": 8,
            "avg_dollar_vol_20d": 1_000_000, "consec_up_days": 6,
            "recent_high": 102, "dist_from_high_pct": -2, "rsi": 68,
        }
        rec = get_recommendation(
            0.5, ctx, regime={"regime": "bull"},
            earnings={"days": 20},
            analyst={"total": 5, "buy": 4, "sell": 0},
            insider={"samples": 2, "sentiment": 0.5},
            news_articles=[{"score": 1, "published_at": "2026-06-06"}],
            pure_technical=True,
        )
        self.assertIn(rec["cls"], ("buy", "strong-buy"))
        self.assertIn(rec["signal"], ("BUY", "STRONG BUY"))
        if rec["cls"] == "strong-buy":
            self.assertGreaterEqual(rec["sizing_confidence"], 70)
        else:
            self.assertEqual(rec["sizing_confidence"], rec["confidence"])

    def test_data_quality_counts_independent_signal_groups(self):
        ctx = {
            "current": 0.0, "ma30": 0.0, "ma7": 0.0, "rsi": 0.0,
            "week_chg_pct": 0.0, "macd_hist": 0.0, "macd_hist_prev": 0.0,
            "bb_pos": 0.0, "stoch_k": 0.0, "stoch_d": 0.0,
            "mom_30d_pct": 0.0, "adx": 0.0, "vol_ratio": 0.0,
            "mfi": 0.0, "weekly_trend_up": False, "rel_str_pct": 0.0,
        }
        rec = get_recommendation(0.0, ctx, regime=None, earnings=None,
                                 analyst=None, insider=None,
                                 allow_live_risk=False)
        self.assertEqual(rec["data_quality"], 1.0)
        self.assertEqual(rec["data_quality_actual_n"], rec["data_quality_expected_n"])
        self.assertNotIn("macd_hist", rec["data_quality_missing_fields"])
        self.assertNotIn("weekly_trend_up", rec["data_quality_missing_fields"])
        self.assertIn("buy_cats", rec["thresholds"])
        self.assertIn("confidence_before_penalties", rec)
        self.assertIn("confidence_after_penalties", rec)
        self.assertIn("confidence_before_floor", rec)
        self.assertIn("confidence_final", rec)
        self.assertIn("confidence_floor_applied", rec)
        self.assertIn("confidence_penalties", rec)


class AdvisorySuggestionTests(unittest.TestCase):
    def _candidate(self, ticker, *, source="watchlist", confidence=90,
                   gross=2.0, net=1.5, edge_source="attribution_v2",
                   edge_samples=80, sector="tech", corr_group="ai"):
        cand = _entry_candidate(
            ticker,
            confidence=confidence,
            cluster="trend",
            source=source,
            score=8.0,
        )
        cand.update({
            "gross_edge_pct": gross,
            "net_edge_pct": net,
            "tradable": True,
            "edge_source": edge_source,
            "edge_samples": edge_samples,
            "sector": sector,
            "corr_group": corr_group,
            "regime_risk_mult": 1.0,
            "cluster_regime_mult": 1.0,
            "catalyst_type": "guidance_raise",
            "catalyst_confirmed": True,
            "catalyst_score_shadow": 0.4,
            "vol_regime": "normal",
        })
        cand["ctx"]["avg_dollar_vol_20d"] = 100_000_000
        cand["ctx"]["vol_ratio"] = 1.5
        return cand

    def test_low_edge_prior_scan_candidates_are_suppressed(self):
        now = int(time.time())
        out = rank_suggestion_candidates(
            [self._candidate(
                "LOWP",
                source="scan",
                confidence=90,
                gross=0.50,
                net=0.40,
                edge_source="confidence_prior",
                edge_samples=0,
            )],
            holdings={},
            top_sectors=[],
            feedback_stats={},
            recent_suggestions={},
            now_ts=now,
            config=DEFAULT_CONFIG,
        )
        self.assertEqual(out, [])

    def test_duplicate_sector_or_corr_group_is_reduced(self):
        now = int(time.time())
        out = rank_suggestion_candidates(
            [
                self._candidate("AAA", sector="tech", corr_group="ai"),
                self._candidate("BBB", sector="tech", corr_group="ai"),
            ],
            holdings={},
            top_sectors=[],
            feedback_stats={},
            recent_suggestions={},
            now_ts=now,
            config=DEFAULT_CONFIG,
        )
        self.assertEqual(len(out), 1)

    def test_recent_suggestion_cooldown_blocks_repeat(self):
        now = int(time.time())
        out = rank_suggestion_candidates(
            [self._candidate("COOL", source="scan", confidence=80)],
            holdings={},
            top_sectors=[],
            feedback_stats={},
            recent_suggestions={"COOL": now - 60},
            now_ts=now,
            config=DEFAULT_CONFIG,
        )
        self.assertEqual(out, [])

    def test_recent_sell_cooldowns_suppress_suggestions(self):
        now = int(time.time())
        recent = rank_suggestion_candidates(
            [self._candidate("SOLD", sector="health", corr_group="med")],
            holdings={},
            top_sectors=[],
            feedback_stats={},
            recent_suggestions={},
            recent_sells={"SOLD": {"ts": now - 60, "reason": "trail"}},
            now_ts=now,
            config=DEFAULT_CONFIG,
        )
        loss = rank_suggestion_candidates(
            [self._candidate("LOSS", sector="energy", corr_group="oil")],
            holdings={},
            top_sectors=[],
            feedback_stats={},
            recent_suggestions={},
            recent_sells={"LOSS": {"ts": now - 13 * 86400, "reason": "loss"}},
            now_ts=now,
            config=DEFAULT_CONFIG,
        )
        self.assertEqual(recent, [])
        self.assertEqual(loss, [])

    def test_suggestion_config_helper_accepts_outer_and_inner_config(self):
        outer = _suggestion_cfg(DEFAULT_CONFIG)
        inner = _suggestion_cfg(DEFAULT_CONFIG["suggestion"])
        self.assertEqual(outer["min_adv_usd"], 15_000_000)
        self.assertEqual(inner["min_adv_usd"], 15_000_000)

    def test_suggestion_noise_gates(self):
        now = int(time.time())
        low_adv = self._candidate("LOWADV")
        low_adv["ctx"]["avg_dollar_vol_20d"] = 12_000_000
        low_vol = self._candidate("LOWVOL")
        low_vol["ctx"]["vol_ratio"] = 1.0
        loss_cd = self._candidate("LOSSY")
        corr = self._candidate("CORR")
        corr["portfolio_variance"] = {"max_pair_corr": 0.80}
        base_cfg = DEFAULT_CONFIG
        self.assertEqual(DEFAULT_CONFIG["suggestion"]["min_adv_usd"], 15_000_000)
        for cand, losses in [
            (low_adv, {}),
            (low_vol, {}),
            (loss_cd, {"LOSSY": now - 60}),
            (corr, {}),
        ]:
            out = rank_suggestion_candidates(
                [cand],
                holdings={},
                top_sectors=[],
                feedback_stats={},
                recent_suggestions={},
                loss_cooldowns=losses,
                now_ts=now,
                config=base_cfg,
            )
            self.assertEqual(out, [])

    def test_suggestion_computed_correlation_gate(self):
        import trading.portfolio_variance as pv

        now = int(time.time())
        old_load = pv.load_close_history
        old_corr = pv.smoothed_corr
        try:
            pv.load_close_history = lambda tickers: {
                t: [100.0, 101.0, 102.0] for t in tickers
            }
            pv.smoothed_corr = lambda *_args, **_kwargs: 0.81
            out = rank_suggestion_candidates(
                [self._candidate("CORR2", sector="energy", corr_group="oil")],
                holdings={"HELD": {"shares": 1, "avg_cost": 1,
                                    "entry_snapshot": {"sector": "tech"},
                                    "corr_group": "software"}},
                top_sectors=[],
                feedback_stats={},
                recent_suggestions={},
                now_ts=now,
                config=DEFAULT_CONFIG,
            )
        finally:
            pv.load_close_history = old_load
            pv.smoothed_corr = old_corr
        self.assertEqual(out, [])

    def test_duplicate_catalyst_type_dedupes_suggestions(self):
        now = int(time.time())
        out = rank_suggestion_candidates(
            [self._candidate("A1", sector="tech", corr_group="a1"),
             self._candidate("A2", sector="health", corr_group="a2")],
            holdings={},
            top_sectors=[],
            feedback_stats={},
            recent_suggestions={},
            now_ts=now,
            config=DEFAULT_CONFIG,
        )
        self.assertEqual(len(out), 1)

    def test_suggestion_logging_schema_idempotent_and_feedback_stats(self):
        import os
        import sqlite3
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "suggestions.sqlite3")
            init_suggestion_db(db)
            init_suggestion_db(db)
            row = {
                "run_id": "run1",
                "ticker": "AAA",
                "feedback_bucket": "scan:trend:tech",
                "suggestion_score": 0.9,
                "showable": True,
            }
            now = int(time.time())
            log_suggestion_run(db, "run1", now, [row], [row])
            record_suggestion_feedback(db, "run1", "AAA", "useful", now)
            stats = load_feedback_stats(db, lookback_days=1)
            recent = load_recent_suggestions(db, lookback_sec=21600)
            conn = sqlite3.connect(db)
            try:
                names = {
                    r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
                    )
                }
            finally:
                conn.close()
            self.assertIn("suggestion_runs", names)
            self.assertIn("idx_suggestion_items_ticker_ts", names)
            self.assertIn("scan:trend:tech", stats)
            self.assertEqual(recent["AAA"], now)

    def test_suggestion_logging_failure_is_noncritical(self):
        import trading.bot as bot

        old_log = bot.log_suggestion_run
        old_prune = bot.prune_suggestion_store
        old_feedback = bot.load_feedback_stats
        old_recent = bot.load_recent_suggestions
        try:
            bot.log_suggestion_run = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("db down")
            )
            bot.prune_suggestion_store = lambda *_args, **_kwargs: None
            bot.load_feedback_stats = lambda *_args, **_kwargs: {}
            bot.load_recent_suggestions = lambda *_args, **_kwargs: {}
            state = {"holdings": {}}
            rows = bot.build_extra_ticker_suggestions(
                state,
                {},
                [self._candidate("SAFE", sector="health", corr_group="health")],
                {"top_sectors": []},
                int(time.time()),
            )
        finally:
            bot.log_suggestion_run = old_log
            bot.prune_suggestion_store = old_prune
            bot.load_feedback_stats = old_feedback
            bot.load_recent_suggestions = old_recent
        self.assertEqual(len(rows), 1)
        self.assertEqual(state["extra_ticker_suggestions"][0]["ticker"], "SAFE")

    def test_build_suggestions_skips_sqlite_when_no_unheld_candidates(self):
        import trading.bot as bot

        old_feedback = bot.load_feedback_stats
        old_recent = bot.load_recent_suggestions
        try:
            bot.load_feedback_stats = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("feedback DB should not load")
            )
            bot.load_recent_suggestions = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("recent DB should not load")
            )
            state = {
                "holdings": {"HELD": {"shares": 1}},
                "extra_ticker_suggestions": [{"ticker": "STALE"}],
            }
            rows = bot.build_extra_ticker_suggestions(
                state,
                {},
                [self._candidate("HELD")],
                {"top_sectors": []},
                int(time.time()),
            )
        finally:
            bot.load_feedback_stats = old_feedback
            bot.load_recent_suggestions = old_recent
        self.assertEqual(rows, [])
        self.assertEqual(state["extra_ticker_suggestions"], [])

    def test_build_suggestions_caches_feedback_stats(self):
        import trading.bot as bot

        cache = {}
        loads = []
        old = {
            "cache_get": bot.cache_get,
            "cache_set": bot.cache_set,
            "load_feedback_stats": bot.load_feedback_stats,
            "load_recent_suggestions": bot.load_recent_suggestions,
            "log_suggestion_run": bot.log_suggestion_run,
            "prune_suggestion_store": bot.prune_suggestion_store,
        }
        try:
            bot.cache_get = lambda key, max_age=None: cache.get(key)
            bot.cache_set = lambda key, value: cache.setdefault(key, value)
            bot.load_feedback_stats = lambda *_args, **_kwargs: loads.append(1) or {}
            bot.load_recent_suggestions = lambda *_args, **_kwargs: {}
            bot.log_suggestion_run = lambda *_args, **_kwargs: None
            bot.prune_suggestion_store = lambda *_args, **_kwargs: None
            state = {"holdings": {}}
            bot.build_extra_ticker_suggestions(
                state,
                {},
                [self._candidate("CACHE1", sector="health", corr_group="med")],
                {"top_sectors": []},
                int(time.time()),
            )
            bot.build_extra_ticker_suggestions(
                state,
                {},
                [self._candidate("CACHE2", sector="energy", corr_group="oil")],
                {"top_sectors": []},
                int(time.time()),
            )
        finally:
            for name, value in old.items():
                setattr(bot, name, value)
        self.assertEqual(len(loads), 1)


class PythonAnywhereHardeningTests(unittest.TestCase):
    def test_buyable_reason_rejection_paths(self):
        import trading.bot as bot

        old_sector = bot.get_sector
        try:
            bot.get_sector = lambda tk: None if tk == "NOSEC" else "tech"
            base = {
                "price": 10,
                "stale": False,
                "rec": {"cls": "buy", "confidence": 70, "catalyst": {}},
                "ctx": {"adx": 30, "rsi": 50, "week_chg_pct": 0,
                        "dist_from_high_pct": 0, "vol_ratio": 1.5,
                        "history_rows": 80, "history_source": "finnhub_daily",
                        "history_status": "ok"},
            }
            self.assertEqual(bot.buyable_reason("X", {"price": 0})[1], "INVALID_PRICE")
            self.assertEqual(bot.buyable_reason("X", {"price": 1, "stale": True})[1], "STALE_CANDIDATE_QUOTE")
            cache_ok = dict(base)
            cache_ok["quote"] = {
                "price": 10,
                "stale": True,
                "cache_used": True,
                "quote_age_seconds": 300,
            }
            cache_ok["stale"] = True
            self.assertEqual(bot.buyable_reason("X", cache_ok, {}, "bull", {})[1], "buyable")
            cache_old = dict(cache_ok)
            cache_old["quote"] = dict(cache_ok["quote"], quote_age_seconds=301)
            self.assertEqual(bot.buyable_reason("X", cache_old, {}, "bull", {})[1], "STALE_CANDIDATE_QUOTE")
            self.assertEqual(
                bot.buyable_reason("X", base, {"X": {"reason": "loss"}}, "bull", {})[1],
                "RECENT_SELL_COOLDOWN:loss",
            )
            self.assertEqual(bot.buyable_reason("NOSEC", base, {}, "bull", {})[1], "buyable")
            bear = dict(base)
            bear["ctx"] = dict(base["ctx"], rsi=45, is_dip=False)
            self.assertEqual(bot.buyable_reason("X", bear, {}, "bear", {})[1],
                             "BEAR_GATE_REQUIRES_DIP_OR_RSI_LT_35")
            neutral = dict(base)
            neutral["ctx"] = dict(base["ctx"], adx=10, is_dip=False)
            neutral["rec"] = dict(base["rec"], confidence=50)
            self.assertEqual(bot.buyable_reason("X", neutral, {}, "neutral", {})[1],
                             "NEUTRAL_ADX_BELOW_GATE_NON_DIP")
            # conf >= 60 bypasses the neutral ADX gate (audit P2-16)
            high_conf_neutral = dict(neutral, rec=dict(base["rec"], confidence=60))
            self.assertEqual(bot.buyable_reason("X", high_conf_neutral, {}, "neutral", {})[1],
                             "buyable")
            knife = dict(base)
            knife["rec"] = {"cls": "buy", "confidence": 70,
                            "catalyst": {"type": "guidance_cut"}}
            knife["ctx"] = dict(base["ctx"], week_chg_pct=-4)
            self.assertEqual(bot.buyable_reason("X", knife, {}, "bull", {})[1], "buyable")
            knife["ctx"] = dict(base["ctx"], week_chg_pct=-5)
            self.assertTrue(bot.buyable_reason("X", knife, {}, "bull", {})[1].startswith("NEGATIVE_CATALYST_FALLING"))
            low_vol = dict(base)
            low_vol["ctx"] = dict(base["ctx"], is_dip=False, vol_ratio=1.0)
            self.assertEqual(bot.buyable_reason("X", low_vol, {}, "bull", {})[1], "buyable")
            very_low_vol = dict(base)
            very_low_vol["ctx"] = dict(base["ctx"], is_dip=False, vol_ratio=0.05)
            self.assertEqual(
                bot.buyable_reason("X", very_low_vol, {}, "bull", {})[1],
                "VERY_LOW_VOLUME_CONFIRMATION",
            )
            low_liq_warning = dict(base)
            low_liq_warning["ctx"] = dict(base["ctx"], avg_dollar_vol_20d=750_000)
            self.assertEqual(bot.buyable_reason("X", low_liq_warning, {}, "bull", {})[1], "buyable")
            low_liq_hard = dict(base)
            low_liq_hard["ctx"] = dict(base["ctx"], avg_dollar_vol_20d=200_000)
            self.assertEqual(bot.buyable_reason("X", low_liq_hard, {}, "bull", {})[1],
                             "LOW_LIQUIDITY_HARD_BLOCK")
            high_atr_warning = dict(base)
            high_atr_warning["ctx"] = dict(base["ctx"], atr_pct=15)
            self.assertEqual(bot.buyable_reason("X", high_atr_warning, {}, "bull", {})[1], "buyable")
            extreme_atr = dict(base)
            extreme_atr["ctx"] = dict(base["ctx"], atr_pct=21)
            self.assertEqual(bot.buyable_reason("X", extreme_atr, {}, "bull", {})[1], "ATR_TOO_HIGH")
        finally:
            bot.get_sector = old_sector

    def test_cached_untrusted_recorded_quote_stays_untrusted(self):
        import market.quotes as quotes

        stale = quotes._annotate_quote_for_execution(
            {
                "price": 10,
                "source": "recorded_price",
                "stale": True,
                "stale_age_sec": 301,
                "execution_trusted": False,
            },
            cache_used=True,
            age_sec=1,
        )

        self.assertFalse(stale["execution_trusted"])
        self.assertTrue(stale["stale"])
        self.assertEqual(stale["quote_age_seconds"], 301)

    def test_no_buy_main_blocker_reducer(self):
        import trading.bot as bot

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        diag["market_open"] = True
        diag["tod_ok"] = False
        bot._set_main_blocker(diag)
        self.assertEqual(diag["main_blocker"], "outside_new_buy_window")

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        diag["market_open"] = True
        diag["tod_ok"] = True
        diag["raw_buy_count"] = 2
        diag["candidate_pool_count"] = 0
        bot._set_main_blocker(diag)
        self.assertEqual(diag["main_blocker"], "weak_raw_buys_only")

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        diag["market_open"] = True
        diag["tod_ok"] = True
        diag["raw_buy_count"] = 2
        diag["display_buy_candidate_count"] = 1
        diag["candidate_pool_count"] = 0
        bot._set_main_blocker(diag)
        self.assertEqual(diag["main_blocker"], "raw_buys_rejected_pre_candidate")

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        diag["market_open"] = True
        diag["tod_ok"] = True
        diag["regime_allow_buys"] = True
        diag["cash_available_after_floor"] = 150
        diag["min_trade_size_effective"] = 200
        bot._set_main_blocker(diag)
        self.assertEqual(diag["main_blocker"], "cash_below_min_position")

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        diag["market_open"] = True
        diag["tod_ok"] = True
        diag["degraded_mode_active"] = True
        diag["data_health_blocks"] = ["STALE_HELD_QUOTE"]
        bot._set_main_blocker(diag)
        self.assertEqual(diag["main_blocker"], "data_health_block")

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        diag["market_open"] = True
        diag["tod_ok"] = True
        diag["paper_trading_locked"] = True
        diag["paper_lock_reason"] = "DAILY_LOSS_LIMIT"
        bot._set_main_blocker(diag)
        self.assertEqual(diag["main_blocker"], "DAILY_LOSS_LIMIT")

    def test_record_skip_preserves_raw_display_and_original_reason(self):
        import trading.bot as bot

        state = {"history": []}
        bot._record_skip(
            state,
            "XYZ",
            "DEGRADED_FINAL_SIZE_TOO_SMALL",
            "BUY",
            58,
            display_signal="BUY_CANDIDATE",
            original_reason="Risk budget target $0 below $100 floor",
            skip_stage="sizing_floor",
            rank_reason_code="FINAL_SIZE_TOO_SMALL",
            gross_edge_pct=0.08,
            net_edge_pct=-1.3,
            required_edge_pct=0.40,
            friction_pct=1.38,
            target_notional=158.40,
            risk_pct=0.7,
        )
        row = state["history"][0]
        self.assertEqual(row["signal"], "BUY_CANDIDATE")
        self.assertEqual(row["raw_signal"], "BUY")
        self.assertEqual(row["display_signal"], "BUY_CANDIDATE")
        self.assertEqual(row["original_reason"], "Risk budget target $0 below $100 floor")
        self.assertEqual(row["skip_stage"], "sizing_floor")
        self.assertEqual(row["rank_reason_code"], "FINAL_SIZE_TOO_SMALL")
        self.assertEqual(row["gross_edge_pct"], 0.08)
        self.assertEqual(row["net_edge_pct"], -1.3)
        self.assertEqual(row["required_edge_pct"], 0.40)
        self.assertEqual(row["friction_pct"], 1.38)
        self.assertEqual(row["target_notional"], 158.40)
        self.assertEqual(row["risk_pct"], 0.7)
        self.assertNotIn("ev_diagnostics", row)
        self.assertNotIn("friction_diagnostics", row)

    def test_tick_log_entry_contains_required_contract_fields(self):
        import trading.bot as bot

        event = bot._tick_log_entry({
            "ts": 123,
            "top_buyable_rejects": [{"rejection_reason": "CONFIDENCE_BELOW_MIN_BUY"}],
            "top_ranked_rejections": [{"rank_reason": "EDGE_TOO_LOW"}],
        })
        for field in bot.REQUIRED_TICK_LOG_FIELDS:
            self.assertIn(field, event)
        self.assertEqual(event["timestamp"], 123)
        self.assertEqual(event["top_rejection_reasons"],
                         ["CONFIDENCE_BELOW_MIN_BUY", "EDGE_TOO_LOW"])

    def test_market_modes_script_does_not_publish_old_degraded_threshold(self):
        source = Path("scripts/check_market_modes.py").read_text(encoding="utf-8")
        self.assertIn("is_execution_candidate", source)
        self.assertIn('"/api/bot/status"', source)
        self.assertIn("test_client", source)
        self.assertIn('"confidence": 40', source)
        self.assertIn('"display_signal_label": "BUY_CANDIDATE"', source)
        for field in (
            "degraded_reject_counts",
            "degraded_buys_today",
            "degraded_max_buys_today",
            "degraded_gross_exposure_pct",
            "degraded_max_gross_exposure_pct",
            "degraded_use_standard_gates_for_testing",
            "degraded_standard_gates_active",
            "degraded_gate_policy",
            "effective_size_mult",
            "effective_min_buy_confidence",
            "normal_ev_gates_required",
            "normal_risk_caps_required",
            "fresh_quote_required",
            "data_health_warnings",
            "regime_data_status",
            "regime_data_warnings",
            "finnhub_key_configured",
            "fmp_key_configured",
            "stooq_status",
        ):
            self.assertIn(f'"{field}"', source)
        self.assertNotIn("DEGRADED_CONFIDENCE_BELOW_65", source)
        self.assertNotIn('"confidence": 64', source)

    def test_candidate_pool_builders_use_execution_candidate_helper(self):
        source = Path("trading/bot.py").read_text(encoding="utf-8")
        start = source.index("candidate_pool = []")
        end = source.index("ranked_candidates = rank_candidates", start)
        body = source[start:end]
        self.assertGreaterEqual(body.count("is_execution_candidate("), 2)
        self.assertNotIn("raw_cls not in", body)
        self.assertNotIn('s.get("rec", {}).get("cls"', body)

    def test_live_buy_path_does_not_call_portfolio_variance(self):
        source = Path("trading/bot.py").read_text(encoding="utf-8")
        start = source.index("ranked_candidates = rank_candidates")
        end = source.index("build_extra_ticker_suggestions", start)
        body = source[start:end]
        self.assertNotIn("candidate_variance_check(", body)
        self.assertNotIn("load_close_history(", body)
        self.assertNotIn("variance_reason(", body)

    def test_paper_loss_lockout_sets_daily_loss_reason(self):
        import trading.bot as bot

        state = {"today_open_equity": 10_000.0}
        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        events = []
        old_log = bot._log_bot_event
        try:
            bot._log_bot_event = lambda event, **payload: events.append((event, payload))
            locked = bot._apply_paper_loss_lockouts(
                state,
                diag,
                {"daily_loss_limit_pct": -0.02, "hard_drawdown_lockout_pct": -0.10},
                9_700.0,
                10_000.0,
                time.time(),
            )
        finally:
            bot._log_bot_event = old_log
        self.assertTrue(locked)
        self.assertTrue(diag["paper_trading_locked"])
        self.assertEqual(diag["paper_lock_reason"], "DAILY_LOSS_LIMIT")
        self.assertEqual(events[0][0], "DAILY_LOSS_LIMIT")

    def test_paper_loss_warning_does_not_lock_new_buys(self):
        import trading.bot as bot

        state = {"today_open_equity": 10_000.0}
        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        locked = bot._apply_paper_loss_lockouts(
            state,
            diag,
            DEFAULT_CONFIG["risk"],
            8_900.0,
            10_000.0,
            time.time(),
        )

        self.assertFalse(locked)
        self.assertFalse(diag["paper_trading_locked"])
        self.assertIn("DAILY_LOSS_WARNING", diag["loss_lockout_warnings"])

    def test_provider_failure_snapshot_marks_rate_limit_recent(self):
        import utils.cache as cache

        old_failures = dict(cache._api_failures)
        old_save = cache._save_provider_health
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            cache.record_api_failure("finnhub", "HTTP 429 Too Many Requests")
            snap = cache.api_failure_snapshot()
        finally:
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)
            cache._save_provider_health = old_save
        self.assertEqual(snap["finnhub"]["status"], "rate_limited")
        self.assertTrue(snap["finnhub"]["rate_limited"])
        self.assertTrue(snap["finnhub"]["rate_limit_recent"])

    def test_pythonanywhere_regime_daily_skips_stooq_and_uses_finnhub_then_fmp(self):
        import pandas as pd
        import market.quotes as quotes

        def df(source):
            out = pd.DataFrame(
                {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
                index=pd.to_datetime(["2026-06-26"]),
            )
            out.attrs["source"] = source
            out.attrs["status"] = "ok"
            return out

        old = {
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "_direct_stooq_daily": quotes._direct_stooq_daily,
            "_finnhub_daily": quotes._finnhub_daily,
            "_fmp_daily": quotes._fmp_daily,
            "cache_get_stale": quotes.cache_get_stale,
        }
        calls = []
        try:
            quotes.PYTHONANYWHERE_MODE = True
            quotes._direct_stooq_daily = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Stooq must be skipped on PythonAnywhere")
            )
            quotes._finnhub_daily = lambda *_args, **_kwargs: calls.append("finnhub") or df("finnhub_daily")
            quotes._fmp_daily = lambda *_args, **_kwargs: calls.append("fmp") or df("fmp_daily")
            quotes.cache_get_stale = lambda *_args, **_kwargs: (quotes.CACHE_MISS, None)
            out = quotes.get_regime_daily("SPY")
            self.assertEqual(out.attrs["source"], "finnhub_daily")
            self.assertEqual(calls, ["finnhub"])

            calls.clear()
            quotes._finnhub_daily = lambda *_args, **_kwargs: calls.append("finnhub") or None
            quotes._fmp_daily = lambda *_args, **_kwargs: calls.append("fmp") or df("fmp_daily")
            out = quotes.get_regime_daily("SPY")
            self.assertEqual(out.attrs["source"], "fmp_daily")
            self.assertEqual(calls, ["finnhub", "fmp"])
        finally:
            for name, value in old.items():
                setattr(quotes, name, value)

    def test_local_regime_daily_may_use_stooq_then_fallbacks(self):
        import pandas as pd
        import market.quotes as quotes

        def df(source):
            out = pd.DataFrame(
                {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
                index=pd.to_datetime(["2026-06-26"]),
            )
            out.attrs["source"] = source
            return out

        old = {
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "_direct_stooq_daily": quotes._direct_stooq_daily,
            "_finnhub_daily": quotes._finnhub_daily,
            "_fmp_daily": quotes._fmp_daily,
        }
        calls = []
        try:
            quotes.PYTHONANYWHERE_MODE = False
            quotes._direct_stooq_daily = lambda *_args, **_kwargs: calls.append("stooq") or df("stooq_regime_daily")
            quotes._finnhub_daily = lambda *_args, **_kwargs: calls.append("finnhub") or df("finnhub_daily")
            quotes._fmp_daily = lambda *_args, **_kwargs: calls.append("fmp") or df("fmp_daily")
            out = quotes.get_regime_daily("SPY")
            self.assertEqual(out.attrs["source"], "stooq_regime_daily")
            self.assertEqual(calls, ["stooq"])

            calls.clear()
            quotes._direct_stooq_daily = lambda *_args, **_kwargs: calls.append("stooq") or None
            quotes._finnhub_daily = lambda *_args, **_kwargs: calls.append("finnhub") or None
            out = quotes.get_regime_daily("SPY")
            self.assertEqual(out.attrs["source"], "fmp_daily")
            self.assertEqual(calls, ["stooq", "finnhub", "fmp"])
        finally:
            for name, value in old.items():
                setattr(quotes, name, value)

    def test_regime_daily_uses_visible_bounded_stale_cache_after_provider_failures(self):
        import pandas as pd
        import market.quotes as quotes

        df = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
            index=pd.to_datetime(["2026-06-26"]),
        )
        old = {
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "_direct_stooq_daily": quotes._direct_stooq_daily,
            "_finnhub_daily": quotes._finnhub_daily,
            "_fmp_daily": quotes._fmp_daily,
            "cache_get_stale": quotes.cache_get_stale,
            "is_market_open": quotes.is_market_open,
        }
        seen = {}
        try:
            quotes.PYTHONANYWHERE_MODE = True
            quotes.is_market_open = lambda: False
            quotes._direct_stooq_daily = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Stooq must be skipped on PythonAnywhere")
            )
            quotes._finnhub_daily = lambda *_args, **_kwargs: None
            quotes._fmp_daily = lambda *_args, **_kwargs: None

            def fake_stale(key, max_age, default):
                seen["max_age"] = max_age
                if key == "fh_daily_SPY":
                    return df, 3600
                return default, None

            quotes.cache_get_stale = fake_stale
            out = quotes.get_regime_daily("SPY")
        finally:
            for name, value in old.items():
                setattr(quotes, name, value)
        self.assertEqual(seen["max_age"], 72 * 3600)
        self.assertEqual(out.attrs["source"], "stale_cache:finnhub_daily")
        self.assertEqual(out.attrs["status"], "stale_cache")
        self.assertIn("STALE_DAILY_CACHE_USED", out.attrs["warnings"])
        self.assertEqual(out.attrs["stale_daily_cache_age_hours"], 1.0)

    def test_fmp_daily_normalizes_and_marks_daily_source(self):
        import json
        import pandas as pd
        import market.quotes as quotes

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps([
                    {"date": "2026-06-26", "open": "2", "high": "3", "low": "1", "close": "2.5", "volume": "200"},
                    {"date": "2026-06-25", "open": "1", "high": "2", "low": "1", "close": "1.5", "volume": "100"},
                    {"date": None, "close": "bad"},
                ]).encode("utf-8")

        old = {
            "FMP_KEY": quotes.FMP_KEY,
            "cache_get": quotes.cache_get,
            "cache_set": quotes.cache_set,
            "should_skip_api": quotes.should_skip_api,
            "record_api_failure": quotes.record_api_failure,
            "record_api_success": quotes.record_api_success,
            "urlopen": quotes.urllib.request.urlopen,
        }
        saved = {}
        requested = {}
        try:
            quotes.FMP_KEY = "secret-key"
            quotes.cache_get = lambda *_args, **_kwargs: quotes.CACHE_MISS
            quotes.cache_set = lambda key, value: saved.setdefault(key, value)
            quotes.should_skip_api = lambda *_args, **_kwargs: False
            quotes.record_api_failure = lambda *_args, **_kwargs: None
            quotes.record_api_success = lambda *_args, **_kwargs: None
            quotes.urllib.request.urlopen = lambda req, timeout=0: requested.setdefault("url", req.full_url) and Response()
            out = quotes._fmp_daily("SPY")
        finally:
            for name, value in old.items():
                if name == "urlopen":
                    quotes.urllib.request.urlopen = value
                else:
                    setattr(quotes, name, value)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(list(out.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(str(out.index[0].date()), "2026-06-25")
        self.assertEqual(str(out.index[-1].date()), "2026-06-26")
        self.assertEqual(out.attrs["source"], "fmp_daily")
        self.assertIn("/stable/historical-price-eod/full?", requested["url"])
        self.assertIn("symbol=SPY", requested["url"])
        self.assertIn("fmp_daily_SPY", saved)

    def test_fmp_daily_rate_limit_opens_global_circuit_and_skips_network(self):
        import market.quotes as quotes
        import utils.cache as cache

        old_quotes = {
            "FMP_KEY": quotes.FMP_KEY,
            "cache_get": quotes.cache_get,
            "cache_set": quotes.cache_set,
            "urlopen": quotes.urllib.request.urlopen,
        }
        old_failures = dict(cache._api_failures)
        old_save = cache._save_provider_health
        calls = []
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            quotes.FMP_KEY = "secret-key"
            quotes.cache_get = lambda *_args, **_kwargs: quotes.CACHE_MISS
            quotes.cache_set = lambda *_args, **_kwargs: None

            def rate_limited(*_args, **_kwargs):
                calls.append("network")
                raise RuntimeError("HTTP 429 Too Many Requests")

            quotes.urllib.request.urlopen = rate_limited
            self.assertIsNone(quotes._fmp_daily("SPY"))
            snap = cache.api_failure_snapshot()[quotes.FMP_DAILY_GLOBAL_ENDPOINT]
            self.assertEqual(snap["status"], "rate_limited")
            self.assertEqual(snap["cooldown_sec"], quotes.FMP_DAILY_RATE_LIMIT_COOLDOWN_SEC)
            self.assertTrue(quotes.fmp_daily_global_circuit_state()["active"])

            def should_not_call(*_args, **_kwargs):
                raise AssertionError("global FMP cooldown should skip network")

            quotes.urllib.request.urlopen = should_not_call
            self.assertIsNone(quotes._fmp_daily("AAPL"))
        finally:
            for name, value in old_quotes.items():
                if name == "urlopen":
                    quotes.urllib.request.urlopen = value
                else:
                    setattr(quotes, name, value)
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)
            cache._save_provider_health = old_save
        self.assertEqual(calls, ["network"])

    def test_fmp_daily_retry_after_sets_global_cooldown(self):
        import market.quotes as quotes
        import utils.cache as cache

        class RateLimitError(RuntimeError):
            code = 429
            headers = {"Retry-After": "120"}

        old_quotes = {
            "FMP_KEY": quotes.FMP_KEY,
            "cache_get": quotes.cache_get,
            "cache_set": quotes.cache_set,
            "urlopen": quotes.urllib.request.urlopen,
        }
        old_failures = dict(cache._api_failures)
        old_save = cache._save_provider_health
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            quotes.FMP_KEY = "secret-key"
            quotes.cache_get = lambda *_args, **_kwargs: quotes.CACHE_MISS
            quotes.cache_set = lambda *_args, **_kwargs: None
            quotes.urllib.request.urlopen = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RateLimitError("HTTP 429 Too Many Requests")
            )

            self.assertIsNone(quotes._fmp_daily("SPY"))
            snap = cache.api_failure_snapshot()[quotes.FMP_DAILY_GLOBAL_ENDPOINT]
            state = quotes.fmp_daily_global_circuit_state()
        finally:
            for name, value in old_quotes.items():
                if name == "urlopen":
                    quotes.urllib.request.urlopen = value
                else:
                    setattr(quotes, name, value)
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)
            cache._save_provider_health = old_save
        self.assertEqual(snap["status"], "rate_limited")
        self.assertEqual(snap["cooldown_sec"], 120)
        self.assertTrue(state["active"])
        self.assertGreater(state["cooldown_remaining_sec"], 0)
        self.assertLessEqual(state["cooldown_remaining_sec"], 120)

    def test_fmp_daily_reads_fresh_cache_before_global_circuit(self):
        import pandas as pd
        import market.quotes as quotes
        import utils.cache as cache

        cached = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
            index=pd.to_datetime(["2026-06-26"]),
        )
        old_quotes = {
            "FMP_KEY": quotes.FMP_KEY,
            "cache_get": quotes.cache_get,
            "urlopen": quotes.urllib.request.urlopen,
        }
        old_failures = dict(cache._api_failures)
        old_save = cache._save_provider_health
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            cache.record_api_failure(
                quotes.FMP_DAILY_GLOBAL_ENDPOINT,
                "HTTP 429 Too Many Requests",
                status="rate_limited",
            )
            quotes.FMP_KEY = "secret-key"
            quotes.cache_get = lambda *_args, **_kwargs: cached
            quotes.urllib.request.urlopen = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("fresh cache should be returned before global FMP cooldown")
            )
            out = quotes._fmp_daily("SPY")
        finally:
            for name, value in old_quotes.items():
                if name == "urlopen":
                    quotes.urllib.request.urlopen = value
                else:
                    setattr(quotes, name, value)
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)
            cache._save_provider_health = old_save
        self.assertIs(out, cached)
        self.assertEqual(out.attrs["source"], "fmp_daily")
        self.assertEqual(out.attrs["provider"], "fmp")

    def test_fmp_daily_global_circuit_expires_to_ok_status(self):
        import time as time_mod
        import market.quotes as quotes
        import utils.cache as cache

        old_failures = dict(cache._api_failures)
        old_save = cache._save_provider_health
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            cache.record_api_failure(
                quotes.FMP_DAILY_GLOBAL_ENDPOINT,
                "HTTP 429 Too Many Requests",
                status="rate_limited",
            )
            cache._api_failures[quotes.FMP_DAILY_GLOBAL_ENDPOINT]["ts"] = (
                time_mod.time() - quotes.FMP_DAILY_RATE_LIMIT_COOLDOWN_SEC - 5
            )
            state = quotes.fmp_daily_global_circuit_state()
        finally:
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)
            cache._save_provider_health = old_save
        self.assertEqual(state["status"], "ok")
        self.assertFalse(state["active"])
        self.assertFalse(state["rate_limited"])

    def test_provider_chain_marks_fmp_endpoint_circuit_as_skip(self):
        import pandas as pd
        import market.quotes as quotes
        import utils.cache as cache

        stale = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
            index=pd.to_datetime(["2026-06-26"]),
        )
        old_quotes = {
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "_finnhub_daily": quotes._finnhub_daily,
            "_fmp_daily": quotes._fmp_daily,
            "cache_get_stale": quotes.cache_get_stale,
            "is_market_open": quotes.is_market_open,
        }
        old_failures = dict(cache._api_failures)
        old_save = cache._save_provider_health
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            for _ in range(3):
                cache.record_api_failure("fmp_daily:AAPL:0", "empty FMP daily response", status="empty_response")
            quotes.PYTHONANYWHERE_MODE = True
            quotes.is_market_open = lambda: False
            quotes._finnhub_daily = lambda *_args, **_kwargs: None
            quotes._fmp_daily = lambda *_args, **_kwargs: None
            quotes.cache_get_stale = lambda key, *_args, **_kwargs: (
                (stale, 3600) if key == "fh_daily_AAPL" else (quotes.CACHE_MISS, None)
            )
            out = quotes._raw_daily("AAPL")
        finally:
            for name, value in old_quotes.items():
                setattr(quotes, name, value)
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)
            cache._save_provider_health = old_save
        fmp_entries = [
            entry for entry in out.attrs["provider_chain_debug"]
            if entry.get("provider") == "fmp_daily"
        ]
        self.assertEqual(fmp_entries[-1]["status"], "skipped_by_circuit")

    def test_pythonanywhere_normal_daily_uses_fmp_after_finnhub_block(self):
        import pandas as pd
        import market.quotes as quotes

        def df(source, rows=80):
            idx = pd.date_range("2026-01-01", periods=rows, freq="B")
            out = pd.DataFrame({
                "Open": range(rows),
                "High": range(rows),
                "Low": range(rows),
                "Close": range(rows),
                "Volume": [100] * rows,
            }, index=idx)
            out.attrs["source"] = source
            out.attrs["status"] = "ok"
            return out

        old = {
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "_direct_stooq_daily": quotes._direct_stooq_daily,
            "_finnhub_daily": quotes._finnhub_daily,
            "_fmp_daily": quotes._fmp_daily,
            "cache_get_stale": quotes.cache_get_stale,
            "record_api_failure": quotes.record_api_failure,
        }
        calls = []
        failures = []
        try:
            quotes.PYTHONANYWHERE_MODE = True
            quotes._direct_stooq_daily = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Stooq must be skipped on PythonAnywhere")
            )

            def blocked(tk, *_args, **_kwargs):
                calls.append(("finnhub", tk))
                quotes.record_api_failure(
                    f"finnhub_daily:{tk}:0",
                    "403 You don't have access to this resource",
                    status="blocked_or_forbidden",
                )
                return None

            quotes._finnhub_daily = blocked
            quotes._fmp_daily = lambda tk, *_args, **_kwargs: calls.append(("fmp", tk)) or df("fmp_daily")
            quotes.cache_get_stale = lambda *_args, **_kwargs: (quotes.CACHE_MISS, None)
            quotes.record_api_failure = lambda *args, **kwargs: failures.append((args, kwargs))
            out = quotes._raw_daily("AAPL")
        finally:
            for name, value in old.items():
                setattr(quotes, name, value)
        self.assertEqual(out.attrs["source"], "fmp_daily")
        self.assertGreater(len(out), 60)
        self.assertIn("Close", out.columns)
        self.assertEqual(calls, [("finnhub", "AAPL"), ("fmp", "AAPL")])
        self.assertEqual(failures[0][1]["status"], "blocked_or_forbidden")

    # ---- Entry 027: global Finnhub daily forbidden circuit --------------------

    def _run_finnhub_daily_with_error(self, error, symbol="AAPL"):
        """Call _finnhub_daily(symbol) with fh.stock_candles raising `error`,
        under a cleared/restored provider-health store. Returns the snapshot."""
        import market.quotes as quotes
        import utils.cache as cache
        import types as _types

        old_fh = quotes.fh
        old_save = cache._save_provider_health
        old_failures = dict(cache._api_failures)
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            quotes.fh = _types.SimpleNamespace(
                stock_candles=lambda *_a, **_k: (_ for _ in ()).throw(error)
            )
            result = quotes._finnhub_daily(symbol, use_cache=False)
            snap = dict(cache.api_failure_snapshot())
            state = quotes.finnhub_daily_global_circuit_state()
            return result, snap, state
        finally:
            quotes.fh = old_fh
            cache._save_provider_health = old_save
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)

    def test_finnhub_daily_403_via_status_code_opens_global_circuit(self):
        import market.quotes as quotes

        class FinnhubForbidden(Exception):
            status_code = 403

            def __str__(self):
                return "FinnhubAPIException(status_code: 403): You don't have access to this resource."

        result, snap, state = self._run_finnhub_daily_with_error(FinnhubForbidden())
        self.assertIsNone(result)
        self.assertIn(quotes.FINNHUB_DAILY_GLOBAL_ENDPOINT, snap)
        gsnap = snap[quotes.FINNHUB_DAILY_GLOBAL_ENDPOINT]
        self.assertEqual(gsnap["status"], "blocked_or_forbidden")
        self.assertEqual(gsnap["cooldown_sec"], quotes.FINNHUB_DAILY_FORBIDDEN_COOLDOWN_SEC)
        self.assertEqual(gsnap["cooldown_sec"], 21_600)
        self.assertTrue(state["active"])
        # Per-symbol entry now uses the 6h forbidden cooldown, not 900s.
        self.assertEqual(snap["finnhub_daily:AAPL:0"]["cooldown_sec"], 21_600)

    def test_finnhub_daily_403_via_message_text_only_opens_global_circuit(self):
        import market.quotes as quotes

        # No .status_code attribute at all — the 403 is only in the message text,
        # exactly like the live log. Must still classify as forbidden (21600s),
        # not the 900s provider-error default.
        err = RuntimeError(
            "FinnhubAPIException(status_code: 403): You don't have access to this resource."
        )
        result, snap, state = self._run_finnhub_daily_with_error(err)
        self.assertIsNone(result)
        self.assertEqual(snap["finnhub_daily:AAPL:0"]["cooldown_sec"], 21_600)
        self.assertEqual(
            snap[quotes.FINNHUB_DAILY_GLOBAL_ENDPOINT]["cooldown_sec"], 21_600
        )
        self.assertTrue(state["active"])

    def test_finnhub_daily_generic_error_uses_900s_and_no_global_circuit(self):
        import market.quotes as quotes

        result, snap, state = self._run_finnhub_daily_with_error(
            RuntimeError("connection timed out")
        )
        self.assertIsNone(result)
        self.assertEqual(snap["finnhub_daily:AAPL:0"]["cooldown_sec"], 900)
        self.assertNotIn(quotes.FINNHUB_DAILY_GLOBAL_ENDPOINT, snap)
        self.assertFalse(state["active"])

    def test_finnhub_daily_rate_limit_uses_1800s_and_no_global_circuit(self):
        import market.quotes as quotes

        class FinnhubRateLimited(Exception):
            status_code = 429

            def __str__(self):
                return "FinnhubAPIException(status_code: 429): API limit reached."

        result, snap, state = self._run_finnhub_daily_with_error(FinnhubRateLimited())
        self.assertIsNone(result)
        self.assertEqual(snap["finnhub_daily:AAPL:0"]["cooldown_sec"], 1_800)
        self.assertNotIn(quotes.FINNHUB_DAILY_GLOBAL_ENDPOINT, snap)
        self.assertFalse(state["active"])

    def test_finnhub_daily_global_circuit_skips_other_symbols_without_network(self):
        import market.quotes as quotes
        import utils.cache as cache
        import types as _types

        old_fh = quotes.fh
        old_save = cache._save_provider_health
        old_failures = dict(cache._api_failures)
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            # Open the global circuit directly.
            cache.record_api_failure(
                quotes.FINNHUB_DAILY_GLOBAL_ENDPOINT,
                "403 forbidden",
                status="blocked_or_forbidden",
                cooldown_sec=quotes.FINNHUB_DAILY_FORBIDDEN_COOLDOWN_SEC,
            )

            def should_not_call(*_a, **_k):
                raise AssertionError("global Finnhub daily circuit should skip network")

            quotes.fh = _types.SimpleNamespace(stock_candles=should_not_call)
            # A *different* symbol, no cache — must return None without a call.
            self.assertIsNone(quotes._finnhub_daily("MSFT", use_cache=False))
            self.assertTrue(quotes.finnhub_daily_global_circuit_state()["active"])
        finally:
            quotes.fh = old_fh
            cache._save_provider_health = old_save
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)

    def test_finnhub_daily_global_circuit_does_not_disable_quotes(self):
        import market.quotes as quotes
        import utils.cache as cache

        old_save = cache._save_provider_health
        old_failures = dict(cache._api_failures)
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            cache.record_api_failure(
                quotes.FINNHUB_DAILY_GLOBAL_ENDPOINT,
                "403 forbidden",
                status="blocked_or_forbidden",
                cooldown_sec=quotes.FINNHUB_DAILY_FORBIDDEN_COOLDOWN_SEC,
            )
            state = quotes.finnhub_daily_global_circuit_state()
            self.assertTrue(state["active"])
            self.assertTrue(state["quote_endpoint_still_enabled"])
            # The Finnhub quote endpoint circuit is a separate key and must be untouched.
            self.assertFalse(quotes.should_skip_api("quote:AAPL", cooldown_sec=120))
            self.assertNotIn("quote:AAPL", cache.api_failure_snapshot())
        finally:
            cache._save_provider_health = old_save
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)

    def test_finnhub_daily_global_circuit_still_reaches_fmp(self):
        import pandas as pd
        import market.quotes as quotes
        import utils.cache as cache
        import types as _types

        idx = pd.date_range("2026-01-01", periods=80, freq="B")
        fmp_df = pd.DataFrame({
            "Open": range(80), "High": range(80), "Low": range(80),
            "Close": range(80), "Volume": [100] * 80,
        }, index=idx)
        fmp_df.attrs["source"] = "fmp_daily"
        fmp_df.attrs["status"] = "ok"

        old = {
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "fh": quotes.fh,
            "_fmp_daily": quotes._fmp_daily,
            "cache_get_stale": quotes.cache_get_stale,
        }
        old_save = cache._save_provider_health
        old_failures = dict(cache._api_failures)
        calls = []
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            cache.record_api_failure(
                quotes.FINNHUB_DAILY_GLOBAL_ENDPOINT,
                "403 forbidden",
                status="blocked_or_forbidden",
                cooldown_sec=quotes.FINNHUB_DAILY_FORBIDDEN_COOLDOWN_SEC,
            )
            quotes.PYTHONANYWHERE_MODE = True
            quotes.fh = _types.SimpleNamespace(
                stock_candles=lambda *_a, **_k: (_ for _ in ()).throw(
                    AssertionError("Finnhub network must be skipped while global circuit is open")
                )
            )
            quotes._fmp_daily = lambda tk, *_a, **_k: calls.append(("fmp", tk)) or fmp_df
            quotes.cache_get_stale = lambda *_a, **_k: (quotes.CACHE_MISS, None)
            out = quotes._raw_daily("AAPL")
        finally:
            for name, value in old.items():
                setattr(quotes, name, value)
            cache._save_provider_health = old_save
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)
        self.assertIsNotNone(out)
        self.assertEqual(out.attrs["source"], "fmp_daily")
        self.assertEqual(calls, [("fmp", "AAPL")])

    # ---- Entry 027: visible MISSING_HISTORY / quote reasoning ------------------

    def test_core_data_blocker_reason_surfaces_missing_history(self):
        import trading.bot as bot

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        diag["history_missing_count"] = 6
        diag["top_missing_history_symbols"] = ["ORCL", "SPOT", "TSLA"]
        reason = bot._core_data_blocker_reason(diag)
        self.assertIn("daily history missing", reason.lower())
        self.assertIn("ORCL", reason)
        self.assertNotEqual(reason, "no BUY signals across watchlist")

    def test_core_data_blocker_reason_empty_when_no_core_blocker(self):
        import trading.bot as bot

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        self.assertEqual(bot._core_data_blocker_reason(diag), "")

    def test_core_data_blocker_reason_mentions_history_and_quote(self):
        import trading.bot as bot

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        diag["history_missing_count"] = 6
        diag["buyable_reject_counts"] = {"INVALID_PRICE": 6}
        reason = bot._core_data_blocker_reason(diag).lower()
        self.assertIn("history", reason)
        self.assertIn("quote", reason)

    def test_set_main_blocker_missing_history_detail_mentions_quote(self):
        import trading.bot as bot

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        diag["market_open"] = True
        diag["tod_ok"] = True
        diag["regime_allow_buys"] = True
        diag["history_missing_count"] = 6
        diag["candidate_pool_count"] = 0
        diag["top_missing_history_symbols"] = ["ORCL", "SPOT"]
        diag["buyable_reject_counts"] = {"INVALID_PRICE": 6}
        bot._set_main_blocker(diag)
        self.assertEqual(diag["main_blocker"], "MISSING_HISTORY")
        self.assertEqual(diag["blocker_stage"], "HISTORY_QUALITY")
        self.assertIn("history missing", diag["blocker_detail"].lower())
        self.assertIn("quote", diag["blocker_detail"].lower())

    def test_set_main_blocker_invalid_price_quote_quality_detail(self):
        import trading.bot as bot

        diag = bot._new_no_buy_diag(int(time.time()), {"cash": 1000}, False, False)
        diag["market_open"] = True
        diag["tod_ok"] = True
        diag["regime_allow_buys"] = True
        diag["candidate_pool_count"] = 0
        diag["buyable_reject_counts"] = {"INVALID_PRICE": 4}
        diag["top_buyable_rejects"] = [{"ticker": "TSLA"}, {"ticker": "META"}]
        bot._set_main_blocker(diag)
        self.assertEqual(diag["main_blocker"], "INVALID_PRICE")
        self.assertEqual(diag["blocker_stage"], "QUOTE_QUALITY")
        self.assertIn("price", diag["blocker_detail"].lower())

    def test_raw_daily_uses_valid_cached_history_before_provider_fetch(self):
        import pandas as pd
        import market.quotes as quotes

        idx = pd.date_range("2026-01-01", periods=80, freq="B")
        cached = pd.DataFrame({
            "Open": range(80),
            "High": range(80),
            "Low": range(80),
            "Close": range(80),
            "Volume": [100] * 80,
        }, index=idx)
        old = {
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "_finnhub_daily": quotes._finnhub_daily,
            "_fmp_daily": quotes._fmp_daily,
            "cache_get_stale": quotes.cache_get_stale,
            "completed_trading_days_since": quotes.completed_trading_days_since,
        }
        seen = []
        try:
            quotes.PYTHONANYWHERE_MODE = True
            quotes.completed_trading_days_since = lambda _d: 0
            quotes._finnhub_daily = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("provider should not be called when valid cache exists")
            )
            quotes._fmp_daily = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("provider should not be called when valid cache exists")
            )

            def fake_cache(key, *_args, **_kwargs):
                seen.append(key)
                if key == "fh_daily_AAPL":
                    return cached, 300
                return quotes.CACHE_MISS, None

            quotes.cache_get_stale = fake_cache
            out = quotes._raw_daily("AAPL")
        finally:
            for name, value in old.items():
                setattr(quotes, name, value)
        self.assertEqual(out.attrs["source"], "finnhub_daily_cache")
        self.assertTrue(out.attrs["history_cache_used"])
        self.assertIn("fh_daily_AAPL", seen)

    def test_raw_daily_respects_shared_history_fetch_budget(self):
        import pandas as pd
        import market.quotes as quotes

        idx = pd.date_range("2026-01-01", periods=80, freq="B")
        fmp = pd.DataFrame({
            "Open": range(80),
            "High": range(80),
            "Low": range(80),
            "Close": range(80),
            "Volume": [100] * 80,
        }, index=idx)
        fmp.attrs.update({"source": "fmp_daily", "provider": "fmp", "status": "ok"})
        old = {
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "_finnhub_daily": quotes._finnhub_daily,
            "_fmp_daily": quotes._fmp_daily,
            "cache_get_stale": quotes.cache_get_stale,
        }
        calls = []
        try:
            quotes.PYTHONANYWHERE_MODE = True
            quotes.cache_get_stale = lambda *_args, **_kwargs: (quotes.CACHE_MISS, None)
            quotes._finnhub_daily = lambda tk, *_args, **_kwargs: calls.append(("finnhub", tk)) or None
            quotes._fmp_daily = lambda tk, *_args, **_kwargs: calls.append(("fmp", tk)) or fmp
            quotes.set_history_fetch_budget(1)
            first = quotes._raw_daily("AAPL")
            second = quotes._raw_daily("MSFT")
        finally:
            quotes.set_history_fetch_budget(None)
            for name, value in old.items():
                setattr(quotes, name, value)
        self.assertEqual(first.attrs["source"], "fmp_daily")
        self.assertIsNone(second)
        self.assertEqual(calls, [("finnhub", "AAPL"), ("fmp", "AAPL")])

    def test_finnhub_daily_403_opens_daily_circuit_only(self):
        import market.quotes as quotes
        import utils.cache as cache

        class Forbidden(RuntimeError):
            code = 403

        old_quotes = {
            "fh": quotes.fh,
            "cache_get": quotes.cache_get,
        }
        old_failures = dict(cache._api_failures)
        old_save = cache._save_provider_health
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True
            quotes.cache_get = lambda *_args, **_kwargs: quotes.CACHE_MISS
            quotes.fh = types.SimpleNamespace(
                stock_candles=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    Forbidden("HTTP 403 Forbidden")
                )
            )
            self.assertIsNone(quotes._finnhub_daily("AAPL"))
            state = quotes.finnhub_daily_circuit_state("AAPL")
            snap = cache.api_failure_snapshot()
        finally:
            quotes.fh = old_quotes["fh"]
            quotes.cache_get = old_quotes["cache_get"]
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)
            cache._save_provider_health = old_save
        self.assertTrue(state["active"])
        self.assertEqual(state["status"], "blocked_or_forbidden")
        self.assertEqual(state["cooldown_sec"], 21600)
        self.assertTrue(state["quote_endpoint_still_enabled"])
        self.assertIn("finnhub_daily:AAPL:0", snap)
        self.assertNotIn("quote:AAPL", snap)

    def test_pythonanywhere_normal_daily_fmp_fallback_is_not_spy_only(self):
        import pandas as pd
        import market.quotes as quotes

        def df(source):
            idx = pd.date_range("2026-01-01", periods=80, freq="B")
            out = pd.DataFrame(
                {"Open": 1, "High": 2, "Low": 1, "Close": 2, "Volume": 100},
                index=idx,
            )
            out.attrs["source"] = source
            return out

        old = {
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "_direct_stooq_daily": quotes._direct_stooq_daily,
            "_finnhub_daily": quotes._finnhub_daily,
            "_fmp_daily": quotes._fmp_daily,
            "cache_get_stale": quotes.cache_get_stale,
        }
        calls = []
        try:
            quotes.PYTHONANYWHERE_MODE = True
            quotes._direct_stooq_daily = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Stooq must be skipped on PythonAnywhere")
            )
            quotes._finnhub_daily = lambda tk, *_args, **_kwargs: calls.append(("finnhub", tk)) or None
            quotes._fmp_daily = lambda tk, *_args, **_kwargs: calls.append(("fmp", tk)) or df("fmp_daily")
            quotes.cache_get_stale = lambda *_args, **_kwargs: (quotes.CACHE_MISS, None)
            for ticker in ("AMD", "TSM", "CAT"):
                out = quotes._raw_daily(ticker)
                self.assertEqual(out.attrs["source"], "fmp_daily")
        finally:
            for name, value in old.items():
                setattr(quotes, name, value)
        self.assertEqual([c for c in calls if c[0] == "fmp"], [("fmp", "AMD"), ("fmp", "TSM"), ("fmp", "CAT")])

    def test_data_manager_get_daily_uses_normal_provider_chain_fmp_fallback(self):
        import pandas as pd
        import market.data_manager as dm
        import market.quotes as quotes

        idx = pd.date_range("2026-01-01", periods=80, freq="B")
        fmp_df = pd.DataFrame(
            {"Open": 1, "High": 2, "Low": 1, "Close": 2, "Volume": 100},
            index=idx,
        )
        fmp_df.attrs.update({"source": "fmp_daily", "provider": "fmp", "status": "ok"})
        old = {
            "dm_cache_get": dm.cache_get,
            "dm_cache_set": dm.cache_set,
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "_direct_stooq_daily": quotes._direct_stooq_daily,
            "_finnhub_daily": quotes._finnhub_daily,
            "_fmp_daily": quotes._fmp_daily,
            "cache_get_stale": quotes.cache_get_stale,
        }
        calls = []
        try:
            dm._DEFAULT_MANAGER.clear()
            dm.cache_get = lambda *_args, **_kwargs: None
            dm.cache_set = lambda *_args, **_kwargs: None
            quotes.PYTHONANYWHERE_MODE = True
            quotes._direct_stooq_daily = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Stooq must be skipped on PythonAnywhere")
            )
            quotes._finnhub_daily = lambda tk, *_args, **_kwargs: calls.append(("finnhub", tk)) or None
            quotes._fmp_daily = lambda tk, *_args, **_kwargs: calls.append(("fmp", tk)) or fmp_df
            quotes.cache_get_stale = lambda *_args, **_kwargs: (quotes.CACHE_MISS, None)
            out = dm.get_daily("AAPL")
        finally:
            dm._DEFAULT_MANAGER.clear()
            dm.cache_get = old["dm_cache_get"]
            dm.cache_set = old["dm_cache_set"]
            for name in (
                "PYTHONANYWHERE_MODE",
                "_direct_stooq_daily",
                "_finnhub_daily",
                "_fmp_daily",
                "cache_get_stale",
            ):
                setattr(quotes, name, old[name])
        self.assertEqual(out.attrs["source"], "fmp_daily")
        self.assertEqual(out.attrs["provider"], "fmp")
        self.assertEqual(calls, [("finnhub", "AAPL"), ("fmp", "AAPL")])

    def test_pythonanywhere_normal_daily_uses_stale_cache_after_provider_failures(self):
        import pandas as pd
        import market.quotes as quotes

        stale = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
            index=pd.to_datetime(["2026-06-26"]),
        )
        old = {
            "PYTHONANYWHERE_MODE": quotes.PYTHONANYWHERE_MODE,
            "_direct_stooq_daily": quotes._direct_stooq_daily,
            "_finnhub_daily": quotes._finnhub_daily,
            "_fmp_daily": quotes._fmp_daily,
            "cache_get_stale": quotes.cache_get_stale,
            "is_market_open": quotes.is_market_open,
        }
        seen = []
        try:
            quotes.PYTHONANYWHERE_MODE = True
            quotes.is_market_open = lambda: False
            quotes._direct_stooq_daily = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Stooq must be skipped on PythonAnywhere")
            )
            quotes._finnhub_daily = lambda *_args, **_kwargs: None
            quotes._fmp_daily = lambda *_args, **_kwargs: None

            def fake_stale(key, max_age, default):
                seen.append(key)
                if key == "fh_daily_AAPL":
                    return stale, 7200
                return default, None

            quotes.cache_get_stale = fake_stale
            out = quotes._raw_daily("AAPL")
        finally:
            for name, value in old.items():
                setattr(quotes, name, value)
        self.assertEqual(out.attrs["source"], "stale_cache:finnhub_daily")
        self.assertEqual(out.attrs["status"], "stale_cache")
        self.assertEqual(out.attrs["stale_daily_cache_age_hours"], 2.0)
        self.assertIn("fh_daily_AAPL", seen)

    def test_fmp_daily_missing_key_records_skipped_missing_key(self):
        import market.quotes as quotes

        old = {
            "FMP_KEY": quotes.FMP_KEY,
            "record_api_failure": quotes.record_api_failure,
        }
        failures = []
        try:
            quotes.FMP_KEY = ""
            quotes.record_api_failure = lambda *args, **kwargs: failures.append((args, kwargs))
            out = quotes._fmp_daily("AAPL")
        finally:
            for name, value in old.items():
                setattr(quotes, name, value)
        self.assertIsNone(out)
        self.assertEqual(failures[0][1]["status"], "skipped_missing_key")

    def test_invalid_fmp_daily_response_does_not_crash(self):
        import json
        import market.quotes as quotes

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps([{"date": "bad-date"}, {"date": "2026-06-26"}]).encode("utf-8")

        old = {
            "FMP_KEY": quotes.FMP_KEY,
            "cache_get": quotes.cache_get,
            "cache_set": quotes.cache_set,
            "should_skip_api": quotes.should_skip_api,
            "record_api_failure": quotes.record_api_failure,
            "record_api_success": quotes.record_api_success,
            "urlopen": quotes.urllib.request.urlopen,
        }
        failures = []
        try:
            quotes.FMP_KEY = "secret-key"
            quotes.cache_get = lambda *_args, **_kwargs: quotes.CACHE_MISS
            quotes.cache_set = lambda *_args, **_kwargs: None
            quotes.should_skip_api = lambda *_args, **_kwargs: False
            quotes.record_api_failure = lambda *args, **kwargs: failures.append((args, kwargs))
            quotes.record_api_success = lambda *_args, **_kwargs: None
            quotes.urllib.request.urlopen = lambda *_args, **_kwargs: Response()
            out = quotes._fmp_daily("AAPL")
        finally:
            for name, value in old.items():
                if name == "urlopen":
                    quotes.urllib.request.urlopen = value
                else:
                    setattr(quotes, name, value)
        self.assertIsNone(out)
        self.assertIn(failures[-1][1]["status"], {"empty_response", "parse_error"})

    def test_append_live_bar_updates_today_row_and_preserves_volume(self):
        import pandas as pd
        import market.quotes as quotes
        import utils.time_utils as time_utils

        today = pd.Timestamp.now().normalize()
        idx = pd.date_range(end=today, periods=25, freq="D")
        df = pd.DataFrame({
            "Open": [10.0] * 25,
            "High": [12.0] * 25,
            "Low": [9.0] * 25,
            "Close": [11.0] * 25,
            "Volume": [1234.0] * 25,
        }, index=idx)
        old = {
            "get_quote": quotes.get_quote,
            "is_market_open": time_utils.is_market_open,
        }
        try:
            time_utils.is_market_open = lambda: True
            quotes.get_quote = lambda _tk: {
                "price": 13.0, "high": 13.5, "low": 8.5,
                "open": 10.5, "stale": False, "execution_trusted": True,
            }
            out = quotes._append_live_bar(df.copy(), "AAPL")
        finally:
            for name, value in old.items():
                if name == "get_quote":
                    quotes.get_quote = value
                else:
                    time_utils.is_market_open = value
        self.assertEqual(float(out["Close"].iloc[-1]), 13.0)
        self.assertEqual(float(out["High"].iloc[-1]), 13.0)
        self.assertEqual(float(out["Low"].iloc[-1]), 9.0)
        self.assertEqual(float(out["Volume"].iloc[-1]), 1234.0)
        self.assertTrue(out.attrs["live_bar_applied"])
        self.assertEqual(out.attrs["live_bar_reason"], "updated_today_row")
        self.assertTrue(out.attrs["quote_fresh"])

    def test_append_live_bar_skips_stale_quote(self):
        import pandas as pd
        import market.quotes as quotes
        import utils.time_utils as time_utils

        idx = pd.date_range(end=pd.Timestamp("2026-06-26"), periods=25, freq="D")
        df = pd.DataFrame({
            "Open": [10.0] * 25, "High": [12.0] * 25, "Low": [9.0] * 25,
            "Close": [11.0] * 25, "Volume": [1234.0] * 25,
        }, index=idx)
        old = {
            "get_quote": quotes.get_quote,
            "is_market_open": time_utils.is_market_open,
        }
        try:
            time_utils.is_market_open = lambda: True
            quotes.get_quote = lambda _tk: {"price": 13.0, "stale": True, "execution_trusted": False}
            out = quotes._append_live_bar(df.copy(), "AAPL")
        finally:
            for name, value in old.items():
                if name == "get_quote":
                    quotes.get_quote = value
                else:
                    time_utils.is_market_open = value
        self.assertEqual(len(out), 25)
        self.assertFalse(out.attrs["live_bar_applied"])
        self.assertEqual(out.attrs["live_bar_reason"], "quote_stale_or_missing")
        self.assertFalse(out.attrs["quote_fresh"])

    def test_get_history_exposes_daily_source_metadata(self):
        import pandas as pd
        import market.history as hist
        import trading.risk as risk

        idx = pd.date_range("2026-01-01", periods=80, freq="B")
        df = pd.DataFrame({
            "Open": range(80),
            "High": range(1, 81),
            "Low": range(80),
            "Close": range(2, 82),
            "Volume": [1000] * 80,
        }, index=idx)
        df.attrs.update({
            "source": "fmp_daily",
            "provider": "fmp",
            "status": "ok",
            "live_bar_applied": True,
            "live_bar_reason": "updated_today_row",
            "quote_fresh": True,
        })
        old = {
            "cache_get": hist.cache_get,
            "cache_set": hist.cache_set,
            "_daily_bars": hist._daily_bars,
            "_append_live_bar": hist._append_live_bar,
            "sector_relative_strength": hist.sector_relative_strength,
            "get_sector": risk.get_sector,
        }
        saved = {}
        try:
            hist.cache_get = lambda *_args, **_kwargs: None
            hist.cache_set = lambda key, value: saved.setdefault(key, value)
            hist._daily_bars = lambda _tk: df
            hist._append_live_bar = lambda frame, _tk: frame
            hist.sector_relative_strength = lambda *_args, **_kwargs: {}
            risk.get_sector = lambda _tk: "tech"
            out = hist.get_history("AAPL")
        finally:
            for name, value in old.items():
                if name == "get_sector":
                    risk.get_sector = value
                else:
                    setattr(hist, name, value)
        self.assertEqual(out["history_source"], "fmp_daily")
        self.assertEqual(out["history_provider"], "fmp")
        self.assertEqual(out["history_rows"], 80)
        self.assertEqual(out["history_last_date"], str(idx[-1].date()))
        self.assertTrue(out["live_bar_applied"])
        self.assertTrue(out["quote_fresh"])
        self.assertIn("h_AAPL", saved)

    def test_get_history_does_not_cache_empty_context(self):
        import market.history as hist

        old = {
            "cache_get": hist.cache_get,
            "cache_set": hist.cache_set,
            "_daily_bars": hist._daily_bars,
            "_ctx_from_recorded": hist._ctx_from_recorded,
        }
        saved = []
        try:
            hist.cache_get = lambda *_args, **_kwargs: None
            hist.cache_set = lambda key, value: saved.append((key, value))
            hist._daily_bars = lambda _tk: None
            hist._ctx_from_recorded = lambda _tk: {}
            out = hist.get_history("AAPL")
        finally:
            for name, value in old.items():
                setattr(hist, name, value)
        self.assertEqual(out["history_status"], "missing")
        self.assertEqual(out["history_rows"], 0)
        self.assertIn("MISSING_HISTORY", out["history_warnings"])
        self.assertEqual(saved, [])

    def test_sector_relative_strength_exposes_missing_history_diagnostics(self):
        import trading.indicators as indicators
        import market.data_manager as dm

        old = {
            "cache_get": indicators.cache_get,
            "cache_set": indicators.cache_set,
            "get_daily": dm.get_daily,
        }
        saved = {}
        try:
            indicators.cache_get = lambda *_args, **_kwargs: None
            indicators.cache_set = lambda key, value: saved.setdefault(key, value)
            dm.get_daily = lambda _tk, full=False: None
            out = indicators.sector_relative_strength("AAPL", "tech", lookback_days=20)
        finally:
            indicators.cache_get = old["cache_get"]
            indicators.cache_set = old["cache_set"]
            dm.get_daily = old["get_daily"]
        self.assertEqual(out["sector_etf"], "XLK")
        self.assertEqual(out["relative_strength_source"], "skipped")
        self.assertEqual(out["relative_strength_skipped_reason"], "missing_or_insufficient_history")
        self.assertIn("relstr_AAPL_XLK_20", saved)

    def test_provider_test_requires_token_and_returns_json(self):
        import types
        import routes.api as api

        old = {
            "require_machine_token": api.require_machine_token,
            "request": api.request,
            "jsonify": api.jsonify,
            "cache_get": api.cache_get,
            "cache_set": api.cache_set,
            "_build_provider_test_payload": api._build_provider_test_payload,
        }
        try:
            api.request = types.SimpleNamespace(args={})
            api.require_machine_token = lambda: ("forbidden", 403)
            self.assertEqual(api.api_provider_test(), ("forbidden", 403))

            api.require_machine_token = lambda: True
            api.jsonify = lambda obj=None, **kwargs: obj if obj is not None else kwargs
            api.cache_get = lambda *_args, **_kwargs: None
            api.cache_set = lambda *_args, **_kwargs: None
            api._build_provider_test_payload = lambda: {
                "environment": {"pythonanywhere_mode": True},
                "providers": {},
                "provider_circuit_state": {},
            }
            out = api.api_provider_test()
        finally:
            for name, value in old.items():
                setattr(api, name, value)
        self.assertIn("environment", out)
        self.assertIn("providers", out)

    def test_provider_test_continues_after_failure_and_caches(self):
        import types
        import routes.api as api

        old = {
            "require_machine_token": api.require_machine_token,
            "request": api.request,
            "jsonify": api.jsonify,
            "cache_get": api.cache_get,
            "cache_set": api.cache_set,
            "_provider_test_definitions": api._provider_test_definitions,
            "api_failure_snapshot": api.api_failure_snapshot,
        }
        cached = {}

        def bad(_start):
            raise RuntimeError("boom token=secret")

        def good(start):
            return {"ok": True, "status": "ok", "source": "good", "runtime_seconds": api._provider_elapsed(start)}

        try:
            api.require_machine_token = lambda: True
            api.request = types.SimpleNamespace(args={"force": "1"})
            api.jsonify = lambda obj=None, **kwargs: obj if obj is not None else kwargs
            api.cache_get = lambda *_args, **_kwargs: None
            api.cache_set = lambda key, value: cached.setdefault(key, value)
            api.api_failure_snapshot = lambda: {}
            api._provider_test_definitions = lambda: [
                ("bad_provider", "bad", bad),
                ("good_provider", "good", good),
            ]
            out = api.api_provider_test()
        finally:
            for name, value in old.items():
                setattr(api, name, value)
        self.assertFalse(out["providers"]["bad_provider"]["ok"])
        self.assertEqual(out["providers"]["bad_provider"]["status"], "error")
        self.assertNotIn("secret", out["providers"]["bad_provider"]["error"])
        self.assertTrue(out["providers"]["good_provider"]["ok"])
        self.assertIn(api._PROVIDER_TEST_CACHE_KEY, cached)

    def test_provider_test_skips_do_not_record_failures(self):
        import time as time_mod
        import routes.api as api
        import utils.cache as cache

        old_api = {
            "PYTHONANYWHERE_MODE": api.PYTHONANYWHERE_MODE,
            "FMP_KEY": api.FMP_KEY,
        }
        old_failures = dict(cache._api_failures)
        try:
            cache._api_failures.clear()
            api.PYTHONANYWHERE_MODE = True
            api.FMP_KEY = ""
            stooq = api._check_stooq_direct_spy(time_mod.time())
            fmp = api._check_fmp_daily_spy(time_mod.time())
            snap = cache.api_failure_snapshot()
        finally:
            for name, value in old_api.items():
                setattr(api, name, value)
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)
        self.assertEqual(stooq["status"], "skipped_on_pythonanywhere")
        self.assertEqual(fmp["status"], "skipped_missing_key")
        self.assertEqual(snap, {})

    def test_provider_test_reports_fmp_global_cooldown_as_skip(self):
        import time as time_mod
        import routes.api as api

        old = {
            "FMP_KEY": api.FMP_KEY,
            "_fmp_daily": api._fmp_daily,
            "fmp_daily_global_circuit_state": api.fmp_daily_global_circuit_state,
            "api_failure_snapshot": api.api_failure_snapshot,
        }
        try:
            api.FMP_KEY = "secret-key"
            api._fmp_daily = lambda *_args, **_kwargs: None
            api.fmp_daily_global_circuit_state = lambda: {
                "active": True,
                "status": "rate_limited",
                "cooldown_remaining_sec": 120,
                "last_429_age_sec": 10,
                "last_error": "HTTP 429",
            }
            api.api_failure_snapshot = lambda: {}
            out = api._check_fmp_daily_spy(time_mod.time())
        finally:
            for name, value in old.items():
                setattr(api, name, value)
        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], "skipped_by_global_rate_limit")
        self.assertTrue(out["rate_limited"])
        self.assertEqual(out["cooldown_remaining_sec"], 120)

    def test_provider_test_marks_cached_daily_result(self):
        import time as time_mod
        import pandas as pd
        import routes.api as api

        df = pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
            index=pd.to_datetime(["2026-06-26"]),
        )
        df.attrs.update({
            "source": "fmp_daily",
            "status": "ok",
            "cache_used": True,
            "cache_max_age_sec": 86400,
        })
        old = {
            "FMP_KEY": api.FMP_KEY,
            "_fmp_daily": api._fmp_daily,
        }
        try:
            api.FMP_KEY = "secret-key"
            api._fmp_daily = lambda *_args, **_kwargs: df
            out = api._check_fmp_daily_spy(time_mod.time())
        finally:
            for name, value in old.items():
                setattr(api, name, value)
        self.assertTrue(out["ok"])
        self.assertEqual(out["status"], "ok")
        self.assertTrue(out["cache_used"])
        self.assertEqual(out["cache_max_age_sec"], 86400)
        self.assertIn("local cache", out["provider_test_note"])

    def test_provider_test_path_does_not_use_executor_map(self):
        source = Path("routes/api.py").read_text(encoding="utf-8")
        start = source.index("def _run_provider_check(")
        end = source.index('@app.route("/api/chart/')
        body = source[start:end]
        self.assertNotIn(".map(", body)
        self.assertIn("executor.submit", body)
        self.assertIn("return_when=FIRST_COMPLETED", body)
        self.assertIn("shutdown(wait=False, cancel_futures=True)", body)
        self.assertIn('_finnhub_daily("SPY", use_cache=True)', body)
        self.assertIn('_fmp_daily("SPY", use_cache=True)', body)

    def test_health_route_is_passive(self):
        import routes.portfolio as portfolio

        old_trigger = portfolio.trigger_bot_if_due
        try:
            portfolio.trigger_bot_if_due = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("health must not trigger bot work")
            )
            response = portfolio.health()
        finally:
            portfolio.trigger_bot_if_due = old_trigger
        self.assertEqual(response[0], "ok")
        self.assertEqual(response[1], 200)

    def test_bot_tick_uses_machine_auth_and_runtime_cap(self):
        import routes.portfolio as portfolio

        calls = {}
        old = {
            "require_machine_token": portfolio.require_machine_token,
            "warm_scan_if_due": portfolio.warm_scan_if_due,
            "run_bot": portfolio.run_bot,
            "jsonify": portfolio.jsonify,
            "PYTHONANYWHERE_MODE": portfolio.PYTHONANYWHERE_MODE,
        }
        try:
            portfolio.PYTHONANYWHERE_MODE = False
            portfolio.require_machine_token = lambda: calls.setdefault("auth", True)
            portfolio.warm_scan_if_due = lambda: calls.setdefault("warm_scan", True)
            portfolio.jsonify = lambda obj=None, **kwargs: obj if obj is not None else kwargs

            def fake_run_bot(**kwargs):
                calls["run_kwargs"] = kwargs
                return {"last_no_buy_diagnostics": {
                    "main_blocker": "weak_raw_buys_only",
                    "trading_mode": "NORMAL_MODE",
                    "degraded_mode_active": False,
                    "data_health_blocks": [],
                    "data_health_warnings": ["SPY_DATA_MISSING"],
                    "raw_buy_count": 2,
                    "display_buy_candidate_count": 0,
                    "candidate_pool_count": 0,
                    "ranked_count": 0,
                    "tradable_count": 0,
                    "scan_age_sec": 10,
                    "scan_fresh_rows_count": 3,
                    "min_buy_confidence": 40,
                    "degraded_size_mult": 0.90,
                    "degraded_use_standard_gates_for_testing": True,
                    "degraded_standard_gates_active": False,
                    "degraded_gate_policy": None,
                    "effective_size_mult": 1.0,
                    "effective_min_buy_confidence": 40,
                    "normal_ev_gates_required": True,
                    "normal_risk_caps_required": True,
                    "fresh_quote_required": True,
                    "degraded_min_confidence": 40,
                    "min_trade_size_effective": 100,
                    "cash_available_after_floor": 7000,
                    "gross_exposure_pct": 0.1,
                    "history_source_counts": {"fmp_daily": 6},
                    "history_missing_count": 0,
                    "history_fmp_fallback_count": 6,
                    "history_finnhub_daily_blocked_count": 2,
                    "history_stale_cache_count": 0,
                    "ticker_signal_debug": [{
                        "ticker": "AAPL",
                        "history_source": "fmp_daily",
                        "history_rows": 80,
                        "quote_source": "finnhub_quote",
                        "execution_eligible": False,
                    }],
                }}, False, "hold"

            portfolio.run_bot = fake_run_bot
            payload = portfolio.bot_tick()
        finally:
            for name, value in old.items():
                setattr(portfolio, name, value)
        self.assertTrue(calls["auth"])
        self.assertTrue(calls["warm_scan"])
        self.assertEqual(calls["run_kwargs"]["max_runtime_sec"],
                         portfolio.BOT_TICK_MAX_RUNTIME_SEC)
        diag = payload["last_no_buy_diagnostics"]
        self.assertEqual(payload["status"], "weak_raw_buys_only")
        self.assertTrue(payload["scan_warm_started"])
        self.assertIn("environment", payload)
        self.assertEqual(diag["trading_mode"], "NORMAL_MODE")
        self.assertEqual(diag["display_buy_candidate_count"], 0)
        self.assertEqual(diag["scan_fresh_rows_count"], 3)
        self.assertEqual(diag["min_buy_confidence"], 40)
        self.assertEqual(diag["degraded_min_confidence"], 40)
        self.assertEqual(diag["degraded_size_mult"], 0.90)
        self.assertTrue(diag["degraded_use_standard_gates_for_testing"])
        self.assertFalse(diag["degraded_standard_gates_active"])
        self.assertIsNone(diag["degraded_gate_policy"])
        self.assertEqual(diag["effective_size_mult"], 1.0)
        self.assertEqual(diag["effective_min_buy_confidence"], 40)
        self.assertTrue(diag["normal_ev_gates_required"])
        self.assertTrue(diag["normal_risk_caps_required"])
        self.assertTrue(diag["fresh_quote_required"])
        self.assertEqual(diag["min_trade_size_effective"], 100)
        self.assertEqual(payload["history_source_counts"], {"fmp_daily": 6})
        self.assertEqual(payload["history_fmp_fallback_count"], 6)
        self.assertEqual(payload["history_finnhub_daily_blocked_count"], 2)
        self.assertEqual(diag["history_source_counts"], {"fmp_daily": 6})
        self.assertEqual(diag["ticker_signal_debug"][0]["history_source"], "fmp_daily")

    def test_bot_warm_history_requires_machine_auth_and_caps_default_batch(self):
        import routes.portfolio as portfolio

        calls = {}
        old = {
            "require_machine_token": portfolio.require_machine_token,
            "request": portfolio.request,
            "jsonify": portfolio.jsonify,
            "warm_history": portfolio.warm_history,
            "load_bot": portfolio.load_bot,
            "load_tickers": portfolio.load_tickers,
        }
        try:
            portfolio.require_machine_token = lambda: ("forbidden", 403)
            self.assertEqual(portfolio.bot_warm_history(), ("forbidden", 403))

            portfolio.require_machine_token = lambda: True
            portfolio.request = types.SimpleNamespace(values={})
            portfolio.jsonify = lambda obj=None, **kwargs: obj if obj is not None else kwargs
            portfolio.load_bot = lambda: {
                "last_no_buy_diagnostics": {
                    "top_missing_history_symbols": ["AAPL", "MSFT", "MU", "CAT"],
                }
            }
            portfolio.load_tickers = lambda: ["XOM", "NVDA"]

            def fake_warm(symbols, max_symbols=None, max_fetches=None):
                calls["symbols"] = symbols
                calls["max_symbols"] = max_symbols
                calls["max_fetches"] = max_fetches
                return {
                    "requested_symbols": symbols,
                    "attempted_symbols": symbols[:max_symbols],
                    "warmed_symbols": symbols[:1],
                    "skipped_symbols": [{"symbol": symbols[-1], "reason": "max_symbols_per_warm_call"}],
                    "failed_symbols": [],
                    "cache_hit_symbols": [],
                    "provider_used_by_symbol": {"AAPL": "fmp_daily"},
                    "rows_by_symbol": {"AAPL": 80},
                    "errors_by_symbol": {},
                    "provider_circuits": {},
                }

            portfolio.warm_history = fake_warm
            payload = portfolio.bot_warm_history()
        finally:
            for name, value in old.items():
                setattr(portfolio, name, value)
        self.assertEqual(calls["symbols"], ["AAPL", "MSFT", "MU"])
        self.assertEqual(calls["max_symbols"], 3)
        self.assertEqual(calls["max_fetches"], 3)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["attempted_symbols"], ["AAPL", "MSFT", "MU"])
        self.assertEqual(payload["max_symbols_per_call"], 3)
        self.assertEqual(payload["max_fetches_per_call"], 3)

    def test_ticker_signal_debug_row_exposes_history_and_execution_reason(self):
        import trading.bot as bot

        cfg = {
            "signal": {"min_buy_confidence": 40},
            "market_data_modes": {"degraded_min_confidence": 40},
        }
        row = bot._ticker_debug_row(
            "AAPL",
            {
                "rec": {
                    "cls": "hold",
                    "score": 2.5,
                    "confidence": 35,
                    "thresholds": {"buy_tot": 4},
                    "cats_pos": 1,
                    "cats_neg": 3,
                    "data_quality": 0.6,
                    "data_quality_actual_n": 3,
                    "data_quality_expected_n": 5,
                    "data_quality_missing_fields": ["mfi"],
                    "confidence_before_floor": 35,
                    "confidence_before_penalties": 42,
                    "confidence_after_penalties": 35,
                    "confidence_final": 35,
                    "confidence_floor_applied": False,
                    "confidence_floor_reason": "raw_class_hold",
                    "confidence_penalties": ["weekly_trend_conflict"],
                },
                "ctx": {
                    "history_source": "fmp_daily",
                    "history_rows": 80,
                    "history_last_date": "2026-06-26",
                    "live_bar_applied": False,
                    "live_bar_reason": "quote_stale_or_missing",
                    "quote_fresh": False,
                },
                "quote": {
                    "source": "finnhub_quote",
                    "price": 12.3,
                    "pct": 1.2,
                    "stale": True,
                },
                "price": 12.3,
                "stale": True,
            },
            cfg,
        )
        self.assertEqual(row["ticker"], "AAPL")
        self.assertEqual(row["history_source"], "fmp_daily")
        self.assertEqual(row["history_rows"], 80)
        self.assertEqual(row["quote_source"], "finnhub_quote")
        self.assertEqual(row["data_quality_actual_n"], 3)
        self.assertEqual(row["data_quality_expected_n"], 5)
        self.assertEqual(row["data_quality_missing_fields"], ["mfi"])
        self.assertEqual(row["confidence_before_floor"], 35)
        self.assertEqual(row["confidence_before_penalties"], 42)
        self.assertEqual(row["confidence_after_penalties"], 35)
        self.assertEqual(row["confidence_final"], 35)
        self.assertFalse(row["confidence_floor_applied"])
        self.assertEqual(row["confidence_floor_reason"], "raw_class_hold")
        self.assertEqual(row["confidence_penalties"], ["weekly_trend_conflict"])
        self.assertIn("sell_score_threshold", row)
        self.assertIn("cats_required_for_buy", row)
        self.assertFalse(row["execution_eligible"])
        self.assertEqual(row["why_not_buy"], "stale_candidate_quote")
        self.assertEqual(row["why_not_execution_eligible"], "raw_class_hold")
        self.assertFalse(row["quote_fresh"])

    def test_bot_tick_skips_scan_warm_on_pythonanywhere_and_reports_timeout(self):
        import routes.portfolio as portfolio

        calls = {}
        old = {
            "require_machine_token": portfolio.require_machine_token,
            "warm_scan_if_due": portfolio.warm_scan_if_due,
            "run_bot": portfolio.run_bot,
            "jsonify": portfolio.jsonify,
            "PYTHONANYWHERE_MODE": portfolio.PYTHONANYWHERE_MODE,
        }
        try:
            portfolio.PYTHONANYWHERE_MODE = True
            portfolio.require_machine_token = lambda: calls.setdefault("auth", True)
            portfolio.warm_scan_if_due = lambda: (_ for _ in ()).throw(
                AssertionError("PA /bot/tick must not warm scan")
            )
            portfolio.jsonify = lambda obj=None, **kwargs: obj if obj is not None else kwargs

            def fake_run_bot(**kwargs):
                calls["run_kwargs"] = kwargs
                return {"last_no_buy_diagnostics": {
                    "main_blocker": "partial_timeout",
                    "partial_result": True,
                    "timeout_reason": "BOT_TICK_MAX_RUNTIME_SEC",
                    "paper_trading_locked": True,
                    "paper_lock_reason": "BOT_TICK_TIMEOUT",
                    "fetch_timeout_tickers": ["AAPL", "NVDA"],
                }}, False, "partial_timeout"

            portfolio.run_bot = fake_run_bot
            payload = portfolio.bot_tick()
        finally:
            for name, value in old.items():
                setattr(portfolio, name, value)

        diag = payload["last_no_buy_diagnostics"]
        self.assertTrue(calls["auth"])
        self.assertEqual(payload["status"], "partial_timeout")
        self.assertFalse(payload["traded"])
        self.assertEqual(payload["last_action"], "partial_timeout")
        self.assertFalse(payload["scan_warm_started"])
        self.assertEqual(diag["timeout_reason"], "BOT_TICK_MAX_RUNTIME_SEC")
        self.assertEqual(diag["paper_lock_reason"], "BOT_TICK_TIMEOUT")
        self.assertEqual(diag["fetch_timeout_tickers"], ["AAPL", "NVDA"])

    def test_bot_fetch_and_scan_paths_are_deadline_safe(self):
        source = Path("trading/bot.py").read_text(encoding="utf-8")
        bot_start = source.index("def _run_bot_locked(")
        scan_start = source.index("def run_scan():")
        bot_body = source[bot_start:scan_start]
        scan_body = source[scan_start:source.index("def warm_scan_if_due")]

        self.assertNotIn(".map(_fetch_one", bot_body)
        self.assertIn("wait(", bot_body)
        self.assertIn("FIRST_COMPLETED", bot_body)
        self.assertIn("shutdown(wait=False, cancel_futures=True)", bot_body)
        self.assertIn("fetch_timeout_tickers", bot_body)
        self.assertIn("_call_with_deadline(lambda: get_market_regime(cfg)", bot_body)
        self.assertIn("_call_with_deadline(get_vix", bot_body)

        self.assertNotIn(".map(_scan_one", scan_body)
        self.assertIn("scan_deadline = time.time() + (20 if PYTHONANYWHERE_MODE else 45)", scan_body)
        self.assertIn("wait(", scan_body)
        self.assertIn("shutdown(wait=False, cancel_futures=True)", scan_body)

    def test_pa_batch_defaults_are_free_tier_safe(self):
        source = Path("utils/deploy_config.py").read_text(encoding="utf-8")
        self.assertIn('PA_TICKERS_PER_BOT_RUN = env_int("PA_TICKERS_PER_BOT_RUN", 6, 1, 8)', source)
        self.assertIn('PA_SCAN_BATCH_SIZE = env_int("PA_SCAN_BATCH_SIZE", 4, 1, 8)', source)

    def test_legacy_sizing_skip_path_preserves_skip_metadata(self):
        source = Path("trading/bot.py").read_text(encoding="utf-8")
        self.assertNotIn('_record_skip(b, cd["ticker"], skip_reason, cd["signal"], cd["confidence"])', source)
        self.assertIn('display_signal=cd.get("display_signal")', source)
        self.assertIn('original_reason=cd.get("original_reason") or skip_reason', source)
        self.assertIn('skip_stage=cd.get("skip_stage") or "sizing_floor"', source)

    def test_get_vix_reports_spy_proxy_source(self):
        import pandas as pd
        import trading.risk as risk

        old_cache_get = risk.cache_get
        old_cache_set = risk.cache_set
        old_daily = risk._regime_daily_bars
        old_append = risk._append_live_bar
        try:
            risk.cache_get = lambda *_args, **_kwargs: None
            risk.cache_set = lambda *_args, **_kwargs: None
            risk._append_live_bar = lambda d, _tk: d
            for source in ("finnhub_daily", "fmp_daily",
                           "stale_cache:finnhub_daily", "stale_cache:fmp_daily"):
                idx = pd.date_range("2026-01-01", periods=80, freq="B")
                df = pd.DataFrame({"Close": [100 + i * 0.4 for i in range(80)]}, index=idx)
                df.attrs["source"] = source
                risk._regime_daily_bars = lambda _tk, frame=df: frame
                out = risk.get_vix()
                self.assertTrue(out["data_ok"])
                self.assertEqual(out["source"], "spy_realized_vol_proxy")
                self.assertEqual(out["spy_data_source"], source)
                self.assertEqual(out["vix_display"], "SPY_REALIZED_VOL_PROXY")
        finally:
            risk.cache_get = old_cache_get
            risk.cache_set = old_cache_set
            risk._regime_daily_bars = old_daily
            risk._append_live_bar = old_append

    def test_get_vix_recomputes_after_failed_cache_when_spy_data_exists(self):
        import pandas as pd
        import trading.risk as risk

        idx = pd.date_range("2026-01-01", periods=80, freq="B")
        df = pd.DataFrame({"Close": [100 + i * 0.4 for i in range(80)]}, index=idx)
        df.attrs["source"] = "fmp_daily"
        old_cache_get = risk.cache_get
        old_cache_set = risk.cache_set
        old_daily = risk._regime_daily_bars
        old_append = risk._append_live_bar
        try:
            risk.cache_get = lambda *_args, **_kwargs: {
                "data_ok": False,
                "data_status": "missing_or_insufficient_spy_history",
                "volatility_window_days": 20,
            }
            risk.cache_set = lambda *_args, **_kwargs: None
            risk._regime_daily_bars = lambda _tk: df
            risk._append_live_bar = lambda d, _tk: d
            out = risk.get_vix()
        finally:
            risk.cache_get = old_cache_get
            risk.cache_set = old_cache_set
            risk._regime_daily_bars = old_daily
            risk._append_live_bar = old_append
        self.assertTrue(out["data_ok"])
        self.assertEqual(out["spy_data_source"], "fmp_daily")
        self.assertEqual(out["vix_display"], "SPY_REALIZED_VOL_PROXY")

    def test_get_history_synthetic_recorded_trust_boundary(self):
        # Audit P1-7: ≥25 recorded closes with a fresh newest point → trusted
        # synthetic ctx; fewer or old points stay an untrusted fallback.
        import market.history as history
        import trading.indicators as indicators
        import trading.risk as risk

        now = int(time.time())
        fresh_pts = [[now - (30 - i) * 3600, 100.0 + i * 0.1] for i in range(30)]
        old_pts = [[now - 12 * 86400 - (30 - i) * 3600, 100.0] for i in range(30)]
        few_pts = [[now - (20 - i) * 3600, 100.0] for i in range(20)]
        old = {
            "cache_get": history.cache_get,
            "cache_set": history.cache_set,
            "_daily_bars": history._daily_bars,
            "load_price_hist": indicators.load_price_hist,
            "get_sector": risk.get_sector,
        }
        try:
            history.cache_get = lambda *_a, **_k: None
            history.cache_set = lambda *_a, **_k: None
            history._daily_bars = lambda _tk: None
            risk.get_sector = lambda _tk: None

            indicators.load_price_hist = lambda: {"SYN": fresh_pts}
            r = history.get_history("SYN")
            self.assertEqual(r["history_source"], "synthetic_recorded")
            self.assertEqual(r["history_status"], "synthetic_recorded")
            self.assertEqual(r["history_rows"], 30)
            self.assertIn("rsi", r)
            self.assertIn("ma30", r)
            import trading.bot as bot
            st = bot._history_execution_status(r)
            self.assertTrue(st["trusted"])
            self.assertIsNone(st["blocker"])

            indicators.load_price_hist = lambda: {"SYN": few_pts}
            r2 = history.get_history("SYN")
            self.assertEqual(r2.get("history_rows", 0), 0)

            indicators.load_price_hist = lambda: {"SYN": old_pts}
            r3 = history.get_history("SYN")
            self.assertEqual(r3.get("history_rows", 0), 0)
        finally:
            history.cache_get = old["cache_get"]
            history.cache_set = old["cache_set"]
            history._daily_bars = old["_daily_bars"]
            indicators.load_price_hist = old["load_price_hist"]
            risk.get_sector = old["get_sector"]

    def test_half_missing_providers_still_rank_candidates(self):
        # Audit P1 goal: with 50% of tickers on synthetic ctx and 50% missing,
        # ranking still produces tradable candidates (no MISSING_HISTORY wall).
        import trading.bot as bot

        good_ctx = {"history_rows": 30, "history_source": "synthetic_recorded",
                    "history_status": "synthetic_recorded", "adx": 30, "rsi": 55,
                    "week_chg_pct": 0, "dist_from_high_pct": 0, "vol_ratio": 1.2,
                    "avg_dollar_vol_20d": 100_000_000, "atr_pct": 2.0}
        dead_ctx = {"history_rows": 0, "history_source": "missing",
                    "history_status": "missing"}
        pool = []
        for i in range(8):
            cand = _entry_candidate(f"MIX{i}", confidence=55, cluster="trend")
            cand["ctx"] = dict(good_ctx if i % 2 == 0 else dead_ctx)
            pool.append(cand)
        buyable = [c for c in pool
                   if bot._history_execution_status(c["ctx"])["blocker"] is None]
        self.assertEqual(len(buyable), 4)
        ranked = rank_candidates(buyable, 10_000, -6, "neutral", 1, 1, 1, {},
                                 min_position_usd=150)
        self.assertTrue(any(c["tradable"] for c in ranked))

    def test_rel_str_spy_fallback_when_sector_etf_missing(self):
        # Audit P1-12: XLK/XLY 402 on PA → benchmark vs SPY, labeled.
        import pandas as pd
        import trading.indicators as ind
        import market.data_manager as dm

        df = pd.DataFrame({"Close": [100.0 + i for i in range(25)]})
        old_get_daily = dm.get_daily
        old_cache_get = ind.cache_get
        old_cache_set = ind.cache_set
        try:
            ind.cache_get = lambda *_a, **_k: None
            ind.cache_set = lambda *_a, **_k: None
            dm.get_daily = lambda tk: None if tk == "XLK" else df
            out = ind.sector_relative_strength("TKR", "tech", lookback_days=20)
        finally:
            dm.get_daily = old_get_daily
            ind.cache_get = old_cache_get
            ind.cache_set = old_cache_set
        self.assertEqual(out["relative_strength_source"], "spy_fallback")
        self.assertEqual(out["rel_str_benchmark"], "SPY")
        self.assertIn("rel_str_pct", out)

    def test_credit_signal_neutral_immediately_on_pa(self):
        # Audit P1-13: HYG/IEI permanently 402 on PA — no fetch spend.
        import trading.risk as risk

        old_mode = risk.PYTHONANYWHERE_MODE
        old_cache_get = risk.cache_get
        try:
            risk.PYTHONANYWHERE_MODE = True
            def _boom(*_a, **_k):
                raise AssertionError("credit_signal touched cache/providers on PA")
            risk.cache_get = _boom
            out = risk.credit_signal()
        finally:
            risk.PYTHONANYWHERE_MODE = old_mode
            risk.cache_get = old_cache_get
        self.assertEqual(out["credit_label"], "neutral")
        self.assertEqual(out["credit_pct"], 50.0)
        self.assertEqual(out["credit_status"], "unavailable_on_pa")

    def test_data_quality_floor_with_core_history_present(self):
        # Audit P1-11: optional categories missing (news/analyst/insider/
        # rel_str/mfi/weekly) must not sink an otherwise-valid BUY: dq >= 0.70.
        core_only_ctx = {
            "current": 105.0, "ma7": 103.0, "ma30": 100.0, "rsi": 60.0,
            "week_chg_pct": 2.0, "adx": 30.0, "mom_30d_pct": 5.0,
            "vol_ratio": 1.4, "avg_dollar_vol_20d": 100_000_000,
            "history_rows": 30, "history_source": "synthetic_recorded",
            "history_status": "synthetic_recorded",
        }
        rec = get_recommendation(0.0, core_only_ctx, regime=None)
        self.assertGreaterEqual(rec["data_quality"], 0.70)
        self.assertLess(rec["data_quality_raw"], 0.70)
        self.assertTrue(rec["data_quality_floor_applied"])

        rec_no_core = get_recommendation(0.0, {}, regime=None)
        self.assertFalse(rec_no_core["data_quality_floor_applied"])

    def _run_stale_holding_tick(self, snapshot_price, snapshot_age_sec,
                                peak=None, avg_cost=100.0):
        # Audit P1-14 harness: one holding, stale live quote, recorded snapshot.
        import trading.bot as bot

        now = time.time()
        state = {
            "cash": 1000.0,
            "starting": 1000.0,
            "holdings": {"STALE1": {
                "shares": 5.0,
                "avg_cost": avg_cost,
                "entry_ts": now - 48 * 3600,
                "peak": peak if peak is not None else avg_cost,
                "trough": avg_cost,
                "peak_pnl_pct": (((peak or avg_cost) - avg_cost) / avg_cost * 100.0),
            }},
            "history": [],
            "last_trade": 0,
            "stopped": False,
        }
        hold_rec = {"cls": "hold", "signal": "HOLD", "confidence": 50,
                    "score": 0, "categories": {}, "reasons": []}
        names = [
            "load_bot", "save_bot", "is_market_open", "in_new_buy_window",
            "load_tickers", "get_market_regime", "get_vix", "get_news",
            "get_history", "get_intraday_context", "get_earnings_soon",
            "get_analyst_rec", "get_insider_sentiment", "get_recommendation",
            "classify_catalyst", "get_quote", "get_sector", "get_corr_group",
            "_scan_snapshot", "load_feedback_stats", "load_recent_suggestions",
            "log_suggestion_run", "prune_suggestion_store", "prune_cache_dir",
            "build_extra_ticker_suggestions", "load_price_hist",
        ]
        old = {n: getattr(bot, n) for n in names}
        try:
            bot.load_bot = lambda: state
            bot.save_bot = lambda _b: None
            bot.is_market_open = lambda: True
            bot.in_new_buy_window = lambda: True
            bot.load_tickers = lambda: []
            bot.get_market_regime = lambda _cfg=None: {
                "regime": "neutral", "regime_effective": "neutral",
                "regime_v3": "normal", "regime_v3_effective": "normal",
                "regime_v3_raw": "normal", "spy_data_ok": True,
                "regime_data_status": "ok", "top_sectors": [],
            }
            bot.get_vix = lambda: {"regime": "NORMAL", "mult": 1.0, "vix": 12.0,
                                   "data_ok": True, "data_status": "ok"}
            bot.get_news = lambda _t: ([], 0.0)
            bot.get_history = lambda _t: {}
            bot.get_intraday_context = lambda _t: {}
            bot.get_earnings_soon = lambda _t: {}
            bot.get_analyst_rec = lambda _t: {}
            bot.get_insider_sentiment = lambda _t: {}
            bot.get_recommendation = lambda *_a, **_k: dict(hold_rec)
            bot.classify_catalyst = lambda *_a, **_k: {}
            bot.get_quote = lambda _t: {"price": 0, "stale": True}
            bot.get_sector = lambda _t: "tech"
            bot.get_corr_group = lambda _t: None
            bot._scan_snapshot = lambda: ([], 0)
            bot.load_feedback_stats = lambda *_a, **_k: {}
            bot.load_recent_suggestions = lambda *_a, **_k: {}
            bot.log_suggestion_run = lambda *_a, **_k: None
            bot.prune_suggestion_store = lambda *_a, **_k: None
            bot.prune_cache_dir = lambda **_k: {"removed": 0, "kept": 0}
            bot.build_extra_ticker_suggestions = lambda *_a, **_k: []
            bot.load_price_hist = lambda: {
                "STALE1": [[int(now - snapshot_age_sec), float(snapshot_price)]]
            }
            out_state, traded, _action = bot._run_bot_locked(force=True)
        finally:
            for name, value in old.items():
                setattr(bot, name, value)
        return out_state, traded

    def test_stale_quote_protective_stop_fires_only_hard_stop(self):
        # Deep loss vs fresh snapshot → ONLY the hard stop fires
        out, traded = self._run_stale_holding_tick(80.0, 20 * 60)
        self.assertTrue(traded)
        self.assertNotIn("STALE1", out["holdings"])
        outcome = out["trade_outcomes"][-1]
        self.assertEqual(outcome["exit_reason"], "stop_loss_stale_quote")

    def test_stale_quote_small_loss_does_not_exit_even_when_aging_due(self):
        # −1%, held 48h (> neutral aging 12h): live aging would exit; stale must not
        out, traded = self._run_stale_holding_tick(99.0, 20 * 60)
        self.assertFalse(traded)
        self.assertIn("STALE1", out["holdings"])

    def test_stale_quote_old_snapshot_halts_completely(self):
        # Snapshot 90 min old (> 60 min freshness) → halt exactly as before
        out, traded = self._run_stale_holding_tick(80.0, 90 * 60)
        self.assertFalse(traded)
        self.assertIn("STALE1", out["holdings"])
        # and no peak/trough mutation happened off the recorded price
        self.assertEqual(out["holdings"]["STALE1"]["peak"], 100.0)

    def test_stale_quote_trail_never_fires(self):
        # +20% with peak +40% → live trail would exit; stale must hold
        out, traded = self._run_stale_holding_tick(120.0, 20 * 60, peak=140.0)
        self.assertFalse(traded)
        self.assertIn("STALE1", out["holdings"])

    def test_record_hold_uses_display_signal_labels(self):
        import trading.bot as bot

        state = {"history": [], "total_trades": 0}
        rec = {
            "cls": "buy",
            "signal": "BUY",
            "confidence": 34,
            "display_signal_label": "BULLISH_LEAN",
        }
        bot._record_hold(state, "no execution", {"AAA": {"rec": rec}})
        row = state["history"][0]
        self.assertEqual(row["action"], "HOLD")
        self.assertIn("AAA:BULLISH_LEAN(34%)", row["signals"])
        self.assertIn("Top signal: AAA BULLISH_LEAN @ 34%", row["reason"])
        self.assertNotIn("AAA:BUY(34%)", row["signals"])

    def test_bot_interval_cooldown_persists_no_buy_diagnostics(self):
        import trading.bot as bot

        state = {
            "cash": 1000.0,
            "starting": 1000.0,
            "holdings": {},
            "history": [],
            "last_trade": time.time(),
            "stopped": False,
        }
        saved = []
        old_load = bot.load_bot
        old_save = bot.save_bot
        old_market_open = bot.is_market_open
        old_prune = bot.prune_cache_dir
        try:
            bot.load_bot = lambda: state
            bot.save_bot = lambda b: saved.append(dict(b))
            bot.is_market_open = lambda: True
            bot.prune_cache_dir = lambda **_kwargs: {"removed": 0, "kept": 0}
            out_state, traded, action = bot._run_bot_locked(force=False)
        finally:
            bot.load_bot = old_load
            bot.save_bot = old_save
            bot.is_market_open = old_market_open
            bot.prune_cache_dir = old_prune
        self.assertFalse(traded)
        self.assertEqual(action, "interval_cooldown")
        diag = out_state["last_no_buy_diagnostics"]
        self.assertEqual(diag["main_blocker"], "interval_cooldown")
        self.assertEqual(diag["min_buy_confidence"], 40)
        self.assertEqual(diag["degraded_min_confidence"], 40)
        self.assertEqual(diag["min_trade_size_effective"], 150.0)
        self.assertEqual(saved[-1]["last_cache_prune"], {"removed": 0, "kept": 0})

    def test_bot_no_trade_run_persists_no_buy_diagnostics(self):
        import trading.bot as bot

        state = {
            "cash": 1000.0,
            "starting": 1000.0,
            "holdings": {},
            "history": [],
            "last_trade": 0,
            "stopped": False,
        }
        saved = []
        old = {
            "load_bot": bot.load_bot,
            "save_bot": bot.save_bot,
            "is_market_open": bot.is_market_open,
            "in_new_buy_window": bot.in_new_buy_window,
            "load_tickers": bot.load_tickers,
            "get_market_regime": bot.get_market_regime,
            "get_vix": bot.get_vix,
            "get_quote": bot.get_quote,
            "_scan_snapshot": bot._scan_snapshot,
            "load_feedback_stats": bot.load_feedback_stats,
            "load_recent_suggestions": bot.load_recent_suggestions,
            "log_suggestion_run": bot.log_suggestion_run,
            "prune_suggestion_store": bot.prune_suggestion_store,
            "prune_cache_dir": bot.prune_cache_dir,
        }
        try:
            bot.load_bot = lambda: state
            bot.save_bot = lambda b: saved.append(dict(b))
            bot.is_market_open = lambda: True
            bot.in_new_buy_window = lambda: True
            bot.load_tickers = lambda: []
            bot.get_market_regime = lambda _cfg=None: {
                "regime": "neutral",
                "regime_effective": "neutral",
                "regime_v3": "normal",
                "regime_v3_effective": "normal",
                "regime_v3_raw": "fallback",
                "top_sectors": [],
            }
            bot.get_vix = lambda: {"regime": "NORMAL", "mult": 1.0, "vix": 12.0}
            bot.get_quote = lambda _ticker: {"price": 0.0, "stale": True}
            bot._scan_snapshot = lambda: ([], 0)
            bot.load_feedback_stats = lambda *_args, **_kwargs: {}
            bot.load_recent_suggestions = lambda *_args, **_kwargs: {}
            bot.log_suggestion_run = lambda *_args, **_kwargs: None
            bot.prune_suggestion_store = lambda *_args, **_kwargs: None
            bot.prune_cache_dir = lambda **_kwargs: {"removed": 0, "kept": 0}
            out_state, traded, _action = bot._run_bot_locked(force=True)
        finally:
            for name, value in old.items():
                setattr(bot, name, value)
        diag = out_state["last_no_buy_diagnostics"]
        self.assertFalse(traded)
        self.assertEqual(diag["candidate_pool_count"], 0)
        self.assertEqual(diag["main_blocker"], "no_buy_candidates")
        self.assertEqual(saved[-1]["last_no_buy_diagnostics"]["main_blocker"],
                         "no_buy_candidates")

    def test_missing_spy_regime_data_enters_degraded_mode(self):
        import trading.bot as bot

        state = {
            "cash": 10_000.0,
            "starting": 10_000.0,
            "holdings": {},
            "history": [],
            "last_trade": 0,
            "stopped": False,
        }
        captured = {}
        old = {
            "load_bot": bot.load_bot,
            "save_bot": bot.save_bot,
            "is_market_open": bot.is_market_open,
            "in_new_buy_window": bot.in_new_buy_window,
            "load_tickers": bot.load_tickers,
            "get_market_regime": bot.get_market_regime,
            "get_vix": bot.get_vix,
            "get_news": bot.get_news,
            "get_history": bot.get_history,
            "get_intraday_context": bot.get_intraday_context,
            "get_earnings_soon": bot.get_earnings_soon,
            "get_analyst_rec": bot.get_analyst_rec,
            "get_insider_sentiment": bot.get_insider_sentiment,
            "get_recommendation": bot.get_recommendation,
            "classify_catalyst": bot.classify_catalyst,
            "get_quote": bot.get_quote,
            "get_sector": bot.get_sector,
            "get_corr_group": bot.get_corr_group,
            "rank_candidates": bot.rank_candidates,
            "_scan_snapshot": bot._scan_snapshot,
            "build_extra_ticker_suggestions": bot.build_extra_ticker_suggestions,
            "prune_cache_dir": bot.prune_cache_dir,
        }
        try:
            bot.load_bot = lambda: state
            bot.save_bot = lambda _b: None
            bot.is_market_open = lambda: True
            bot.in_new_buy_window = lambda: True
            bot.load_tickers = lambda: ["AAA"]
            bot.get_market_regime = lambda _cfg=None: {
                "regime": "neutral",
                "regime_effective": "neutral",
                "regime_v3": "neutral",
                "regime_v3_effective": "neutral",
                "regime_v3_raw": "fallback",
                "regime_v3_fallback": True,
                "regime_v3_reason": "fallback neutral: missing_spy_history",
                "spy_data_ok": False,
                "regime_data_status": "missing_spy_history",
                "regime_data_fallback": True,
                "spy_rows": 0,
                "spy_last_date": None,
                "spy_mom_label": "1M / 22 trading days",
                "top_sectors": [],
            }
            bot.get_vix = lambda: {
                "regime": "NORMAL",
                "mult": 1.0,
                "vix": 12.0,
                "data_ok": True,
                "data_status": "ok",
                "source": "spy_realized_vol_proxy",
                "volatility_source": "spy_realized_vol_proxy",
                "vix_display": "SPY_REALIZED_VOL_PROXY",
            }
            bot.get_news = lambda _tk: ([], 0.0)
            bot.get_history = lambda _tk: {
                "adx": 30, "rsi": 55, "week_chg_pct": 0,
                "dist_from_high_pct": 0, "avg_dollar_vol_20d": 100_000_000,
                "vol_ratio": 1.5, "history_rows": 80,
                "history_source": "finnhub_daily", "history_status": "ok",
            }
            bot.get_intraday_context = lambda _tk: {}
            bot.get_earnings_soon = lambda _tk: {}
            bot.get_analyst_rec = lambda _tk: {}
            bot.get_insider_sentiment = lambda _tk: {}
            bot.get_recommendation = lambda *_args, **_kwargs: {
                "cls": "buy", "signal": "BUY", "confidence": 80,
                "score": 4.0, "categories": {"trend": 2.0},
            }
            bot.classify_catalyst = lambda *_args, **_kwargs: {}
            bot.get_quote = lambda tk: {
                "price": 100.0 if tk == "AAA" else 0.0,
                "stale": tk != "AAA",
            }
            bot.get_sector = lambda _tk: "tech"
            bot.get_corr_group = lambda _tk: None
            bot._scan_snapshot = lambda: ([], 0)
            bot.build_extra_ticker_suggestions = lambda *_args, **_kwargs: []
            bot.prune_cache_dir = lambda **_kwargs: {"removed": 0, "kept": 0}

            def fake_rank(_cands, _pt, _stop, _regime, vix_mult, *_args, **_kwargs):
                captured["vix_mult"] = vix_mult
                captured["mode_size_mult"] = _kwargs.get("mode_size_mult")
                captured["mode_size_reason"] = _kwargs.get("mode_size_reason")
                return []

            bot.rank_candidates = fake_rank
            out_state, _traded, _action = bot._run_bot_locked(force=True)
        finally:
            for name, value in old.items():
                setattr(bot, name, value)
        # Audit P1-6: missing SPY now truthfully reports DEGRADED_MODE, and the
        # standard-gates parity path keeps every effective gate/size at NORMAL.
        self.assertEqual(captured["vix_mult"], 1.0)
        self.assertEqual(captured["mode_size_mult"], 1.0)
        self.assertIsNone(captured["mode_size_reason"])
        diag = out_state["last_no_buy_diagnostics"]
        self.assertTrue(diag["regime_allow_buys"])
        self.assertEqual(diag["trading_mode"], "DEGRADED_MODE")
        self.assertTrue(diag["degraded_mode_active"])
        self.assertTrue(diag["degraded_use_standard_gates_for_testing"])
        self.assertTrue(diag["degraded_standard_gates_active"])
        self.assertEqual(diag["degraded_gate_policy"], "standard_gates_for_testing")
        self.assertEqual(diag["effective_size_mult"], 1.0)
        self.assertEqual(diag["effective_min_buy_confidence"], 40)
        self.assertTrue(diag["normal_ev_gates_required"])
        self.assertTrue(diag["normal_risk_caps_required"])
        self.assertTrue(diag["fresh_quote_required"])
        self.assertEqual(diag["data_health_blocks"], ["SPY_DATA_MISSING"])
        self.assertIn("SPY_DATA_MISSING", diag["data_health_warnings"])
        self.assertTrue(diag["regime_data_fallback"])
        self.assertEqual(diag["regime_data_size_mult"], 1.0)

    def test_scan_payload_cache_prevents_buy_pass_refetch(self):
        import trading.bot as bot

        state = {
            "cash": 10_000.0,
            "starting": 10_000.0,
            "holdings": {},
            "history": [],
            "last_trade": 0,
            "stopped": False,
        }
        payload = {
            "arts": [],
            "sent": 0.0,
            "ctx": {
                "adx": 30,
                "rsi": 55,
                "week_chg_pct": 0,
                "dist_from_high_pct": 0,
                "avg_dollar_vol_20d": 100_000_000,
                "vol_ratio": 1.5,
                "history_rows": 80,
                "history_source": "finnhub_daily",
                "history_status": "ok",
            },
            "earn": {},
            "analyst": {},
            "insider": {},
            "quote": {"price": 100.0, "stale": False},
            "price": 100.0,
            "stale": False,
            "rec": {
                "cls": "buy",
                "signal": "BUY",
                "confidence": 80,
                "score": 4.0,
                "categories": {"trend": 2.0},
                "catalyst": {},
                "reasons": ["cached scan payload"],
            },
        }
        captured = {}
        refetched = []
        old = {
            "load_bot": bot.load_bot,
            "save_bot": bot.save_bot,
            "is_market_open": bot.is_market_open,
            "in_new_buy_window": bot.in_new_buy_window,
            "load_tickers": bot.load_tickers,
            "get_market_regime": bot.get_market_regime,
            "get_vix": bot.get_vix,
            "get_news": bot.get_news,
            "get_history": bot.get_history,
            "get_earnings_soon": bot.get_earnings_soon,
            "get_analyst_rec": bot.get_analyst_rec,
            "get_insider_sentiment": bot.get_insider_sentiment,
            "get_recommendation": bot.get_recommendation,
            "classify_catalyst": bot.classify_catalyst,
            "get_quote": bot.get_quote,
            "get_sector": bot.get_sector,
            "get_corr_group": bot.get_corr_group,
            "rank_candidates": bot.rank_candidates,
            "_scan_snapshot": bot._scan_snapshot,
            "build_extra_ticker_suggestions": bot.build_extra_ticker_suggestions,
            "cache_get": bot.cache_get,
            "prune_cache_dir": bot.prune_cache_dir,
        }
        try:
            bot.load_bot = lambda: state
            bot.save_bot = lambda _b: None
            bot.is_market_open = lambda: True
            bot.in_new_buy_window = lambda: True
            bot.load_tickers = lambda: []
            bot.get_market_regime = lambda _cfg=None: {
                "regime": "bull",
                "regime_effective": "bull",
                "regime_v3": "normal",
                "regime_v3_effective": "normal",
                "regime_v3_raw": "normal",
                "spy_data_ok": True,
                "regime_data_status": "ok",
                "top_sectors": [],
            }
            bot.get_vix = lambda: {
                "regime": "NORMAL",
                "mult": 1.0,
                "vix": 12.0,
                "data_ok": True,
                "data_status": "ok",
            }

            def fail_fetch(name):
                def _inner(tk, *_args, **_kwargs):
                    if tk == "SCANX":
                        refetched.append(name)
                        raise AssertionError(f"{name} refetched {tk}")
                    return {} if name != "get_news" else ([], 0.0)
                return _inner

            bot.get_news = fail_fetch("get_news")
            bot.get_history = fail_fetch("get_history")
            bot.get_earnings_soon = fail_fetch("get_earnings_soon")
            bot.get_analyst_rec = fail_fetch("get_analyst_rec")
            bot.get_insider_sentiment = fail_fetch("get_insider_sentiment")
            bot.get_recommendation = fail_fetch("get_recommendation")
            bot.classify_catalyst = fail_fetch("classify_catalyst")

            def quote(tk):
                if tk == "SCANX":
                    refetched.append("get_quote")
                    raise AssertionError("quote refetched SCANX")
                return {"price": 400.0, "stale": False}

            bot.get_quote = quote
            bot.get_sector = lambda _tk: "tech"
            bot.get_corr_group = lambda _tk: None
            bot._scan_snapshot = lambda: (
                [{"ticker": "SCANX", "direction": 1, "price": 100.0,
                  "confidence": 80, "score": 4.0, "pct": 3.0}],
                int(time.time()),
            )
            bot.cache_get = (
                lambda key, max_age=None: payload if key == "scan_payload_SCANX" else None
            )
            bot.build_extra_ticker_suggestions = lambda *_args, **_kwargs: []
            bot.prune_cache_dir = lambda **_kwargs: {"removed": 0, "kept": 0}

            def fake_rank(cands, *_args, **_kwargs):
                captured["tickers"] = [c["ticker"] for c in cands]
                return []

            bot.rank_candidates = fake_rank
            out_state, traded, _action = bot._run_bot_locked(force=True)
        finally:
            for name, value in old.items():
                setattr(bot, name, value)
        self.assertFalse(traded)
        self.assertEqual(refetched, [])
        self.assertEqual(captured["tickers"], ["SCANX"])
        self.assertEqual(out_state["last_no_buy_diagnostics"]["scan_payload_misses"], 0)

    def test_api_bot_status_is_safe_and_path_free(self):
        import app as _app  # noqa: F401
        import routes.api as api

        old_load = api.load_bot
        old_jsonify = api.jsonify
        try:
            api.jsonify = lambda obj=None, **kwargs: obj if obj is not None else kwargs
            api.load_bot = lambda: {
                "last_no_buy_diagnostics": {
                    "main_blocker": "no_buy_candidates",
                    "candidate_pool_count": 0,
                    "raw_buy_count": 1,
                    "display_buy_candidate_count": 0,
                    "history_source_counts": {"stale_cache": 2},
                    "history_fmp_attempted_count": 3,
                    "history_fmp_skipped_count": 2,
                    "history_fmp_rate_limited_count": 1,
                    "history_fmp_global_circuit_skipped_count": 2,
                    "fmp_daily_global_circuit_status": "rate_limited",
                    "fmp_daily_rate_limited": True,
                    "fmp_daily_cooldown_remaining_sec": 120,
                    "fmp_daily_last_429_age_sec": 15,
                    "data_health_blocks": [],
                    "data_health_warnings": ["SPY_DATA_MISSING"],
                    "trading_mode": "NORMAL_MODE",
                    "degraded_mode_active": False,
                    "degraded_mode_reason": None,
                    "min_buy_confidence": 40,
                    "degraded_size_mult": 0.90,
                    "degraded_use_standard_gates_for_testing": True,
                    "degraded_standard_gates_active": False,
                    "degraded_gate_policy": None,
                    "effective_size_mult": 1.0,
                    "effective_min_buy_confidence": 40,
                    "normal_ev_gates_required": True,
                    "normal_risk_caps_required": True,
                    "fresh_quote_required": True,
                    "degraded_min_confidence": 40,
                    "min_trade_size_effective": 100,
                    "degraded_reject_counts": {"DEGRADED_CONFIDENCE_BELOW_MIN": 1},
                    "volatility_value": 14.2,
                    "regime_source": "neutral_missing_spy",
                    "top_buyable_rejects": [
                        {"ticker": "AAA", "rejection_reason": "VERY_LOW_VOLUME_CONFIRMATION"}
                    ],
                    "ticker_signal_debug": [
                        {"ticker": f"T{i}"} for i in range(api.PA_TICKERS_PER_BOT_RUN + 3)
                    ],
                    "partial_result": True,
                    "timeout_reason": "max_runtime",
                    "timeout_stage": "fetch",
                    "fetch_timeout_tickers": ["ZZZ"],
                    "tick_runtime_seconds": 0.25,
                    "storage_base_dir": r"C:\secret",
                    "bot_file_path": r"C:\secret\bot_state.json",
                },
                "last_state_write_ts": 123,
                "last_cache_prune": {"removed": 1, "kept": 2},
            }
            out = api.api_bot_status()
        finally:
            api.load_bot = old_load
            api.jsonify = old_jsonify
        self.assertEqual(out["last_no_buy_diagnostics"]["main_blocker"],
                         "no_buy_candidates")
        self.assertEqual(out["last_no_buy_diagnostics"]["raw_buy_count"], 1)
        self.assertEqual(out["last_no_buy_diagnostics"]["data_health_blocks"], [])
        self.assertEqual(out["last_no_buy_diagnostics"]["data_health_warnings"],
                         ["SPY_DATA_MISSING"])
        self.assertEqual(out["last_no_buy_diagnostics"]["trading_mode"],
                         "NORMAL_MODE")
        self.assertFalse(out["last_no_buy_diagnostics"]["degraded_mode_active"])
        self.assertEqual(out["last_no_buy_diagnostics"]["min_buy_confidence"], 40)
        self.assertEqual(out["last_no_buy_diagnostics"]["degraded_min_confidence"], 40)
        self.assertTrue(out["last_no_buy_diagnostics"]["degraded_use_standard_gates_for_testing"])
        self.assertFalse(out["last_no_buy_diagnostics"]["degraded_standard_gates_active"])
        self.assertEqual(out["last_no_buy_diagnostics"]["degraded_gate_policy"],
                         None)
        self.assertEqual(out["last_no_buy_diagnostics"]["effective_size_mult"], 1.0)
        self.assertEqual(out["last_no_buy_diagnostics"]["effective_min_buy_confidence"], 40)
        self.assertTrue(out["last_no_buy_diagnostics"]["normal_ev_gates_required"])
        self.assertTrue(out["last_no_buy_diagnostics"]["normal_risk_caps_required"])
        self.assertTrue(out["last_no_buy_diagnostics"]["fresh_quote_required"])
        self.assertEqual(out["last_no_buy_diagnostics"]["min_trade_size_effective"], 100)
        self.assertEqual(out["last_no_buy_diagnostics"]["volatility_value"], 14.2)
        self.assertEqual(out["last_no_buy_diagnostics"]["regime_source"],
                         "neutral_missing_spy")
        self.assertEqual(out["last_no_buy_diagnostics"]["tick_runtime_seconds"], 0.25)
        self.assertEqual(out["last_no_buy_diagnostics"]["history_fmp_attempted_count"], 3)
        self.assertEqual(out["last_no_buy_diagnostics"]["history_fmp_skipped_count"], 2)
        self.assertEqual(out["last_no_buy_diagnostics"]["history_fmp_rate_limited_count"], 1)
        self.assertEqual(out["last_no_buy_diagnostics"]["history_fmp_global_circuit_skipped_count"], 2)
        self.assertEqual(out["last_no_buy_diagnostics"]["fmp_daily_global_circuit_status"], "rate_limited")
        self.assertTrue(out["last_no_buy_diagnostics"]["fmp_daily_rate_limited"])
        self.assertEqual(out["last_no_buy_diagnostics"]["fmp_daily_cooldown_remaining_sec"], 120)
        self.assertEqual(out["last_no_buy_diagnostics"]["fmp_daily_last_429_age_sec"], 15)
        self.assertEqual(len(out["last_no_buy_diagnostics"]["ticker_signal_debug"]),
                         api.PA_TICKERS_PER_BOT_RUN)
        self.assertTrue(out["last_no_buy_diagnostics"]["partial_result"])
        self.assertEqual(out["last_no_buy_diagnostics"]["timeout_reason"], "max_runtime")
        self.assertEqual(out["last_no_buy_diagnostics"]["timeout_stage"], "fetch")
        self.assertEqual(out["last_no_buy_diagnostics"]["fetch_timeout_tickers"], ["ZZZ"])
        self.assertNotIn("bot_file_path", str(out))
        self.assertNotIn("storage_base_dir", str(out))

    def test_profit_calendar_aggregates_equity_and_trades_by_month(self):
        import app as _app  # noqa: F401
        import routes.api as api
        from datetime import datetime

        def ts(day, hour, minute=0):
            return int(datetime(2026, 6, day, hour, minute, tzinfo=api.BOT_CALENDAR_TZ).timestamp())

        payload = api.build_profit_calendar_payload({
            "equity_history": [
                [ts(3, 9, 30), 10000.00, 0.0, [["AAPL", 2, 101.0]], False],
                [ts(3, 15, 59), 10075.00, 0.75, [["AAPL", 2, 102.5], ["MSFT", 1, 300.0]], False],
                [ts(4, 9, 30), 10075.00, 0.75, [], False],
                [ts(4, 15, 59), 10050.00, 0.5, [], False],
            ],
            "history": [
                {"action": "BUY", "ticker": "AAPL", "shares": 2, "price": 101.0, "ts": ts(3, 10), "time_et": "Jun 03 10:00"},
                {"action": "SELL", "ticker": "MSFT", "shares": 1, "price": 300.0, "pnl_usd": 5.0, "ts": ts(3, 14), "time_et": "Jun 03 14:00"},
                {"action": "BUY", "ticker": "OLD", "shares": 1, "price": 1.0, "time": "06/01 09:30"},
            ],
            "holdings": {},
        }, 2026, 6)

        day3 = payload["days"][2]
        self.assertEqual(day3["date"], "2026-06-03")
        self.assertEqual(day3["status"], "profit")
        self.assertEqual(day3["pnl_usd"], 75.0)
        self.assertEqual(day3["pnl_pct"], 0.75)
        self.assertEqual(day3["holdings"], ["AAPL", "MSFT"])
        self.assertEqual(day3["trades_opened"], 1)
        self.assertEqual(day3["trades_closed"], 1)

        day4 = payload["days"][3]
        self.assertEqual(day4["status"], "loss")
        self.assertEqual(day4["pnl_usd"], -25.0)
        self.assertIn("Holdings snapshot unavailable", day4["note"])

        self.assertEqual(payload["summary"]["monthly_pnl_usd"], 50.0)
        self.assertEqual(payload["summary"]["win_days"], 1)
        self.assertEqual(payload["summary"]["loss_days"], 1)
        self.assertEqual(payload["summary"]["active_days"], 2)
        self.assertEqual(payload["data_quality"]["legacy_trade_rows_without_ts"], 1)

    def test_profit_calendar_empty_state_is_safe(self):
        import app as _app  # noqa: F401
        import routes.api as api

        payload = api.build_profit_calendar_payload({
            "equity_history": [],
            "history": [],
            "holdings": {},
        }, 2026, 2)

        self.assertEqual(len(payload["days"]), 28)
        self.assertIsNone(payload["summary"]["monthly_pnl_usd"])
        self.assertEqual(payload["summary"]["active_days"], 0)
        self.assertTrue(all(day["status"] == "neutral" for day in payload["days"]))
        self.assertTrue(all(day["note"] == "No trading data for this day." for day in payload["days"]))

    def test_profit_calendar_rejects_invalid_month(self):
        import app as _app  # noqa: F401
        import routes.api as api

        old_request = api.request
        old_jsonify = api.jsonify
        try:
            api.request = types.SimpleNamespace(args={"month": "2026-13"})
            api.jsonify = lambda obj=None, **kwargs: obj if obj is not None else kwargs
            body, status = api.api_bot_profit_calendar()
            self.assertEqual(status, 400)
            self.assertIn("Invalid month", body["error"])
            api.request = types.SimpleNamespace(args={"month": "0000-01"})
            body, status = api.api_bot_profit_calendar()
            self.assertEqual(status, 400)
        finally:
            api.request = old_request
            api.jsonify = old_jsonify

    def test_profit_calendar_route_is_read_only(self):
        source = Path("routes/api.py").read_text(encoding="utf-8")
        body = source[
            source.index("def _parse_profit_calendar_month"):
            source.index('@app.route("/api/bot/status")')
        ]
        for forbidden in ("save_bot", "run_bot", "trigger_bot_if_due", "warm_scan", "get_quote", "_finnhub_daily", "_fmp_daily"):
            self.assertNotIn(forbidden, body)

    def test_daily_bar_callers_use_data_manager_not_stooq_imports(self):
        files = [
            "market/history.py",
            "market/charts.py",
            "trading/indicators.py",
            "trading/portfolio_variance.py",
            "trading/risk.py",
            "trading/regime_v3.py",
            "trading/backtest.py",
            "routes/api.py",
        ]
        for file_name in files:
            source = Path(file_name).read_text(encoding="utf-8")
            self.assertNotIn("from market.quotes import _stooq_daily", source)
            self.assertNotIn("_stooq_daily(", source)

    def test_pa_daily_bars_use_shared_provider_chain_with_finnhub_first(self):
        import pandas as pd
        import market.quotes as quotes

        old_pa = quotes.PYTHONANYWHERE_MODE
        old_direct = quotes._direct_stooq_daily
        old_fh_daily = quotes._finnhub_daily
        old_fmp_daily = quotes._fmp_daily
        old_stale = quotes.cache_get_stale
        df = pd.DataFrame({"Close": [101.0]}, index=pd.to_datetime(["2026-06-26"]))
        df.attrs.update({"source": "finnhub_daily", "provider": "finnhub", "status": "ok"})
        try:
            quotes.PYTHONANYWHERE_MODE = True
            quotes._direct_stooq_daily = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Stooq must be skipped on PythonAnywhere")
            )
            quotes._finnhub_daily = lambda tk, full=False: df
            quotes._fmp_daily = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("FMP should not be called after valid Finnhub daily bars")
            )
            quotes.cache_get_stale = lambda *_args, **_kwargs: (quotes.CACHE_MISS, None)
            out = quotes._stooq_daily("AAA", full=True)
        finally:
            quotes.PYTHONANYWHERE_MODE = old_pa
            quotes._direct_stooq_daily = old_direct
            quotes._finnhub_daily = old_fh_daily
            quotes._fmp_daily = old_fmp_daily
            quotes.cache_get_stale = old_stale
        self.assertEqual(out.attrs["source"], "finnhub_daily")
        self.assertEqual(out.attrs["provider"], "finnhub")

    def test_health_trigger_throttles_duplicate_runs(self):
        import utils.threading_utils as tu

        old_async = tu._trigger_bot_async
        old_status = dict(tu._BOT_STATUS)
        old_enabled = tu.BOT_ENABLED
        calls = []
        try:
            tu.BOT_ENABLED = True
            tu._trigger_bot_async = lambda **kwargs: calls.append(kwargs) or True
            tu._BOT_STATUS["last_run_ts"] = time.time()
            self.assertFalse(tu.trigger_bot_if_due(min_interval=60))
            self.assertEqual(calls, [])
            self.assertFalse(tu.trigger_bot_if_due(min_interval=1))
            self.assertEqual(calls, [])
            self.assertTrue(tu.trigger_bot_if_due(force=True, min_interval=999))
            self.assertEqual(len(calls), 1)
            tu._BOT_STATUS["last_run_ts"] = time.time() - tu.BOT_INTERVAL - 1
            self.assertTrue(tu.trigger_bot_if_due(min_interval=1))
            self.assertEqual(len(calls), 2)
        finally:
            tu.BOT_ENABLED = old_enabled
            tu._trigger_bot_async = old_async
            tu._BOT_STATUS.clear()
            tu._BOT_STATUS.update(old_status)

    def test_pa_staging_prioritizes_holdings_and_rotates_watchlist(self):
        import trading.bot as bot

        old_pa = bot.PYTHONANYWHERE_MODE
        old_batch = bot.PA_TICKERS_PER_BOT_RUN
        try:
            bot.PYTHONANYWHERE_MODE = True
            bot.PA_TICKERS_PER_BOT_RUN = 3
            state = {}
            first = bot._pa_stage_tickers(["A", "B", "C", "D"], ["H1", "H2"], state)
            second = bot._pa_stage_tickers(["A", "B", "C", "D"], ["H1", "H2"], state)
            self.assertEqual(first, ["H1", "H2", "A"])
            self.assertEqual(second, ["H1", "H2", "B"])
            state = {}
            self.assertEqual(bot._pa_stage_tickers(["SPY", "A", "QQQ"], [], state), ["A"])
            self.assertEqual(bot._pa_stage_tickers(["SPY", "A"], ["SPY"], state), ["SPY", "A"])
        finally:
            bot.PYTHONANYWHERE_MODE = old_pa
            bot.PA_TICKERS_PER_BOT_RUN = old_batch

    def test_active_config_pa_overrides_without_mutating_default(self):
        import utils.deploy_config as dc

        old_pa = dc.PYTHONANYWHERE_MODE
        base_hash = config_hash(DEFAULT_CONFIG)
        try:
            dc.PYTHONANYWHERE_MODE = True
            cfg = active_config()
            self.assertEqual(cfg["suggestion"]["min_adv_usd"], 15_000_000)
            self.assertEqual(cfg["suggestion"]["min_net_edge_pct"], 0.50)
            self.assertEqual(cfg["correlation"]["lookback_days"], 20)
            self.assertEqual(config_hash(DEFAULT_CONFIG), base_hash)
        finally:
            dc.PYTHONANYWHERE_MODE = old_pa

    def test_pythonanywhere_runtime_defaults_are_free_tier_safe(self):
        import importlib
        import os
        import utils.deploy_config as deploy_config

        names = ("PA_TICKERS_PER_BOT_RUN", "PA_SCAN_BATCH_SIZE")
        old_env = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ.pop(name, None)
            deploy_config = importlib.reload(deploy_config)
            self.assertEqual(deploy_config.PA_TICKERS_PER_BOT_RUN, 6)
            self.assertEqual(deploy_config.PA_SCAN_BATCH_SIZE, 4)

            os.environ["PA_TICKERS_PER_BOT_RUN"] = "99"
            os.environ["PA_SCAN_BATCH_SIZE"] = "99"
            deploy_config = importlib.reload(deploy_config)
            self.assertEqual(deploy_config.PA_TICKERS_PER_BOT_RUN, 8)
            self.assertEqual(deploy_config.PA_SCAN_BATCH_SIZE, 8)
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            importlib.reload(deploy_config)

    def test_pa_state_caps_trim_oversized_state(self):
        import utils.deploy_config as dc
        import utils.storage as storage

        old_pa = dc.PYTHONANYWHERE_MODE
        try:
            dc.PYTHONANYWHERE_MODE = True
            state = {
                "equity_history": [[i, 1, 0, [], False] for i in range(700)],
                "attribution_events": [{"i": i} for i in range(700)],
                "exit_attribution_events": [{"i": i} for i in range(700)],
                "candidate_observations": [{"i": i} for i in range(700)],
            }
            storage._cap_bot_state(state)
            self.assertLessEqual(len(state["equity_history"]), 400)
            self.assertLessEqual(len(state["attribution_events"]), 500)
            self.assertLessEqual(len(state["exit_attribution_events"]), 500)
            self.assertLessEqual(len(state["candidate_observations"]), 300)
        finally:
            dc.PYTHONANYWHERE_MODE = old_pa

    def test_equity_history_cap_constant_applied(self):
        import trading.bot as bot

        old_market_open = bot.is_market_open
        try:
            bot.is_market_open = lambda: True
            state = {
                "cash": 1000.0,
                "starting": 1000.0,
                "holdings": {},
                "equity_history": [[i * 61, 1000.0, 0.0, [], False]
                                   for i in range(bot.EQUITY_HISTORY_MAX + 25)],
            }
            bot.record_equity_snapshot(state, 1001.0)
        finally:
            bot.is_market_open = old_market_open
        self.assertLessEqual(len(state["equity_history"]), bot.EQUITY_HISTORY_MAX)

    def test_suggestion_feedback_route_requires_admin(self):
        source = Path("routes/portfolio.py").read_text(encoding="utf-8")
        start = source.index('def bot_suggestion_feedback():')
        end = source.index('@app.route("/health")')
        body = source[start:end]
        self.assertIn("require_admin_token()", body)
        self.assertIn('@app.route("/bot/suggestion-feedback", methods=["POST"])', source)

    def test_bot_tick_route_requires_admin_and_returns_json(self):
        source = Path("routes/portfolio.py").read_text(encoding="utf-8")
        start = source.index('def bot_tick():')
        end = source.index('@app.route("/bot/suggestion-feedback"')
        body = source[start:end]
        self.assertIn('@app.route("/bot/tick", methods=["GET", "POST"])', source)
        self.assertIn("require_machine_token()", body)
        self.assertIn("max_runtime_sec=BOT_TICK_MAX_RUNTIME_SEC", body)
        self.assertIn("max_runtime_seconds", body)

    def test_price_history_atomic_updates_preserve_tickers(self):
        import os
        import tempfile
        from concurrent.futures import ThreadPoolExecutor
        import utils.storage as storage

        old_file = storage.PRICE_HIST_FILE
        try:
            with tempfile.TemporaryDirectory() as td:
                storage.PRICE_HIST_FILE = os.path.join(td, "price_history.json")
                tickers = [f"T{i}" for i in range(12)]
                with ThreadPoolExecutor(max_workers=6) as ex:
                    list(ex.map(
                        lambda t: storage.append_price_snapshot(
                            t, 100.0, min_interval=0, limit=10
                        ),
                        tickers,
                    ))
                self.assertEqual(set(storage.load_price_hist().keys()), set(tickers))
        finally:
            storage.PRICE_HIST_FILE = old_file

    def test_cache_prune_and_api_circuit_breaker(self):
        import os
        import tempfile
        import time as time_mod
        import utils.cache as cache

        old_dir = cache._CACHE_DIR
        old_persistent = cache.PERSISTENT_CACHE
        old_failures = dict(cache._api_failures)
        old_provider_health_file = cache._PROVIDER_HEALTH_FILE
        try:
            with tempfile.TemporaryDirectory() as td:
                cache._CACHE_DIR = td
                cache._PROVIDER_HEALTH_FILE = os.path.join(td, "provider_health.json")
                cache._api_failures.clear()
                cache.PERSISTENT_CACHE = True
                old_path = os.path.join(td, "old.pkl")
                new_path = os.path.join(td, "new.pkl")
                Path(old_path).write_bytes(b"x")
                Path(new_path).write_bytes(b"x")
                old_ts = time_mod.time() - 10 * 86400
                os.utime(old_path, (old_ts, old_ts))
                out = cache.prune_cache_dir(max_files=1, max_age_sec=7 * 86400)
                self.assertGreaterEqual(out["removed"], 1)
                cache.record_api_failure("endpoint:test")
                self.assertFalse(cache.should_skip_api("endpoint:test", cooldown_sec=300))
                cache.record_api_failure("endpoint:test", "429 too many requests")
                cache.record_api_failure("endpoint:test")
                self.assertTrue(cache.should_skip_api("endpoint:test", cooldown_sec=300))
                snap = cache.api_failure_snapshot()
                self.assertIn("endpoint:test", snap)
                self.assertTrue(snap["endpoint:test"]["rate_limited"])
                self.assertTrue(snap["endpoint:test"]["rate_limit_recent"])
                self.assertEqual(snap["endpoint:test"]["status"], "provider_error")
                cache.record_api_success("endpoint:test")
                self.assertFalse(cache.should_skip_api("endpoint:test", cooldown_sec=300))

                cache.cache_set("bt_sweep_payload", {"x": 1})
                self.assertFalse(any("bt_sweep_payload" in name for name in os.listdir(td)))
                cache.cache_set("plain_payload", {"x": 1})
                self.assertTrue(any("plain_payload" in name for name in os.listdir(td)))
        finally:
            cache._CACHE_DIR = old_dir
            cache.PERSISTENT_CACHE = old_persistent
            cache._PROVIDER_HEALTH_FILE = old_provider_health_file
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)

    def test_finnhub_enrichment_fetchers_respect_circuit_breaker(self):
        import types
        import trading.risk as risk
        import market.sentiment as sentiment

        class _FailRiskFinnhub:
            def earnings_calendar(self, *_args, **_kwargs):
                raise AssertionError("earnings fetch should have been skipped")

            def recommendation_trends(self, *_args, **_kwargs):
                raise AssertionError("analyst fetch should have been skipped")

            def stock_insider_sentiment(self, *_args, **_kwargs):
                raise AssertionError("insider fetch should have been skipped")

        class _FailNewsFinnhub:
            def company_news(self, *_args, **_kwargs):
                raise AssertionError("news fetch should have been skipped")

        old_risk = {
            "cache_get": risk.cache_get,
            "cache_set": risk.cache_set,
            "should_skip_api": risk.should_skip_api,
            "fh": risk.fh,
        }
        old_sentiment = {
            "cache_get": sentiment.cache_get,
            "cache_set": sentiment.cache_set,
            "should_skip_api": sentiment.should_skip_api,
            "fh": sentiment.fh,
        }
        try:
            risk.cache_get = lambda *_args, **_kwargs: None
            risk.cache_set = lambda *_args, **_kwargs: None
            risk.should_skip_api = lambda endpoint, **_kwargs: True
            risk.fh = _FailRiskFinnhub()
            self.assertEqual(risk.get_earnings_soon("AAA"), {"soon": False, "date": None})
            self.assertEqual(risk.get_analyst_rec("AAA")["total"], 0)
            self.assertEqual(risk.get_insider_sentiment("AAA")["samples"], 0)

            sentiment.cache_get = lambda *_args, **_kwargs: None
            sentiment.cache_set = lambda *_args, **_kwargs: None
            sentiment.should_skip_api = lambda endpoint, **_kwargs: True
            sentiment.fh = _FailNewsFinnhub()
            fake_yf = types.SimpleNamespace(
                Ticker=lambda *_args, **_kwargs: types.SimpleNamespace(news=[])
            )
            with patch.dict(sys.modules, {"yfinance": fake_yf}):
                self.assertEqual(sentiment.get_news("AAA"), ([], 0.0))
        finally:
            for name, value in old_risk.items():
                setattr(risk, name, value)
            for name, value in old_sentiment.items():
                setattr(sentiment, name, value)

    def test_data_manager_reuses_memory_cache(self):
        import pandas as pd
        from market.data_manager import DataManager

        calls = []

        def source(ticker, full=False):
            calls.append((ticker, full))
            return pd.DataFrame({"Close": [1.0, 2.0]})

        mgr = DataManager(max_items=2, disk_ttl_sec=3600)
        first = mgr.get_daily("AAA", source_func=source)
        second = mgr.get_daily("AAA", source_func=source)
        self.assertEqual(len(calls), 1)
        self.assertEqual(list(first["Close"]), list(second["Close"]))

    def test_data_manager_does_not_cache_stale_cache_frames(self):
        import pandas as pd
        import market.data_manager as dm
        from market.data_manager import DataManager

        stale = pd.DataFrame({"Close": [1.0, 2.0]})
        stale.attrs.update({"source": "stale_cache:fmp_daily", "status": "stale_cache"})
        fresh = pd.DataFrame({"Close": [3.0, 4.0]})
        fresh.attrs.update({"source": "fmp_daily", "status": "ok"})

        old = {
            "cache_get": dm.cache_get,
            "cache_set": dm.cache_set,
        }
        saved = []
        calls = []

        def source(ticker, full=False):
            calls.append((ticker, full))
            return stale if len(calls) == 1 else fresh

        try:
            dm.cache_get = lambda *_args, **_kwargs: None
            dm.cache_set = lambda key, value: saved.append((key, value))
            mgr = DataManager(max_items=2, disk_ttl_sec=3600)
            first = mgr.get_daily("AAA", source_func=source)
            second = mgr.get_daily("AAA", source_func=source)
        finally:
            dm.cache_get = old["cache_get"]
            dm.cache_set = old["cache_set"]
        self.assertEqual(first.attrs["status"], "stale_cache")
        self.assertEqual(second.attrs["source"], "fmp_daily")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(saved), 1)
        self.assertIs(saved[0][1], fresh)


class TradingV1BacktestSmokeTests(unittest.TestCase):
    def test_backtest_execution_price_helpers(self):
        import pandas as pd
        import trading.backtest as bt

        dates = pd.date_range("2025-01-01", periods=2, freq="B")
        df = pd.DataFrame({
            "Open": [100.0, 105.0],
            "High": [102.0, 108.0],
            "Low": [99.0, 101.0],
            "Close": [101.0, 106.0],
            "Volume": [1_000_000, 1_000_000],
        }, index=dates)
        self.assertEqual(bt._entry_price_for_next_bar(df, list(dates), 0), 105.0)
        self.assertIsNone(bt._entry_price_for_next_bar(df, list(dates), 1))
        self.assertEqual(bt._exit_price_for_bar(bt._bar_at(df, dates[1]), stop_price=102.0), 102.0)
        self.assertEqual(bt._exit_price_for_bar(bt._bar_at(df, dates[1]), take_profit_price=107.0), 107.0)
        self.assertEqual(bt._exit_price_for_bar(bt._bar_at(df, dates[1])), 106.0)

    def test_synthetic_backtest_smoke_no_network(self):
        import pandas as pd
        from trading.backtest import _simulate

        dates = pd.date_range("2025-01-01", periods=80, freq="B")

        def frame(start, step):
            closes = [start + i * step for i in range(len(dates))]
            return pd.DataFrame({
                "Open": closes,
                "High": [c * 1.01 for c in closes],
                "Low": [c * 0.99 for c in closes],
                "Close": closes,
                "Volume": [2_000_000 for _ in closes],
            }, index=dates)

        panel = {"AAA": frame(50, 0.45), "BBB": frame(40, 0.10)}
        spy = frame(400, 0.20)
        run = _simulate(panel, spy, list(dates), {}, "fixed", learn=False,
                        window=40, edge_stats={})
        self.assertIn("final_equity", run)
        self.assertIn("attribution_buckets", run["edge_stats"])
        self.assertGreater(len(run["equity_curve"]), 0)

    def test_backtest_metrics_include_concentration_and_hold_days(self):
        import pandas as pd
        import trading.backtest as bt

        dates = list(pd.date_range("2025-01-01", periods=3, freq="B"))
        run = {
            "equity_curve": [(dates[0], 10_000), (dates[-1], 10_500)],
            "final_equity": 10_500,
            "commission_per_trade": 0.99,
            "trades": [
                {"ticker": "AAPL", "exit_reason": "trail", "pnl_pct": 5.0, "days_held": 2},
                {"ticker": "MSFT", "exit_reason": "trail", "pnl_pct": 1.0, "days_held": 1},
            ],
        }
        out = bt._metrics(run, dates)
        self.assertEqual(out["trade_count"], 2)
        self.assertEqual(out["avg_hold_days"], 1.5)
        self.assertGreater(out["top_ticker_profit_contribution_pct"], 50)
        self.assertTrue(out["profit_concentration_rejected"])


class TradingBacktestParityV2Tests(unittest.TestCase):
    def _panel(self, periods=75):
        import pandas as pd

        dates = pd.date_range("2025-01-01", periods=periods, freq="B")

        def frame(start, step):
            closes = [start + i * step for i in range(len(dates))]
            return pd.DataFrame({
                "Open": closes,
                "High": [c * 1.01 for c in closes],
                "Low": [c * 0.99 for c in closes],
                "Close": closes,
                "Volume": [3_000_000 for _ in closes],
            }, index=dates)

        return {
            "AAPL": frame(100, 0.30),
            "MSFT": frame(200, 0.20),
        }, frame(400, 0.10), frame(350, 0.12), list(dates)

    def test_rolling_window_generator(self):
        import pandas as pd
        from trading.backtest import _rolling_windows

        dates = list(pd.date_range("2025-01-01", periods=100, freq="B"))
        windows = _rolling_windows(dates, train_days=20, test_days=10, step_days=5)
        self.assertEqual(len(windows), 15)
        self.assertEqual(windows[0]["train_dates"][0], dates[0])
        self.assertEqual(windows[0]["test_dates"][0], dates[20])
        self.assertEqual(windows[-1]["test_dates"][-1], dates[99])

    def test_null_provider_and_mode_flags(self):
        from trading.backtest import NullHistoricalSignalProvider, _mode_flags

        provider = NullHistoricalSignalProvider()
        self.assertEqual(provider.name, "NullHistoricalSignalProvider")
        self.assertEqual(provider.news("AAPL", "2025-01-01"), ([], 0.0))
        self.assertFalse(_mode_flags("technical_only")["regime"])
        self.assertTrue(_mode_flags("technical_regime")["regime"])
        self.assertTrue(_mode_flags("technical_regime_news")["news"])
        self.assertTrue(_mode_flags("full_current")["earnings"])

    def test_require_external_history_fails_for_null_provider(self):
        from trading.backtest import _run_rolling_backtest

        panel, spy, qqq, dates = self._panel()
        out = _run_rolling_backtest(
            panel, spy, dates, qqq_df=qqq, modes=("full_current",),
            train_days=40, test_days=10, step_days=25,
            require_external_history=True,
        )
        self.assertEqual(out["error"], "external_history_required")
        self.assertEqual(out["provider_name"], "NullHistoricalSignalProvider")

    def test_candidate_log_sampling(self):
        from trading.backtest import _append_candidate_log

        logs = []
        for i in range(3):
            _append_candidate_log(logs, {"i": i}, log_sample_size=2,
                                  include_full_logs=False)
        self.assertEqual(len(logs), 2)
        logs = []
        for i in range(3):
            _append_candidate_log(logs, {"i": i}, log_sample_size=2,
                                  include_full_logs=True)
        self.assertEqual(len(logs), 3)

    def test_rolling_backtest_all_modes_no_network(self):
        import trading.backtest as bt

        panel, spy, qqq, dates = self._panel()
        old_get_sector = bt.get_sector
        bt.get_sector = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network sector fallback called")
        )
        try:
            out = bt._run_rolling_backtest(
                panel, spy, dates, qqq_df=qqq, train_days=40,
                test_days=10, step_days=25, log_sample_size=5,
            )
        finally:
            bt.get_sector = old_get_sector
        self.assertEqual(out["mode"], "rolling")
        self.assertIn("technical_only", out["summary_by_mode"])
        self.assertIn("full_current", out["coverage_by_mode"])
        self.assertIn("SPY", out["benchmarks"])
        self.assertIn("QQQ", out["benchmarks"])
        self.assertIn("non_price_historical_provider_empty", out["caveats"])
        self.assertFalse(out["data_source_audit"]["used_live_api"])
        self.assertLessEqual(len(out["candidate_logs_sample"]), 5)
        if out["candidate_logs_sample"]:
            self.assertIn("portfolio_variance", out["candidate_logs_sample"][0])
            self.assertIn("config_hash", out["candidate_logs_sample"][0])
            self.assertIn("regime_v3", out["candidate_logs_sample"][0])
            self.assertIn("catalyst_type", out["candidate_logs_sample"][0])
        self.assertIn("config", out)
        self.assertIn("hash", out["config"])

    def test_sensitivity_sweep_is_advisory_and_stresses_friction(self):
        import trading.backtest as bt

        panel, spy, qqq, dates = self._panel(periods=80)
        cfg = merge_config({"backtest.min_sweep_trades": 0})
        cfg["sweep_params"] = {"signal.min_buy_confidence": [40, 60]}
        cfg["preset_sweeps"] = {"current": {}}
        out = bt._run_sensitivity_sweep(
            panel, spy, dates, qqq_df=qqq, train_days=40,
            test_days=10, step_days=30, window=40, base_config=cfg,
            sweep="one_at_a_time",
        )
        self.assertTrue(out["advisory_only"])
        self.assertFalse(out["live_config_mutated"])
        self.assertEqual(out["mode"], "full_current")
        self.assertGreaterEqual(len(out["results"]), 1)
        self.assertIn("friction_2x_survives", out["results"][0])
        self.assertIn("recommendations", out)

    def test_sweep_rejects_top_ticker_profit_concentration_unless_accepted(self):
        import trading.backtest as bt

        rolling = {
            "summary_by_mode": {
                "full_current": {
                    "windows": 2,
                    "avg_total_return_pct": 4.0,
                    "avg_max_drawdown_pct": 3.0,
                    "avg_sharpe": 1.1,
                    "avg_profit_factor": 1.4,
                    "total_trades": 40,
                    "net_win_rate_pct": 55.0,
                    "max_top_ticker_profit_contribution_pct": 51.0,
                }
            },
            "windows": [
                {"results": {"full_current": {"total_return_pct": 3.0}}},
                {"results": {"full_current": {"total_return_pct": 5.0}}},
            ],
        }
        stressed = {
            "summary_by_mode": {
                "full_current": {"avg_total_return_pct": 1.0}
            }
        }
        cfg = merge_config({"backtest": {"min_sweep_trades": 0}})
        row = bt._sweep_row("x", "preset", {}, cfg, rolling, stressed)
        self.assertIn("profit_too_concentrated", row["rejections"])
        self.assertFalse(row["recommended"])

        accepted_cfg = merge_config({
            "backtest": {
                "min_sweep_trades": 0,
                "accept_profit_concentration": True,
            }
        })
        accepted = bt._sweep_row("x", "preset", {}, accepted_cfg,
                                 rolling, stressed)
        self.assertNotIn("profit_too_concentrated", accepted["rejections"])

    def test_rolling_train_edge_cache_reuses_repeated_train_window(self):
        import pandas as pd
        import trading.backtest as bt

        dates = list(pd.date_range("2025-01-01", periods=8, freq="B"))
        df = pd.DataFrame({"Close": [100 + i for i in range(len(dates))]},
                          index=dates)
        windows = [
            {
                "index": 0,
                "train_period": ("2025-01-01", "2025-01-03"),
                "test_period": ("2025-01-06", "2025-01-07"),
                "train_dates": dates[:3],
                "test_dates": dates[3:5],
            },
            {
                "index": 1,
                "train_period": ("2025-01-01", "2025-01-03"),
                "test_period": ("2025-01-08", "2025-01-09"),
                "train_dates": dates[:3],
                "test_dates": dates[5:7],
            },
        ]
        calls = []
        old_windows = bt._rolling_windows
        old_learn = bt._learn_forward_edges
        old_sim = bt._simulate
        try:
            bt._rolling_windows = lambda *_args, **_kwargs: windows
            bt._learn_forward_edges = (
                lambda *_args, **_kwargs: calls.append(1) or {}
            )
            bt._simulate = lambda _panel, _spy, run_dates, *_args, **_kwargs: {
                "equity_curve": [(d, 10_000.0) for d in run_dates],
                "trades": [],
                "final_equity": 10_000.0,
                "coverage": {},
            }
            bt._run_rolling_backtest(
                {"AAA": df}, df, dates, modes=("technical_only", "technical_only"),
                train_days=3, test_days=2, step_days=2,
            )
        finally:
            bt._rolling_windows = old_windows
            bt._learn_forward_edges = old_learn
            bt._simulate = old_sim
        self.assertEqual(len(calls), 1)


class TradingV1TemplateTests(unittest.TestCase):
    def test_bot_tooltip_title_uses_point_time(self):
        template = Path("templates/bot.html").read_text(encoding="utf-8")
        self.assertIn("title: (items)", template)
        self.assertIn("labels[items[0].dataIndex]", template)
        self.assertIn("label: (c) => '$' + c.parsed.y", template)

    def test_bot_template_contains_profit_calendar_controls(self):
        template = Path("templates/bot.html").read_text(encoding="utf-8")
        self.assertIn('id="profitCalendarOpen"', template)
        self.assertIn('id="profitCalendarBackdrop"', template)
        self.assertIn('/api/bot/profit-calendar', template)
        self.assertIn("pc-dot profit", template)
        self.assertIn("No trading data for this day.", template)

    def test_market_dashboard_shows_regime_data_fallback(self):
        template = Path("templates/index.html").read_text(encoding="utf-8")
        self.assertIn("regime.spy_data_ok", template)
        self.assertIn("fallback neutral", template)
        self.assertIn("1M / 22 trading days", template)
        self.assertNotIn("% / 30d", template)

    def test_stock_detail_binds_sentiment_and_context(self):
        source = Path("routes/dashboard.py").read_text(encoding="utf-8")
        start = source.index("def stock_detail(ticker):")
        end = source.index('@app.route("/top")')
        body = source[start:end]
        self.assertIn('sent = snap["sentiment"]', body)
        self.assertIn('ctx = snap["ctx"]', body)
        self.assertIn("sentiment=sent", body)
        self.assertIn("ctx=ctx", body)

    def test_public_pages_do_not_trigger_bot_work(self):
        bot_source = Path("trading/bot.py").read_text(encoding="utf-8")
        render_body = bot_source[
            bot_source.index("def _render_bot_page(read_only):"):
            bot_source.index("# ── Market-wide scan")
        ]
        self.assertNotIn("trigger_bot_if_due", render_body)
        self.assertNotIn("start_scheduler_once", render_body)

        dashboard_source = Path("routes/dashboard.py").read_text(encoding="utf-8")
        index_body = dashboard_source[
            dashboard_source.index("def index():"):
            dashboard_source.index('@app.route("/stock/<ticker>")')
        ]
        self.assertNotIn("trigger_bot_if_due", index_body)
        self.assertNotIn("start_scheduler_once", index_body)

        portfolio_source = Path("routes/portfolio.py").read_text(encoding="utf-8")
        simulator_body = portfolio_source[
            portfolio_source.index("def simulator():"):
            portfolio_source.index('@app.route("/simulator/buy"')
        ]
        health_body = portfolio_source[
            portfolio_source.index("def health():"):
        ]
        self.assertNotIn("trigger_bot_if_due", simulator_body)
        self.assertNotIn("trigger_bot_if_due", health_body)
        self.assertIn('return "ok"', health_body)


class RatchetPeakOnPyramidTests(unittest.TestCase):
    """B5: pyramid adds carry the ratchet peak, expressed against the new blended cost."""

    def test_carried_peak_pnl_pct(self):
        import trading.bot as bot
        # New position: peak ≈ price ≈ avg_cost → ~0 (matches the old reset behavior).
        self.assertEqual(bot._carried_peak_pnl_pct(None, 100.0, 100.0), 0.0)
        # Pyramid add: carry the old peak PRICE but express it vs the NEW blended cost.
        self.assertAlmostEqual(
            bot._carried_peak_pnl_pct(120.0, 100.0, 110.0),
            round((120.0 - 110.0) / 110.0 * 100, 3),
        )
        # Basis adjustment must be strictly below the stale old-basis value — this is what
        # prevents a too-high lock from tripping an immediate ratchet exit after adding.
        self.assertLess(
            bot._carried_peak_pnl_pct(120.0, 100.0, 110.0),
            (120.0 - 100.0) / 100.0 * 100,
        )
        # Zero-guard: no divide-by-zero on a bad avg cost.
        self.assertEqual(bot._carried_peak_pnl_pct(120.0, 100.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
