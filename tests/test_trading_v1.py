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
from trading.sizing import entry_cluster, evaluate_candidate, rank_candidates  # noqa: E402


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
        modes = DEFAULT_CONFIG["market_data_modes"]
        self.assertEqual(risk["max_new_buys_per_tick"], 1)
        self.assertEqual(risk["max_new_buys_per_day"], 2)
        self.assertEqual(risk["max_positions"], 8)
        self.assertEqual(risk["max_position_pct"], 0.08)
        self.assertEqual(risk["max_gross_exposure_pct"], 0.70)
        self.assertEqual(risk["min_cash_reserve_pct"], 0.30)
        self.assertEqual(risk["daily_loss_limit_pct"], -0.02)
        self.assertEqual(risk["hard_drawdown_lockout_pct"], -0.10)
        self.assertFalse(kelly["enabled"])
        self.assertEqual(kelly["min_samples"], 100)
        self.assertEqual(kelly["max_mult"], 1.0)
        self.assertEqual(modes["proxy_size_mult"], 0.85)
        self.assertEqual(modes["degraded_size_mult"], 0.70)
        self.assertEqual(modes["degraded_min_confidence"], 65)

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
        lo["ctx"]["vol_ratio"] = 1.0
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
        self.assertEqual(classify_display_signal("buy", 54), "WATCH_OR_LEAN")
        self.assertEqual(classify_display_signal("buy", 55), "BUY_CANDIDATE")

    def test_confidence_prior_watchlist_warmup_requires_min_confidence(self):
        cand = self._candidate("WARM", 35, "trend", atr_pct=2.0, score=4.0)
        cand["rec"]["cls"] = "strong-buy"
        cand["rec"]["signal"] = "STRONG BUY"
        cand["rec"]["sizing_confidence"] = 55
        out = evaluate_candidate(
            cand,
            total_equity=10_000,
            regime_stop_pct=-6,
            regime_kind="bull",
            vix_mult=1,
            streak_mult=1,
            kelly_mult=1,
            edge_stats={},
            min_position_usd=400,
        )
        self.assertFalse(out["tradable"])
        self.assertIn("EV gate", out["rank_reason"])
        self.assertGreaterEqual(out["risk"]["target_notional"], 400)
        self.assertEqual(out["sizing_confidence"], 55)

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
        self.assertEqual(DEFAULT_CONFIG["signal"]["min_buy_confidence"], 55)
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

    def test_negative_votes_fall_back_to_global_bucket(self):
        state = {}
        cand = _entry_candidate("NEG", cluster="mixed")
        cand["rec"]["categories"] = {"overbought": -2.0}
        event = record_entry_event(state, cand, "skipped", "warning", ts=123,
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
            record_entry_event(state, cand, "skipped", "test", ts=now + i,
                               regime="neutral")
        self.assertEqual(len(state["attribution_events"]), 150)

        for i in range(10):
            cand = _entry_candidate(f"B{i:03d}", cluster="trend")
            cand["rec"]["categories"] = {"dip": 2.0}
            record_entry_event(state, cand, "skipped", "test", ts=now + 500 + i,
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
                record_entry_event(state, cand, "skipped", "test", ts=now + i,
                                   regime="neutral")
            self.assertEqual(len(state["attribution_events"]), 100)
        finally:
            dc.PYTHONANYWHERE_MODE = old_pa


class TradingV2IntegrationTests(unittest.TestCase):
    def test_forward_update_fills_due_horizons_and_bucket(self):
        state = {}
        now = int(time.time())
        cand = _entry_candidate("AAA", cluster="trend", friction_pct=0.1)
        cand["benchmark_prices"] = {"SPY": 100.0, "QQQ": 200.0}
        event = record_entry_event(
            state,
            cand,
            "skipped",
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
            "skipped",
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
            self.assertGreaterEqual(rec["sizing_confidence"], 55)
        else:
            self.assertEqual(rec["sizing_confidence"], rec["confidence"])


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
                        "dist_from_high_pct": 0, "vol_ratio": 1.5},
            }
            self.assertEqual(bot.buyable_reason("X", {"price": 0})[1], "INVALID_PRICE")
            self.assertEqual(bot.buyable_reason("X", {"price": 1, "stale": True})[1], "STALE_CANDIDATE_QUOTE")
            self.assertEqual(
                bot.buyable_reason("X", base, {"X": {"reason": "loss"}}, "bull", {})[1],
                "RECENT_SELL_COOLDOWN:loss",
            )
            self.assertEqual(bot.buyable_reason("NOSEC", base, {}, "bull", {})[1], "MISSING_SECTOR")
            bear = dict(base)
            bear["ctx"] = dict(base["ctx"], rsi=45, is_dip=False)
            self.assertEqual(bot.buyable_reason("X", bear, {}, "bear", {})[1],
                             "BEAR_GATE_REQUIRES_DIP_OR_RSI_LT_35")
            neutral = dict(base)
            neutral["ctx"] = dict(base["ctx"], adx=10, is_dip=False)
            self.assertEqual(bot.buyable_reason("X", neutral, {}, "neutral", {})[1],
                             "NEUTRAL_ADX_BELOW_20_NON_DIP")
            knife = dict(base)
            knife["rec"] = {"cls": "buy", "confidence": 70,
                            "catalyst": {"type": "guidance_cut"}}
            knife["ctx"] = dict(base["ctx"], week_chg_pct=-4)
            self.assertTrue(bot.buyable_reason("X", knife, {}, "bull", {})[1].startswith("NEGATIVE_CATALYST_FALLING"))
            low_vol = dict(base)
            low_vol["ctx"] = dict(base["ctx"], is_dip=False, vol_ratio=1.0)
            self.assertEqual(bot.buyable_reason("X", low_vol, {}, "bull", {})[1], "buyable")
            very_low_vol = dict(base)
            very_low_vol["ctx"] = dict(base["ctx"], is_dip=False, vol_ratio=0.5)
            self.assertEqual(
                bot.buyable_reason("X", very_low_vol, {}, "bull", {})[1],
                "VERY_LOW_VOLUME_CONFIRMATION",
            )
        finally:
            bot.get_sector = old_sector

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
        self.assertEqual(diag["main_blocker"], "raw_buys_rejected_pre_candidate")

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
        self.assertEqual(snap["finnhub"]["status"], "degraded")
        self.assertTrue(snap["finnhub"]["rate_limited"])
        self.assertTrue(snap["finnhub"]["rate_limit_recent"])

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
            "run_bot": portfolio.run_bot,
            "jsonify": portfolio.jsonify,
        }
        try:
            portfolio.require_machine_token = lambda: calls.setdefault("auth", True)
            portfolio.jsonify = lambda obj=None, **kwargs: obj if obj is not None else kwargs

            def fake_run_bot(**kwargs):
                calls["run_kwargs"] = kwargs
                return {"last_no_buy_diagnostics": {"main_blocker": "partial_timeout"}}, False, "partial_timeout"

            portfolio.run_bot = fake_run_bot
            payload = portfolio.bot_tick()
        finally:
            for name, value in old.items():
                setattr(portfolio, name, value)
        self.assertTrue(calls["auth"])
        self.assertEqual(calls["run_kwargs"]["max_runtime_sec"],
                         portfolio.BOT_TICK_MAX_RUNTIME_SEC)
        self.assertEqual(payload["status"], "partial_timeout")

    def test_get_vix_reports_spy_proxy_source(self):
        import pandas as pd
        import trading.risk as risk

        idx = pd.date_range("2026-01-01", periods=80, freq="B")
        df = pd.DataFrame({"Close": [100 + i * 0.4 for i in range(80)]}, index=idx)
        old_cache_get = risk.cache_get
        old_cache_set = risk.cache_set
        old_daily = risk._regime_daily_bars
        old_append = risk._append_live_bar
        try:
            risk.cache_get = lambda *_args, **_kwargs: None
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
        self.assertEqual(out["source"], "spy_realized_vol_proxy")
        self.assertEqual(out["vix_display"], "proxy")

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
        self.assertEqual(out_state["last_no_buy_diagnostics"]["main_blocker"],
                         "interval_cooldown")
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
                "vix_display": "proxy",
            }
            bot.get_news = lambda _tk: ([], 0.0)
            bot.get_history = lambda _tk: {
                "adx": 30, "rsi": 55, "week_chg_pct": 0,
                "dist_from_high_pct": 0, "avg_dollar_vol_20d": 100_000_000,
                "vol_ratio": 1.5,
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
        self.assertEqual(captured["vix_mult"], 1.0)
        self.assertEqual(captured["mode_size_mult"], 0.70)
        self.assertEqual(captured["mode_size_reason"], "DEGRADED_MODE")
        diag = out_state["last_no_buy_diagnostics"]
        self.assertTrue(diag["regime_allow_buys"])
        self.assertEqual(diag["trading_mode"], "DEGRADED_MODE")
        self.assertTrue(diag["degraded_mode_active"])
        self.assertIn("SPY_DATA_MISSING", diag["data_health_blocks"])
        self.assertTrue(diag["regime_data_fallback"])
        self.assertEqual(diag["regime_data_size_mult"], 0.70)

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
        try:
            api.load_bot = lambda: {
                "last_no_buy_diagnostics": {
                    "main_blocker": "no_buy_candidates",
                    "candidate_pool_count": 0,
                    "raw_buy_count": 1,
                    "display_buy_candidate_count": 0,
                    "data_health_blocks": ["SPY_DATA_MISSING"],
                    "trading_mode": "DEGRADED_MODE",
                    "degraded_mode_active": True,
                    "degraded_mode_reason": "SPY_DATA_MISSING",
                    "degraded_size_mult": 0.70,
                    "degraded_min_confidence": 65,
                    "degraded_reject_counts": {"DEGRADED_LOW_VOLUME_BLOCKED": 1},
                    "volatility_value": 14.2,
                    "regime_source": "degraded_fallback",
                    "top_buyable_rejects": [
                        {"ticker": "AAA", "rejection_reason": "VERY_LOW_VOLUME_CONFIRMATION"}
                    ],
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
        self.assertEqual(out["last_no_buy_diagnostics"]["main_blocker"],
                         "no_buy_candidates")
        self.assertEqual(out["last_no_buy_diagnostics"]["raw_buy_count"], 1)
        self.assertEqual(out["last_no_buy_diagnostics"]["data_health_blocks"],
                         ["SPY_DATA_MISSING"])
        self.assertEqual(out["last_no_buy_diagnostics"]["trading_mode"],
                         "DEGRADED_MODE")
        self.assertTrue(out["last_no_buy_diagnostics"]["degraded_mode_active"])
        self.assertEqual(out["last_no_buy_diagnostics"]["volatility_value"], 14.2)
        self.assertEqual(out["last_no_buy_diagnostics"]["regime_source"],
                         "degraded_fallback")
        self.assertEqual(out["last_no_buy_diagnostics"]["tick_runtime_seconds"], 0.25)
        self.assertNotIn("bot_file_path", str(out))
        self.assertNotIn("storage_base_dir", str(out))

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

    def test_pa_daily_bars_use_finnhub_adapter(self):
        import market.quotes as quotes

        old_pa = quotes.PYTHONANYWHERE_MODE
        old_fh_daily = quotes._finnhub_daily
        try:
            quotes.PYTHONANYWHERE_MODE = True
            quotes._finnhub_daily = lambda tk, full=False: {
                "ticker": tk, "full": full, "source": "finnhub"
            }
            self.assertEqual(
                quotes._stooq_daily("AAA", full=True),
                {"ticker": "AAA", "full": True, "source": "finnhub"},
            )
        finally:
            quotes.PYTHONANYWHERE_MODE = old_pa
            quotes._finnhub_daily = old_fh_daily

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
                self.assertEqual(snap["endpoint:test"]["status"], "degraded")
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
        cfg["sweep_params"] = {"signal.min_buy_confidence": [55, 60]}
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


if __name__ == "__main__":
    unittest.main()
