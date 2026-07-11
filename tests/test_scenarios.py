"""Scenario tests: does the bot make the right call in a given situation?

Covers the Learning v3 overhaul: decision scenarios (uptrend/downtrend/knife/
neutral-trend gate), learning mechanics (decay, skip-learning, adaptive edge
floor, confidence calibration), the session-anchored profit calendar, and the
Google-first quote chain. unittest only (pytest is not installed).
"""
import os
import time
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

# This file sorts first under `unittest discover`, so it must establish the
# same test-mode env defaults test_stabilization.py relies on BEFORE any app
# module import caches its config.
os.environ.setdefault("CATSIUM_TEST_MODE", "1")
os.environ.setdefault("BOT_ENABLED", "0")
os.environ.setdefault("FINNHUB_KEY", "test-finnhub")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")

# Reuse the main suite's dependency stubs (finnhub/yfinance installed as fakes
# before app imports). Import the SAME module instance unittest discover uses
# (top-level name) so its import-time side effects run exactly once.
try:
    import test_trading_v1  # noqa: F401  (discover imports it top-level)
except ImportError:  # direct `python -m unittest tests.test_scenarios`
    import tests.test_trading_v1  # noqa: F401

import pandas as pd

from trading import attribution as A
from trading.bot import buyable_reason, _record_candidate_observation
from trading.attribution import TRACK_TOP_SKIPS_PER_CYCLE
from trading.config import DEFAULT_CONFIG
from trading.indicators import _ctx_from_series
from trading.signals import get_recommendation

ET = ZoneInfo("America/New_York")
NOW = int(time.time())


def _close_series(pattern, days=70, start=100.0):
    """Zigzag path from a repeating list of daily % moves (realistic RSI —
    a perfectly monotonic path pins RSI to 0/100 and tests the wrong thing)."""
    vals = [start]
    for i in range(days):
        vals.append(vals[-1] * (1 + pattern[i % len(pattern)] / 100.0))
    idx = pd.date_range(end="2026-07-10", periods=len(vals), freq="B")
    return pd.Series(vals, index=idx).round(2)


UPTREND = [1.1, 0.7, -0.4]      # net ~ +0.5%/day with pullbacks
DOWNTREND = [-1.1, -0.7, 0.4]   # mirror image


def _rec_for(pattern, sent=0.0):
    close = _close_series(pattern)
    ctx = _ctx_from_series(close)
    return get_recommendation(sent, ctx, pure_technical=True,
                              allow_live_risk=False, config=DEFAULT_CONFIG), ctx


def _signal_snapshot(ctx_overrides=None, rec_overrides=None, price=100.0):
    """Minimal snapshot for buyable_reason gate tests."""
    ctx = {
        "atr_pct": 2.0, "avg_dollar_vol_20d": 100_000_000, "rsi": 55,
        "adx": 30, "vol_ratio": 1.0, "week_chg_pct": 0.5,
        "dist_from_high_pct": -2.0, "is_dip": False,
        "current": price, "ma30": price * 0.97,
        "history_rows": 70, "history_status": "ok", "history_source": "test",
    }
    ctx.update(ctx_overrides or {})
    rec = {
        "signal": "BUY", "cls": "buy", "confidence": 65, "score": 4.0,
        "categories": {"trend": 2.0}, "reasons": [],
    }
    rec.update(rec_overrides or {})
    return {"ctx": ctx, "rec": rec, "price": price, "stale": False,
            "execution_trusted": True, "arts": []}


class DecisionScenarioTests(unittest.TestCase):
    """The bot should lean the right way for clear-cut price paths."""

    def test_uptrend_is_not_a_sell_and_beats_downtrend(self):
        rec_up, ctx_up = _rec_for(UPTREND, sent=0.3)
        rec_down, _ = _rec_for(DOWNTREND, sent=-0.3)
        self.assertNotIn(rec_up["cls"], ("sell", "strong-sell"))
        self.assertGreater(ctx_up["current"], ctx_up["ma30"])
        # monotonicity: identical structure, opposite direction -> the
        # uptrend must always score higher than the downtrend
        self.assertGreater(rec_up["score"], rec_down["score"])

    def test_downtrend_is_never_a_strong_buy(self):
        rec, ctx = _rec_for(DOWNTREND, sent=-0.3)
        self.assertNotEqual(rec["cls"], "strong-buy")
        self.assertLess(ctx["current"], ctx["ma30"])

    def test_falling_knife_blocked_by_hard_gate(self):
        s = _signal_snapshot({"week_chg_pct": -6.0, "dist_from_high_pct": -12.0})
        ok, reason = buyable_reason("KNF", s, regime_kind="neutral",
                                    cfg=DEFAULT_CONFIG)
        self.assertFalse(ok)
        self.assertIn("FALLING_KNIFE", reason)

    def test_neutral_trend_continuation_needs_confirmation(self):
        # ADX 22 (above the old 20 gate) + flat volume -> blocked by the new
        # evidence-backed gate; the same setup with ADX 27 or hot volume passes.
        weak = _signal_snapshot({"adx": 22, "vol_ratio": 1.0})
        ok, reason = buyable_reason("TRD", weak, regime_kind="neutral",
                                    cfg=DEFAULT_CONFIG)
        self.assertFalse(ok)
        self.assertEqual(reason, "NEUTRAL_TREND_NEEDS_CONFIRMATION")

        strong_adx = _signal_snapshot({"adx": 27, "vol_ratio": 1.0})
        ok, reason = buyable_reason("TRD", strong_adx, regime_kind="neutral",
                                    cfg=DEFAULT_CONFIG)
        self.assertTrue(ok, reason)

        hot_volume = _signal_snapshot({"adx": 22, "vol_ratio": 1.4})
        ok, reason = buyable_reason("TRD", hot_volume, regime_kind="neutral",
                                    cfg=DEFAULT_CONFIG)
        self.assertTrue(ok, reason)

    def test_confirmation_gate_respects_kill_switch(self):
        cfg = dict(DEFAULT_CONFIG)
        cfg["signal"] = dict(DEFAULT_CONFIG.get("signal", {}))
        cfg["signal"]["neutral_trend_confirmation_enabled"] = False
        weak = _signal_snapshot({"adx": 22, "vol_ratio": 1.0})
        ok, reason = buyable_reason("TRD", weak, regime_kind="neutral", cfg=cfg)
        self.assertTrue(ok, reason)

    def test_dip_setup_bypasses_trend_confirmation(self):
        dip = _signal_snapshot({"adx": 15, "vol_ratio": 1.0, "is_dip": True,
                                "rsi": 32})
        ok, reason = buyable_reason("DIP", dip, regime_kind="neutral",
                                    cfg=DEFAULT_CONFIG)
        self.assertTrue(ok, reason)

    def test_bear_regime_requires_dip_or_oversold(self):
        s = _signal_snapshot({"rsi": 55})
        ok, reason = buyable_reason("BR", s, regime_kind="bear",
                                    cfg=DEFAULT_CONFIG)
        self.assertFalse(ok)
        self.assertIn("BEAR_GATE", reason)


class LearningMechanicsTests(unittest.TestCase):
    """Decay forgets, skips are learned from, gates adapt in both directions."""

    def test_decay_halves_bucket_weight_after_half_life(self):
        b = A.init_entry_bucket({})
        b = A._update_bucket(b, 1.0, 1.0, 1.0, 0.0, 0.5,
                             NOW - A.DECAY_HALF_LIFE_SEC, decision="executed")
        n_before = b["n"]
        decayed = A._decay_entry_bucket(dict(b), NOW)
        self.assertAlmostEqual(decayed["n"], n_before * 0.5, places=2)
        # alpha/beta decay toward the prior, never below it
        self.assertGreaterEqual(decayed["alpha"], A.ALPHA_PRIOR)
        self.assertGreaterEqual(decayed["beta"], A.BETA_PRIOR)

    def test_old_events_are_pruned_from_state(self):
        state = {"attribution_events": [
            {"ts": NOW - A.EVENT_MAX_AGE_SEC - 86400, "price": 100.0,
             "ticker": "OLD", "forward_returns": {f"{d}d": 1.0 for d in (1, 3, 5, 10)},
             "bucketed_horizons": ["5d"], "category_votes": {}},
            {"ts": NOW - 86400, "price": 100.0, "ticker": "NEW",
             "forward_returns": {}, "bucketed_horizons": [],
             "category_votes": {}},
        ]}
        A.update_forward_outcomes(state, lambda tk: None, ts=NOW)
        tickers = [e["ticker"] for e in state["attribution_events"]]
        self.assertNotIn("OLD", tickers)
        self.assertIn("NEW", tickers)

    def test_skip_learning_bounded_per_tick(self):
        b = {"attribution_events": [], "attribution_buckets": {}}
        tick_ts = NOW
        recorded = 0
        for i in range(TRACK_TOP_SKIPS_PER_CYCLE + 4):
            cand = {"ticker": f"SK{i:02d}", "cluster": "momentum",
                    "price": 50.0, "rec": {"confidence": 60, "categories": {}},
                    "ctx": {}, "source": "scan"}
            if _record_candidate_observation(b, cand, "skipped", "EDGE_TOO_LOW",
                                             tick_ts, "neutral"):
                recorded += 1
        self.assertEqual(recorded, TRACK_TOP_SKIPS_PER_CYCLE)
        self.assertEqual(len(b["attribution_events"]), TRACK_TOP_SKIPS_PER_CYCLE)
        self.assertTrue(all(e["decision"] == "skipped"
                            for e in b["attribution_events"]))

    def test_adaptive_floor_raises_when_executed_entries_lose(self):
        state = {"attribution_events": [
            {"ts": NOW - 86400, "decision": "executed", "confidence": 60,
             "forward_returns": {"3d": -1.5}}
        ] * 12}
        out = A.adaptive_edge_floor(state, base_floor=0.40, ts=NOW)
        self.assertGreater(out["floor"], 0.40)
        self.assertIn("raise", out["notes"])

    def test_adaptive_floor_lowers_when_skipped_candidates_win(self):
        events = ([{"ts": NOW - 86400, "decision": "executed", "confidence": 60,
                    "forward_returns": {"3d": 0.1}}] * 8
                  + [{"ts": NOW - 86400, "decision": "skipped", "confidence": 60,
                      "forward_returns": {"3d": 1.6}}] * 8)
        out = A.adaptive_edge_floor({"attribution_events": events},
                                    base_floor=0.40, ts=NOW)
        self.assertLess(out["floor"], 0.40)
        self.assertIn("lower", out["notes"])

    def test_calibration_replaces_formula_with_evidence(self):
        # 30 fresh samples: the winning 62-bin maps to a positive edge, the
        # losing 52-bin maps to zero (the "confidence is noise" finding).
        events = ([{"ts": NOW - 86400, "decision": "executed", "confidence": 62,
                    "forward_returns": {"3d": 1.2}}] * 15
                  + [{"ts": NOW - 86400, "decision": "executed", "confidence": 52,
                      "forward_returns": {"3d": -0.9}}] * 15)
        state = {"attribution_events": events}
        good = A.calibrated_prior_edge(state, 62, ts=NOW)
        bad = A.calibrated_prior_edge(state, 52, ts=NOW)
        self.assertIsNotNone(good)
        self.assertGreater(good, 0.5)
        self.assertEqual(bad, 0.0)

    def test_calibration_declines_when_sample_too_thin(self):
        state = {"attribution_events": [
            {"ts": NOW - 86400, "decision": "executed", "confidence": 62,
             "forward_returns": {"3d": 1.0}}
        ] * 5}
        self.assertIsNone(A.calibrated_prior_edge(state, 62, ts=NOW))


class SessionCalendarTests(unittest.TestCase):
    """Daily P/L = session end - session start (09:30-16:00 ET; 9:30pm-4am SGT)."""

    @staticmethod
    def _ts(y, m, d, hh, mm):
        return int(datetime(y, m, d, hh, mm, tzinfo=ET).timestamp())

    def _payload(self, equity_history):
        import routes.api as api
        bot = {"equity_history": equity_history, "history": [], "holdings": {}}
        return api.build_profit_calendar_payload(bot, 2026, 7)

    def test_session_pnl_is_end_minus_start_within_session(self):
        eh = [
            [self._ts(2026, 7, 7, 9, 31), 10000.0, 0.0, [], False],
            [self._ts(2026, 7, 7, 12, 0), 10040.0, 0.4, [], False],
            [self._ts(2026, 7, 7, 15, 59), 10025.0, 0.25, [], False],
        ]
        day = next(d for d in self._payload(eh)["days"] if d["date"] == "2026-07-07")
        self.assertTrue(day["has_data"])
        self.assertEqual(day["start_equity"], 10000.0)
        self.assertEqual(day["end_equity"], 10025.0)
        self.assertEqual(day["pnl_usd"], 25.0)

    def test_out_of_session_snapshots_are_excluded(self):
        eh = [
            [self._ts(2026, 7, 7, 9, 31), 10000.0, 0.0, [], False],
            [self._ts(2026, 7, 7, 15, 59), 10025.0, 0.25, [], False],
            # 16:30 ET is after the close (+ tolerance) — must not shift the end
            [self._ts(2026, 7, 7, 16, 30), 9000.0, -10.0, [], False],
        ]
        payload = self._payload(eh)
        day = next(d for d in payload["days"] if d["date"] == "2026-07-07")
        self.assertEqual(day["end_equity"], 10025.0)
        self.assertEqual(payload["data_quality"]["session_excluded_snapshots"], 1)

    def test_session_sgt_label_matches_user_window(self):
        day = next(d for d in self._payload([])["days"] if d["date"] == "2026-07-07")
        self.assertIsNotNone(day["session_sgt"])
        self.assertIn("21:30", day["session_sgt"])   # 09:30 ET (EDT) = 9:30pm SGT
        self.assertIn("04:00", day["session_sgt"])   # 16:00 ET = 4am SGT next day
        self.assertIn("SGT", day["session_sgt"])
        # weekend day has no session
        sat = next(d for d in self._payload([])["days"] if d["date"] == "2026-07-04")
        self.assertIsNone(sat["session_sgt"])

    def test_half_day_close_excludes_afternoon_snapshots(self):
        import routes.api as api
        ts_ok = int(datetime(2026, 11, 27, 12, 30, tzinfo=ET).timestamp())
        ts_late = int(datetime(2026, 11, 27, 14, 0, tzinfo=ET).timestamp())
        bot = {"equity_history": [
            [ts_ok, 10000.0, 0.0, [], False],
            [ts_late, 10100.0, 1.0, [], False],
        ], "history": [], "holdings": {}}
        payload = api.build_profit_calendar_payload(bot, 2026, 11)
        day = next(d for d in payload["days"] if d["date"] == "2026-11-27")
        self.assertEqual(day["end_equity"], 10000.0)


class GoogleQuoteTests(unittest.TestCase):
    FIXTURE = (
        '<html><body><div data-last-price="255.46" data-currency-code="USD">'
        "</div><div>Previous close</div><div class=\"P6K39c\">$252.20</div>"
        "<div>Day range</div><div class=\"P6K39c\">$251.00 - $256.10</div>"
        "</body></html>"
    )

    def test_parse_quote_html(self):
        from market.google_quotes import parse_quote_html
        q = parse_quote_html(self.FIXTURE)
        self.assertEqual(q["price"], 255.46)
        self.assertEqual(q["prev"], 252.20)
        self.assertEqual(q["low"], 251.00)
        self.assertEqual(q["high"], 256.10)
        self.assertAlmostEqual(q["pct"], (255.46 - 252.20) / 252.20 * 100, places=3)
        self.assertEqual(q["source"], "google_quote")

    def test_parse_returns_none_without_price(self):
        from market.google_quotes import parse_quote_html
        self.assertIsNone(parse_quote_html("<html>captcha page</html>"))

    def test_quote_chain_google_first_then_finnhub_fallback(self):
        from types import SimpleNamespace
        import market.quotes as quotes
        import market.google_quotes as gq
        from utils import cache

        old_google = gq.google_quote
        old_fh = quotes.fh
        old_failures = dict(cache._api_failures)
        old_save = cache._save_provider_health
        try:
            cache._api_failures.clear()
            cache._save_provider_health = lambda: True

            # Google succeeds -> finnhub must not be called
            gq.google_quote = lambda tk: {
                "price": 101.0, "change": 1.0, "pct": 1.0, "high": 102.0,
                "low": 100.0, "open": 0.0, "prev": 100.0,
                "source": "google_quote",
            }
            def _boom(tk):
                raise AssertionError("finnhub called despite google success")
            quotes.fh = SimpleNamespace(quote=_boom)
            r = quotes._fetch_quote_once("AAPL")
            self.assertEqual(r["source"], "google_quote")
            self.assertEqual(r["provider_used"], "google_quote")

            # Google empty -> falls through to finnhub
            gq.google_quote = lambda tk: None
            quotes.fh = SimpleNamespace(
                quote=lambda tk: {"c": 55.0, "d": 1.0, "dp": 2.0, "h": 56.0,
                                  "l": 54.0, "o": 55.0, "pc": 54.0})
            r = quotes._fetch_quote_once("AAPL")
            self.assertEqual(r["source"], "finnhub_quote")
            statuses = [a["status"] for a in r["provider_attempts"]]
            self.assertIn("empty", statuses)   # google tried first

            # Google blocked -> global circuit opens, finnhub still serves
            from market.google_quotes import GoogleQuoteBlocked
            def _blocked(tk):
                raise GoogleQuoteBlocked("http_429")
            gq.google_quote = _blocked
            r = quotes._fetch_quote_once("AAPL")
            self.assertEqual(r["source"], "finnhub_quote")
            self.assertTrue(quotes.google_quote_global_circuit_state()["active"])

            # circuit active -> google skipped without calling it
            def _never(tk):
                raise AssertionError("google called while global circuit open")
            gq.google_quote = _never
            r = quotes._fetch_quote_once("AAPL")
            self.assertEqual(r["source"], "finnhub_quote")
        finally:
            gq.google_quote = old_google
            quotes.fh = old_fh
            cache._api_failures.clear()
            cache._api_failures.update(old_failures)
            cache._save_provider_health = old_save


if __name__ == "__main__":
    unittest.main()
