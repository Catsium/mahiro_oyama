import ast
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("CATSIUM_TEST_MODE", "1")
os.environ.setdefault("BOT_ENABLED", "0")
os.environ.setdefault("FINNHUB_KEY", "test-finnhub")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")


def _has_real_flask():
    try:
        import flask
        return hasattr(flask.Flask, "test_client")
    except Exception:
        return False


HAS_REAL_FLASK = _has_real_flask()


class ConfigTests(unittest.TestCase):
    def test_test_mode_supplies_placeholder_secrets(self):
        from utils.config import get_finnhub_key, get_flask_secret_key, redacted_config

        self.assertTrue(get_finnhub_key())
        self.assertTrue(get_flask_secret_key())
        cfg = redacted_config()
        self.assertTrue(cfg["test_mode"])
        self.assertNotIn("test-secret", json.dumps(cfg))

    def test_production_secret_missing_fails(self):
        env = os.environ.copy()
        env.pop("FLASK_SECRET_KEY", None)
        env["CATSIUM_TEST_MODE"] = "0"
        code = (
            "import os; "
            "os.environ.pop('FLASK_SECRET_KEY', None); "
            "os.environ['CATSIUM_TEST_MODE']='0'; "
            "from utils.config import get_flask_secret_key; "
            "get_flask_secret_key()"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("FLASK_SECRET_KEY is required", proc.stderr)

    def test_keepalive_default_interval_is_one_minute(self):
        import keepalive

        self.assertEqual(keepalive.INTERVAL, 60)


class CacheAndQuoteTests(unittest.TestCase):
    def test_none_cache_entry_is_distinct_from_miss(self):
        from utils.cache import CACHE_MISS, cache_get, cache_set

        cache_set("unit-none", None)
        self.assertIsNone(cache_get("unit-none", default=CACHE_MISS))
        self.assertIs(cache_get("unit-missing", default=CACHE_MISS), CACHE_MISS)

    def test_stooq_negative_cache_avoids_repeat_fetch(self):
        from market import quotes

        symbol = "NEGZUNIT"
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")) as fetch:
            self.assertIsNone(quotes._stooq_daily(symbol))
            self.assertIsNone(quotes._stooq_daily(symbol))
        self.assertEqual(fetch.call_count, 1)

    def test_pythonanywhere_news_skips_yfinance_fallback(self):
        from market import sentiment

        class FakeFinnhub:
            def company_news(self, *_args, **_kwargs):
                return []

        with patch.object(sentiment, "PYTHONANYWHERE_MODE", True):
            with patch.object(sentiment, "fh", FakeFinnhub()):
                with patch.dict(sys.modules, {"yfinance": None}):
                    self.assertEqual(sentiment.get_news("PANONEUNIT"), ([], 0.0))

    def test_pa_hot_modules_do_not_import_yfinance_at_module_load(self):
        for path in (
            "routes/api.py",
            "routes/dashboard.py",
            "market/charts.py",
            "market/history.py",
            "market/quotes.py",
            "market/sentiment.py",
            "trading/risk.py",
        ):
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            top_imports = [
                alias.name
                for node in tree.body
                if isinstance(node, ast.Import)
                for alias in node.names
            ]
            self.assertNotIn("yfinance", top_imports, path)


class StorageTests(unittest.TestCase):
    def test_ticker_save_load_dedupes_and_validates(self):
        import utils.storage as storage

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tickers.json")
            with patch.object(storage, "TICKERS_FILE", path):
                storage.save_tickers(["aapl", "BAD!", "AAPL", "msft", ""])
                self.assertEqual(storage.load_tickers(), ["AAPL", "MSFT"])

    def test_corrupt_bot_state_refuses_overwrite(self):
        import utils.storage as storage

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bot_state.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{bad json")
            with patch.object(storage, "BOT_FILE", path):
                state = storage.load_bot()
                self.assertTrue(state["stopped"])
                self.assertIn("_load_error", state)
                self.assertFalse(storage.save_bot(state))
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "{bad json")

    def test_bot_file_lock_is_exclusive(self):
        import utils.storage as storage

        with tempfile.TemporaryDirectory() as td:
            lock_path = os.path.join(td, "bot_state.json.lock")
            with patch.object(storage, "BOT_LOCK_FILE", lock_path):
                with storage.acquire_bot_file_lock():
                    with self.assertRaises(TimeoutError):
                        with storage.acquire_bot_file_lock(block=False):
                            pass

    def test_atomic_json_write_is_thread_safe(self):
        import utils.storage as storage

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "state.json")
            errors = []

            def worker(i):
                if not storage.save_json_atomic(path, {"i": i, "payload": [i] * 5}):
                    errors.append(i)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("i", data)
            self.assertIsInstance(data["i"], int)

    def test_legacy_bot_state_is_normalized_and_seeded(self):
        import utils.storage as storage

        raw = {
            "cash": 9000,
            "starting": 10000,
            "holdings": {"AAPL": {"shares": 2, "avg_cost": 100}},
            "history": [{"action": "BUY"}, {"action": "SELL"}, {"action": "HOLD"}],
            "trade_outcomes": [{"pnl_pct": 1.2}, {"pnl_pct": -0.5}],
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bot_state.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(raw, f)
            with patch.object(storage, "BOT_FILE", path):
                state = storage.load_bot()
        self.assertEqual(state["schema_version"], storage.BOT_SCHEMA_VERSION)
        self.assertEqual(state["total_trades"], 2)
        self.assertEqual(state["wins_total"], 1)
        self.assertEqual(state["losses_total"], 1)
        self.assertIn("commission_invested", state["holdings"]["AAPL"])


@unittest.skipUnless(HAS_REAL_FLASK, "Flask not installed")
class SimulatorRouteTests(unittest.TestCase):
    def setUp(self):
        from app import app

        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_buy_rejects_stale_or_zero_live_price(self):
        with patch("routes.portfolio.load_tickers", return_value=["AAPL"]):
            with patch("routes.portfolio.get_quote", return_value={"price": 0, "stale": True}):
                response = self.client.post(
                    "/simulator/buy",
                    data={"ticker": "AAPL", "shares": "1"},
                    follow_redirects=False,
                )
        self.assertEqual(response.status_code, 302)
        with self.client.session_transaction() as sess:
            self.assertEqual(sess["pf"]["holdings"], {})

    def test_sell_rejects_stale_or_zero_live_price(self):
        with self.client.session_transaction() as sess:
            sess["pf"] = {
                "cash": 0,
                "holdings": {"AAPL": {"shares": 2.0, "avg_cost": 100.0}},
                "history": [],
            }
        with patch("routes.portfolio.load_tickers", return_value=["AAPL"]):
            with patch("routes.portfolio.get_quote", return_value={"price": 0, "stale": True}):
                response = self.client.post(
                    "/simulator/sell",
                    data={"ticker": "AAPL", "shares": "1"},
                    follow_redirects=False,
                )
        self.assertEqual(response.status_code, 302)
        with self.client.session_transaction() as sess:
            self.assertEqual(sess["pf"]["holdings"]["AAPL"]["shares"], 2.0)


@unittest.skipUnless(HAS_REAL_FLASK, "Flask not installed")
class AuthRouteTests(unittest.TestCase):
    def setUp(self):
        from app import app

        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_admin_query_token_matches_docs_and_strips_url_secret(self):
        patches = [
            patch("utils.auth.ADMIN_TOKEN", "secret"),
            patch("routes.portfolio._render_bot_page", return_value="ok"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        response = self.client.get("/botcontrol?token=secret&tab=bot")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/botcontrol?tab=bot")

        response = self.client.get(response.headers["Location"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "ok")

    def test_admin_post_session_requires_csrf(self):
        with patch("utils.auth.ADMIN_TOKEN", "secret"):
            with self.client.session_transaction() as sess:
                sess["admin_ok"] = True
                sess["csrf_token"] = "known"
            response = self.client.post("/bot/reset", data={"starting": "10000"})
        self.assertEqual(response.status_code, 403)


@unittest.skipUnless(HAS_REAL_FLASK, "Flask not installed")
class DashboardRouteTests(unittest.TestCase):
    def setUp(self):
        from app import app

        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_scan_universe_stock_detail_allowed_outside_watchlist(self):
        fake_rec = {
            "signal": "BUY",
            "cls": "buy",
            "confidence": 80,
            "score": 2,
            "reasons": ["unit"],
            "categories": {},
        }
        patches = [
            patch("routes.dashboard.load_tickers", return_value=["AAPL"]),
            patch("routes.dashboard.get_quote", return_value={"price": 10, "stale": False}),
            patch("routes.dashboard.get_news", return_value=([], 0.0)),
            patch("routes.dashboard.get_history", return_value={}),
            patch("routes.dashboard.get_market_regime", return_value={"regime": "bull"}),
            patch("routes.dashboard.get_earnings_soon", return_value={}),
            patch("routes.dashboard.get_analyst_rec", return_value={}),
            patch("routes.dashboard.get_insider_sentiment", return_value={}),
            patch("routes.dashboard.get_recommendation", return_value=fake_rec),
            patch("routes.dashboard.render_template", return_value="ok"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        response = self.client.get("/stock/NVDA")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "ok")

    def test_pythonanywhere_dashboard_caps_visible_tickers_only(self):
        tickers = ["AAPL", "MSFT", "NVDA"]
        snapshot_calls = []
        rendered = {}

        def fake_snapshot(ticker, **_kwargs):
            snapshot_calls.append(ticker)
            return {
                "quote": {"price": 1, "pct": 0, "change": 0, "high": 1, "low": 1, "prev": 1},
                "sentiment": 0,
                "rec": {"signal": "HOLD", "cls": "hold", "confidence": 0, "score": 0},
            }

        def fake_render(_template, **kwargs):
            rendered.update(kwargs)
            return "ok"

        patches = [
            patch("routes.dashboard.PYTHONANYWHERE_MODE", True),
            patch("routes.dashboard.PA_PAGE_TICKER_LIMIT", 2),
            patch("routes.dashboard.load_tickers", return_value=tickers),
            patch("routes.dashboard.get_market_regime", return_value={"regime": "neutral", "spy_mom_30d": 0}),
            patch("routes.dashboard.signal_snapshot", side_effect=fake_snapshot),
            patch("routes.dashboard.get_scan", return_value=([], 0)),
            patch("routes.dashboard.render_template", side_effect=fake_render),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "ok")
        self.assertEqual(snapshot_calls, ["AAPL", "MSFT"])
        self.assertEqual(rendered["tickers"], tickers)
        self.assertEqual(rendered["visible_tickers"], ["AAPL", "MSFT"])
        self.assertEqual(rendered["hidden_ticker_count"], 1)


@unittest.skipUnless(HAS_REAL_FLASK, "Flask not installed")
class ApiRouteTests(unittest.TestCase):
    def setUp(self):
        from app import app

        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_invalid_chart_ticker_returns_400(self):
        response = self.client.get("/api/chart/BAD!/1d")
        self.assertEqual(response.status_code, 400)

    def test_equity_route_buckets_without_empty_daily_reset(self):
        points = [[1, 10000, 0, []], [2, 10001, 0.01, []], [61, 10002, 0.02, []]]
        state = {"equity_history": points, "starting": 10000}
        with patch("routes.api.load_bot", return_value=state):
            response = self.client.get("/api/bot/equity?range=1d")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["range"], "1d")
        self.assertEqual(len(data["points"]), 2)

    def test_heavy_backtest_routes_require_admin(self):
        with patch("utils.auth.ADMIN_TOKEN", "secret"):
            for path in (
                "/api/backtest/portfolio?allow_heavy=1",
                "/api/backtest/AAPL",
                "/api/signal_validation/AAPL",
            ):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 403, path)


@unittest.skipUnless(HAS_REAL_FLASK, "Flask not installed")
class TriggerRouteTests(unittest.TestCase):
    def setUp(self):
        from app import app

        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_public_get_routes_do_not_trigger_bot_work(self):
        fake_rec = {"signal": "HOLD", "cls": "hold", "confidence": 0, "score": 0}
        patches = [
            patch("routes.dashboard.load_tickers", return_value=["AAPL"]),
            patch("routes.dashboard.get_market_regime", return_value={"regime": "neutral", "spy_mom_30d": 0}),
            patch("routes.dashboard.signal_snapshot", return_value={
                "quote": {"price": 1, "pct": 0, "change": 0, "high": 1, "low": 1, "prev": 1},
                "sentiment": 0,
                "rec": fake_rec,
            }),
            patch("routes.dashboard.get_scan", return_value=([], 0)),
            patch("routes.dashboard.render_template", return_value="ok"),
            patch("routes.portfolio.pf_state", return_value=({"cash": 0, "holdings": {}, "history": []}, [], 0, 0, 0)),
            patch("routes.portfolio.bot_state", return_value=({"cash": 0}, [], 0, 0, 0)),
            patch("routes.portfolio.load_tickers", return_value=[]),
            patch("routes.portfolio.get_market_regime", return_value={"regime": "neutral"}),
            patch("routes.portfolio.render_template", return_value="ok"),
            patch("trading.bot.bot_state", return_value=({"cash": 0, "history": []}, [], 0, 0, 0)),
            patch("trading.bot.render_template", return_value="ok"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        trigger_patch = patch("routes.portfolio.trigger_bot_if_due")
        trigger = trigger_patch.start()
        self.addCleanup(trigger_patch.stop)

        for path in ("/", "/simulator", "/bot"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
        trigger.assert_not_called()

    def test_health_is_the_public_trigger_route(self):
        with patch("routes.portfolio.start_scheduler_once") as scheduler:
            with patch("routes.portfolio.trigger_bot_if_due") as trigger:
                with patch("routes.portfolio.warm_scan_if_due") as scan:
                    response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "ok")
        scheduler.assert_called_once_with()
        trigger.assert_called_once_with(force=False)
        scan.assert_called_once_with()

    def test_bot_reset_refuses_corrupt_loaded_state(self):
        patches = [
            patch("utils.auth.ADMIN_TOKEN", "secret"),
            patch("routes.portfolio.load_bot", return_value={"_load_error": "bad json", "stopped": True}),
            patch("routes.portfolio.save_bot"),
        ]
        save = None
        for p in patches:
            started = p.start()
            if p is patches[2]:
                save = started
            self.addCleanup(p.stop)

        response = self.client.post("/bot/reset?token=secret", data={"starting": "10000"})
        self.assertEqual(response.status_code, 302)
        save.assert_not_called()

    def test_bot_run_reports_when_trigger_not_started(self):
        patches = [
            patch("utils.auth.ADMIN_TOKEN", "secret"),
            patch("routes.portfolio.is_market_open", return_value=True),
            patch("routes.portfolio.trigger_bot_if_due", return_value=False),
            patch("routes.portfolio.BOT_ENABLED", False),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        response = self.client.post("/bot/run?token=secret")
        self.assertEqual(response.status_code, 302)
        with self.client.session_transaction() as sess:
            flashes = sess.get("_flashes", [])
        self.assertIn(("warning", "Bot is disabled by configuration; no run started."), flashes)


class BotAccountingTests(unittest.TestCase):
    def test_partial_trim_allocates_commission_and_records_outcome(self):
        from trading.sizing import _partial_trim

        state = {
            "cash": 0.0,
            "history": [],
            "holdings": {},
            "trade_outcomes": [],
            "wins_total": 0,
            "losses_total": 0,
            "total_trades": 0,
            "total_costs_usd": 0.0,
            "signal_attribution": {},
            "signal_weights": {},
        }
        holding = {
            "shares": 10.0,
            "avg_cost": 10.0,
            "commission_invested": 1.0,
            "entry_categories": {"trend": 1.0},
            "entry_snapshot": {"market_regime": "bull", "confidence": 70},
        }
        rec = {"signal": "BUY", "confidence": 70, "reasons": ["unit"], "categories": {}}
        trimmed = _partial_trim(
            state,
            "AAPL",
            holding,
            12.0,
            {},
            rec,
            [],
            0.5,
            "unit partial",
            exit_reason_key="partial_take",
        )
        self.assertEqual(trimmed, 5.0)
        self.assertEqual(state["holdings"]["AAPL"]["shares"], 5.0)
        self.assertAlmostEqual(state["holdings"]["AAPL"]["commission_invested"], 0.5)
        self.assertEqual(state["trade_outcomes"][-1]["exit_reason"], "partial_take")
        self.assertEqual(state["wins_total"], 1)
        self.assertGreater(state["history"][0]["pnl_usd"], 0)

    def test_run_bot_test_mode_has_no_save_side_effect(self):
        from trading.bot import run_bot
        from utils.storage import default_bot_state

        with patch("trading.bot.load_bot", return_value=default_bot_state()):
            with patch("trading.bot.save_bot") as save:
                _state, traded, action = run_bot(force=True)
        self.assertFalse(traded)
        self.assertEqual(action, "bot_disabled")
        save.assert_not_called()

    def test_scheduler_is_noop_in_test_mode(self):
        import utils.threading_utils as threading_utils

        threading_utils._BOT_STATUS["scheduler_started"] = False
        threading_utils.start_scheduler_once()
        self.assertFalse(threading_utils._BOT_STATUS["scheduler_started"])


class ScanTests(unittest.TestCase):
    def test_run_scan_uses_mocked_data_and_skips_stale_quotes(self):
        import trading.bot as bot

        def quote(ticker):
            if ticker == "BAD":
                return {"price": 0, "stale": True}
            return {"price": 100.0, "change": 1.0, "pct": 1.0, "stale": False}

        def recommendation(*_args, **_kwargs):
            return {
                "signal": "BUY",
                "cls": "buy",
                "confidence": 75,
                "score": 3,
                "categories": {"trend": 1.0},
                "reasons": ["unit"],
            }

        with patch.object(bot, "SCAN_UNIVERSE", ["AAA", "BAD"]):
            with patch("trading.bot.get_market_regime", return_value={"regime": "bull"}):
                with patch("trading.bot.get_quote", side_effect=quote):
                    with patch("trading.bot.get_news", return_value=([], 0.0)):
                        with patch("trading.bot.get_history", return_value={"rsi": 50}):
                            with patch("trading.bot.get_earnings_soon", return_value={}):
                                with patch("trading.bot.get_analyst_rec", return_value={}):
                                    with patch("trading.bot.get_insider_sentiment", return_value={}):
                                        with patch("trading.bot.get_recommendation", side_effect=recommendation):
                                            rows = bot.run_scan()
        self.assertEqual([r["ticker"] for r in rows], ["AAA"])
        self.assertIsInstance(rows[0]["ts"], int)


if __name__ == "__main__":
    unittest.main()
