"""Learning v3 migration smoke test against the REAL local bot state.

Loads data/bot_state.json into memory (read-only — never saved back) and runs
the new decay / prune / adaptive-floor / calibration / session-calendar paths
over it, proving the v3 code tolerates the legacy on-disk schema. Skips
cleanly on machines without a local state file.
"""
import json
import os
import time
import unittest

os.environ.setdefault("CATSIUM_TEST_MODE", "1")
os.environ.setdefault("BOT_ENABLED", "0")
os.environ.setdefault("FINNHUB_KEY", "test-finnhub")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")

STATE_PATH = os.path.join("data", "bot_state.json")


@unittest.skipUnless(os.path.exists(STATE_PATH), "no local bot_state.json")
class MigrationDryRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(STATE_PATH, encoding="utf-8") as f:
            cls.state = json.load(f)  # in-memory copy; never written back

    def test_learning_v3_runs_over_legacy_state(self):
        from trading import attribution as A
        b = json.loads(json.dumps(self.state))  # deep copy
        A.ensure_attribution_state(b)
        now = int(time.time())
        A.update_forward_outcomes(b, lambda tk: None, ts=now)
        A.update_exit_post_outcomes(b, lambda tk: None, ts=now)
        for e in b["attribution_events"]:
            self.assertLessEqual(now - int(e.get("ts") or now), A.EVENT_MAX_AGE_SEC)

        floor = A.adaptive_edge_floor(b, base_floor=0.40, ts=now)
        self.assertGreaterEqual(floor["floor"], 0.20)
        self.assertLessEqual(floor["floor"], 1.00)
        A.calibrated_prior_edge(b, 62, ts=now)  # must not raise on legacy rows

        for key, bucket in list(b.get("attribution_buckets", {}).items()):
            decayed = A._decay_entry_bucket(dict(bucket), now + 14 * 86400)
            self.assertLessEqual(decayed["n"], float(bucket.get("n", 0) or 0) + 1e-9,
                                 f"decay must not grow n for {key}")
            self.assertGreaterEqual(decayed["alpha"], A.ALPHA_PRIOR)

        A.exit_profile(b, "neutral", "momentum")  # no KeyErrors on legacy schema

    def test_session_calendar_over_legacy_state(self):
        import routes.api as api
        b = json.loads(json.dumps(self.state))
        payload = api.build_profit_calendar_payload(b, 2026, 7)
        self.assertIn("session_excluded_snapshots", payload["data_quality"])
        for day in payload["days"]:
            if day["has_data"]:
                self.assertAlmostEqual(
                    round(day["start_equity"] + day["pnl_usd"], 2),
                    day["end_equity"], places=2,
                    msg=f"pnl must equal end-start for {day['date']}")
            if day["session_sgt"]:
                self.assertIn("SGT", day["session_sgt"])


if __name__ == "__main__":
    unittest.main()
