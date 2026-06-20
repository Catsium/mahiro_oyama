"""SQLite advisory suggestion telemetry.

Stores UI impressions and feedback only. Trading state stays in JSON.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager

from trading.attribution import bucket_feedback_key
from trading.config import active_config


def _cfg() -> dict:
    return active_config().get("suggestion", {})


def get_connection(db_path: str) -> sqlite3.Connection:
    path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _connect(db_path: str):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_suggestion_db(db_path: str) -> None:
    cfg = _cfg()
    with _connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            f"PRAGMA wal_autocheckpoint={int(cfg.get('sqlite_wal_autocheckpoint_pages', 200))}"
        )
        conn.execute(
            f"PRAGMA journal_size_limit={int(cfg.get('sqlite_journal_size_limit_bytes', 16777216))}"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS suggestion_runs (
                run_id TEXT PRIMARY KEY,
                ts INTEGER NOT NULL,
                emitted_count INTEGER NOT NULL DEFAULT 0,
                ranked_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS suggestion_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                ts INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                bucket_key TEXT NOT NULL,
                emitted INTEGER NOT NULL DEFAULT 0,
                suggestion_score REAL,
                showable INTEGER NOT NULL DEFAULT 0,
                show_reason TEXT,
                payload_json TEXT,
                FOREIGN KEY(run_id) REFERENCES suggestion_runs(run_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS suggestion_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                ticker TEXT NOT NULL,
                bucket_key TEXT NOT NULL,
                action TEXT NOT NULL,
                ts INTEGER NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestion_items_ticker_ts "
            "ON suggestion_items(ticker, ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestion_items_run "
            "ON suggestion_items(run_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestion_feedback_action_ts "
            "ON suggestion_feedback(action, ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestion_feedback_ticker_ts "
            "ON suggestion_feedback(ticker, ts)"
        )


def log_suggestion_run(db_path: str, run_id: str, ts: int, emitted: list[dict],
                       all_ranked: list[dict] | None = None) -> None:
    init_suggestion_db(db_path)
    ranked = all_ranked if all_ranked is not None else emitted
    emitted_tickers = {str(row.get("ticker") or "").upper() for row in emitted}
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO suggestion_runs(run_id, ts, emitted_count, ranked_count) "
            "VALUES (?, ?, ?, ?)",
            (run_id, int(ts), len(emitted), len(ranked)),
        )
        for row in ranked:
            ticker = str(row.get("ticker") or "").upper()
            if not ticker:
                continue
            bucket = row.get("feedback_bucket") or bucket_feedback_key(row)
            conn.execute(
                "INSERT INTO suggestion_items("
                "run_id, ts, ticker, bucket_key, emitted, suggestion_score, "
                "showable, show_reason, payload_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    int(ts),
                    ticker,
                    bucket,
                    1 if ticker in emitted_tickers else 0,
                    row.get("suggestion_score"),
                    1 if row.get("showable", ticker in emitted_tickers) else 0,
                    row.get("show_reason"),
                    json.dumps(row, default=str, sort_keys=True),
                ),
            )


def record_suggestion_feedback(db_path: str, run_id: str, ticker: str,
                               action: str, ts: int) -> None:
    init_suggestion_db(db_path)
    ticker = str(ticker or "").upper()
    action = str(action or "").lower()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT bucket_key FROM suggestion_items "
            "WHERE run_id = ? AND ticker = ? ORDER BY ts DESC LIMIT 1",
            (run_id, ticker),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT bucket_key FROM suggestion_items "
                "WHERE ticker = ? ORDER BY ts DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        bucket = row["bucket_key"] if row else f"feedback:unknown:{ticker or 'unknown'}"
        conn.execute(
            "INSERT INTO suggestion_feedback(run_id, ticker, bucket_key, action, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, ticker, bucket, action, int(ts)),
        )


def load_feedback_stats(db_path: str, lookback_days: int = 90) -> dict:
    if not os.path.exists(os.path.abspath(db_path)):
        return {}
    cutoff = int(time.time()) - int(lookback_days) * 86400
    out = {}
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT bucket_key, action, COUNT(*) AS n "
            "FROM suggestion_feedback WHERE ts >= ? "
            "GROUP BY bucket_key, action",
            (cutoff,),
        ).fetchall()
    for row in rows:
        bucket = out.setdefault(row["bucket_key"], {
            "n": 0,
            "useful": 0,
            "weak": 0,
            "hide": 0,
            "alpha": 1,
            "beta": 1,
            "score": 0.5,
        })
        action = row["action"]
        n = int(row["n"] or 0)
        bucket["n"] += n
        if action in ("useful", "weak", "hide"):
            bucket[action] += n
    for bucket in out.values():
        bucket["alpha"] = 1 + bucket.get("useful", 0)
        bucket["beta"] = 1 + bucket.get("weak", 0) + bucket.get("hide", 0)
        bucket["score"] = round(
            bucket["alpha"] / max(1, bucket["alpha"] + bucket["beta"]),
            4,
        )
    return out


def load_recent_suggestions(db_path: str, lookback_sec: int = 21600) -> dict[str, int]:
    if not os.path.exists(os.path.abspath(db_path)):
        return {}
    cutoff = int(time.time()) - int(lookback_sec)
    out: dict[str, int] = {}
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT ticker, MAX(ts) AS ts FROM ("
            "SELECT ticker, ts FROM suggestion_items WHERE emitted = 1 AND ts >= ? "
            "UNION ALL "
            "SELECT ticker, ts FROM suggestion_feedback WHERE action = 'hide' AND ts >= ?"
            ") GROUP BY ticker",
            (cutoff, cutoff),
        ).fetchall()
    for row in rows:
        ticker = row["ticker"]
        ts = int(row["ts"] or 0)
        out[ticker] = max(ts, out.get(ticker, 0))
    return out


def prune_suggestion_store(db_path: str, *, log_retention_days: int = 90,
                           feedback_retention_days: int = 180) -> None:
    if not os.path.exists(os.path.abspath(db_path)):
        return
    now = int(time.time())
    log_cutoff = now - int(log_retention_days) * 86400
    feedback_cutoff = now - int(feedback_retention_days) * 86400
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM suggestion_items WHERE ts < ?", (log_cutoff,))
        conn.execute("DELETE FROM suggestion_runs WHERE ts < ?", (log_cutoff,))
        conn.execute("DELETE FROM suggestion_feedback WHERE ts < ?", (feedback_cutoff,))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
