"""File-system storage for bot state, watchlist, and locally-recorded prices.

Round-3 change: data files now live in `mahiro_oyama/data/` rather than the
project root. BASE_DIR resolves to mahiro_oyama/; DATA_DIR = mahiro_oyama/data/.
The folder is created on import if missing — first deploy after the reorg
won't error out.
"""
import json
import os
import re
import threading
import time
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # mahiro_oyama/
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

BOT_FILE        = os.path.join(DATA_DIR, "bot_state.json")
TICKERS_FILE    = os.path.join(DATA_DIR, "tickers.json")
PRICE_HIST_FILE = os.path.join(DATA_DIR, "price_history.json")
SUGGESTION_DB_FILE = os.path.join(DATA_DIR, "suggestions.sqlite3")
BOT_LOCK_FILE = BOT_FILE + ".lock"

DEFAULT_TICKERS = ["AAPL", "NVDA", "MSFT", "GOOG"]
BOT_SCHEMA_VERSION = 1
_storage_lock = threading.RLock()
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,7}$")


@contextmanager
def acquire_bot_file_lock(block=True, timeout=0, stale_sec=15 * 60):
    """Cross-process bot-state lock for PA web workers + scheduled tasks."""
    start = time.time()
    fd = None
    while True:
        try:
            fd = os.open(BOT_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, f"{os.getpid()} {int(time.time())}".encode("ascii"))
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(BOT_LOCK_FILE) > stale_sec:
                    os.remove(BOT_LOCK_FILE)
                    continue
            except FileNotFoundError:
                continue
            except Exception:
                pass
            if not block or (timeout and time.time() - start >= timeout):
                raise TimeoutError("bot state is locked")
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            os.remove(BOT_LOCK_FILE)
        except FileNotFoundError:
            pass


def _clean_tickers(tks):
    out = []
    for t in tks or []:
        tk = str(t or "").upper().strip()
        if tk and tk not in out and _TICKER_RE.match(tk):
            out.append(tk)
    return out


def default_bot_state():
    return {"cash": 10000.0, "starting": 10000.0, "holdings": {}, "history": [],
            "last_trade": 0, "equity_history": [], "total_trades": 0,
            "wins_total": 0, "losses_total": 0, "total_costs_usd": 0.0,
            "schema_version": BOT_SCHEMA_VERSION,
            "candidate_observations": [], "edge_stats": {},
            "last_candidate_rankings": [], "extra_ticker_suggestions": [],
            "last_portfolio_variance_checks": [],
            "last_regime_v3": {}, "regime_v3_state": {},
            "attribution_events": [],
            "attribution_buckets": {}, "exit_attribution_events": [],
            "exit_attribution_buckets": {},
            "attribution_status": {"legacy_archived": True}}


def _load_error_state(e):
    b = default_bot_state()
    b["stopped"] = True
    b["_load_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return b


def save_json_atomic(path, data):
    """Write to <path>.tmp in the same directory, then os.replace() to swap.
    os.replace is atomic on POSIX and Windows; the file is never half-written."""
    tmp = path + ".tmp"
    with _storage_lock:
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False


# ── Tickers (watchlist) ──────────────────────────────────────────────────────
def load_tickers():
    with _storage_lock:
        try:
            with open(TICKERS_FILE, encoding="utf-8") as f:
                tks = _clean_tickers(json.load(f))
                return tks if tks else DEFAULT_TICKERS.copy()
        except Exception:
            return DEFAULT_TICKERS.copy()


def save_tickers(tks):
    save_json_atomic(TICKERS_FILE, _clean_tickers(tks))


# ── Locally-recorded price history (chart fallback when yfinance is dead) ───
def load_price_hist():
    with _storage_lock:
        try:
            with open(PRICE_HIST_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


def save_price_hist(d):
    save_json_atomic(PRICE_HIST_FILE, d)


def append_price_snapshot(tk, price, min_interval=60, limit=6000):
    """Atomic load-modify-save for quote workers updating price history."""
    if not price or price <= 0:
        return False
    import time
    with _storage_lock:
        d = load_price_hist()
        pts = d.get(tk, [])
        ts = int(time.time())
        if pts and ts - pts[-1][0] < min_interval:
            return False
        pts.append([ts, round(float(price), 2)])
        d[tk] = pts[-limit:]
        return save_json_atomic(PRICE_HIST_FILE, d)


# ── Bot state ────────────────────────────────────────────────────────────────
def load_bot():
    try:
        with _storage_lock:
            with open(BOT_FILE, encoding="utf-8") as f:
                b = json.load(f)
        # Round-3 fix: total_trades is a monotonic counter independent of the
        # history list (which is pruned to 50). On legacy state files without
        # this key, seed it from the current history's BUY+SELL count so we
        # don't lose the existing count on first load.
        if "total_trades" not in b:
            b["total_trades"] = sum(
                1 for t in b.get("history", [])
                if t.get("action") in ("BUY", "SELL")
            )
        # Bug #2: all-time win/loss counters were previously seeded in
        # _render_bot_page, which used to persist via bot_state()'s save_bot. That
        # save was removed (lost-update race), so seed here instead — from the
        # capped trade_outcomes, mirroring total_trades. run_bot persists on its
        # next tick; in the meantime reads see a consistent value.
        if "wins_total" not in b or "losses_total" not in b:
            _oc = b.get("trade_outcomes", [])
            b["wins_total"]   = sum(1 for o in _oc if o.get("pnl_pct", 0) > 0)
            b["losses_total"] = sum(1 for o in _oc if o.get("pnl_pct", 0) <= 0)
        b["schema_version"] = BOT_SCHEMA_VERSION
        for h in (b.get("holdings") or {}).values():
            if isinstance(h, dict):
                h.setdefault("commission_invested", 0.0)
        b.setdefault("total_costs_usd", 0.0)
        b.setdefault("candidate_observations", [])
        b.setdefault("edge_stats", {})
        b.setdefault("last_candidate_rankings", [])
        b.setdefault("extra_ticker_suggestions", [])
        b.setdefault("last_portfolio_variance_checks", [])
        b.setdefault("last_regime_v3", {})
        b.setdefault("regime_v3_state", {})
        b.setdefault("attribution_events", [])
        b.setdefault("attribution_buckets", {})
        b.setdefault("exit_attribution_events", [])
        b.setdefault("exit_attribution_buckets", {})
        b.setdefault("attribution_status", {"legacy_archived": True})
        return b
    except FileNotFoundError:
        return default_bot_state()
    except Exception:
        return _load_error_state(__import__("sys").exc_info()[1])


def save_bot(b):
    if isinstance(b, dict) and b.get("_load_error"):
        try:
            print(f"[storage] refusing to overwrite bot state after load error: {b.get('_load_error')}")
        except Exception:
            pass
        return False
    _cap_bot_state(b)
    if isinstance(b, dict):
        b["last_state_write_ts"] = int(time.time())
    return save_json_atomic(BOT_FILE, b)


def _cap_bot_state(b):
    """Keep JSON state bounded for PA free disk and fast reads."""
    if not isinstance(b, dict):
        return b
    try:
        from utils.deploy_config import PYTHONANYWHERE_MODE
    except Exception:
        PYTHONANYWHERE_MODE = False
    caps = {
        # 350 so the UI's 100 buys + 100 sells + 100 hold/skip rows coexist
        "history": 350,
        "equity_history": 400 if PYTHONANYWHERE_MODE else 2000,
        "trade_outcomes": 200,
        "candidate_observations": 300 if PYTHONANYWHERE_MODE else 600,
        "attribution_events": 500 if PYTHONANYWHERE_MODE else 800,
        "exit_attribution_events": 500 if PYTHONANYWHERE_MODE else 800,
        "last_candidate_rankings": 50 if PYTHONANYWHERE_MODE else 100,
        "extra_ticker_suggestions": 10,
        "last_portfolio_variance_checks": 50 if PYTHONANYWHERE_MODE else 100,
    }
    for key, cap in caps.items():
        val = b.get(key)
        if isinstance(val, list) and len(val) > cap:
            b[key] = val[-cap:] if key == "equity_history" else val[:cap]
    return b


def storage_debug_info():
    return {
        "storage_base_dir": DATA_DIR,
        "bot_file_path": BOT_FILE,
        "tickers_file_path": TICKERS_FILE,
        "price_history_file_path": PRICE_HIST_FILE,
        "suggestion_db_file_path": SUGGESTION_DB_FILE,
    }
