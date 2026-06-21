"""Concurrency primitives for the bot.

PA free-tier uses UptimeRobot /health plus trigger_bot_if_due(); the infinite
scheduler thread is local-dev / paid-host only.
"""
import threading
import time

from utils.deploy_config import BOT_TRIGGER_MIN_INTERVAL, PYTHONANYWHERE_MODE
from utils.config import BOT_ENABLED, SCHEDULER_ENABLED, TEST_MODE

_bot_run_lock = threading.Lock()

_BOT_STATUS = {
    "last_run_ts": 0,
    "last_action": "",
    "last_traded": False,
    "scheduler_started": False,
}
_trigger_lock = threading.Lock()

BOT_INTERVAL = 2 * 60


def _trigger_bot_async(force=False, user_forced=False):
    """Fire run_bot() on a background daemon thread. Returns immediately."""
    if _bot_run_lock.locked():
        return False
    from trading.bot import run_bot

    def _target():
        try:
            run_bot(force=force, user_forced=user_forced)
        except Exception as e:
            _BOT_STATUS.update({
                "last_error": f"{type(e).__name__}: {str(e)[:200]}",
                "last_error_ts": int(time.time()),
                "last_action": "error",
                "last_traded": False,
            })
            try:
                print(f"[bot-trigger] run_bot error: {type(e).__name__}: {e}")
            except Exception:
                pass

    try:
        threading.Thread(
            target=_target,
            daemon=True,
        ).start()
        return True
    except Exception:
        if PYTHONANYWHERE_MODE:
            return False
        try:
            run_bot(force=force, user_forced=user_forced)
        except Exception:
            pass
        return True


def trigger_bot_if_due(force=False, user_forced=False, min_interval=None):
    """PA-safe trigger: return quickly unless a run is due and idle."""
    if not BOT_ENABLED:
        return False
    interval = BOT_TRIGGER_MIN_INTERVAL if min_interval is None else min_interval
    now = time.time()
    with _trigger_lock:
        if _bot_run_lock.locked():
            return False
        if not force and now - _BOT_STATUS.get("last_run_ts", 0) < interval:
            return False
        _BOT_STATUS["last_trigger_attempt_ts"] = int(now)
        return _trigger_bot_async(force=force, user_forced=user_forced)


def _bot_scheduler_loop():
    """Local/paid background loop for bot decisions and scan warming."""
    from trading.bot import run_bot
    while True:
        time.sleep(BOT_INTERVAL)
        try:
            run_bot(force=False)
        except Exception as e:
            try:
                print(f"[scheduler] run_bot error: {type(e).__name__}: {e}")
            except Exception:
                pass
        try:
            from trading.bot import _scan_snapshot, _refresh_scan_background, SCAN_CACHE_TTL
            data, ts = _scan_snapshot()
            if (not data) or (time.time() - ts >= SCAN_CACHE_TTL * 0.8):
                threading.Thread(target=_refresh_scan_background, daemon=True).start()
        except Exception:
            pass


def start_scheduler_once():
    """Start local/paid scheduler once; no-op in PA free-tier mode."""
    if PYTHONANYWHERE_MODE or TEST_MODE or not SCHEDULER_ENABLED:
        _BOT_STATUS["scheduler_started"] = False
        return
    if _BOT_STATUS["scheduler_started"]:
        return
    _BOT_STATUS["scheduler_started"] = True
    try:
        t = threading.Thread(target=_bot_scheduler_loop, daemon=True, name="bot-scheduler")
        t.start()
    except Exception:
        # Health pings can still call trigger_bot_if_due().
        pass
