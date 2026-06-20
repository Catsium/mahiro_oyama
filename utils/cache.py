"""Thread-safe in-memory cache shared across the app.

Single lock guards a module-level dict. Used by every data fetcher (quotes,
news, history, intraday, market regime, etc.) and by the scan worker pool.
"""
import threading
import time
import os
import pickle
import re

from utils.deploy_config import PERSISTENT_CACHE
from utils.storage import DATA_DIR

CACHE_TTL = 60   # default — 1-min stock data freshness
CACHE_MISS = object()

_cache: dict = {}
_cache_lock = threading.Lock()
_api_failures: dict = {}
_CACHE_DIR = os.path.join(DATA_DIR, "cache")


def _safe_key(k):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(k))[:180]


def _cache_path(k):
    return os.path.join(_CACHE_DIR, _safe_key(k) + ".pkl")


def _read_disk(k, ttl):
    if not PERSISTENT_CACHE:
        return CACHE_MISS
    try:
        path = _cache_path(k)
        if time.time() - os.path.getmtime(path) >= ttl:
            return CACHE_MISS
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return CACHE_MISS


def _write_disk(k, v):
    if not PERSISTENT_CACHE or v is None:
        return
    if str(k).startswith(("backtest_", "bt_", "stooq_full_")):
        return
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = _cache_path(k)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        pass


def cache_get(k, max_age=None, default=None):
    """Return cached value or None. max_age overrides CACHE_TTL for entries with
    different freshness needs (VIX uses 3600s, analyst uses 3600s, scan uses 90s).
    """
    ttl = CACHE_TTL if max_age is None else max_age
    with _cache_lock:
        e = _cache.get(k)
        if e and time.time() - e[1] < ttl:
            return e[0]
    disk_value = _read_disk(k, ttl)
    if disk_value is not CACHE_MISS:
        with _cache_lock:
            _cache[k] = (disk_value, time.time())
        return disk_value
    return default


def cache_set(k, v):
    with _cache_lock:
        _cache[k] = (v, time.time())
    _write_disk(k, v)


def prune_cache_dir(max_files=300, max_age_sec=7 * 86400):
    if not os.path.isdir(_CACHE_DIR):
        return {"removed": 0, "kept": 0}
    now = time.time()
    removed = 0
    entries = []
    try:
        for name in os.listdir(_CACHE_DIR):
            path = os.path.join(_CACHE_DIR, name)
            if not os.path.isfile(path):
                continue
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                continue
            if now - mtime > max_age_sec:
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass
            else:
                entries.append((mtime, path))
        entries.sort(reverse=True)
        for _, path in entries[int(max_files):]:
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass
        return {"removed": removed, "kept": min(len(entries), int(max_files))}
    except Exception:
        return {"removed": removed, "kept": 0}


def should_skip_api(endpoint, cooldown_sec=300, failure_threshold=3):
    key = str(endpoint or "")
    with _cache_lock:
        rec = _api_failures.get(key)
        if not rec:
            return False
        count = int(rec.get("count", 0) or 0)
        if count < int(failure_threshold or 1):
            return False
        return time.time() - rec.get("ts", 0) < float(cooldown_sec or 0)


def record_api_failure(endpoint):
    key = str(endpoint or "")
    with _cache_lock:
        rec = _api_failures.setdefault(key, {"count": 0, "ts": 0})
        rec["count"] = int(rec.get("count", 0)) + 1
        rec["ts"] = time.time()
    return rec


def record_api_success(endpoint):
    key = str(endpoint or "")
    with _cache_lock:
        _api_failures.pop(key, None)


def api_failure_snapshot():
    now = time.time()
    with _cache_lock:
        return {
            key: {
                "count": int(rec.get("count", 0) or 0),
                "age_sec": int(now - float(rec.get("ts", 0) or 0)),
            }
            for key, rec in _api_failures.items()
        }
