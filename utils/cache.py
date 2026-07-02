"""Thread-safe in-memory cache shared across the app.

Single lock guards a module-level dict. Used by every data fetcher (quotes,
news, history, intraday, market regime, etc.) and by the scan worker pool.
"""
import threading
import time
import os
import pickle
import re
import json

from utils.deploy_config import PERSISTENT_CACHE
from utils.storage import DATA_DIR

CACHE_TTL = 60   # default — 1-min stock data freshness
CACHE_MISS = object()

_cache: dict = {}
_cache_lock = threading.Lock()
_api_failures: dict = {}
_CACHE_DIR = os.path.join(DATA_DIR, "cache")
_PROVIDER_HEALTH_FILE = os.path.join(DATA_DIR, "provider_health.json")
_provider_health_last_error = None


def _load_provider_health():
    try:
        with open(_PROVIDER_HEALTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_provider_health():
    global _provider_health_last_error
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = _PROVIDER_HEALTH_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_api_failures, f, sort_keys=True)
        os.replace(tmp, _PROVIDER_HEALTH_FILE)
        _provider_health_last_error = None
        return True
    except Exception:
        _provider_health_last_error = "provider_health_write_failed"
        return False


def _looks_rate_limited(error):
    text = str(error or "").lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def sanitize_provider_error(error):
    """Return a bounded provider error string with obvious secret material removed."""
    if error is None:
        return None
    text = str(error)
    text = re.sub(r"(?i)(apikey|api_key|token|key)=([^&\s]+)", r"\1=<redacted>", text)
    text = re.sub(
        r"(?i)\b(FINNHUB_KEY|FMP_KEY|FMP_API_KEY|ADMIN_TOKEN)\b\s*=?\s*[^&\s]+",
        r"\1=<redacted>",
        text,
    )
    return text[:300]


def _classify_provider_status(error=None, status=None):
    if status:
        return str(status)
    text = str(error or "").lower()
    if _looks_rate_limited(error):
        return "rate_limited"
    if ("403" in text or "forbidden" in text or "blocked" in text
            or "don't have access" in text or "do not have access" in text
            or "access to this resource" in text):
        return "blocked_or_forbidden"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "empty" in text or "no data" in text:
        return "empty_response"
    if "parse" in text or "jsondecode" in text or "decode" in text:
        return "parse_error"
    return "provider_error"


_api_failures.update(_load_provider_health())

SKIP_PROVIDER_STATUSES = {
    "skipped_on_pythonanywhere",
    "skipped_missing_key",
    "skipped_by_global_rate_limit",
    "skipped_by_circuit",
}


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


def cache_get_stale(k, max_age, default=CACHE_MISS):
    """Return a successful cached value and age even after its normal TTL expires."""
    now = time.time()
    with _cache_lock:
        e = _cache.get(k)
        if e and e[0] is not None:
            age = now - float(e[1] or 0)
            if PERSISTENT_CACHE:
                try:
                    age = max(age, now - os.path.getmtime(_cache_path(k)))
                except Exception:
                    pass
            if age <= float(max_age or 0):
                return e[0], age
    if not PERSISTENT_CACHE:
        return default, None
    try:
        path = _cache_path(k)
        age = now - os.path.getmtime(path)
        if age > float(max_age or 0):
            return default, None
        with open(path, "rb") as f:
            value = pickle.load(f)
        if value is None:
            return default, None
        with _cache_lock:
            _cache[k] = (value, now - age)
        return value, age
    except Exception:
        return default, None


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
        if rec.get("status") in {"ok", "healthy", *SKIP_PROVIDER_STATUSES}:
            return False
        now = time.time()
        recent = [
            float(ts) for ts in (rec.get("failures") or [])
            if now - float(ts or 0) <= 600
        ]
        if not recent and rec.get("ts"):
            recent = [float(rec.get("ts") or 0)] * int(rec.get("count", 0) or 0)
        rec["failures"] = recent[-20:]
        rec["count"] = len(rec["failures"])
        if len(rec["failures"]) < int(failure_threshold or 1):
            return False
        return now - rec.get("ts", 0) < float(cooldown_sec or 0)


def api_cooldown_state(endpoint, cooldown_sec=300, active_statuses=None):
    """Return passive circuit state for an endpoint without mutating counters."""
    key = str(endpoint or "")
    active_statuses = set(active_statuses or {"rate_limited"})
    now = time.time()
    with _cache_lock:
        rec = dict(_api_failures.get(key) or {})
    status = rec.get("status") or "ok"
    ts = float(rec.get("ts", 0) or 0)
    age_sec = int(max(0, now - ts)) if ts else None
    effective_cooldown = float(rec.get("cooldown_sec") or cooldown_sec or 0)
    active = bool(ts and status in active_statuses and (now - ts) < effective_cooldown)
    rate_limited = bool(rec.get("rate_limited") or status == "rate_limited")
    return {
        "status": status,
        "active": active,
        "rate_limited": rate_limited,
        "cooldown_remaining_sec": int(max(0, effective_cooldown - (age_sec or 0))) if active else 0,
        "cooldown_sec": int(effective_cooldown),
        "last_429_age_sec": age_sec if rate_limited else None,
        "last_error": rec.get("last_error"),
        "count": int(rec.get("count", 0) or 0),
    }


def record_api_failure(endpoint, error=None, status=None, cooldown_sec=None):
    key = str(endpoint or "")
    resolved_status = _classify_provider_status(error, status)
    if resolved_status in SKIP_PROVIDER_STATUSES:
        now = time.time()
        rec = {
            "count": 0,
            "ts": now,
            "failures": [],
            "status": resolved_status,
            "rate_limited": False,
            "last_error": sanitize_provider_error(error),
            "persisted": True,
        }
        if cooldown_sec is not None:
            try:
                rec["cooldown_sec"] = int(max(0, float(cooldown_sec)))
            except Exception:
                pass
        with _cache_lock:
            _api_failures[key] = rec
            rec["persisted"] = _save_provider_health()
        return rec
    with _cache_lock:
        now = time.time()
        rec = _api_failures.setdefault(key, {"count": 0, "ts": 0, "failures": []})
        failures = [
            float(ts) for ts in (rec.get("failures") or [])
            if now - float(ts or 0) <= 600
        ]
        failures.append(now)
        rec["failures"] = failures[-20:]
        rec["count"] = len(rec["failures"])
        rec["ts"] = now
        rec["last_error"] = sanitize_provider_error(error) if error is not None else rec.get("last_error")
        rec["rate_limited"] = bool(resolved_status == "rate_limited" or rec.get("rate_limited"))
        rec["status"] = resolved_status
        if cooldown_sec is not None:
            try:
                rec["cooldown_sec"] = int(max(0, float(cooldown_sec)))
            except Exception:
                pass
        rec["persisted"] = _save_provider_health()
    return rec


def record_api_success(endpoint):
    key = str(endpoint or "")
    with _cache_lock:
        rec = _api_failures.setdefault(key, {"count": 0, "ts": time.time(), "failures": []})
        rec.update({
            "count": 0,
            "failures": [],
            "ts": time.time(),
            "status": "ok",
            "rate_limited": False,
            "last_error": None,
        })
        rec.pop("cooldown_sec", None)
        rec["persisted"] = _save_provider_health()


def api_failure_snapshot():
    now = time.time()
    with _cache_lock:
        snapshot = {
            key: {
                "count": int(rec.get("count", 0) or 0),
                "age_sec": int(now - float(rec.get("ts", 0) or 0)),
                "status": rec.get("status") or "degraded",
                "rate_limited": bool(rec.get("rate_limited")),
                "last_error": rec.get("last_error"),
                "persisted": bool(rec.get("persisted", True)),
                "cooldown_sec": rec.get("cooldown_sec"),
            }
            for key, rec in _api_failures.items()
            if rec.get("status") not in {"ok", "healthy", *SKIP_PROVIDER_STATUSES}
            or int(rec.get("count", 0) or 0) > 0
        }
        for rec in snapshot.values():
            rec["rate_limit_recent"] = bool(rec.get("rate_limited") and rec.get("age_sec", 0) <= 600)
        if _provider_health_last_error:
            snapshot["_provider_health_persistence"] = {
                "count": 1,
                "age_sec": 0,
                "status": "degraded",
                "rate_limited": False,
                "rate_limit_recent": False,
                "last_error": _provider_health_last_error,
                "persisted": False,
            }
        return snapshot
