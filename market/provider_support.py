"""Per-symbol daily-history provider support registry (audit P0-5).

FMP free tier returns a permanent 402 Payment Required for a subset of
symbols (Finnhub daily is 403 for everything). Symbols marked unsupported are
skipped by the PA rotation — unless recorded snapshots can decide them — so
the per-tick fetch budget only goes to decidable names.

Weekly re-probe with no scheduler: once `reprobe_after_ts` passes,
`is_supported` returns True again and the next natural fetch attempt either
clears the mark (success) or rolls it forward (another 402).
"""
import json
import os
import threading
import time

from utils.storage import DATA_DIR

PROVIDER_SUPPORT_FILE = os.path.join(DATA_DIR, "provider_support.json")
REPROBE_SEC = 7 * 86400

_lock = threading.Lock()
_state = None


def _load():
    global _state
    if _state is None:
        try:
            with open(PROVIDER_SUPPORT_FILE, encoding="utf-8") as f:
                data = json.load(f)
                _state = data if isinstance(data, dict) else {}
        except Exception:
            _state = {}
    return _state


def _save():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = PROVIDER_SUPPORT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_state, f, sort_keys=True)
        os.replace(tmp, PROVIDER_SUPPORT_FILE)
    except Exception:
        pass  # registry is an optimization; never break the fetch path


def _norm(sym):
    return str(sym or "").upper().strip()


def mark_unsupported(sym, reason):
    sym = _norm(sym)
    if not sym:
        return
    now = int(time.time())
    with _lock:
        _load()[sym] = {
            "status": "unsupported",
            "reason": str(reason)[:120],
            "marked_ts": now,
            "reprobe_after_ts": now + REPROBE_SEC,
        }
        _save()


def mark_supported(sym):
    sym = _norm(sym)
    with _lock:
        if sym in _load():
            del _state[sym]
            _save()


def is_supported(sym):
    """True when the symbol is unmarked or its weekly re-probe is due."""
    with _lock:
        entry = _load().get(_norm(sym))
    if not entry:
        return True
    return time.time() >= float(entry.get("reprobe_after_ts") or 0)


def unsupported_symbols():
    """Currently-skipped symbols (marked and not yet due for re-probe)."""
    now = time.time()
    with _lock:
        return sorted(
            s for s, e in _load().items()
            if now < float((e or {}).get("reprobe_after_ts") or 0)
        )
