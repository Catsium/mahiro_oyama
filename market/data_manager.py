"""Bounded daily-bar cache manager.

Keeps repeated daily OHLCV callers from hammering the same low-level source in
one bot run, while still using the existing disk cache helpers for PA restarts.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

from utils.cache import cache_get, cache_set


class DataManager:
    def __init__(self, max_items=80, disk_ttl_sec=6 * 3600):
        self.max_items = int(max_items)
        self.disk_ttl_sec = int(disk_ttl_sec)
        self._lock = threading.RLock()
        self._daily = OrderedDict()

    def _key(self, ticker, full=False):
        return (str(ticker or "").upper(), bool(full))

    def _copy(self, value):
        try:
            return value.copy()
        except Exception:
            return value

    def _is_stale_cache(self, value):
        attrs = getattr(value, "attrs", {}) or {}
        source = str(attrs.get("source") or "")
        return attrs.get("status") == "stale_cache" or source.startswith("stale_cache:")

    def _source(self, ticker, full=False):
        from market import quotes
        return quotes._raw_daily(ticker, full=full)

    def get_daily(self, ticker, full=False, source_func=None):
        key = self._key(ticker, full)
        now = time.time()
        with self._lock:
            rec = self._daily.get(key)
            if rec:
                value, ts = rec
                if self._is_stale_cache(value):
                    self._daily.pop(key, None)
                elif now - ts < self.disk_ttl_sec:
                    self._daily.move_to_end(key)
                    return self._copy(value)
                else:
                    self._daily.pop(key, None)

        disk_key = f"dm_daily_{key[0]}_{'full' if key[1] else 'tail'}"
        value = cache_get(disk_key, max_age=self.disk_ttl_sec)
        if self._is_stale_cache(value):
            value = None
        if value is None:
            src = source_func or self._source
            value = src(key[0], full=key[1])
            if value is not None and not self._is_stale_cache(value):
                cache_set(disk_key, value)

        if value is None:
            return None

        if self._is_stale_cache(value):
            return self._copy(value)

        with self._lock:
            self._daily[key] = (value, now)
            self._daily.move_to_end(key)
            while len(self._daily) > self.max_items:
                self._daily.popitem(last=False)
        return self._copy(value)

    def clear(self):
        with self._lock:
            self._daily.clear()


_DEFAULT_MANAGER = DataManager()


def get_daily(ticker, full=False):
    return _DEFAULT_MANAGER.get_daily(ticker, full=full)
