"""Process-local rate limiter for Finnhub free-tier calls."""
import threading
import time
from collections import deque

from utils.deploy_config import FINNHUB_CALLS_PER_MINUTE, FINNHUB_CALLS_PER_SECOND


_lock = threading.Lock()
_minute_calls = deque()
_second_calls = deque()


def acquire_finnhub_slot(block=True):
    """Reserve one Finnhub call inside rolling second and minute windows."""
    minute_limit = max(1, FINNHUB_CALLS_PER_MINUTE)
    second_limit = max(1, FINNHUB_CALLS_PER_SECOND)
    while True:
        wait_for = 0.0
        with _lock:
            now = time.time()
            while _minute_calls and now - _minute_calls[0] >= 60:
                _minute_calls.popleft()
            while _second_calls and now - _second_calls[0] >= 1:
                _second_calls.popleft()
            if len(_minute_calls) < minute_limit and len(_second_calls) < second_limit:
                _minute_calls.append(now)
                _second_calls.append(now)
                return True
            if not block:
                return False
            minute_wait = 60 - (now - _minute_calls[0]) if len(_minute_calls) >= minute_limit else 0
            second_wait = 1 - (now - _second_calls[0]) if len(_second_calls) >= second_limit else 0
            wait_for = max(0.05, minute_wait, second_wait)
        time.sleep(min(wait_for, 5.0))
