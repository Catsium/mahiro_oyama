"""Process-local rate limiter for Finnhub free-tier calls."""
import threading
import time
from collections import deque

from utils.deploy_config import FINNHUB_CALLS_PER_MINUTE


_lock = threading.Lock()
_calls = deque()


def acquire_finnhub_slot(block=True):
    """Reserve one Finnhub call inside a rolling 60-second window."""
    limit = max(1, FINNHUB_CALLS_PER_MINUTE)
    while True:
        wait_for = 0.0
        with _lock:
            now = time.time()
            while _calls and now - _calls[0] >= 60:
                _calls.popleft()
            if len(_calls) < limit:
                _calls.append(now)
                return True
            if not block:
                return False
            wait_for = max(0.05, 60 - (now - _calls[0]) + 0.05)
        time.sleep(min(wait_for, 5.0))
