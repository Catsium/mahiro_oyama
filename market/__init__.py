"""Market data layer - Finnhub client singleton."""
try:
    import finnhub
except Exception:
    finnhub = None

from utils.deploy_config import FINNHUB_KEY
from utils.rate_limit import acquire_finnhub_slot


class _MissingFinnhubClient:
    def __getattr__(self, name):
        if not finnhub:
            raise RuntimeError("finnhub package is not installed")
        raise RuntimeError("FINNHUB_KEY is not configured")


class _RateLimitedFinnhubClient:
    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            acquire_finnhub_slot(block=True)
            return attr(*args, **kwargs)

        return _wrapped


fh = (
    _RateLimitedFinnhubClient(finnhub.Client(api_key=FINNHUB_KEY))
    if finnhub and FINNHUB_KEY else _MissingFinnhubClient()
)
