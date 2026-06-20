"""Route-facing signal snapshots.

In PA mode, list pages must not cold-fetch every ticker through the only worker.
They read cache and show HOLD/0 placeholders until /health-staged runs warm data.
Single-ticker pages can still request live data.
"""
from market.history import get_history
from market.quotes import get_quote
from market.sentiment import get_news
from trading.risk import get_earnings_soon, get_analyst_rec, get_insider_sentiment
from trading.signals import get_recommendation
from utils.cache import cache_get
from utils.deploy_config import PYTHONANYWHERE_MODE


EMPTY_QUOTE = {
    "price": 0,
    "change": 0,
    "pct": 0,
    "high": 0,
    "low": 0,
    "open": 0,
    "prev": 0,
    "stale": True,
}


def signal_snapshot(ticker, regime=None, live=True, owned=0):
    t = ticker.upper()
    if live or not PYTHONANYWHERE_MODE:
        q = get_quote(t)
        arts, sent = get_news(t)
        ctx = get_history(t)
        earn = get_earnings_soon(t)
        analyst = get_analyst_rec(t)
        insider = get_insider_sentiment(t)
    else:
        q = cache_get(f"q_{t}", max_age=3600) or dict(EMPTY_QUOTE)
        arts, sent = cache_get(f"n_{t}", max_age=6 * 3600) or ([], 0.0)
        ctx = cache_get(f"h_{t}", max_age=6 * 3600) or {}
        earn = cache_get(f"earn_{t}", max_age=24 * 3600) or {"soon": False, "date": None}
        analyst = cache_get(f"ar_{t}", max_age=24 * 3600) or {
            "net": 0.0, "buy": 0, "hold": 0, "sell": 0, "total": 0, "age_hours": None
        }
        insider = cache_get(f"is_{t}", max_age=24 * 3600) or {
            "sentiment": 0.0, "samples": 0, "age_hours": None
        }
    rec = get_recommendation(sent, ctx, regime=regime, earnings=earn,
                             analyst=analyst, insider=insider,
                             news_articles=arts)
    return {
        "ticker": t,
        "quote": q,
        "articles": arts,
        "sentiment": sent,
        "ctx": ctx,
        "earnings": earn,
        "analyst": analyst,
        "insider": insider,
        "rec": rec,
        "price": q.get("price", 0),
        "owned": owned,
    }
