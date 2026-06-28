"""News fetching (Finnhub company_news → yfinance fallback) + headline scoring.

VADER lives here (with a finance-tuned lexicon overlay). `score_headline` is
re-exported from trading.signals so the recommendation engine can call it; we
just define it once.
"""
import re
from datetime import datetime, timedelta

from market import fh
from utils.cache import (
    cache_get, cache_set, record_api_failure, record_api_success, should_skip_api,
)
from utils.deploy_config import PYTHONANYWHERE_MODE


# VADER sentiment — much better than keyword matching. Falls back if missing.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None


# Fallback keyword sets (used only if VADER fails to load)
BULL = {"surge", "surged", "rally", "rallied", "beat", "beats", "record", "growth",
        "gain", "gains", "upgrade", "upgraded", "outperform", "bullish", "strong",
        "profit", "boom", "soar", "soared", "positive", "exceed", "exceeded",
        "breakout", "high", "rise", "rose", "buy"}
BEAR = {"fall", "fell", "drop", "dropped", "miss", "missed", "decline", "declined",
        "downgrade", "downgraded", "underperform", "bearish", "weak", "loss",
        "losses", "crash", "crashed", "negative", "lawsuit", "investigation",
        "layoff", "cut", "cuts", "concern", "sell", "low"}

# Finance-specific lexicon extensions for VADER (tunes domain vocabulary)
_FIN_LEXICON = {
    "upgrade": 2.5, "upgraded": 2.5, "outperform": 2.5, "outperforms": 2.5,
    "downgrade": -2.5, "downgraded": -2.5, "underperform": -2.5,
    "beat": 1.8, "beats": 1.8, "surpass": 1.8, "surpassed": 1.8,
    "miss": -1.8, "missed": -1.8, "misses": -1.8,
    "lawsuit": -2.0, "investigation": -2.0, "subpoena": -2.2,
    "bullish": 2.5, "bearish": -2.5,
    "guidance": 0.0, "buyback": 1.8, "dividend": 1.3,
    "layoff": -1.8, "layoffs": -1.8, "bankruptcy": -3.5,
    "rally": 2.0, "crash": -3.0, "plunge": -2.5, "soar": 2.5,
    "tariff": -1.2, "tariffs": -1.2, "sanction": -1.5, "sanctions": -1.5,
    "earnings": 0.0, "profit": 1.5, "loss": -1.5,
}
if _VADER:
    try: _VADER.lexicon.update(_FIN_LEXICON)
    except Exception: pass


def score_headline(t):
    """Returns a discrete score in -2..+2 from VADER (if available) or keyword fallback."""
    if not t:
        return 0
    if _VADER:
        c = _VADER.polarity_scores(t).get("compound", 0)
        if   c >=  0.55: return 2
        if   c >=  0.15: return 1
        if   c <= -0.55: return -2
        if   c <= -0.15: return -1
        return 0
    # Keyword fallback
    w = set(re.findall(r"\b\w+\b", t.lower()))
    b, s = len(w & BULL), len(w & BEAR)
    return 1 if b > s else (-1 if s > b else 0)


def get_news(tk):
    """News + aggregate sentiment for ticker. Tries Finnhub company_news first
    (free tier, reliable on shared IPs); falls back to yfinance.news (often
    empty since Yahoo blocked anonymous scraping)."""
    c = cache_get(f"n_{tk}", max_age=3600)
    if c is not None:
        return c
    arts, weighted_scores, weights = [], [], []
    today = datetime.now().date()

    def _ingest(title, pub, link, src, pub_ts=None):
        if not title: return
        weight = 0.5
        try:
            if pub:
                age_days = (today - datetime.strptime(pub, "%Y-%m-%d").date()).days
                if   age_days <= 0: weight = 1.0
                elif age_days <= 1: weight = 0.8
                elif age_days <= 3: weight = 0.55
                elif age_days <= 7: weight = 0.35
                else:               weight = 0.2
        except Exception: pass
        s = score_headline(title)
        weighted_scores.append(s * weight); weights.append(weight)
        # pub_ts (epoch seconds, UTC) is threaded through so signals.py can derive
        # news_age_hours at hour resolution — the date-string `pub` only gives days.
        arts.append({"title": title, "score": s, "pub": pub, "pub_ts": pub_ts,
                     "link": link, "source": src, "weight": round(weight, 2)})

    # 1) Finnhub company_news — primary source
    endpoint = "finnhub_news"
    if not should_skip_api(endpoint, cooldown_sec=600):
        try:
            start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
            end   = today.strftime("%Y-%m-%d")
            raw = fh.company_news(tk, _from=start, to=end) or []
            record_api_success(endpoint)
            for it in raw[:20]:
                title = it.get("headline", "")
                ts = it.get("datetime")
                pub_ts = None
                try:
                    if ts:
                        pub_ts = int(ts)
                        pub = datetime.utcfromtimestamp(pub_ts).strftime("%Y-%m-%d")
                    else:
                        pub = ""
                except Exception:
                    pub, pub_ts = "", None
                link = it.get("url", "")
                src  = it.get("source", "")
                _ingest(title, pub, link, src, pub_ts=pub_ts)
        except Exception as e:
            record_api_failure(endpoint, e)
            try: print(f"[news] finnhub {tk} failed: {type(e).__name__}: {e}")
            except Exception: pass

    # 2) yfinance fallback — only if Finnhub returned nothing
    if not arts and not PYTHONANYWHERE_MODE:
        try:
            import yfinance as yf
            raw = yf.Ticker(tk).news or []
            for it in raw[:20]:
                ct = it.get("content", {})
                pub_ts = None
                if isinstance(ct, dict):
                    title = ct.get("title", "")
                    pub_raw = (ct.get("pubDate", "") or "")
                    pub   = pub_raw[:10]
                    # yfinance pubDate is ISO-8601 (e.g. "2024-05-01T13:30:00Z").
                    # Parse to epoch so the decay sees real hours, not just the date.
                    if pub_raw:
                        try:
                            iso = pub_raw.replace("Z", "+00:00")
                            pub_ts = int(datetime.fromisoformat(iso).timestamp())
                        except Exception:
                            pub_ts = None
                    link  = (ct.get("canonicalUrl") or {}).get("url", "")
                    src   = (ct.get("provider") or {}).get("displayName", "")
                else:
                    title, pub, link, src = it.get("title", ""), "", it.get("link", ""), ""
                _ingest(title, pub, link, src, pub_ts=pub_ts)
        except Exception:
            pass

    avg = (sum(weighted_scores) / sum(weights)) if weights else 0.0
    r = (arts, round(avg, 3))
    cache_set(f"n_{tk}", r)
    return r
