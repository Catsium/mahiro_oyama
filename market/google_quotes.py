"""Google Finance quote scraper — first-choice QUOTE provider.

Clean-room implementation (the suggested GPL repo was not imported).
Google Finance exposes quotes only — no historical candles — so this module
never participates in the daily-history chain.

Failure model: any HTTP 429 / captcha marker means Google is rate-limiting the
shared PythonAnywhere proxy IP; callers open a GLOBAL circuit and demote to
Finnhub. Ordinary parse/network errors stay per-symbol.
"""
import re
import urllib.error
import urllib.request

from utils.cache import cache_get, cache_set

GOOGLE_QUOTE_TIMEOUT_SEC = 4
# Most US tickers resolve on one of these; the winner is cached per symbol.
GOOGLE_EXCHANGES = ("NASDAQ", "NYSE", "NYSEARCA", "NYSEAMERICAN")
_EXCH_CACHE_TTL_SEC = 30 * 86400
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_PRICE_RE = re.compile(r'data-last-price="([0-9][0-9,]*\.?[0-9]*)"')
_PREV_RE = re.compile(
    r"Previous close</div>\s*<div[^>]*>\$?([0-9][0-9,]*\.?[0-9]*)")
_RANGE_RE = re.compile(
    r"Day range</div>\s*<div[^>]*>\$?([0-9][0-9,]*\.?[0-9]*)\s*-\s*"
    r"\$?([0-9][0-9,]*\.?[0-9]*)")


class GoogleQuoteBlocked(RuntimeError):
    """Google is rate-limiting / captcha-walling us — open the global circuit."""


def _num(text):
    try:
        return float(str(text).replace(",", ""))
    except Exception:
        return 0.0


def _fetch_page(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA,
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=GOOGLE_QUOTE_TIMEOUT_SEC) as resp:
            body = resp.read(1_500_000).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code in (429, 503):
            raise GoogleQuoteBlocked(f"http_{e.code}")
        if e.code == 404:
            return None  # wrong exchange guess — try the next one
        raise
    low = body[:4000].lower()
    if "unusual traffic" in low or "/sorry/" in low or "captcha" in low:
        raise GoogleQuoteBlocked("captcha_interstitial")
    return body


def parse_quote_html(body):
    """Extract a quote dict from a Google Finance quote page. Price required;
    prev/high/low best-effort (0 when the stats block isn't parseable)."""
    m = _PRICE_RE.search(body or "")
    if not m:
        return None
    price = _num(m.group(1))
    if price <= 0:
        return None
    prev_m = _PREV_RE.search(body)
    prev = _num(prev_m.group(1)) if prev_m else 0.0
    range_m = _RANGE_RE.search(body)
    low = _num(range_m.group(1)) if range_m else 0.0
    high = _num(range_m.group(2)) if range_m else 0.0
    change = round(price - prev, 4) if prev > 0 else 0.0
    pct = round((price - prev) / prev * 100, 4) if prev > 0 else 0.0
    return {"price": price, "change": change, "pct": pct,
            "high": high, "low": low, "open": 0.0, "prev": prev,
            "source": "google_quote"}


def google_quote(tk):
    """Return a quote dict (same shape as the Finnhub mapping) or None.

    Raises GoogleQuoteBlocked on rate-limit/captcha so the caller can open the
    endpoint-global circuit. Other errors bubble as ordinary exceptions.
    """
    tk = str(tk or "").upper().strip()
    if not tk or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", tk):
        return None
    exch_key = f"google_exch_{tk}"
    cached_exch = cache_get(exch_key, max_age=_EXCH_CACHE_TTL_SEC)
    order = ([cached_exch] if cached_exch else []) + [
        e for e in GOOGLE_EXCHANGES if e != cached_exch
    ]
    for exch in order:
        body = _fetch_page(f"https://www.google.com/finance/quote/{tk}:{exch}")
        if body is None:
            continue
        q = parse_quote_html(body)
        if q:
            if exch != cached_exch:
                cache_set(exch_key, exch)
            return q
        # Page loaded but had no price element: symbol not on this exchange —
        # Google serves a generic page rather than 404 for some misses.
    return None
