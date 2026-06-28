"""Deployment/environment knobs.

Defaults are local-dev friendly. PythonAnywhere should set PYTHONANYWHERE_MODE=1
plus real secrets in the web app environment.
"""
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_BASE_DIR = os.environ.get("APP_BASE_DIR", BASE_DIR)


def env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default, min_value=None, max_value=None):
    try:
        value = int(os.environ.get(name, default))
    except Exception:
        value = int(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _detect_pythonanywhere():
    markers = (
        "PYTHONANYWHERE_DOMAIN",
        "PYTHONANYWHERE_SITE",
        "PYTHONANYWHERE_USERNAME",
    )
    if any(os.environ.get(k) for k in markers):
        return True
    server = os.environ.get("SERVER_SOFTWARE", "").lower()
    return "pythonanywhere" in server


PYTHONANYWHERE_MODE = env_bool("PYTHONANYWHERE_MODE", _detect_pythonanywhere())

FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "").strip()
FMP_KEY = (os.environ.get("FMP_KEY", "").strip()
           or os.environ.get("FMP_API_KEY", "").strip())
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "").strip()
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()

BOT_TRIGGER_MIN_INTERVAL = env_int("BOT_TRIGGER_MIN_INTERVAL", 60, 30, 900)
FINNHUB_CALLS_PER_MINUTE = env_int("FINNHUB_CALLS_PER_MINUTE", 50, 1, 50)
FINNHUB_CALLS_PER_SECOND = env_int("FINNHUB_CALLS_PER_SECOND", 25, 1, 25)

# PA free-tier staging. Holdings are always included; this caps extra tickers.
PA_TICKERS_PER_BOT_RUN = env_int("PA_TICKERS_PER_BOT_RUN", 6, 1, 8)
PA_SCAN_BATCH_SIZE = env_int("PA_SCAN_BATCH_SIZE", 4, 1, 8)
PA_PAGE_TICKER_LIMIT = env_int("PA_PAGE_TICKER_LIMIT", 12, 1, 56)

# Persistent cache survives web-worker reloads. Local dev can opt in explicitly.
PERSISTENT_CACHE = env_bool("PERSISTENT_CACHE", PYTHONANYWHERE_MODE)
