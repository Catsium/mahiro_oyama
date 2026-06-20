"""Runtime configuration.

Secrets are never embedded. Tests can opt into deterministic placeholder
values with CATSIUM_TEST_MODE=1.
"""
import os


def _truthy(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


CATSIUM_ENV = os.environ.get("CATSIUM_ENV", "development").strip().lower()
TEST_MODE = _truthy("CATSIUM_TEST_MODE", False)
BOT_ENABLED = _truthy("BOT_ENABLED", not TEST_MODE)
SCHEDULER_ENABLED = _truthy("SCHEDULER_ENABLED", False)
DEBUG = _truthy("FLASK_DEBUG", CATSIUM_ENV == "development" and not TEST_MODE)
KEEPALIVE_URL = os.environ.get(
    "KEEPALIVE_URL",
    "https://catsiumsama.pythonanywhere.com/health",
).strip()


def get_required_env(name, *, test_value=None):
    if TEST_MODE and test_value is not None:
        return os.environ.get(name, test_value)
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is required. Set {name} in the environment "
            "or run with CATSIUM_TEST_MODE=1 for tests."
        )
    return value


def get_finnhub_key():
    return get_required_env("FINNHUB_KEY", test_value="test-finnhub-key")


def get_flask_secret_key():
    return get_required_env("FLASK_SECRET_KEY", test_value="test-flask-secret")


def redacted_config():
    return {
        "catsium_env": CATSIUM_ENV,
        "test_mode": TEST_MODE,
        "bot_enabled": BOT_ENABLED,
        "scheduler_enabled": SCHEDULER_ENABLED,
        "debug": DEBUG,
        "keepalive_url": KEEPALIVE_URL,
        "finnhub_key": {"configured": bool(os.environ.get("FINNHUB_KEY")) or TEST_MODE},
        "flask_secret_key": {"configured": bool(os.environ.get("FLASK_SECRET_KEY")) or TEST_MODE},
    }
