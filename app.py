"""Flask app entry point. Initializes the Flask `app` object, then imports
the route modules so their @app.route decorators bind.

The route modules use `from app import app` — this works because Python
caches the partially-loaded `app` module by the time the imports execute,
so they get the already-created `app` object.

Templates keep working with bare `url_for('index')` etc. (no Blueprint
namespacing required).
"""
from flask import Flask
from utils.deploy_config import ADMIN_TOKEN, FINNHUB_KEY, FLASK_SECRET_KEY, PYTHONANYWHERE_MODE

app = Flask(__name__)
if PYTHONANYWHERE_MODE:
    missing = [
        name for name, value in (
            ("FINNHUB_KEY", FINNHUB_KEY),
            ("ADMIN_TOKEN", ADMIN_TOKEN),
            ("FLASK_SECRET_KEY", FLASK_SECRET_KEY),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Set {', '.join(missing)} for PythonAnywhere deployment.")
app.secret_key = FLASK_SECRET_KEY or "dev-only-change-me"


@app.context_processor
def _inject_csrf_token():
    from flask import session
    from utils.auth import csrf_token
    return {"csrf_token": csrf_token() if session.get("admin_ok") else ""}

# Import route modules AFTER `app` is bound so their @app.route decorators bind correctly.
# noqa: E402,F401 — deliberately late-imported.
from routes import dashboard, api, portfolio   # noqa: E402,F401


if __name__ == "__main__":
    # Local-dev entry: start scheduler eagerly, run dev server.
    from utils.threading_utils import start_scheduler_once
    start_scheduler_once()
    app.run(debug=True, port=5000)
