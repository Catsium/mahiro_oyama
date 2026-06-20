import secrets
from urllib.parse import urlencode

from flask import abort, redirect, request, session

from utils.deploy_config import ADMIN_TOKEN, PYTHONANYWHERE_MODE


def _tokenless_redirect():
    args = request.args.to_dict(flat=False)
    args.pop("token", None)
    args.pop("admin_token", None)
    qs = urlencode(args, doseq=True)
    return redirect(request.path + (f"?{qs}" if qs else ""))


def csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(24)
        session.modified = True
    return session["csrf_token"]


def require_admin_token():
    """Require ADMIN_TOKEN in PA mode; allow local dev when unset.

    Accepted once through query/form/header, then remembered in the Flask session.
    """
    if not ADMIN_TOKEN:
        if PYTHONANYWHERE_MODE:
            abort(403)
        return True
    token = (
        request.headers.get("X-Admin-Token")
        or request.args.get("admin_token")
        or request.args.get("token")
        or request.form.get("admin_token")
        or request.form.get("token")
    )
    if token == ADMIN_TOKEN:
        session["admin_ok"] = True
        csrf_token()
        session.modified = True
        if request.method == "GET" and (
            request.args.get("token") == token or request.args.get("admin_token") == token
        ):
            return _tokenless_redirect()
        return True
    if session.get("admin_ok"):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
            if sent != session.get("csrf_token"):
                abort(403)
        return True
    abort(403)
    return False
