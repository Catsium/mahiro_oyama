"""PythonAnywhere WSGI entrypoint.

Configure the web app's WSGI file to import `application` from here, or paste
this file's contents into PythonAnywhere's generated WSGI file.
"""
import os
import sys

PROJECT_HOME = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
if PROJECT_HOME not in sys.path:
    sys.path.insert(0, PROJECT_HOME)
os.chdir(PROJECT_HOME)

from app import app as application  # noqa: E402
