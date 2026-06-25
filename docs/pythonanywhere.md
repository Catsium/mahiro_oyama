# PythonAnywhere Deployment

This app is a paper-trading bot. Keep `PYTHONANYWHERE_MODE=1` on PythonAnywhere
so worker counts, cache use, scan batches, and state caps stay free-tier safe.

## First Setup

```bash
cd /home/<username>
git clone <repo-url> mahiro_oyama
cd /home/<username>/mahiro_oyama
python3.10 -m venv /home/<username>/.virtualenvs/mahiro
/home/<username>/.virtualenvs/mahiro/bin/pip install -r requirements.txt
mkdir -p data
```

## Web Tab

Set these values in the PythonAnywhere Web tab:

```text
Source code: /home/<username>/mahiro_oyama
Working directory: /home/<username>/mahiro_oyama
Virtualenv: /home/<username>/.virtualenvs/mahiro
WSGI file: /var/www/<username>_pythonanywhere_com_wsgi.py
```

In the WSGI file, either import `application` from `pythonanywhere_wsgi.py` or
paste this block before importing the app:

```python
import os
import sys

os.environ["APP_BASE_DIR"] = "/home/<username>/mahiro_oyama"
os.environ["PYTHONANYWHERE_MODE"] = "1"
os.environ["PERSISTENT_CACHE"] = "1"
os.environ["FINNHUB_KEY"] = "<secret>"
os.environ["ADMIN_TOKEN"] = "<secret>"
os.environ["FLASK_SECRET_KEY"] = "<secret>"

project_home = os.environ["APP_BASE_DIR"]
if project_home not in sys.path:
    sys.path.insert(0, project_home)
os.chdir(project_home)

from app import app as application
```

Reload the web app after editing the WSGI file.

## Scheduled Bot Trigger

If your PythonAnywhere account supports scheduled tasks, add:

```bash
/home/<username>/.virtualenvs/mahiro/bin/python /home/<username>/mahiro_oyama/bot_task.py
```

Run it every few minutes during market hours if your plan allows that frequency.
The script itself skips work when the market is closed.

## Free-Tier Keepalive Fallback

Free accounts can use an external uptime monitor or always-on machine to hit
`/health`, which warms the app and triggers the throttled bot check:

```bash
cd /home/<username>/mahiro_oyama
KEEPALIVE_URL=https://<username>.pythonanywhere.com/health python keepalive.py
```

By default `keepalive.py` pings every 60 seconds. Override with
`KEEPALIVE_INTERVAL=<seconds>` if your monitor needs a different cadence.
On an external Linux box, run the same command inside a `tmux` session or systemd
service. UptimeRobot can also call every minute:

```text
https://<username>.pythonanywhere.com/health
```

The 60-second ping only keeps the web worker warm. Bot work is still throttled
to the bot interval, so duplicate pings do not start cooldown-only bot passes.

## Admin Checks

Open:

```text
https://<username>.pythonanywhere.com/botcontrol?token=<ADMIN_TOKEN>
```

Check the `//why no buy?` panel. On the admin page it shows the active storage
paths, including `bot_state.json` and `tickers.json`, so you can confirm WSGI and
scheduled tasks are using the same repo folder.

## Update Existing Deploy

```bash
cd /home/<username>/mahiro_oyama
git pull
/home/<username>/.virtualenvs/mahiro/bin/pip install -r requirements.txt
touch /var/www/<username>_pythonanywhere_com_wsgi.py
```

Then reload the web app from the Web tab.

## References

- PythonAnywhere scheduled tasks: <https://help.pythonanywhere.com/pages/ScheduledTasks/>
- PythonAnywhere environment variables: <https://helpdev.pythonanywhere.com/pages/EnvironmentVariables/>
- PythonAnywhere virtualenvs: <https://help.pythonanywhere.com/pages/VirtualenvsExplained/>
