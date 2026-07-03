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
# Optional daily/history fallback only:
# os.environ["FMP_KEY"] = "<secret>"
os.environ["ADMIN_TOKEN"] = "<secret>"
os.environ["FLASK_SECRET_KEY"] = "<secret>"

project_home = os.environ["APP_BASE_DIR"]
if project_home not in sys.path:
    sys.path.insert(0, project_home)
os.chdir(project_home)

from app import app as application
```

For paper bot ticks, start with these free-tier-safe scan settings:

```python
os.environ["PA_TICKERS_PER_BOT_RUN"] = "6"
os.environ["PA_SCAN_BATCH_SIZE"] = "4"
```

`BOT_TICK_MAX_RUNTIME_SEC` is fixed in code at `25` seconds. Keep
PythonAnywhere bot scans under that wall-clock cap rather than increasing batch
sizes until requests time out.

If authenticated `/bot/tick?token=<ADMIN_TOKEN>` consistently returns in under
10-12 seconds, raise `PA_TICKERS_PER_BOT_RUN` to `8`. Do not jump to
full-watchlist scanning on PythonAnywhere free.

Reload the web app after editing the WSGI file.

Provider rules are governed by the root `AGENTS.md`: on PythonAnywhere free,
Stooq and yfinance are not live decision sources. Finnhub is primary; FMP is an
optional daily/history fallback only.

Current controlled-testing trading policy: `DEGRADED_MODE` stays visible when
SPY/regime or volatility data is missing, but the default
`degraded_use_standard_gates_for_testing=True` uses standard gates with effective
size `1.0`, effective min buy confidence `40`, normal EV gates, normal risk caps,
and fresh quote requirements. The stored `degraded_size_mult=0.90` profile is a
rollback setting when that flag is explicitly disabled.

## Scheduled Bot Trigger

If your PythonAnywhere account supports scheduled tasks, add:

```bash
/home/<username>/.virtualenvs/mahiro/bin/python /home/<username>/mahiro_oyama/bot_task.py
```

Run it every few minutes during market hours if your plan allows that frequency.
The script itself skips work when the market is closed.

## Free-Tier Keepalive Fallback

Free accounts can use an external uptime monitor or always-on machine to hit
`/health`, which only keeps the web app warm:

```bash
cd /home/<username>/mahiro_oyama
KEEPALIVE_URL=https://<username>.pythonanywhere.com/health python keepalive.py
```

By default `keepalive.py` pings every 60 seconds. Override with
`KEEPALIVE_INTERVAL=<seconds>` if your monitor needs a different cadence.
On an external Linux box, run the same command inside a `tmux` session or systemd
service. UptimeRobot can also call `/health` every minute:

```text
https://<username>.pythonanywhere.com/health
```

The 60-second ping only keeps the web worker warm; it does not trigger bot work.
For machine-triggered bot ticks, call the authenticated route from a trusted
scheduler or monitor:

```text
https://<username>.pythonanywhere.com/bot/tick?token=<ADMIN_TOKEN>
```

Provider diagnostics are available at:

```text
https://<username>.pythonanywhere.com/api/provider-test?token=<ADMIN_TOKEN>
```

Use `force=1` only when you intentionally want to bypass the 60-second diagnostic
cache.

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
