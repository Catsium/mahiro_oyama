"""Optional paid-plan bot loop.

PythonAnywhere free accounts cannot use Always-On tasks; the free deployment
uses UptimeRobot hitting /health instead. This file remains for paid accounts or
external machines that want a long-running loop.
"""
import sys, os, time, traceback

BASE_DIR = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from trading.bot import run_bot
from utils.time_utils import is_market_open

INTERVAL = 10 * 60  # 10 minutes

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Bot loop started at {BASE_DIR}.", flush=True)

while True:
    try:
        if is_market_open():
            b, traded, last_action = run_bot(force=False)
            stamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{stamp}] traded={traded} | {last_action or 'no trade'} | cash=${b['cash']:.2f}", flush=True)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Market closed — sleeping.", flush=True)
    except Exception:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR:", flush=True)
        traceback.print_exc()
    time.sleep(INTERVAL)
