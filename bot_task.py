"""Single-shot bot trigger for PythonAnywhere scheduled task."""
import sys, os

BASE_DIR = os.environ.get("APP_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from trading.bot import run_bot
from utils.time_utils import is_market_open

if is_market_open():
    b, traded, last_action = run_bot(force=False)
    print("Traded:", traded, "|", last_action)
else:
    print("Market closed, skipping.")
