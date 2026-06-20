"""
Keep the PythonAnywhere web app warm AND trigger bot trades.
Run on any always-on machine (laptop, Pi, etc.):
    python keepalive.py
"""
import time
import urllib.request
import urllib.error
import os

URL      = os.environ.get("KEEPALIVE_URL", "http://127.0.0.1:5000/health")
INTERVAL = int(os.environ.get("KEEPALIVE_INTERVAL", "60"))


def main():
    while True:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "keepalive/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                print(f"[{stamp}] {r.status} {URL}", flush=True)
        except urllib.error.URLError as e:
            print(f"[{stamp}] ERROR {e}", flush=True)
        except Exception as e:
            print(f"[{stamp}] ERROR {type(e).__name__}: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
