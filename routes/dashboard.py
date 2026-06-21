"""Public-facing browse routes: /, /stock/<ticker>, /top, /top/refresh, /diag."""
import time
from datetime import datetime
from flask import render_template, redirect, url_for, flash, jsonify

from app import app
from market import fh
from market.history import get_history
from market.quotes import get_quote
from market.sentiment import get_news
from market.snapshots import signal_snapshot
from trading.bot import SCAN_UNIVERSE, get_scan, top_refresh_clear
from trading.risk import (
    get_analyst_rec, get_earnings_soon, get_insider_sentiment, get_market_regime,
)
from trading.signals import get_recommendation
from utils.auth import require_admin_token
from utils.deploy_config import PA_PAGE_TICKER_LIMIT, PYTHONANYWHERE_MODE
from utils.storage import load_tickers
from utils.time_utils import is_market_open


@app.route("/")
def index():
    tickers = load_tickers()
    visible_tickers = tickers[:PA_PAGE_TICKER_LIMIT] if PYTHONANYWHERE_MODE else tickers
    hidden_ticker_count = max(0, len(tickers) - len(visible_tickers))
    regime = get_market_regime()
    cards = []
    for t in visible_tickers:
        snap = signal_snapshot(t, regime=regime, live=not PYTHONANYWHERE_MODE)
        cards.append({"ticker": t, "quote": snap["quote"],
                      "sentiment": snap["sentiment"], "rec": snap["rec"]})
    scan_rows, scan_ts = get_scan()
    top_confidence = sorted(
        [r for r in scan_rows if r["direction"] > 0 and r["price"] > 0],
        key=lambda r: (-r["confidence"], -r["score"])
    )[:10]
    top_gainers = sorted(
        [r for r in scan_rows if r["price"] > 0],
        key=lambda r: -r["pct"]
    )[:10]
    scan_age_min = int((time.time() - scan_ts) / 60) if scan_ts else None
    return render_template("index.html", cards=cards, tickers=tickers,
                           visible_tickers=visible_tickers,
                           hidden_ticker_count=hidden_ticker_count,
                           regime=regime,
                           top_confidence=top_confidence, top_gainers=top_gainers,
                           scan_age_min=scan_age_min, scan_total=len(scan_rows),
                           now=datetime.now(), market_open=is_market_open())


def _stock_snapshot(ticker, regime):
    q = get_quote(ticker)
    arts, sent = get_news(ticker)
    ctx = get_history(ticker)
    rec = get_recommendation(
        sent, ctx, regime=regime, earnings=get_earnings_soon(ticker),
        analyst=get_analyst_rec(ticker), insider=get_insider_sentiment(ticker),
        news_articles=arts,
    )
    return {"quote": q, "articles": arts, "sentiment": sent, "ctx": ctx, "rec": rec}


@app.route("/stock/<ticker>")
def stock_detail(ticker):
    ticker = ticker.upper()
    tickers = load_tickers()
    if ticker not in tickers and ticker not in SCAN_UNIVERSE:
        flash(f"{ticker} not in watchlist. Add it from the Dashboard.", "warning")
        return redirect(url_for("index"))
    regime = get_market_regime()
    snap = _stock_snapshot(ticker, regime)
    q = snap["quote"]; arts = snap["articles"]; rec = snap["rec"]
    sent = snap["sentiment"]; ctx = snap["ctx"]
    sup = next((a for a in arts if (rec["score"] >= 0 and a["score"] > 0) or
                                    (rec["score"] <  0 and a["score"] < 0)),
               arts[0] if arts else None)
    return render_template("stock.html", ticker=ticker, quote=q, articles=arts,
                           sentiment=sent, ctx=ctx, rec=rec, support=sup,
                           now=datetime.now(), market_open=is_market_open())


@app.route("/top")
def top_picks():
    rows, ts = get_scan()
    age_min = int((time.time() - ts) / 60) if ts else 0
    bullish = [r for r in rows if r["direction"] > 0][:20]
    bearish = [r for r in rows if r["direction"] < 0][-10:][::-1]
    watchlist = load_tickers()
    return render_template("top.html", bullish=bullish, bearish=bearish,
                           age_min=age_min, total_scanned=len(rows),
                           watchlist=watchlist, market_open=is_market_open(),
                           now=datetime.now())


@app.route("/top/refresh", methods=["POST"])
def top_refresh():
    require_admin_token()
    top_refresh_clear()
    flash("Refreshing scan — this may take a minute…", "info")
    return redirect(url_for("top_picks"))


@app.route("/diag")
def diag():
    info = {"pythonanywhere_mode": PYTHONANYWHERE_MODE}
    try:
        import pandas as _pd
        info["pandas_version"] = _pd.__version__
    except Exception as e:
        info["pandas_version"] = f"ERROR: {e}"
    if PYTHONANYWHERE_MODE:
        info["yfinance_version"] = "SKIPPED in PythonAnywhere mode"
        info["yfinance_test_AAPL"] = "SKIPPED in PythonAnywhere mode"
    else:
        try:
            import yfinance as yf
            info["yfinance_version"] = getattr(yf, "__version__", "?")
            h = yf.Ticker("AAPL").history(period="5d")
            info["yfinance_test_AAPL"] = f"OK ({len(h)} rows)" if not h.empty else "EMPTY"
        except Exception as e:
            info["yfinance_test_AAPL"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
            info.setdefault("yfinance_version", f"ERROR: {e}")
    try:
        q = fh.quote("AAPL")
        info["finnhub_test_AAPL"] = f"OK (price ${q.get('c', 0)})"
    except Exception as e:
        info["finnhub_test_AAPL"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
    info["tickers"] = load_tickers()
    info["market_open"] = is_market_open()
    return jsonify(info)
