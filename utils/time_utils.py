"""Market-hours + timezone formatting helpers.

NYSE hours: 09:30-16:00 ET (half-days close 13:00 ET). is_market_open() handles
DST automatically via zoneinfo, respects full + half-day holiday tables.
Conservative fallback: if zoneinfo is unavailable, assume closed.
"""
from datetime import datetime, timedelta

# NYSE holidays maintained manually (NYSE publishes annually)
NYSE_HOLIDAYS_FULL = {
    # 2025
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
# Half-days close at 1pm ET (Black Friday, Christmas Eve, July 3 some years)
NYSE_HOLIDAYS_HALF = {
    "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
    "2027-11-26",
}


def is_market_open():
    """Return True if NYSE regular session is currently open in America/New_York.
    Handles DST automatically via zoneinfo; respects full holidays + half-days."""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # Conservative fallback: assume closed if zoneinfo isn't available
        return False
    if et.weekday() >= 5:                                # Saturday/Sunday
        return False
    date_str = et.strftime("%Y-%m-%d")
    if date_str in NYSE_HOLIDAYS_FULL:                    # full close
        return False
    open_mins  = 9 * 60 + 30                              # 09:30 ET
    close_mins = 13 * 60 if date_str in NYSE_HOLIDAYS_HALF else 16 * 60   # half-day: 13:00 ET
    cur = et.hour * 60 + et.minute
    return open_mins <= cur <= close_mins


def is_trading_day(d):
    return d.weekday() < 5 and d.strftime("%Y-%m-%d") not in NYSE_HOLIDAYS_FULL


def previous_trading_day(d):
    cur = d - timedelta(days=1)
    while not is_trading_day(cur):
        cur -= timedelta(days=1)
    return cur


def latest_completed_trading_day():
    try:
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return previous_trading_day(datetime.utcnow().date())
    today = et.date()
    close_mins = 13 * 60 if today.strftime("%Y-%m-%d") in NYSE_HOLIDAYS_HALF else 16 * 60
    cur = et.hour * 60 + et.minute
    if is_trading_day(today) and cur >= close_mins:
        return today
    return previous_trading_day(today)


def completed_trading_days_since(d):
    latest = latest_completed_trading_day()
    if d is None or d >= latest:
        return 0
    cur = d + timedelta(days=1)
    days = 0
    while cur <= latest:
        if is_trading_day(cur):
            days += 1
        cur += timedelta(days=1)
    return days


def in_new_buy_window():
    """#3 Round-5: True only inside the calmer mid-session window for NEW buys.
    Skips the volatile first 15 min (09:30-09:45) and last 30 min (15:30-16:00).
    Half-days (close 13:00) clamp the late edge to 12:45. Sells are NOT gated by
    this — stops/trails must run at open and close."""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return False
    if et.weekday() >= 5:
        return False
    date_str = et.strftime("%Y-%m-%d")
    if date_str in NYSE_HOLIDAYS_FULL:
        return False
    cur = et.hour * 60 + et.minute
    start = 9 * 60 + 30                                   # 09:30 ET
    if date_str in NYSE_HOLIDAYS_HALF:
        end = 13 * 60                                    # 13:00 ET half-day close
    else:
        end = 16 * 60                                    # 16:00 ET close
    return start <= cur <= end


def _fmt_times(ts=None):
    """Return (ET_string, SGT_string) for a Unix timestamp."""
    import time
    if ts is None:
        ts = time.time()
    try:
        from zoneinfo import ZoneInfo
        et  = datetime.fromtimestamp(ts, ZoneInfo("America/New_York"))
        sgt = datetime.fromtimestamp(ts, ZoneInfo("Asia/Singapore"))
        return et.strftime("%m/%d %H:%M ET"), sgt.strftime("%m/%d %H:%M SGT")
    except Exception:
        et  = datetime.utcfromtimestamp(ts - 4 * 3600).strftime("%m/%d %H:%M ET")
        sgt = datetime.utcfromtimestamp(ts + 8 * 3600).strftime("%m/%d %H:%M SGT")
        return et, sgt
