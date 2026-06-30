"""Pure exit-math helpers — the single source of truth for how positions are exited.

Shared by the live bot (`trading.bot` SELL pass) and the walk-forward backtest
(`trading.backtest`) so both price exits with IDENTICAL rules and can't drift apart.
No Flask, no network, no state — numbers in, numbers out, fully unit-testable.

What lives here:
  - round_trip_cost_pct   : commission (buy+sell) as a % of a position's value
  - breakeven_lock_pct    : cost-aware ratchet floor (A1 — replaces the old +0.1% strangle)
  - dynamic_stop_pct      : ATR-scaled hard stop (median-of-trend aware)
  - dynamic_trail_width   : ATR-scaled trailing-stop width (intraday → daily fallback, A5)
"""


def round_trip_cost_pct(pos_val, commission):
    """Round-trip commission (one buy + one sell) as a percent of position value.

    A $100 position at $0.99/trade -> 2*0.99/100*100 = 1.98%. This is the floor a
    winning exit must clear to be net-positive — the basis for the cost-aware ratchet
    and the minimum-position-size rule.
    """
    if not pos_val or pos_val <= 0:
        return 0.0
    return 100.0 * 2.0 * commission / pos_val


def breakeven_lock_pct(peak_pnl_pct, rt_cost_pct, *, min_arm=3.0, arm_cost_mult=3.0,
                       lock_fraction=0.5, cost_buffer=0.3):
    """Cost-aware ratchet floor (A1). Returns the positive stop level (% gain) to lock
    in, or None if the position hasn't peaked enough to arm the ratchet.

    Replaces the old fixed `max(dynamic_stop, 0.1)` breakeven, which sold any position
    that ever touched +1.5% the instant it dipped to +0.1% — a NET LOSS after the
    round-trip commission on any position under ~$2k, and a hard cap on every winner.

    Rules:
      - Arm only once peak P&L >= max(min_arm, arm_cost_mult * rt_cost_pct): a real move
        that clears several multiples of the round-trip cost (tiny positions, which have
        a large rt_cost_pct, therefore arm later — exactly when they should).
      - Once armed, lock the GREATER of (rt_cost_pct + cost_buffer) and
        (lock_fraction * peak). The lock ratchets UP as the peak grows and is always
        net-positive after commissions, so we never sell green at a loss while still
        letting the position keep running on the ATR trail above the lock.
    """
    arm_at = max(min_arm, arm_cost_mult * rt_cost_pct)
    if peak_pnl_pct < arm_at:
        return None
    return max(rt_cost_pct + cost_buffer, peak_pnl_pct * lock_fraction)


def dynamic_stop_pct(eff_atr_for_stop, regime_stop, *, floor=-15.0, ceiling=-1.5,
                     atr_mult=2.0):
    """ATR-scaled hard stop, as a negative % from avg cost.

    `eff_atr_for_stop` is the max of the position's current ATR% and its
    median-of-trend ATR% (the wider, noise-tolerant value). Falls back to the
    regime-static stop when no ATR is available. Clamped to [floor, ceiling].
    """
    if eff_atr_for_stop and eff_atr_for_stop > 0:
        return max(floor, min(ceiling, -(eff_atr_for_stop * atr_mult)))
    return regime_stop


def dynamic_trail_width(intra_atr, daily_atr, regime_static, *, lo=1.5, hi=6.0,
                        intra_mult=2.0, daily_mult=1.0, daily_lo=2.0, daily_hi=8.0):
    """Trailing-stop width (positive %). Returns (width, source_label).

    Prefer 2× intraday ATR (live, tight). When intraday ATR is unavailable — it is
    ALWAYS 0 on PythonAnywhere, where yfinance intraday is blocked (A5) — fall back to
    daily ATR so the trail stays volatility-scaled instead of collapsing to a flat
    regime constant. Final fallback: the regime-static width.
    """
    if intra_atr and intra_atr > 0:
        w = max(lo, min(hi, intra_atr * intra_mult))
        return w, f"2×intra-ATR ({intra_atr:.2f}%) → {w:.1f}%"
    if daily_atr and daily_atr > 0:
        w = max(daily_lo, min(daily_hi, daily_atr * daily_mult))
        return w, f"daily-ATR ({daily_atr:.2f}%) → {w:.1f}%"
    return regime_static, f"regime-static {regime_static:.1f}%"
