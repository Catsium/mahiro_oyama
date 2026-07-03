"""Signal aggregation + attribution multipliers.

get_recommendation builds the categorized vote-weighted recommendation. It does
NOT itself touch the network — it consumes ctx (from market.history), sent (from
market.sentiment.get_news), and per-category external blocks (analyst, insider,
regime). Pure transformation function.

Legacy `_attribute_outcome` remains for archived diagnostics only. Live learning
comes from trading.attribution forward-return buckets.
"""
import time
from datetime import datetime

from utils.storage import load_bot
from trading.config import DEFAULT_CONFIG

# Re-export score_headline so callers can `from trading.signals import score_headline`
from market.sentiment import score_headline  # noqa: F401


def load_signal_weights():
    """Return live V2 attribution weights. Legacy `signal_weights` are archived."""
    try:
        b = load_bot()
        from trading.attribution import attribution_signal_weights
        return attribution_signal_weights(b)
    except Exception:
        return {}


def _regime_key(regime):
    """Normalize a regime dict / string to one of 'bull' / 'bear' / 'neutral'.
    Used for regime-specific weight lookup (algorithm #7)."""
    if isinstance(regime, dict):
        r = (regime.get("regime") or "neutral").lower()
    else:
        r = str(regime or "neutral").lower()
    if "bull" in r: return "bull"
    if "bear" in r: return "bear"
    return "neutral"


def classify_display_signal(raw_cls, confidence):
    """Human label for signal strength; execution still uses raw cls + gates."""
    cls = str(raw_cls or "hold").lower()
    try:
        conf = float(confidence or 0)
    except Exception:
        conf = 0.0
    if cls in ("buy", "strong-buy"):
        if conf < 40:
            return "BULLISH_LEAN"
        if conf < 70:
            return "BUY_CANDIDATE"
        return "STRONG_BUY_CANDIDATE"
    if cls == "sell":
        return "SELL"
    if cls == "strong-sell":
        return "STRONG_SELL"
    return "HOLD"


def _with_confidence_metadata(rec):
    conf = rec.get("confidence", 0)
    cls = str(rec.get("cls") or "hold").lower()
    reason = f"raw_class_{cls}"
    rec.setdefault("score_total", rec.get("score", 0))
    rec.setdefault("confidence_before_penalties", conf)
    rec.setdefault("confidence_after_penalties", conf)
    rec.setdefault("confidence_before_floor", conf)
    rec.setdefault("confidence_final", conf)
    rec.setdefault("confidence_floor_candidate", False)
    rec.setdefault("confidence_floor_applied", False)
    rec.setdefault("confidence_floor_reason", reason)
    rec.setdefault("confidence_floor_blockers", [reason])
    rec.setdefault("confidence_penalties", [])
    return rec


def _decay_multiplier(age_hours, half_life_hours):
    """Exponential decay: signal value is multiplied by 0.5 every half_life_hours."""
    if age_hours <= 0:
        return 1.0
    return 0.5 ** (age_hours / max(half_life_hours, 0.1))


def _present(ctx, key):
    if not ctx or key not in ctx:
        return False
    value = ctx.get(key)
    if value is None:
        return False
    try:
        return not bool(value != value)
    except Exception:
        return True


def get_recommendation(sent, ctx, regime=None, earnings=None, analyst=None, insider=None,
                       news_age_hours=None, news_articles=None, pure_technical=False,
                       weights=None, allow_live_risk=True, config=None):
    """Categorized signal aggregator with cross-category confirmation.
      TREND          — price vs MA30, MA7/MA30 cross, ADX, weekly posture (#1.4)
      MOMENTUM       — RSI+MACD primary, BB/Stoch only confirm/disagree
      VOLUME         — vol ratio, MFI, VWAP
      DIP            — distance from 20d high
      REL_STR        — sector-relative strength (#1.3)
      NEWS           — news sentiment (decay 6h)
      ANALYST        — analyst consensus (decay 14d)
      INSIDER        — insider sentiment (decay 21d, halved in #1.5)
      REGIME         — broad market direction (SPY)

    NOTE: the POLITICIAN (US congress disclosure) category was removed — a 45-day
    lagged, late-filed dataset has no timing edge, and sector flow is already
    captured by REL_STR. Pure noise removal.
    """
    cfg = config or DEFAULT_CONFIG
    signal_cfg = cfg.get("signal", {})
    catalyst_cfg = cfg.get("catalyst", {})
    earnings_risk = None
    if earnings and earnings.get("soon"):
        days_until = None
        try:
            ed = datetime.strptime(str(earnings.get("date")), "%Y-%m-%d").date()
            days_until = (ed - datetime.now().date()).days
        except Exception:
            pass
        earnings_risk = {
            "soon": True,
            "date": earnings.get("date"),
            "days_until": days_until,
            "policy": "warning_and_mild_size_penalty" if days_until is None or days_until <= 1 else "warning_only",
            "hard_block_enabled": bool(catalyst_cfg.get("earnings_hard_block_enabled", False)),
        }

    # `weights` lets the backtest pass frozen/trained weights (or {} for defaults)
    # without touching the live bot_state.json; live callers leave it None.
    weights = load_signal_weights() if weights is None else weights
    # Round-4 algorithm #7: prefer regime-specific weight (e.g. "bull:momentum")
    # before falling back to plain "momentum" before defaulting to 1.0
    _cur_regime = _regime_key(regime) if regime else "neutral"
    def w(cat):
        return weights.get(f"{_cur_regime}:{cat}",
               weights.get(cat, 1.0))

    # Derive news age from freshest article if not provided. Prefer the epoch
    # (pub_ts, UTC seconds) for true hour resolution; fall back to the date string
    # (day resolution) only when the epoch is missing. time.time() is UTC so the
    # subtraction is timezone-safe.
    if news_articles and news_age_hours is None:
        today = datetime.now().date()
        now_ts = time.time()
        ages = []
        for a in news_articles:
            pub_ts = a.get("pub_ts")
            if pub_ts:
                ages.append(max(0.0, (now_ts - float(pub_ts)) / 3600.0))
                continue
            pub = (a.get("pub") or "")[:10]
            if pub:
                try:
                    d = datetime.strptime(pub, "%Y-%m-%d").date()
                    ages.append(max(0.0, (today - d).total_seconds() / 3600.0))
                except Exception:
                    pass
        if ages:
            news_age_hours = min(ages)

    # Pure-technical mode (backtest path)
    if pure_technical:
        sent = 0.0; news_articles = None; news_age_hours = None
        analyst = None; insider = None

    reasons = []
    if earnings_risk:
        reasons.append(
            f"Earnings risk {earnings_risk.get('date') or 'soon'} - warning only for paper testing"
        )
    cat_signals = {"trend": [], "momentum": [], "volume": [], "dip": [],
                   "news": [], "analyst": [], "insider": [],
                   "rel_str": [], "regime": []}
    is_dip_flag = False
    momentum_uncertainty = 0.0

    if ctx:
        cur, ma7, ma30, rsi, wc = ctx["current"], ctx["ma7"], ctx["ma30"], ctx["rsi"], ctx["week_chg_pct"]

        # ── TREND ──────────────────────────────────────────────────────
        gap_pct = (cur - ma30) / ma30 * 100 if ma30 else 0
        if   gap_pct >  3:  s = 2;  reasons.append(f"Price ${cur:.2f} is {gap_pct:+.1f}% above 30-day MA — strong uptrend")
        elif gap_pct >  0:  s = 1;  reasons.append(f"Price ${cur:.2f} is {gap_pct:+.1f}% above 30-day MA — uptrend")
        elif gap_pct > -3:  s = -1; reasons.append(f"Price ${cur:.2f} is {gap_pct:.1f}% below 30-day MA — mild downtrend")
        else:               s = -2; reasons.append(f"Price ${cur:.2f} is {gap_pct:.1f}% below 30-day MA — strong downtrend")
        cat_signals["trend"].append(s)

        # Round-7 de-dup #1: cross_pct (ma7 vs ma30) almost always agrees with
        # gap_pct (price vs ma30) — voting both double-counts "trend up". Only emit
        # a cross VOTE when it DIVERGES from gap_pct (the rare crossover/reversal
        # moment = real new info); otherwise it's a reason string only.
        cross_pct = (ma7 - ma30) / ma30 * 100 if ma30 else 0
        gap_dir = 1 if gap_pct > 0 else (-1 if gap_pct < 0 else 0)
        cross_dir = 1 if cross_pct > 0 else (-1 if cross_pct < 0 else 0)
        if gap_dir != 0 and cross_dir != 0 and gap_dir != cross_dir:
            # Divergence: price above MA30 but MA7 below (or vice versa) — dampen.
            cat_signals["trend"].append(cross_dir)
            reasons.append(f"⚠️ MA divergence: 7-day MA {cross_pct:+.1f}% vs 30-day while price {gap_pct:+.1f}% — possible turn")
        else:
            reasons.append(f"7-day MA {cross_pct:+.1f}% vs 30-day (confirms price position)")

        if "adx" in ctx and "mom_30d_pct" in ctx:
            adx = ctx["adx"]; m = ctx["mom_30d_pct"]
            if   adx >= 35 and m > 0: cat_signals["trend"].append(1);  reasons.append(f"ADX {adx:.0f} — very strong uptrend")
            elif adx >= 35 and m < 0: cat_signals["trend"].append(-1); reasons.append(f"ADX {adx:.0f} — very strong downtrend")
            elif adx < 15:            cat_signals["trend"].append(0);  reasons.append(f"ADX {adx:.0f} — choppy/range-bound")

        # #1.4 Weekly higher-timeframe confirmation
        if "weekly_trend_up" in ctx:
            wtu = ctx["weekly_trend_up"]
            wmb = ctx.get("weekly_macd_bullish", False)
            wrs = ctx.get("weekly_rsi", 50)
            if wtu and wmb:
                cat_signals["trend"].append(1)
                reasons.append(f"Weekly trend up + MACD bullish (RSI {wrs:.0f}) — confirms daily")
            elif (not wtu) and (not wmb):
                cat_signals["trend"].append(-2)
                reasons.append(f"⚠️ Weekly trend DOWN + MACD bearish (RSI {wrs:.0f}) — daily signal fights the tape")
            elif wtu and not wmb:
                cat_signals["trend"].append(0)
                reasons.append(f"Weekly trend up but MACD weakening (RSI {wrs:.0f}) — mixed higher TF")
            else:
                cat_signals["trend"].append(0)
                reasons.append(f"Weekly trend down but MACD turning up (RSI {wrs:.0f}) — possible base")

        # ── MOMENTUM (RSI+MACD primary, BB/Stoch confirm) ─────────────
        rsi_sig = macd_sig = bb_sig = stoch_sig = 0
        if   rsi <= 25: rsi_sig = 2;  reasons.append(f"RSI {rsi} — extreme oversold")
        elif rsi <= 35: rsi_sig = 1;  reasons.append(f"RSI {rsi} — oversold")
        elif rsi <= 65: rsi_sig = 0
        elif rsi <= 75: rsi_sig = -1; reasons.append(f"RSI {rsi} — overbought")
        else:           rsi_sig = -2; reasons.append(f"RSI {rsi} — extreme overbought")

        macd_present = "macd_hist" in ctx
        if macd_present:
            mh, mhp = ctx["macd_hist"], ctx["macd_hist_prev"]
            if   mh > 0 and mhp <= 0: macd_sig = 2;  reasons.append(f"MACD just crossed bullish (hist {mh:+.3f})")
            elif mh < 0 and mhp >= 0: macd_sig = -2; reasons.append(f"MACD just crossed bearish (hist {mh:+.3f})")
            elif mh > 0 and mh > mhp: macd_sig = 1;  reasons.append("MACD bullish & strengthening")
            elif mh > 0:              macd_sig = 0
            elif mh < 0 and mh < mhp: macd_sig = -1; reasons.append("MACD bearish & weakening")
            else:                     macd_sig = 0

        if "bb_pos" in ctx:
            bp = ctx["bb_pos"]
            if   bp <= 0.05: bb_sig = 2
            elif bp <= 0.20: bb_sig = 1
            elif bp >= 0.95: bb_sig = -2
            elif bp >= 0.80: bb_sig = -1
            else:            bb_sig = 0

        if "stoch_k" in ctx and "stoch_d" in ctx:
            k, dd = ctx["stoch_k"], ctx["stoch_d"]
            if   k < 20 and k > dd: stoch_sig = 2
            elif k < 25:            stoch_sig = 1
            elif k > 80 and k < dd: stoch_sig = -2
            elif k > 75:            stoch_sig = -1
            else:                   stoch_sig = 0

        primary_sigs = [rsi_sig]
        if macd_present: primary_sigs.append(macd_sig)
        primary_vote = sum(primary_sigs) / len(primary_sigs)
        cat_signals["momentum"].append(primary_vote)
        reasons.append(f"Momentum primary: RSI={rsi_sig:+d}{', MACD=' + str(macd_sig) if macd_present else ''} → {primary_vote:+.2f}")

        all_osc = [rsi_sig, macd_sig, bb_sig, stoch_sig]
        non_zero = [s for s in all_osc if s != 0]
        if len(non_zero) >= 2:
            signs = [1 if s > 0 else -1 for s in non_zero]
            alignment = abs(sum(signs)) / len(signs)
            momentum_uncertainty = round(1.0 - alignment, 2)
            if momentum_uncertainty >= 0.5:
                reasons.append(f"⚠️ Oscillators split (BB={bb_sig:+d}, Stoch={stoch_sig:+d} vs primary) — uncertainty={momentum_uncertainty:.2f}")
        else:
            momentum_uncertainty = 0.0

        # Round-7 de-dup #2: week_chg_pct tracks the same recent-momentum axis as
        # MACD — voting both double-counts momentum. Keep it as a reason only; its
        # gap-up-risk role already feeds the trend-exhaustion confidence penalty.
        if   wc >  6:   reasons.append(f"+{wc:.1f}% week — strong upward momentum (context, not a vote)")
        elif wc >  2:   reasons.append(f"+{wc:.1f}% week — steady gains")
        elif wc < -6:   reasons.append(f"{wc:.1f}% week — sharp decline (context)")
        elif wc < -2:   reasons.append(f"{wc:.1f}% week — weakening")

        # ── VOLUME ─────────────────────────────────────────────────────
        if "vol_ratio" in ctx and "mom_30d_pct" in ctx:
            vr = ctx["vol_ratio"]; m = ctx["mom_30d_pct"]
            if   vr >= 1.5 and m > 0:  cat_signals["volume"].append(1);  reasons.append(f"Volume {vr:.1f}× normal on uptrend — strong buying conviction")
            elif vr >= 1.5 and m < 0:  cat_signals["volume"].append(-1); reasons.append(f"Volume {vr:.1f}× normal on downtrend — heavy distribution")
            elif vr <= 0.6 and m > 0:  cat_signals["volume"].append(-1); reasons.append(f"Uptrend on thin volume ({vr:.1f}×) — weak conviction")

        if "mfi" in ctx:
            m = ctx["mfi"]
            if   m <= 20: cat_signals["volume"].append(2);  reasons.append(f"MFI {m:.0f} — capital flowing in at oversold levels")
            elif m <= 30: cat_signals["volume"].append(1);  reasons.append(f"MFI {m:.0f} — money flow oversold")
            elif m >= 80: cat_signals["volume"].append(-2); reasons.append(f"MFI {m:.0f} — capital distribution at overbought levels")
            elif m >= 70: cat_signals["volume"].append(-1); reasons.append(f"MFI {m:.0f} — money flow overbought")

        # Round-7 de-dup #3 (+ bug 7): the daily-bar "vwap_dist_pct" is a 20-day
        # VWMA distance, NOT intraday institutional VWAP — the old label was
        # misleading and it overlapped vol_ratio. Real intraday VWMA distance lives
        # in get_intraday_context (used for trail sizing). Dropped as a daily vote.

        # ── DIP ────────────────────────────────────────────────────────
        if ctx.get("is_dip"):
            is_dip_flag = True
            d = ctx.get("dist_from_high_pct", 0)
            cat_signals["dip"].append(2)
            reasons.append(f"💰 Dip detected — price {d:+.1f}% below recent 20d high ${ctx.get('recent_high', 0):.2f}, RSI<45 with MACD turning up")
        elif "dist_from_high_pct" in ctx and ctx["dist_from_high_pct"] <= -20:
            cat_signals["dip"].append(-1)
            reasons.append(f"Price {ctx['dist_from_high_pct']:+.1f}% below recent 20d high — possible falling-knife")

        # Round-4 algorithm #5: Trend exhaustion. 5+ consecutive up days WITH
        # price >5% above MA30 = parabolic/late-trend setup. Vote against the
        # dip category (which gates buy aggressiveness) and surface it in the
        # reasons so the user sees why a hot stock isn't getting full size.
        consec_up = ctx.get("consec_up_days", 0)
        if consec_up >= 5 and gap_pct > 5:
            cat_signals["dip"].append(-2)
            reasons.append(f"⚠️ Trend exhaustion: {consec_up} consecutive up days + {gap_pct:+.0f}% above MA30 (parabolic)")
        elif consec_up >= 5:
            reasons.append(f"Note: {consec_up} consecutive up days — momentum extended")

        # ── REL_STR (#1.3) ─────────────────────────────────────────────
        if "rel_str_pct" in ctx:
            rsp = ctx["rel_str_pct"]
            etf = ctx.get("sector_etf", "?")
            if   rsp >=  5: cat_signals["rel_str"].append(2);  reasons.append(f"Strong rel-strength: {rsp:+.1f}% vs {etf} over 20d")
            elif rsp >=  2: cat_signals["rel_str"].append(1);  reasons.append(f"Outperforming sector: {rsp:+.1f}% vs {etf} over 20d")
            elif rsp <= -5: cat_signals["rel_str"].append(-2); reasons.append(f"⚠️ Severe lag vs sector: {rsp:+.1f}% vs {etf} over 20d")
            elif rsp <= -2: cat_signals["rel_str"].append(-1); reasons.append(f"Lagging sector: {rsp:+.1f}% vs {etf} over 20d")

    # ── NEWS (decay half-life 6h) ───────────────────────────────────────
    has_news = (news_age_hours is not None) or (sent != 0.0)
    if has_news:
        news_decay = _decay_multiplier(news_age_hours if news_age_hours is not None else 12, 6)
        sent_decayed = sent * news_decay
        if   sent_decayed >  0.5:  cat_signals["news"].append(2);  reasons.append(f"Very bullish news ({sent:+.2f}, decay×{news_decay:.2f})")
        elif sent_decayed >  0.2:  cat_signals["news"].append(1);  reasons.append(f"Bullish news ({sent:+.2f})")
        elif sent_decayed > -0.2:  cat_signals["news"].append(0);  reasons.append(f"Neutral news coverage ({sent:+.2f})")
        elif sent_decayed > -0.5:  cat_signals["news"].append(-1); reasons.append(f"Bearish news ({sent:+.2f})")
        else:                      cat_signals["news"].append(-2); reasons.append(f"Very bearish news ({sent:+.2f})")

    # ── REGIME ──────────────────────────────────────────────────────────
    if regime:
        rg = regime.get("regime")
        m = regime.get("spy_mom_30d") or 0
        if   rg == "bull": cat_signals["regime"].append(1);  reasons.append(f"Broad market bullish (SPY {m:+.1f}% / 30d) — tailwind")
        elif rg == "bear": cat_signals["regime"].append(-1); reasons.append(f"Broad market bearish (SPY {m:+.1f}% / 30d) — headwind")

    # ── ANALYST (decay 14d) ─────────────────────────────────────────────
    if analyst and analyst.get("total", 0) > 0:
        age_h = analyst.get("age_hours")
        analyst_decay = 0.5 if age_h is None else _decay_multiplier(age_h, 14 * 24)
        net = analyst["net"] * analyst_decay
        if   net >=  0.5: cat_signals["analyst"].append(2);  reasons.append(f"Strong analyst BUY ({analyst['buy']}B / {analyst['hold']}H / {analyst['sell']}S, age {age_h/24 if age_h else '?'}d, decay×{analyst_decay:.2f})")
        elif net >=  0.2: cat_signals["analyst"].append(1);  reasons.append(f"Analyst lean bullish (decay×{analyst_decay:.2f})")
        elif net <= -0.5: cat_signals["analyst"].append(-2); reasons.append(f"Strong analyst SELL (decay×{analyst_decay:.2f})")
        elif net <= -0.2: cat_signals["analyst"].append(-1); reasons.append(f"Analyst lean bearish (decay×{analyst_decay:.2f})")

    # ── INSIDER (decay 21d, halved in #1.5) ────────────────────────────
    if insider and insider.get("samples", 0) > 0:
        age_h = insider.get("age_hours")
        insider_decay = 0.5 if age_h is None else _decay_multiplier(age_h, 21 * 24)
        s = insider["sentiment"] * insider_decay
        if   s >=  0.4:  cat_signals["insider"].append(2);  reasons.append(f"Heavy insider BUYING (MSPR {insider['sentiment']:+.2f}, age {age_h/24 if age_h else '?'}d, decay×{insider_decay:.2f})")
        elif s >=  0.15: cat_signals["insider"].append(1);  reasons.append(f"Insider buying tilt (decay×{insider_decay:.2f})")
        elif s <= -0.4:  cat_signals["insider"].append(-2); reasons.append(f"Heavy insider SELLING (decay×{insider_decay:.2f})")
        elif s <= -0.15: cat_signals["insider"].append(-1); reasons.append(f"Insider selling tilt (decay×{insider_decay:.2f})")

    # ── Per-category votes ─────────────────────────────────────────────
    cat_votes = {}
    for cat, vals in cat_signals.items():
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        cat_votes[cat] = round(avg * w(cat), 3)

    cats_pos = sum(1 for v in cat_votes.values() if v >= 0.5)
    cats_neg = sum(1 for v in cat_votes.values() if v <= -0.5)
    tot = round(sum(cat_votes.values()), 3)

    # Round-5 #1: regime-conditional thresholds. Bull = easier to buy / harder
    # to sell; bear = harder to buy / easier to sell; neutral = unchanged.
    rk = _regime_key(regime) if regime else "neutral"
    if rk == "bull":
        sb_t, sb_c, b_t, b_c = 3.5, 2, 1.25, 2     # buy side looser
        ss_t, ss_c, s_t, s_c = -4.5, 3, -1.75, 2   # sell side tighter
    elif rk == "bear":
        sb_t, sb_c, b_t, b_c = 5.0, 4, 2.0, 3      # buy side tighter
        ss_t, ss_c, s_t, s_c = -3.5, 2, -1.25, 2   # sell side looser
    else:
        sb_t, sb_c, b_t, b_c = 4.0, 3, 1.5, 2      # neutral = current
        ss_t, ss_c, s_t, s_c = -4.0, 3, -1.5, 2

    if   tot >= sb_t and cats_pos >= sb_c: sig, cls = "STRONG BUY",  "strong-buy"
    elif tot >= b_t  and cats_pos >= b_c:  sig, cls = "BUY",         "buy"
    elif tot <= ss_t and cats_neg >= ss_c: sig, cls = "STRONG SELL", "strong-sell"
    elif tot <= s_t  and cats_neg >= s_c:  sig, cls = "SELL",        "sell"
    else:                                   sig, cls = "HOLD",        "hold"

    # ── Dynamic signal denominator ────────────────────────────────────
    # Round-7: base is now 2 (gap-vs-MA30 + momentum-primary are the only always-on
    # votes). Was 4 when MA-cross and week-change also always voted; both are now
    # conditional / reason-only (de-dup #1, #2), so the denominator drops to match.
    baseline_tech = 0
    data_quality_present_n = 0
    data_quality_expected_n = 0
    data_quality_missing_fields = []
    if ctx:
        data_groups = [
            ("daily_history", ("current", "ma7", "ma30", "rsi", "week_chg_pct")),
            ("adx_or_mom_30d_pct", ("adx", "mom_30d_pct")),
            ("vol_ratio_or_mom_30d_pct", ("vol_ratio", "mom_30d_pct")),
            ("mfi", ("mfi",)),
            ("weekly_trend_up", ("weekly_trend_up",)),
            ("rel_str_pct", ("rel_str_pct",)),
        ]
        for label, keys in data_groups:
            data_quality_expected_n += 1
            if all(_present(ctx, key) for key in keys):
                data_quality_present_n += 1
            else:
                data_quality_missing_fields.append(label)
        baseline_tech += 2
        if _present(ctx, "adx") and _present(ctx, "mom_30d_pct"):
            baseline_tech += 1
        if _present(ctx, "vol_ratio") and _present(ctx, "mom_30d_pct"):
            baseline_tech += 1
        if _present(ctx, "mfi"):             baseline_tech += 1
        # Round-7: vwap daily vote dropped (de-dup #3) — no longer in denominator.
        if _present(ctx, "weekly_trend_up"): baseline_tech += 1
        if _present(ctx, "rel_str_pct"):     baseline_tech += 1
    else:
        data_quality_expected_n = 1
        data_quality_missing_fields.append("daily_history")
    expected_n = max(1, baseline_tech)
    if has_news:
        expected_n += 1
        data_quality_expected_n += 1
        data_quality_present_n += 1
    if analyst and analyst.get("total", 0) > 0:
        expected_n += 1
        data_quality_expected_n += 1
        data_quality_present_n += 1
    if insider and insider.get("samples", 0) > 0:
        expected_n += 1
        data_quality_expected_n += 1
        data_quality_present_n += 1
    if regime:
        expected_n += 1
        data_quality_expected_n += 1
        data_quality_present_n += 1

    n_raw = sum(len(v) for v in cat_signals.values())
    data_quality = min(1.0, data_quality_present_n / max(1, data_quality_expected_n))

    # ── Confidence ─────────────────────────────────────────────────────
    max_plausible = 14.0
    if cls == "hold":
        n_total = max(n_raw, 1)
        bulls = sum(1 for vals in cat_signals.values() for s in vals if s > 0)
        bears = sum(1 for vals in cat_signals.values() for s in vals if s < 0)
        score_neutrality = max(0.0, 1.0 - abs(tot) / 1.5)
        disagreement = (min(bulls, bears) / n_total) if n_total else 0
        conf = round(20 + score_neutrality * 30 + data_quality * 30 - disagreement * 25)
        conf = max(10, min(75, conf))
    else:
        strength = min(1.0, abs(tot) / max_plausible)
        cat_bonus = min(1.0, max(cats_pos, cats_neg) / 4)
        uncertainty_penalty = 1.0 - 0.5 * momentum_uncertainty
        conf = round((35 + strength * 45 + cat_bonus * 10) * data_quality * uncertainty_penalty
                     + 12 * (1 - data_quality))
        conf = max(15, min(95, conf))
    confidence_before_penalties = conf

    if not ctx:
        reasons.insert(0, "Limited data — only news sentiment available. Collecting price snapshots locally as fallback.")

    # Round-4 algorithm #2: uncertainty penalties. Markets are probabilistic;
    # a setup with strong agreement can still fail in adverse conditions.
    # These shave confidence in known-risky environments. Only applied to
    # non-HOLD signals (HOLD already has its own neutrality calc).
    penalty = 0
    penalty_notes = []
    confidence_penalties = []
    if cls != "hold":
        # Market-volatility gate: high realized vol = unstable price action. get_vix()
        # now returns SPY 20d annualized realized-vol % (yfinance ^VIX is dead on PA).
        # cached, cheap. Skipped in pure_technical (backtest) mode — it reads the LIVE
        # current vol, which would contaminate historical bars with today's regime.
        if not pure_technical and allow_live_risk:
            try:
                from trading.risk import get_vix
                vix = get_vix() or {}
                v = vix.get("vix", 0) or 0
                if v > 28:
                    penalty += 25
                    penalty_notes.append(f"realized-vol {v:.0f}% PANIC −25")
                    confidence_penalties.append("volatility_panic")
                elif v > 18:
                    penalty += 12
                    penalty_notes.append(f"realized-vol {v:.0f}% elevated −12")
                    confidence_penalties.append("volatility_elevated")
            except Exception:
                pass
        if ctx:
            adv = ctx.get("avg_dollar_vol_20d", 0) or 0
            atr = ctx.get("atr_pct", 0) or 0
            low_adv_warning = float(signal_cfg.get("low_avg_dollar_volume_warning", 1_000_000))
            high_atr_warning = float(signal_cfg.get("high_atr_warning_pct", 10.0))
            if 0 < adv < low_adv_warning:
                penalty_notes.append(f"low liquidity ADV ${adv/1e6:.1f}M warning")
                confidence_penalties.append("low_liquidity_warning")
            if atr > high_atr_warning:
                penalty_notes.append(f"high ATR {atr:.1f}% warning")
                confidence_penalties.append("high_atr_warning")
            # Higher-timeframe conflict: daily bullish but weekly bearish
            if tot > 0 and ctx.get("weekly_trend_up") is False:
                penalty += 15
                penalty_notes.append("daily↑ vs weekly↓ −15")
                confidence_penalties.append("weekly_trend_conflict")
            # Gap-up extension: consecutive up days × MA30 gap proxy
            consec = ctx.get("consec_up_days", 0)
            gap = (ctx.get("current", 0) - ctx.get("ma30", 0)) / ctx.get("ma30", 1) * 100 if ctx.get("ma30") else 0
            if consec >= 5 and gap > 5 and tot > 0:
                penalty += 15
                penalty_notes.append(f"gap-up extended ({consec} green, {gap:+.0f}% vs MA30) −15")
                confidence_penalties.append("gap_extension")
    if penalty:
        conf = max(15, conf - penalty)
        reasons.append(f"⚠️ Confidence penalties: {' · '.join(penalty_notes)} → final {conf}%")
    confidence_after_penalties = conf

    floor_blockers = []
    if cls in ("buy", "strong-buy"):
        if tot < b_t:
            floor_blockers.append("score_below_buy_threshold")
        if cats_pos < b_c:
            floor_blockers.append("insufficient_positive_categories")
        if not ctx:
            floor_blockers.append("missing_history")
        if data_quality < 0.60:
            floor_blockers.append("low_data_quality")
        if cats_neg > 1:
            floor_blockers.append("high_negative_category_count")
        for code in ("volatility_panic", "low_liquidity", "high_atr"):
            if code in confidence_penalties:
                floor_blockers.append(code)
    confidence_before_floor = conf
    confidence_floor_candidate = cls in ("buy", "strong-buy") and not floor_blockers

    # #7: sizing_confidence - floor strong signals to 70 for the sizing path only,
    # V1: post-penalty labels must match usable confidence.
    if cls in ("strong-buy", "strong-sell"):
        sizing_confidence = max(conf, 70)
    else:
        sizing_confidence = conf

    return {
        "signal": sig, "cls": cls, "confidence": conf, "score": tot,
        "sizing_confidence": sizing_confidence,
        "max_score": max_plausible, "reasons": reasons,
        "data_quality": data_quality, "is_dip": is_dip_flag,
        "categories": cat_votes, "cats_pos": cats_pos, "cats_neg": cats_neg,
        "expected_n": expected_n, "n_raw": n_raw,
        "data_quality_actual_n": data_quality_present_n,
        "data_quality_expected_n": data_quality_expected_n,
        "data_quality_missing_fields": data_quality_missing_fields,
        "penalty": penalty, "penalty_notes": penalty_notes,
        "score_total": tot,
        "confidence_before_penalties": confidence_before_penalties,
        "confidence_after_penalties": confidence_after_penalties,
        "confidence_before_floor": confidence_before_floor,
        "confidence_final": conf,
        "confidence_floor_candidate": confidence_floor_candidate,
        "confidence_floor_applied": False,
        "confidence_floor_reason": (
            "signal_floor_candidate" if confidence_floor_candidate
            else (floor_blockers[0] if floor_blockers else None)
        ),
        "confidence_floor_blockers": floor_blockers,
        "earnings_risk": earnings_risk or {},
        "confidence_penalties": confidence_penalties,
        "thresholds": {"regime": rk, "strong_buy_tot": sb_t, "buy_tot": b_t,
                       "strong_buy_cats": sb_c, "buy_cats": b_c,
                       "strong_sell_tot": ss_t, "sell_tot": s_t,
                       "strong_sell_cats": ss_c, "sell_cats": s_c},
    }


def _attribute_outcome(b, ticker, pnl_pct, entry_reasons, entry_regime=None):
    """Update per-category signal weights from a closed trade's P&L.

    Round-4 algorithm #7: now buckets by (regime, category). The entry's market
    regime is read from entry_reasons (a dict that includes the categories at
    entry) OR from the explicit `entry_regime` arg. We update BOTH the
    regime-specific bucket AND the aggregate plain-category bucket, so:
      • signal_weights["bull:trend"] = regime-specific weight (Round-4 lookup)
      • signal_weights["trend"]      = aggregate, used as fallback if no regime data
    """
    attribution = b.setdefault("signal_attribution", {})
    weights = b.setdefault("signal_weights", {})
    cats_at_entry = entry_reasons.get("categories", {}) if isinstance(entry_reasons, dict) else {}
    if not cats_at_entry:
        return

    # Determine entry regime — passed in (from holdings entry_snapshot) OR via
    # entry_reasons dict OR default neutral
    reg = entry_regime or (entry_reasons.get("regime") if isinstance(entry_reasons, dict) else None)
    regime_norm = _regime_key(reg)

    try:
        from scipy.stats import beta as _beta_dist
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    def _update_bucket(key, vote_magnitude, won):
        bucket = attribution.setdefault(key, {"wins": 0.0, "losses": 0.0,
                                              "pnl_sum": 0.0, "weighted_n": 0.0})
        # A6: seed α/β from the PRE-increment wins/losses (legacy back-compat) BEFORE
        # applying this trade. The old order seeded from post-increment counts and then
        # added the magnitude again, double-counting the first update per bucket.
        if "alpha" not in bucket or "beta" not in bucket:
            bucket["alpha"] = round(bucket.get("wins", 0) + 1.0, 4)
            bucket["beta"]  = round(bucket.get("losses", 0) + 1.0, 4)
        if won:
            bucket["wins"]  = bucket.get("wins", 0) + vote_magnitude
            bucket["alpha"] = round(bucket["alpha"] + vote_magnitude, 4)
        else:
            bucket["losses"] = bucket.get("losses", 0) + vote_magnitude
            bucket["beta"]   = round(bucket["beta"] + vote_magnitude, 4)
        bucket["pnl_sum"] = round(bucket.get("pnl_sum", 0) + pnl_pct * vote_magnitude, 3)
        bucket["weighted_n"] = round(bucket.get("weighted_n", 0) + vote_magnitude, 3)
        if HAVE_SCIPY:
            p_edge = 1.0 - float(_beta_dist.cdf(0.5, bucket["alpha"], bucket["beta"]))
            bucket["p_edge"] = round(p_edge, 4)
            new_w = 0.5 + p_edge
            weights[key] = round(max(0.5, min(1.5, new_w)), 3)
        else:
            if bucket["weighted_n"] >= 30:
                avg_pnl = bucket["pnl_sum"] / bucket["weighted_n"]
                raw_w = 1.0 + max(-0.5, min(0.5, avg_pnl / 10.0))
                old_w = weights.get(key, 1.0)
                weights[key] = round(old_w * 0.85 + raw_w * 0.15, 3)

    won = pnl_pct > 0
    for cat, vote in cats_at_entry.items():
        if vote == 0:
            continue
        magnitude = min(1.0, abs(vote) / 2.0)
        # Update BOTH the regime-specific bucket AND the aggregate
        _update_bucket(f"{regime_norm}:{cat}", magnitude, won)
        _update_bucket(cat, magnitude, won)
