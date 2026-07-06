"""Position-sizing primitives: cost model, EV ranking, and partial-trim helper.

Round-5 cost model: dropped Almgren-Chriss bps slippage from LIVE trading.
At $10k paper scale it was invisible noise AND it made the equity chart drop
more than the stated commission on every buy (markup baked into effective price).
Now: flat $0.99 commission per trade, no price markup. So a buy nets exactly
−$0.99 in equity and the chart reflects it. `slippage_bps` stays defined because
the backtest endpoint (routes/api.py) still references it. The live fill path
does not mark up prices, but EV ranking still includes model slippage in
friction so tiny-edge trades stay blocked.
"""
import time

from utils.cache import cache_get, cache_set  # noqa: F401 - kept for future Kelly utils
from trading.config import DEFAULT_CONFIG
from trading.catalysts import catalyst_cluster
from trading.exits import dynamic_stop_pct
from trading.exit_ladders import normalize_cluster

# Back-compat aliases (slippage_bps still used by backtest only)
SLIPPAGE_BPS       = 5.0
SLIPPAGE_BASE_BPS  = 5.0
SLIPPAGE_IMPACT_K  = 10.0

# Flat per-trade commission — the ONLY live trading cost now.
COMMISSION_PER_TRADE = 0.99
KNIFE_CATALYST_TYPES = {
    "earnings_miss",
    "guidance_cut",
    "regulatory_risk",
    "lawsuit_investigation",
}

# Partial profit-taking (#2.4) — enabled Round-5
PARTIAL_TAKE_ENABLED  = True
PARTIAL_TAKE_PCT      = 6.0   # trim once peak P&L crosses +6%
PARTIAL_TAKE_FRACTION = 0.30  # trim 30% of remaining (audit P4: let winners run)

# V1 risk budget. Values are percent of total equity at risk per trade before
# confidence/source/Kelly modifiers.
REGIME_RISK_PCT = {
    "bull": 1.0,
    "neutral": 0.7,
    "bear": 0.4,
}
SOURCE_RISK_SCALE = {
    "watchlist": 1.0,
    "scan": 0.85,
}
FORWARD_HORIZONS_DAYS = (1, 3, 5, 10)
EDGE_HORIZON = "5d"
EDGE_MIN_SAMPLES = 8


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_regime_kind(regime):
    if isinstance(regime, dict):
        regime = regime.get("regime_effective") or regime.get("regime")
    r = str(regime or "neutral").lower()
    return r if r in REGIME_RISK_PCT else "neutral"


def entry_cluster(rec, ctx=None):
    """Small entry taxonomy used for EV learning and later cluster-specific exits."""
    rec = rec or {}
    ctx = ctx or {}
    cats = rec.get("categories") or {}
    rsi = ctx.get("rsi", 50) or 50
    mh = ctx.get("macd_hist", 0) or 0
    mhp = ctx.get("macd_hist_prev", 0) or 0
    vol_ratio = ctx.get("vol_ratio", 1.0) or 1.0
    mom_30d = ctx.get("mom_30d_pct", 0) or 0
    week_chg = ctx.get("week_chg_pct", 0) or 0
    cur = ctx.get("current", 0) or 0
    ma30 = ctx.get("ma30", 0) or 0
    reasons = " ".join(str(r) for r in rec.get("reasons", [])).lower()
    cat_cluster = catalyst_cluster(rec.get("catalyst") or {})
    if cat_cluster:
        return cat_cluster

    catalyst_vote = float(cats.get("news", 0) or 0)
    catalyst_text = any(k in reasons for k in ("news", "earnings", "guidance", "catalyst"))
    price_confirmed = (
        vol_ratio >= 1.3
        and (mom_30d >= 2.0 or week_chg >= 2.0 or (cur > 0 and ma30 > 0 and cur > ma30))
        and mh >= mhp
    )
    if (catalyst_vote > 0 or catalyst_text) and price_confirmed:
        return "news_catalyst_confirmed"

    if rec.get("is_dip") or ctx.get("is_dip"):
        return "dip"
    if rsi >= 65 and mh > mhp and vol_ratio >= 1.05:
        return "breakout"
    positives = [(cat, val) for cat, val in cats.items() if val > 0]
    if not positives:
        return "mixed"
    top_cat, _ = max(positives, key=lambda kv: kv[1])
    if top_cat in ("trend", "rel_str"):
        return "trend_continuation"
    if top_cat in ("momentum", "volume"):
        return "momentum"
    if top_cat in ("news", "analyst", "insider"):
        return "mixed"
    return normalize_cluster(top_cat or "mixed")


def confidence_scale(confidence):
    """Map buy confidence into paper-trade risk buckets."""
    try:
        c = float(confidence or 0)
    except Exception:
        c = 0.0
    if c < 40:
        return 0.0
    if c < 45:
        return 0.80
    if c < 50:
        return 0.95
    if c < 55:
        return 1.00
    if c < 65:
        return 1.10
    if c < 70:
        return 1.25
    return 1.60


def stop_distance_pct(ctx, regime_stop_pct):
    ctx = ctx or {}
    atr_pct = ctx.get("atr_pct", 0) or 0
    stop_pct = dynamic_stop_pct(atr_pct, regime_stop_pct)
    return round(max(1.0, abs(stop_pct)), 4)


def risk_budget_pct(regime_kind, vix_mult, streak_mult, kelly_mult, confidence,
                    source="watchlist", config=None):
    regime_kind = normalize_regime_kind(regime_kind)
    cfg = config or DEFAULT_CONFIG
    risk_cfg = cfg.get("risk", {})
    kelly_cfg = cfg.get("kelly", {})
    base_map = risk_cfg.get("base_risk_pct_by_regime") or REGIME_RISK_PCT
    base = float(base_map.get(regime_kind, REGIME_RISK_PCT[regime_kind]))
    vix = _clamp(float(vix_mult if vix_mult is not None else 1.0), 0.0, 1.5)
    streak = _clamp(float(streak_mult if streak_mult is not None else 1.0), 0.0, 1.5)
    kelly = _clamp(
        float(kelly_mult if kelly_mult is not None else 1.0),
        float(kelly_cfg.get("min_mult", 0.10)),
        float(kelly_cfg.get("max_mult", 1.50)),
    )
    source_scale = SOURCE_RISK_SCALE.get(source, 1.0)
    conf_scale = confidence_scale(confidence)
    pre_kelly = base * vix * streak * conf_scale * source_scale
    risk_pct = pre_kelly * kelly
    return {
        "regime": regime_kind,
        "base_pct": round(base, 4),
        "vix_mult": round(vix, 4),
        "streak_mult": round(streak, 4),
        "kelly_mult": round(kelly, 4),
        "confidence_scale": conf_scale,
        "source_scale": round(source_scale, 4),
        "pre_kelly_pct": round(pre_kelly, 4),
        "risk_pct": round(risk_pct, 4),
    }


def risk_target_notional(total_equity, ctx, regime_stop_pct, regime_kind, vix_mult,
                         streak_mult, kelly_mult, confidence, source="watchlist",
                         config=None):
    risk = risk_budget_pct(regime_kind, vix_mult, streak_mult, kelly_mult,
                           confidence, source, config=config)
    stop_pct = stop_distance_pct(ctx, regime_stop_pct)
    risk_dollars = float(total_equity or 0.0) * risk["risk_pct"] / 100.0
    notional = risk_dollars / (stop_pct / 100.0) if stop_pct > 0 else 0.0
    risk.update({
        "stop_distance_pct": round(stop_pct, 4),
        "risk_dollars": round(risk_dollars, 2),
        "target_notional": round(max(0.0, notional), 2),
    })
    return risk


def apply_size_mult(risk, mult, reason):
    mult = _clamp(float(mult or 1.0), 0.0, 1.0)
    if mult >= 1.0:
        return risk
    risk.setdefault("size_penalties", []).append(reason)
    risk["risk_pct"] = round(risk["risk_pct"] * mult, 4)
    risk["risk_dollars"] = round(risk["risk_dollars"] * mult, 2)
    risk["target_notional"] = round(risk["target_notional"] * mult, 2)
    risk["size_mult"] = round(float(risk.get("size_mult", 1.0)) * mult, 4)
    return risk


def apply_size_factor(risk, mult, reason):
    """Apply a candidate sizing factor; paper confidence buckets may exceed 1.0."""
    mult = _clamp(float(mult or 1.0), 0.0, 2.0)
    if mult == 1.0:
        return risk
    if reason:
        risk.setdefault("size_penalties", []).append(reason)
    risk["risk_pct"] = round(risk["risk_pct"] * mult, 4)
    risk["risk_dollars"] = round(risk["risk_dollars"] * mult, 2)
    risk["target_notional"] = round(risk["target_notional"] * mult, 2)
    risk["size_mult"] = round(float(risk.get("size_mult", 1.0)) * mult, 4)
    return risk


def exit_quality_size_mult(edge_stats, regime_kind, cluster):
    buckets = (edge_stats or {}).get("exit_attribution_buckets") or {}
    key = f"{normalize_regime_kind(regime_kind)}:{cluster or 'mixed'}"
    b = buckets.get(key) or {}
    n = int(b.get("n", 0) or 0)
    too_late = float(b.get("too_late_rate_pct", 0) or 0)
    if too_late > 45 and n >= 50:
        return 0.75
    if too_late > 30 and n >= 30:
        return 0.85
    return 1.0


def _spread_proxy_details(ctx, source="watchlist"):
    """Round-trip spread proxy from liquidity when bid/ask is unavailable."""
    adv = (ctx or {}).get("avg_dollar_vol_20d", 0) or 0
    if adv >= 50_000_000:
        pct = 0.03
        bucket = "adv_ge_50m"
    elif adv >= 5_000_000:
        pct = 0.06
        bucket = "adv_ge_5m"
    elif adv > 0:
        pct = 0.15
        bucket = "adv_lt_5m"
    else:
        pct = 0.10
        bucket = "missing_adv"
    base_pct = pct
    if source == "scan":
        pct += 0.02
    return {
        "pct": round(pct, 4),
        "base_pct": round(base_pct, 4),
        "source_addon_pct": round(pct - base_pct, 4),
        "bucket": bucket,
        "source": "proxy_adv_bucket",
        "avg_dollar_vol_20d": adv,
    }


def spread_proxy_pct(ctx, source="watchlist"):
    return _spread_proxy_details(ctx, source)["pct"]


def build_friction_diagnostics(notional_usd, ctx, commission=COMMISSION_PER_TRADE,
                               source="watchlist"):
    notional = max(float(notional_usd or 0.0), 1.0)
    commission_usd = float(commission or 0.0)
    round_trip_commission = 2.0 * commission_usd
    commission_pct = round_trip_commission / notional * 100.0
    model_slippage_pct = 2.0 * slippage_bps(notional, ctx) / 100.0
    spread = _spread_proxy_details(ctx, source)
    total = commission_pct + model_slippage_pct + spread["pct"]
    return {
        "notional_usd": round(notional, 2),
        "commission_per_trade_usd": round(commission_usd, 4),
        "round_trip_commission_usd": round(round_trip_commission, 4),
        "commission_pct": round(commission_pct, 4),
        "model_slippage_pct": round(model_slippage_pct, 4),
        "slippage_pct": round(model_slippage_pct, 4),
        "spread_proxy_pct": spread["pct"],
        "spread_pct": spread["pct"],
        "total_pct": round(total, 4),
        "avg_dollar_vol_20d": spread["avg_dollar_vol_20d"],
        "spread_source": spread["source"],
        "spread_liquidity_bucket": spread["bucket"],
        "source": source,
        "source_spread_addon_pct": spread["source_addon_pct"],
        "components_sum_check_pct": round(
            round(commission_pct, 4)
            + round(model_slippage_pct, 4)
            + spread["pct"],
            4,
        ),
    }


def estimate_friction_pct(notional_usd, ctx, commission=COMMISSION_PER_TRADE,
                          source="watchlist"):
    diag = build_friction_diagnostics(notional_usd, ctx, commission, source)
    return {
        "total_pct": diag["total_pct"],
        "commission_pct": diag["commission_pct"],
        "slippage_pct": diag["slippage_pct"],
        "spread_pct": diag["spread_pct"],
    }


def edge_key(regime_kind, cluster):
    return f"{normalize_regime_kind(regime_kind)}:{cluster or 'mixed'}"


def confidence_prior_edge_pct(confidence, score=0):
    """Fallback gross edge prior until forward-return samples exist."""
    try:
        conf = float(confidence or 0)
    except Exception:
        conf = 0.0
    try:
        sc = float(score or 0)
    except Exception:
        sc = 0.0
    edge = max(0.0, (conf - 42.0) * 0.06) + max(0.0, sc) * 0.05
    return round(_clamp(edge, 0.0, 4.0), 4)


def build_edge_diagnostics(edge, rec, friction, attr_edge=None):
    rec = rec or {}
    friction = friction or {}
    if attr_edge:
        sources = list(attr_edge.get("sources") or [])
        samples = sum(float(s.get("n_effective", 0) or 0) for s in sources)
        return {
            "edge_source": "attribution_v2",
            "edge_horizon": EDGE_HORIZON,
            "edge_samples": round(samples, 2),
            "gross_edge_pct": round(float(edge.get("gross_edge_pct", 0.0) or 0.0), 4),
            "net_edge_pct": round(float(attr_edge.get("net_edge_pct", 0.0) or 0.0), 4),
            "required_edge_pct": round(float(attr_edge.get("required_edge_pct", 0.0) or 0.0), 4),
            "friction_safety_mult": 1.5,
            "success_threshold_pct": 0.5,
            "sources": sources,
        }
    try:
        conf = float(rec.get("confidence", 0) or 0)
    except Exception:
        conf = 0.0
    try:
        score = float(rec.get("score", 0) or 0)
    except Exception:
        score = 0.0
    confidence_component = max(0.0, (conf - 42.0) * 0.06)
    score_component = max(0.0, score) * 0.05
    return {
        "edge_source": "confidence_prior",
        "edge_horizon": EDGE_HORIZON,
        "edge_samples": int(edge.get("edge_samples", 0) or 0),
        "gross_edge_pct": round(float(edge.get("gross_edge_pct", 0.0) or 0.0), 4),
        "confidence": rec.get("confidence"),
        "score": rec.get("score"),
        "confidence_component_pct": round(confidence_component, 4),
        "score_component_pct": round(score_component, 4),
        "formula": "max(0,(confidence-42)*0.06)+max(0,score)*0.05",
        "clamp_min_pct": 0.0,
        "clamp_max_pct": 4.0,
        "why_prior_used": "no_live_attribution_bucket",
        "friction_total_pct": friction.get("total_pct"),
    }


def rank_reason_code_for(*, risk_sized, gross, required_edge, net,
                         min_net_edge, warmup_watchlist, ev_pass,
                         require_net_positive=False):
    if not risk_sized:
        return "FINAL_SIZE_TOO_SMALL"
    if ev_pass:
        return "EV_RISK_PASS"
    # warmup checked before the edge gates: a warmup-allowed candidate is
    # tradable even when gross <= required, so the code must say so honestly
    if warmup_watchlist:
        return "WARMUP_CONFIDENCE_PRIOR_ALLOWED"
    if gross <= required_edge:
        return "EDGE_TOO_LOW"
    if net < min_net_edge or (require_net_positive and net <= 0):
        return "NET_EDGE_TOO_LOW_AFTER_COSTS"
    return "EV_RISK_PASS"


def estimate_gross_edge_pct(edge_stats, regime_kind, cluster, confidence, score=0,
                            horizon=EDGE_HORIZON, min_samples=EDGE_MIN_SAMPLES):
    key = edge_key(regime_kind, cluster)
    bucket = ((edge_stats or {}).get(key) or {}).get(horizon) or {}
    n = int(bucket.get("n", 0) or 0)
    if n >= min_samples:
        return {
            "gross_edge_pct": round(float(bucket.get("avg_return_pct", 0.0) or 0.0), 4),
            "edge_source": key,
            "edge_samples": n,
            "edge_horizon": horizon,
        }
    return {
        "gross_edge_pct": confidence_prior_edge_pct(confidence, score),
        "edge_source": "confidence_prior",
        "edge_samples": n,
        "edge_horizon": horizon,
    }


def evaluate_candidate(candidate, total_equity, regime_stop_pct, regime_kind,
                       vix_mult, streak_mult, kelly_mult, edge_stats,
                       min_position_usd=100.0,
                       commission=COMMISSION_PER_TRADE, config=None,
                       mode_size_mult=1.0, mode_size_reason=None):
    cfg = config or DEFAULT_CONFIG
    signal_cfg = cfg.get("signal", {})
    rec = candidate.get("rec") or {}
    ctx = candidate.get("ctx") or {}
    source = candidate.get("source") or "watchlist"
    cluster = candidate.get("cluster") or entry_cluster(rec, ctx)
    size_confidence = rec.get("sizing_confidence", rec.get("confidence", 0))
    risk = risk_target_notional(total_equity, ctx, regime_stop_pct, regime_kind,
                                vix_mult, streak_mult, kelly_mult,
                                size_confidence, source, config=cfg)
    candidate_size_mult = float(candidate.get("candidate_size_multiplier", 1.0) or 1.0)
    if candidate_size_mult != 1.0:
        risk["pre_candidate_size_target_notional"] = risk.get("target_notional")
        apply_size_factor(
            risk,
            candidate_size_mult,
            candidate.get("candidate_type") or "candidate_size_multiplier",
        )
    warnings = []
    catalyst_cfg = cfg.get("catalyst", {}) if isinstance(cfg, dict) else DEFAULT_CONFIG.get("catalyst", {})
    avg_dvol = float(ctx.get("avg_dollar_vol_20d", 0) or 0)
    if 0 < avg_dvol < float(signal_cfg.get("low_avg_dollar_volume_warning", 1_000_000)):
        warnings.append("LOW_AVG_DOLLAR_VOLUME_WARNING")
    vol_ratio = float(ctx.get("vol_ratio", 1.0) or 1.0)
    if not bool(ctx.get("is_dip")):
        very_low_ratio = float(signal_cfg.get("very_low_volume_ratio_size_penalty", 0.30))
        low_ratio = float(signal_cfg.get("low_volume_warning_ratio", 0.70))
        if 0 < vol_ratio < very_low_ratio:
            warnings.append("VERY_LOW_VOLUME_SIZE_PENALTY")
            apply_size_mult(
                risk,
                float(signal_cfg.get("very_low_volume_size_multiplier", 0.75)),
                "VERY_LOW_VOLUME_SIZE_PENALTY",
            )
        elif vol_ratio < low_ratio:
            warnings.append("LOW_VOLUME_PENALTY_ONLY")
            apply_size_mult(
                risk,
                float(signal_cfg.get("low_volume_size_multiplier", 0.90)),
                "LOW_VOLUME_PENALTY_ONLY",
            )
    atr_pct = float(ctx.get("atr_pct", 0) or 0)
    if atr_pct > float(signal_cfg.get("high_atr_warning_pct", 10.0)):
        hard_atr = float(signal_cfg.get("extreme_atr_hard_block_pct", 20.0))
        if atr_pct > hard_atr:
            warnings.append("EXTREME_ATR_SIZE_PENALTY")
            apply_size_mult(
                risk,
                float(signal_cfg.get("extreme_atr_size_multiplier", 0.65)),
                "EXTREME_ATR_SIZE_PENALTY",
            )
        else:
            warnings.append("HIGH_ATR_SIZE_PENALTY")
            apply_size_mult(
                risk,
                float(signal_cfg.get("high_atr_size_multiplier", 0.85)),
                "HIGH_ATR_SIZE_PENALTY",
            )
    earnings_risk = (candidate.get("earn") or rec.get("earnings_risk") or {})
    if earnings_risk and earnings_risk.get("soon"):
        days = earnings_risk.get("days_until")
        try:
            days = int(days)
        except Exception:
            days = None
        if days is not None and days <= 1:
            mult_key = "earnings_same_day_size_multiplier" if days <= 0 else "earnings_tomorrow_size_multiplier"
            warnings.append("EARNINGS_RISK_SIZE_PENALTY")
            apply_size_mult(
                risk,
                float(catalyst_cfg.get(mult_key, 0.90)),
                "EARNINGS_RISK_SIZE_PENALTY",
            )
        elif days is not None and days <= 3:
            warnings.append("EARNINGS_RISK_WARNING")
    catalyst = rec.get("catalyst") or {}
    catalyst_type = catalyst.get("type")
    week_chg = float(ctx.get("week_chg_pct", 0) or 0)
    if catalyst_type in KNIFE_CATALYST_TYPES and week_chg > float(catalyst_cfg.get("negative_catalyst_falling_threshold_pct", -5.0)):
        warnings.append("NEGATIVE_CATALYST_SIZE_PENALTY")
        apply_size_mult(
            risk,
            float(catalyst_cfg.get("negative_catalyst_size_multiplier", 0.90)),
            "NEGATIVE_CATALYST_SIZE_PENALTY",
        )
    regime_mult = float(candidate.get("regime_risk_mult", 1.0) or 1.0)
    cluster_mult = float(candidate.get("cluster_regime_mult", 1.0) or 1.0)
    combined_regime_mult = max(0.0, min(1.25, regime_mult * cluster_mult))
    if combined_regime_mult != 1.0:
        risk["regime_risk_mult"] = round(regime_mult, 4)
        risk["cluster_regime_mult"] = round(cluster_mult, 4)
        risk["combined_regime_mult"] = round(combined_regime_mult, 4)
        risk["risk_pct"] = round(risk["risk_pct"] * combined_regime_mult, 4)
        risk["risk_dollars"] = round(risk["risk_dollars"] * combined_regime_mult, 2)
        risk["target_notional"] = round(risk["target_notional"] * combined_regime_mult, 2)
    eq_mult = exit_quality_size_mult(edge_stats, regime_kind, cluster)
    if eq_mult < 1.0:
        apply_size_mult(risk, eq_mult, "exit quality too-late penalty")
    if float(mode_size_mult or 1.0) < 1.0:
        risk["pre_mode_target_notional"] = risk.get("target_notional")
        apply_size_mult(risk, mode_size_mult, mode_size_reason or "MARKET_DATA_MODE")
    friction = estimate_friction_pct(max(risk["target_notional"], 1.0), ctx,
                                     commission=commission, source=source)
    attr_edge = None
    if isinstance(edge_stats, dict) and "attribution_buckets" in edge_stats:
        try:
            from trading.attribution import expected_edge_for_candidate
            attr_edge = expected_edge_for_candidate(
                edge_stats, regime_kind, cluster, rec.get("categories", {}),
                friction.get("total_pct", 0.0),
            )
        except Exception:
            attr_edge = None
    if attr_edge:
        edge = {
            "gross_edge_pct": attr_edge["gross_edge_pct"],
            "edge_source": attr_edge["edge_source"],
            "edge_samples": sum(
                float(s.get("n_effective", 0) or 0)
                for s in (attr_edge.get("sources") or [])
            ),
            "edge_horizon": EDGE_HORIZON,
            "required_edge_pct": attr_edge.get("required_edge_pct"),
        }
    else:
        legacy_edges = edge_stats.get("edge_stats", {}) if (
            isinstance(edge_stats, dict) and "attribution_buckets" in edge_stats
        ) else edge_stats
        edge = estimate_gross_edge_pct(legacy_edges, regime_kind, cluster,
                                       rec.get("confidence", 0), rec.get("score", 0))
    gross = edge["gross_edge_pct"]
    net = gross - friction["total_pct"]
    stop_pct = risk["stop_distance_pct"]
    ev_score = net / stop_pct if stop_pct > 0 else -999.0
    required_edge = max(
        edge.get("required_edge_pct", friction["total_pct"]),
        float(signal_cfg.get("min_expected_edge_pct", 0.0) or 0.0),
    )
    min_net_edge = float(signal_cfg.get("min_net_edge_pct", -999.0))
    require_net_positive = bool(signal_cfg.get("require_net_edge_positive", False))
    risk_sized = risk["target_notional"] >= min_position_usd
    ev_pass = gross > required_edge and net >= min_net_edge and (not require_net_positive or net > 0)
    expected_net_profit_usd = max(risk["target_notional"], 0.0) * net / 100.0
    edge_to_required_ratio = (
        gross / required_edge
        if required_edge and required_edge > 0
        else (1.0 if gross > 0 else 0.0)
    )
    quote_execution_trusted = bool(candidate.get("execution_trusted", True))
    relaxed_ev_enabled = bool(signal_cfg.get("paper_ev_relaxation_enabled", False))
    display_signal = rec.get("display_signal_label")
    candidate_type = candidate.get("candidate_type") or rec.get("candidate_type")
    if display_signal == "STRONG_BUY_CANDIDATE":
        min_expected_net_profit = float(signal_cfg.get("strong_buy_min_expected_net_profit_usd", 0.10))
    elif display_signal == "BUY_CANDIDATE":
        min_expected_net_profit = float(signal_cfg.get("buy_candidate_min_expected_net_profit_usd", 0.25))
    elif display_signal == "BULLISH_LEAN" or candidate_type == "weak_paper_test_candidate":
        min_expected_net_profit = float(signal_cfg.get("bullish_lean_min_expected_net_profit_usd", 0.50))
    elif display_signal == "HOLD" or candidate_type == "experimental_near_buy_candidate":
        min_expected_net_profit = float(signal_cfg.get("high_confidence_hold_min_expected_net_profit_usd", 0.75))
    else:
        min_expected_net_profit = float(signal_cfg.get(
            "default_min_expected_net_profit_usd",
            signal_cfg.get("min_expected_net_profit_usd", 0.25),
        ))
    relaxed_ev_pass = bool(
        relaxed_ev_enabled
        and risk_sized
        and quote_execution_trusted
        and expected_net_profit_usd >= min_expected_net_profit
        and net > 0
    )
    # Warm-up trades are information purchases (audit P0-1): no net>0 needed,
    # but higher confidence floor and a size haircut applied below.
    warmup_watchlist = (
        source == "watchlist"
        and edge.get("edge_source") == "confidence_prior"
        and rec.get("cls") in ("buy", "strong-buy")
        and rec.get("confidence", 0) >= float(signal_cfg.get("warmup_min_confidence", 48))
        and gross > 0
    )
    if warmup_watchlist and not (ev_pass or relaxed_ev_pass) and risk_sized:
        apply_size_mult(risk, float(signal_cfg.get("warmup_size_mult", 0.60)),
                        "WARMUP_CONFIDENCE_PRIOR_SIZE")
        # flat commission → friction % moves with notional; keep diagnostics honest
        friction = estimate_friction_pct(max(risk["target_notional"], 1.0), ctx,
                                         commission=commission, source=source)
        net = gross - friction["total_pct"]
        ev_score = net / stop_pct if stop_pct > 0 else -999.0
        expected_net_profit_usd = max(risk["target_notional"], 0.0) * net / 100.0
        risk_sized = risk["target_notional"] >= min_position_usd
    tradable = risk_sized and (ev_pass or relaxed_ev_pass or warmup_watchlist)
    rank_reason_code = rank_reason_code_for(
        risk_sized=risk_sized,
        gross=gross,
        required_edge=required_edge,
        net=net,
        min_net_edge=min_net_edge,
        warmup_watchlist=warmup_watchlist,
        ev_pass=ev_pass or relaxed_ev_pass,
        require_net_positive=require_net_positive,
    )
    if not risk_sized:
        reason = (f"Risk budget target ${risk['target_notional']:.0f} below "
                  f"${float(min_position_usd):.0f} floor")
    elif ev_pass:
        reason = "EV/risk pass"
    elif relaxed_ev_pass:
        reason = "relaxed paper EV pass"
    elif warmup_watchlist:
        reason = "warm-up confidence prior allowed"
    elif gross <= required_edge:
        reason = (f"EV gate: edge {gross:.2f}% <= friction "
                  f"{friction['total_pct']:.2f}%"
                  + (f" / required {required_edge:.2f}%" if required_edge != friction["total_pct"] else ""))
    elif require_net_positive and net <= 0:
        reason = f"EV gate: net edge {net:.2f}% must be positive"
    elif net < min_net_edge:
        reason = f"EV gate: net edge {net:.2f}% < min {min_net_edge:.2f}%"
    else:
        reason = "EV/risk pass"
    friction_diag = build_friction_diagnostics(
        max(risk["target_notional"], 1.0),
        ctx,
        commission=commission,
        source=source,
    )
    edge_diag = build_edge_diagnostics(edge, rec, friction, attr_edge=attr_edge)
    ev_diag = {
        "gross_edge_pct": round(gross, 4),
        "friction_total_pct": friction["total_pct"],
        "required_edge_pct": round(required_edge, 4),
        "net_edge_pct": round(net, 4),
        "min_expected_edge_pct": float(signal_cfg.get("min_expected_edge_pct", 0.25)),
        "min_net_edge_pct": min_net_edge,
        "require_net_edge_positive": require_net_positive,
        "paper_ev_relaxation_enabled": relaxed_ev_enabled,
        "expected_net_profit_usd": round(expected_net_profit_usd, 4),
        "min_expected_net_profit_usd": min_expected_net_profit,
        "min_required_expected_net_profit_usd": min_expected_net_profit,
        "edge_to_required_ratio": round(edge_to_required_ratio, 6),
        "edge_gap_to_required_pct": round(gross - required_edge, 4),
        "net_gap_to_min_pct": round(net - min_net_edge, 4),
        "ev_score": round(ev_score, 6),
        "stop_distance_pct": stop_pct,
        "risk_sized": bool(risk_sized),
        "ev_pass": bool(ev_pass),
        "relaxed_ev_pass": bool(relaxed_ev_pass),
        "ev_gate_passed": bool(ev_pass or relaxed_ev_pass),
        "ev_blocker_code": None if (ev_pass or relaxed_ev_pass) else (
            "EDGE_TOO_LOW" if gross <= required_edge else "NET_EDGE_TOO_LOW_AFTER_COSTS"
        ),
        "quote_execution_trusted": quote_execution_trusted,
        "warmup_watchlist": bool(warmup_watchlist),
        "tradable": bool(tradable),
        "rank_reason": reason,
        "rank_reason_code": rank_reason_code,
    }
    out = dict(candidate)
    out.update({
        "cluster": cluster,
        "risk": risk,
        "friction": friction,
        "friction_diagnostics": friction_diag,
        "edge_diagnostics": edge_diag,
        "ev_diagnostics": ev_diag,
        "gross_edge_pct": round(gross, 4),
        "net_edge_pct": round(net, 4),
        "ev_score": round(ev_score, 6),
        "tradable": bool(tradable),
        "rank_reason": reason,
        "rank_reason_code": rank_reason_code,
        "edge_source": edge["edge_source"],
        "trade_bucket": candidate.get("trade_bucket") or (
            "confidence_prior" if edge["edge_source"] == "confidence_prior" else "validated"
        ),
        "edge_samples": edge["edge_samples"],
        "edge_horizon": edge["edge_horizon"],
        "required_edge_pct": round(required_edge, 4),
        "sizing_confidence": size_confidence,
        "candidate_type": candidate.get("candidate_type"),
        "candidate_size_multiplier": candidate_size_mult,
        "warnings": warnings,
    })
    return out


def rank_candidates(candidates, total_equity, regime_stop_pct, regime_kind,
                    vix_mult, streak_mult, kelly_mult, edge_stats,
                    min_position_usd=100.0,
                    commission=COMMISSION_PER_TRADE, config=None,
                    mode_size_mult=1.0, mode_size_reason=None):
    ranked = [
        evaluate_candidate(c, total_equity, regime_stop_pct, regime_kind, vix_mult,
                           streak_mult, kelly_mult, edge_stats, min_position_usd,
                           commission, config=config, mode_size_mult=mode_size_mult,
                           mode_size_reason=mode_size_reason)
        for c in candidates
    ]
    ranked.sort(key=lambda c: (c.get("tradable", False), c.get("ev_score", -999),
                              c.get("net_edge_pct", -999),
                              (c.get("rec") or {}).get("confidence", 0)),
                reverse=True)
    return ranked


def sector_rotation_bonus(candidate: dict, sector_lookup, *,
                          top_sectors=None,
                          weak_sectors=None) -> float:
    """Small advisory bonus for suggestions; execution EV ranking stays separate."""
    top = set(top_sectors or [])
    weak = set(weak_sectors or [])
    sector = candidate.get("sector")
    if not sector:
        try:
            if callable(sector_lookup):
                sector = sector_lookup(candidate.get("ticker"))
            elif isinstance(sector_lookup, dict):
                sector = sector_lookup.get(candidate.get("ticker"))
        except Exception:
            sector = None
    if not sector:
        return 0.0
    if sector in top:
        return 0.05
    if sector in weak:
        return -0.05
    return 0.0


def aggregate_forward_return(edge_stats, regime_kind, cluster, horizon_key,
                             return_pct, decision):
    stats = edge_stats.setdefault(edge_key(regime_kind, cluster), {})
    bucket = stats.setdefault(horizon_key, {
        "n": 0, "wins": 0, "sum_return_pct": 0.0,
        "executed_n": 0, "skipped_n": 0,
    })
    ret = float(return_pct or 0.0)
    bucket["n"] = int(bucket.get("n", 0)) + 1
    bucket["wins"] = int(bucket.get("wins", 0)) + (1 if ret > 0 else 0)
    bucket["sum_return_pct"] = round(float(bucket.get("sum_return_pct", 0.0)) + ret, 4)
    if decision == "executed":
        bucket["executed_n"] = int(bucket.get("executed_n", 0)) + 1
    else:
        bucket["skipped_n"] = int(bucket.get("skipped_n", 0)) + 1
    bucket["avg_return_pct"] = round(bucket["sum_return_pct"] / bucket["n"], 4)
    bucket["hit_rate_pct"] = round(bucket["wins"] / bucket["n"] * 100.0, 2)
    return bucket


def slippage_bps(notional_usd, ctx):
    """One-side slippage in bps. BACKTEST-ONLY now — live bot uses flat
    COMMISSION_PER_TRADE. Returns base + K × (notional / ADV) × 10000."""
    base = SLIPPAGE_BASE_BPS
    adv = (ctx or {}).get("avg_dollar_vol_20d", 0) or 0
    if adv > 0:
        return base + SLIPPAGE_IMPACT_K * (notional_usd / adv) * 10000
    return base


def _partial_trim(b, t, h, pr, ctx, rec, arts, frac, reason, exit_reason_key="partial_take"):
    """Trim `frac` (0-1) of position `t` at price `pr`. Shared by partial-take
    (#2.4) and signal-degradation (#4.3) exits. Round-5: flat $0.99 commission,
    no price markup — proceeds = trim_sh*pr − 0.99. Records realized profit so
    the SELL row can show P&L. Returns shares trimmed (0 if no-op)."""
    from trading.bot import _record_trade   # lazy: bot.py imports sizing module
    original_sh = h["shares"]
    trim_sh = round(original_sh * frac, 4)
    if trim_sh <= 0:
        return 0
    avg_cost = h.get("avg_cost", 0) or 0
    entry_commission = float(h.get("commission_invested", 0) or 0) * (trim_sh / original_sh)
    realized = trim_sh * pr - trim_sh * avg_cost - entry_commission - COMMISSION_PER_TRADE
    b["total_costs_usd"] = round(b.get("total_costs_usd", 0) + COMMISSION_PER_TRADE, 2)
    b["cash"] += trim_sh * pr - COMMISSION_PER_TRADE
    _record_trade(b, "SELL", t, trim_sh, pr, rec, arts, reason, pnl_usd=round(realized, 2))
    h["shares"] = round(h["shares"] - trim_sh, 4)
    if "commission_invested" in h:
        h["commission_invested"] = round(float(h.get("commission_invested") or 0) - entry_commission, 4)
    h["peak"]   = round(pr, 4)
    b["holdings"][t] = h
    cost_basis = trim_sh * avg_cost + entry_commission
    net_pnl_pct = (realized / cost_basis * 100) if cost_basis else 0
    b.setdefault("trade_outcomes", []).append({
        "ticker": t,
        "pnl_pct": round(net_pnl_pct, 2),
        "gross_pnl_pct": round(((pr - avg_cost) / avg_cost * 100) if avg_cost else 0, 2),
        "exit_reason": exit_reason_key,
        "ts": int(time.time()),
        "entry_regime": (h.get("entry_snapshot") or {}).get("market_regime"),
        "entry_confidence": (h.get("entry_snapshot") or {}).get("confidence"),
    })
    b["trade_outcomes"] = b["trade_outcomes"][-100:]
    if net_pnl_pct > 0:
        b["wins_total"] = b.get("wins_total", 0) + 1
    else:
        b["losses_total"] = b.get("losses_total", 0) + 1
    return trim_sh
