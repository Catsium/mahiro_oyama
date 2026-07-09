"""Forward-return attribution for entries and exits.

This module owns the V2 learning state. Entry learning uses forward returns, not
closed trade P&L. Exit learning measures capture quality separately.
"""
import time
import uuid

from trading.exit_ladders import normalize_cluster


ALPHA_PRIOR = 3
BETA_PRIOR = 4
EMA_ALPHA = 0.08
ENTRY_LIVE_N = 60
EXIT_LIVE_N = 60
FULL_TRUST_N = 120
SUCCESS_THRESHOLD_PCT = 0.5
FAILURE_THRESHOLD_PCT = -1.0
STRONG_SUCCESS_PCT = 2.0
FRICTION_SAFETY_MULT = 1.5
LEARNING_STRENGTH = 0.25
MAIN_HORIZON = "5d"
FORWARD_HORIZONS_DAYS = (1, 3, 5, 10)
MAX_ATTRIBUTION_EVENTS = 800
MAX_EXIT_EVENTS = 600
TRACK_TOP_SKIPS_PER_CYCLE = 5
FRESHNESS_HALF_LIFE_SEC = 45 * 86400
MIN_STALE_TRUST_MULT = 0.30

CATEGORY_CLAMP = (0.85, 1.15)
REGIME_CATEGORY_CLAMP = (0.75, 1.25)
CLUSTER_CATEGORY_CLAMP = (0.70, 1.30)
LOW_SAMPLE_CLAMP = (0.95, 1.05)

ENTRY_BLEND = (
    ("specific", 0.50),
    ("cluster", 0.25),
    ("category", 0.15),
    ("global", 0.10),
)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def now_ts():
    return int(time.time())


def max_entry_events():
    try:
        from utils.deploy_config import PYTHONANYWHERE_MODE
        return 500 if PYTHONANYWHERE_MODE else 800
    except Exception:
        return MAX_ATTRIBUTION_EVENTS


def max_exit_events():
    try:
        from utils.deploy_config import PYTHONANYWHERE_MODE
        return 500 if PYTHONANYWHERE_MODE else 800
    except Exception:
        return MAX_EXIT_EVENTS


def max_bucket_events():
    try:
        from utils.deploy_config import PYTHONANYWHERE_MODE
        return 100 if PYTHONANYWHERE_MODE else 150
    except Exception:
        return 150


def _cap_events_by_bucket(events, cap=None):
    cap = int(cap or max_bucket_events())
    counts = {}
    kept = []
    for event in events or []:
        votes = event.get("category_votes") or {}
        categories = []
        for k, v in votes.items():
            try:
                keep_vote = v is None or float(v or 0.0) > 0.0
            except Exception:
                keep_vote = False
            if keep_vote:
                categories.append(str(k).lower())
        if not categories:
            categories = [event.get("exit_reason") or event.get("category") or "all"]
        key = (
            normalize_regime(event.get("regime")),
            event.get("cluster") or "mixed",
            ",".join(sorted(categories)),
        )
        n = counts.get(key, 0)
        if n >= cap:
            continue
        counts[key] = n + 1
        kept.append(event)
    return kept


def normalize_regime(regime):
    if isinstance(regime, dict):
        regime = regime.get("regime_effective") or regime.get("regime")
    r = str(regime or "neutral").lower()
    if "bull" in r:
        return "bull"
    if "bear" in r:
        return "bear"
    return "neutral"


def ensure_attribution_state(state):
    state.setdefault("attribution_events", [])
    state.setdefault("attribution_buckets", {})
    state.setdefault("exit_attribution_events", [])
    state.setdefault("exit_attribution_buckets", {})
    status = state.setdefault("attribution_status", {})
    status.setdefault("entry_live_min_n", ENTRY_LIVE_N)
    status.setdefault("exit_live_min_n", EXIT_LIVE_N)
    status.setdefault("legacy_archived", True)
    status.setdefault("entry_live", False)
    status.setdefault("exit_live", False)
    return state


def init_entry_bucket(bucket=None):
    bucket = bucket or {}
    bucket.setdefault("alpha", ALPHA_PRIOR)
    bucket.setdefault("beta", BETA_PRIOR)
    bucket.setdefault("n", 0)
    bucket.setdefault("successes", 0)
    bucket.setdefault("failures", 0)
    bucket.setdefault("neutral", 0)
    bucket.setdefault("strong_successes", 0)
    bucket.setdefault("sum_net_ret_pct", 0.0)
    bucket.setdefault("sum_raw_ret_pct", 0.0)
    bucket.setdefault("sum_win_pct", 0.0)
    bucket.setdefault("sum_loss_pct", 0.0)
    bucket.setdefault("wins", 0)
    bucket.setdefault("losses", 0)
    bucket.setdefault("executed_n", 0)
    bucket.setdefault("skipped_n", 0)
    bucket.setdefault("ema_net_ret_pct", 0.0)
    bucket.setdefault("ema_raw_ret_pct", 0.0)
    bucket.setdefault("ema_mfe_pct", 0.0)
    bucket.setdefault("ema_mae_pct", 0.0)
    bucket.setdefault("avg_signal_strength", 0.0)
    bucket.setdefault("last_updated", None)
    return bucket


def p_success(bucket):
    b = init_entry_bucket(bucket)
    return b["alpha"] / max(1.0, b["alpha"] + b["beta"])


def freshness_trust_mult(bucket=None, ts=None):
    b = bucket or {}
    last = ts if ts is not None else b.get("last_updated")
    if not last:
        return 1.0
    age = max(0.0, now_ts() - int(last))
    return clamp(0.5 ** (age / FRESHNESS_HALF_LIFE_SEC),
                 MIN_STALE_TRUST_MULT, 1.0)


def sample_trust(n=None, min_n=ENTRY_LIVE_N, full_n=FULL_TRUST_N,
                 bucket=None, ts=None):
    b = bucket or {}
    if n is None:
        n = b.get("n", 0)
    n = int(n or 0)
    if n < min_n:
        return 0.0
    trust = clamp(0.25 + 0.75 * ((n - min_n) / max(1, full_n - min_n)), 0.25, 1.0)
    trust *= freshness_trust_mult(b, ts=ts)
    if int(b.get("executed_n", 0) or 0) < 10 and int(b.get("skipped_n", 0) or 0) > int(b.get("executed_n", 0) or 0):
        trust = min(trust, 0.50)
    return clamp(trust, 0.0, 1.0)


def bucket_score(bucket):
    b = init_entry_bucket(bucket)
    trust = sample_trust(bucket=b)
    if trust <= 0:
        return 0.0
    prob_score = (p_success(b) - 0.50) * 2.0
    ret_score = clamp((b.get("ema_net_ret_pct", 0.0) or 0.0) / 2.0, -1.0, 1.0)
    raw = 0.45 * prob_score + 0.55 * ret_score
    return clamp(raw, -1.0, 1.0) * trust


def attribution_multiplier(bucket, clamp_range=CATEGORY_CLAMP):
    b = init_entry_bucket(bucket)
    lo, hi = LOW_SAMPLE_CLAMP if b.get("n", 0) < ENTRY_LIVE_N else clamp_range
    return round(clamp(1.0 + LEARNING_STRENGTH * bucket_score(b), lo, hi), 4)


def avg_win_loss(bucket):
    b = init_entry_bucket(bucket)
    avg_win = b["sum_win_pct"] / b["wins"] if b["wins"] else 0.0
    avg_loss = abs(b["sum_loss_pct"] / b["losses"]) if b["losses"] else 0.0
    payoff = avg_win / avg_loss if avg_loss else None
    return avg_win, avg_loss, payoff


def expected_gross_edge(bucket):
    b = init_entry_bucket(bucket)
    avg_win, avg_loss, _ = avg_win_loss(b)
    if b.get("n", 0) < ENTRY_LIVE_N or (not avg_win and not avg_loss):
        return None
    p = p_success(b)
    return p * avg_win - (1.0 - p) * avg_loss


def category_strength(vote):
    try:
        return clamp(abs(float(vote)) / 2.0, 0.0, 1.0)
    except Exception:
        return 0.0


def bucket_key(regime, cluster, category):
    return f"{normalize_regime(regime)}:{cluster or 'mixed'}:{category or 'all'}"


def bucket_feedback_key(candidate: dict) -> str:
    """Bucket advisory UI feedback separately from market-outcome attribution."""
    cand = candidate or {}
    source = str(cand.get("source") or "unknown").lower()
    cluster = str(cand.get("cluster") or "mixed").lower()
    group = cand.get("corr_group")
    sector = cand.get("sector")
    if not group:
        try:
            from trading.risk import get_corr_group
            group = get_corr_group(cand.get("ticker") or "")
        except Exception:
            group = None
    if not sector:
        try:
            from trading.risk import get_sector
            sector = get_sector(cand.get("ticker") or "")
        except Exception:
            sector = None
    return f"{source}:{cluster}:{group or sector or 'unknown'}"


def bucket_keys(regime, cluster, category):
    r = normalize_regime(regime)
    c = cluster or "mixed"
    cat = category or "all"
    return {
        "specific": f"{r}:{c}:{cat}",
        "cluster": f"all:{c}:{cat}",
        "category": f"all:all:{cat}",
        "global": "all:all:all",
    }


def _update_bucket(bucket, net_ret_pct, raw_ret_pct, mfe_pct, mae_pct, signal_strength, ts,
                   decision=None):
    b = init_entry_bucket(bucket)
    net = float(net_ret_pct or 0.0)
    raw = float(raw_ret_pct or 0.0)
    mfe = float(mfe_pct or 0.0)
    mae = float(mae_pct or 0.0)
    strength = float(signal_strength or 0.0)
    if net > SUCCESS_THRESHOLD_PCT:
        b["alpha"] += 1
        b["successes"] += 1
        if net > STRONG_SUCCESS_PCT:
            b["strong_successes"] += 1
    elif net < FAILURE_THRESHOLD_PCT:
        b["beta"] += 1
        b["failures"] += 1
    else:
        b["neutral"] += 1
    old_n = b["n"]
    b["n"] = old_n + 1
    b["sum_net_ret_pct"] = round(b["sum_net_ret_pct"] + net, 4)
    b["sum_raw_ret_pct"] = round(b["sum_raw_ret_pct"] + raw, 4)
    if net > 0:
        b["wins"] += 1
        b["sum_win_pct"] = round(b["sum_win_pct"] + net, 4)
    elif net < 0:
        b["losses"] += 1
        b["sum_loss_pct"] = round(b["sum_loss_pct"] + net, 4)
    if old_n == 0:
        b["ema_net_ret_pct"] = round(net, 4)
        b["ema_raw_ret_pct"] = round(raw, 4)
        b["ema_mfe_pct"] = round(mfe, 4)
        b["ema_mae_pct"] = round(mae, 4)
        b["avg_signal_strength"] = round(strength, 4)
    else:
        b["ema_net_ret_pct"] = round((1 - EMA_ALPHA) * b["ema_net_ret_pct"] + EMA_ALPHA * net, 4)
        b["ema_raw_ret_pct"] = round((1 - EMA_ALPHA) * b["ema_raw_ret_pct"] + EMA_ALPHA * raw, 4)
        b["ema_mfe_pct"] = round((1 - EMA_ALPHA) * b["ema_mfe_pct"] + EMA_ALPHA * mfe, 4)
        b["ema_mae_pct"] = round((1 - EMA_ALPHA) * b["ema_mae_pct"] + EMA_ALPHA * mae, 4)
        b["avg_signal_strength"] = round(
            (b["avg_signal_strength"] * old_n + strength) / b["n"], 4
        )
    avg_win, avg_loss, payoff = avg_win_loss(b)
    b["avg_win_pct"] = round(avg_win, 4)
    b["avg_loss_pct"] = round(avg_loss, 4)
    b["payoff_ratio"] = round(payoff, 4) if payoff is not None else None
    b["p_success"] = round(p_success(b), 4)
    if decision == "executed":
        b["executed_n"] += 1
    elif decision:
        b["skipped_n"] += 1
    b["score"] = round(bucket_score(b), 4)
    b["multiplier"] = attribution_multiplier(b)
    b["live"] = b["n"] >= ENTRY_LIVE_N
    b["last_updated"] = int(ts)
    return b


def update_entry_buckets(state, event, horizon=MAIN_HORIZON):
    ensure_attribution_state(state)
    if horizon not in (event.get("forward_returns") or {}):
        return 0
    if horizon in event.setdefault("bucketed_horizons", []):
        return 0
    raw_ret = event["forward_returns"][horizon]
    friction = float(event.get("friction_pct") or 0.0)
    net_ret = raw_ret - friction
    mfe = event.get("mfe_pct", 0.0)
    mae = event.get("mae_pct", 0.0)
    regime = event.get("regime")
    cluster = event.get("cluster")
    votes = event.get("category_votes") or {}
    buckets = state.setdefault("attribution_buckets", {})
    updated = 0
    seen_global = False
    positive_seen = False
    for cat, vote in votes.items():
        try:
            vote_f = float(vote or 0.0)
        except Exception:
            continue
        if vote_f <= 0:
            continue
        positive_seen = True
        strength = category_strength(vote_f)
        for key in bucket_keys(regime, cluster, cat).values():
            if key == "all:all:all":
                if seen_global:
                    continue
                seen_global = True
            buckets[key] = _update_bucket(
                buckets.setdefault(key, {}), net_ret, raw_ret, mfe, mae, strength,
                event.get("ts") or now_ts(), decision=event.get("decision"),
            )
            updated += 1
    if not positive_seen:
        key = "all:all:all"
        buckets[key] = _update_bucket(
            buckets.setdefault(key, {}), net_ret, raw_ret, mfe, mae, 0.0,
            event.get("ts") or now_ts(), decision=event.get("decision"),
        )
        updated += 1
    event["bucketed_horizons"].append(horizon)
    state["attribution_buckets"] = buckets
    _refresh_status(state)
    return updated


def record_entry_event(state, candidate, decision, reason, ts=None, regime=None):
    ensure_attribution_state(state)
    # "skipped" events fill forward-return edge buckets without trading
    # (audit P0-2); caller caps them at TRACK_TOP_SKIPS_PER_CYCLE per tick.
    if decision not in ("executed", "skipped"):
        return None
    rec = candidate.get("rec") or {}
    ctx = candidate.get("ctx") or {}
    risk = candidate.get("risk") or {}
    friction = candidate.get("friction") or {}
    event = {
        "event_id": candidate.get("event_id") or uuid.uuid4().hex,
        "ts": int(ts or now_ts()),
        "ticker": candidate.get("ticker"),
        "source": candidate.get("source") or "watchlist",
        "decision": decision,
        "was_executed": decision == "executed",
        "skip_reason": reason if decision != "executed" else None,
        "reason": reason,
        "cluster": candidate.get("cluster") or "mixed",
        "regime": normalize_regime(regime or candidate.get("regime")),
        "vol_regime": candidate.get("vol_regime") or ctx.get("vol_regime"),
        "price": round(float(candidate.get("price") or 0.0), 4),
        "signal_label": rec.get("signal"),
        "confidence": rec.get("confidence"),
        "score": rec.get("score"),
        "expected_edge_pct": candidate.get("gross_edge_pct"),
        "net_edge_pct": candidate.get("net_edge_pct"),
        "friction_pct": friction.get("total_pct") if friction else candidate.get("friction_pct"),
        "rank_reason_code": candidate.get("rank_reason_code"),
        "required_edge_pct": candidate.get("required_edge_pct"),
        "edge_source": candidate.get("edge_source"),
        "edge_samples": candidate.get("edge_samples"),
        "edge_horizon": candidate.get("edge_horizon"),
        "friction_diagnostics": candidate.get("friction_diagnostics"),
        "edge_diagnostics": candidate.get("edge_diagnostics"),
        "ev_diagnostics": candidate.get("ev_diagnostics"),
        "atr_pct": ctx.get("atr_pct"),
        "risk_pct": risk.get("risk_pct"),
        "target_notional": risk.get("target_notional"),
        "category_votes": rec.get("categories", {}),
        "forward_returns": {},
        "relative_returns": {},
        "bucketed_horizons": [],
        "mfe_pct": 0.0,
        "mae_pct": 0.0,
        "entry_benchmarks": candidate.get("benchmark_prices") or {},
        "portfolio_variance": candidate.get("portfolio_variance"),
        "config_hash": candidate.get("config_hash"),
        "regime_v3": candidate.get("regime_v3"),
        "regime_reason": candidate.get("regime_reason"),
        "regime_risk_mult": candidate.get("regime_risk_mult"),
        "cluster_regime_mult": candidate.get("cluster_regime_mult"),
        "top_sectors": candidate.get("top_sectors"),
        "catalyst_type": candidate.get("catalyst_type"),
        "catalyst_score_shadow": candidate.get("catalyst_score_shadow"),
        "catalyst_confirmed": candidate.get("catalyst_confirmed"),
    }
    events = state.setdefault("attribution_events", [])
    for old in events[:80]:
        if (old.get("ticker") == event["ticker"]
                and old.get("decision") == event["decision"]
                and old.get("cluster") == event["cluster"]
                and old.get("source") == event["source"]
                and old.get("ts", 0) >= event["ts"] - 6 * 3600):
            return old
    events.insert(0, event)
    state["attribution_events"] = _cap_events_by_bucket(events)[:max_entry_events()]
    return event


def update_forward_outcomes(state, price_lookup, ts=None, benchmark_lookup=None):
    ensure_attribution_state(state)
    ts = int(ts or now_ts())
    updated = 0
    keep = []
    for event in state.get("attribution_events", []):
        fwd = event.setdefault("forward_returns", {})
        done = all(f"{d}d" in fwd for d in FORWARD_HORIZONS_DAYS)
        age_sec = ts - event.get("ts", 0)
        if done and age_sec > 35 * 86400:
            keep.append(event)
            continue
        keep.append(event)
        price = float(event.get("price") or 0.0)
        if price <= 0:
            continue
        cur = price_lookup(event.get("ticker"))
        if cur is None or cur <= 0:
            continue
        ret = (cur - price) / price * 100.0
        event["mfe_pct"] = round(max(event.get("mfe_pct", ret), ret), 4)
        event["mae_pct"] = round(min(event.get("mae_pct", ret), ret), 4)
        elapsed_days = age_sec / 86400.0
        for days in FORWARD_HORIZONS_DAYS:
            hkey = f"{days}d"
            if elapsed_days < days or hkey in fwd:
                continue
            fwd[hkey] = round(ret, 4)
            if benchmark_lookup:
                rels = event.setdefault("relative_returns", {})
                for bench in ("SPY", "QQQ"):
                    start = (event.get("entry_benchmarks") or {}).get(bench)
                    if not start:
                        continue
                    now_px = benchmark_lookup(bench)
                    if now_px:
                        bret = (now_px - start) / start * 100.0
                        rels[f"{bench}_{hkey}"] = round(ret - bret, 4)
            if hkey == MAIN_HORIZON:
                update_entry_buckets(state, event, hkey)
            updated += 1
    state["attribution_events"] = _cap_events_by_bucket(keep)[:max_entry_events()]
    _refresh_status(state)
    return updated


def blended_bucket(state, regime, cluster, category):
    ensure_attribution_state(state)
    buckets = state.get("attribution_buckets", {})
    keys = bucket_keys(regime, cluster, category)
    out = {
        "n_effective": 0.0,
        "gross_edge_sum": 0.0,
        "weight_sum": 0.0,
        "score_sum": 0.0,
        "parts": [],
    }
    for level, base_w in ENTRY_BLEND:
        key = keys[level]
        b = buckets.get(key)
        if not b:
            continue
        trust = sample_trust(bucket=b)
        if trust <= 0:
            out["parts"].append({"key": key, "n": b.get("n", 0), "live": False})
            continue
        edge = expected_gross_edge(b)
        if edge is None:
            continue
        w = base_w * trust
        out["gross_edge_sum"] += edge * w
        out["score_sum"] += bucket_score(b) * w
        out["weight_sum"] += w
        out["n_effective"] += b.get("n", 0) * base_w
        out["parts"].append({"key": key, "n": b.get("n", 0), "live": True,
                             "edge": round(edge, 4), "weight": round(w, 4)})
    if out["weight_sum"] <= 0:
        out["live"] = False
        out["gross_edge_pct"] = None
        return out
    out["live"] = out["n_effective"] >= ENTRY_LIVE_N
    out["gross_edge_pct"] = round(out["gross_edge_sum"] / out["weight_sum"], 4)
    out["score"] = round(out["score_sum"] / out["weight_sum"], 4)
    out["n_effective"] = round(out["n_effective"], 2)
    return out


def expected_edge_for_candidate(state, regime, cluster, category_votes, friction_pct=0.0):
    ensure_attribution_state(state)
    votes = category_votes or {}
    weighted = 0.0
    total_w = 0.0
    sources = []
    for cat, vote in votes.items():
        try:
            vote_f = float(vote or 0.0)
        except Exception:
            continue
        if vote_f <= 0:
            continue
        strength = category_strength(vote_f)
        b = blended_bucket(state, regime, cluster, cat)
        if not b.get("live") or b.get("gross_edge_pct") is None:
            continue
        weighted += b["gross_edge_pct"] * strength
        total_w += strength
        sources.append({"category": cat, "edge": b["gross_edge_pct"],
                        "n_effective": b["n_effective"]})
    if total_w <= 0:
        return None
    gross = weighted / total_w
    required = max(SUCCESS_THRESHOLD_PCT, float(friction_pct or 0.0) * FRICTION_SAFETY_MULT)
    return {
        "gross_edge_pct": round(gross, 4),
        "net_edge_pct": round(gross - float(friction_pct or 0.0), 4),
        "required_edge_pct": round(required, 4),
        "live": gross >= required,
        "edge_source": "attribution_v2",
        "sources": sources,
    }


def attribution_signal_weights(state):
    ensure_attribution_state(state)
    weights = {}
    buckets = state.get("attribution_buckets", {})
    for key, bucket in buckets.items():
        parts = key.split(":")
        if len(parts) != 3:
            continue
        regime, cluster, category = parts
        if category == "all":
            continue
        if regime == "all" and cluster == "all":
            weights[category] = attribution_multiplier(bucket, CATEGORY_CLAMP)
        elif cluster == "all":
            weights[f"{regime}:{category}"] = attribution_multiplier(bucket, REGIME_CATEGORY_CLAMP)
    return weights


def init_exit_bucket(bucket=None):
    bucket = bucket or {}
    bucket.setdefault("n", 0)
    bucket.setdefault("capture_n", 0)
    bucket.setdefault("sum_capture_ratio", 0.0)
    bucket.setdefault("sum_realized_pnl_pct", 0.0)
    bucket.setdefault("sum_mfe_pct", 0.0)
    bucket.setdefault("sum_mae_pct", 0.0)
    bucket.setdefault("too_late_n", 0)
    bucket.setdefault("too_early_n", 0)
    bucket.setdefault("post_3d_n", 0)
    bucket.setdefault("last_updated", None)
    bucket.setdefault("live", False)
    return bucket


def exit_bucket_key(regime, cluster, exit_reason=None):
    base = f"{normalize_regime(regime)}:{cluster or 'mixed'}"
    return f"{base}:{exit_reason}" if exit_reason else base


def _capture_ratio(realized_pnl_pct, mfe_pct):
    if mfe_pct and mfe_pct > 0:
        return float(realized_pnl_pct or 0.0) / mfe_pct
    return None


def record_exit_event(state, ticker, holding, exit_reason, realized_pnl_pct, price, ts=None,
                      regime=None):
    ensure_attribution_state(state)
    ts = int(ts or now_ts())
    avg = holding.get("avg_cost", 0) or 0
    peak = holding.get("peak", price) or price
    trough = holding.get("trough", avg) or avg
    mfe = (peak - avg) / avg * 100.0 if avg else 0.0
    mae = (trough - avg) / avg * 100.0 if avg else 0.0
    realized = float(realized_pnl_pct or 0.0)
    capture = _capture_ratio(realized, mfe)
    peak_giveback = max(0.0, mfe - realized)
    too_late = bool(mfe > 3.0 and (capture is None or capture < 0.15 or peak_giveback > mfe * 0.50))
    event = {
        "event_id": uuid.uuid4().hex,
        "ts": ts,
        "ticker": ticker,
        "price": price,
        "regime": normalize_regime(regime or (holding.get("entry_snapshot") or {}).get("market_regime")),
        "cluster": holding.get("entry_cluster") or (holding.get("entry_snapshot") or {}).get("entry_cluster") or "mixed",
        "exit_reason": exit_reason or "other",
        "realized_pnl_pct": round(realized, 4),
        "mfe_pct": round(mfe, 4),
        "mae_pct": round(mae, 4),
        "peak_giveback_pct": round(peak_giveback, 4),
        "capture_ratio": round(capture, 4) if capture is not None else None,
        "held_hours": round((ts - holding.get("entry_ts", ts)) / 3600.0, 2),
        "post_exit_returns": {},
        "post_bucketed": [],
        "too_late": too_late,
        "too_early": False,
        "exit_ladder_profile": holding.get("exit_ladder_profile"),
        "shadow_old_exit": holding.get("shadow_old_exit"),
        "old_exit_shadow_pnl_pct": (
            (holding.get("shadow_old_exit") or {}).get("pnl_pct")
        ),
        "new_exit_pnl_pct": round(realized, 4),
        "active_exit_rule": exit_reason or "other",
    }
    state.setdefault("exit_attribution_events", []).insert(0, event)
    state["exit_attribution_events"] = _cap_events_by_bucket(
        state["exit_attribution_events"]
    )[:max_exit_events()]
    update_exit_bucket(state, event)
    return event


def update_exit_bucket(state, event):
    buckets = state.setdefault("exit_attribution_buckets", {})
    keys = [
        exit_bucket_key(event.get("regime"), event.get("cluster")),
        exit_bucket_key(event.get("regime"), event.get("cluster"), event.get("exit_reason")),
    ]
    for key in keys:
        b = init_exit_bucket(buckets.setdefault(key, {}))
        b["n"] += 1
        cap = event.get("capture_ratio")
        if cap is not None:
            b["capture_n"] += 1
            b["sum_capture_ratio"] = round(b["sum_capture_ratio"] + cap, 4)
        b["sum_realized_pnl_pct"] = round(b["sum_realized_pnl_pct"] + event.get("realized_pnl_pct", 0), 4)
        b["sum_mfe_pct"] = round(b["sum_mfe_pct"] + event.get("mfe_pct", 0), 4)
        b["sum_mae_pct"] = round(b["sum_mae_pct"] + event.get("mae_pct", 0), 4)
        if event.get("too_late"):
            b["too_late_n"] += 1
        b["avg_capture_ratio"] = (
            round(b["sum_capture_ratio"] / b["capture_n"], 4)
            if b["capture_n"] else None
        )
        b["avg_realized_pnl_pct"] = round(b["sum_realized_pnl_pct"] / b["n"], 4)
        b["avg_mfe_pct"] = round(b["sum_mfe_pct"] / b["n"], 4)
        b["avg_mae_pct"] = round(b["sum_mae_pct"] / b["n"], 4)
        b["too_late_rate_pct"] = round(b["too_late_n"] / b["n"] * 100.0, 2)
        b["too_early_rate_pct"] = round(b["too_early_n"] / b["n"] * 100.0, 2)
        b["live"] = b["n"] >= EXIT_LIVE_N
        b["last_updated"] = event.get("ts")
        buckets[key] = b
    _refresh_status(state)


def _update_exit_post_bucket(state, event, hkey, ret):
    buckets = state.setdefault("exit_attribution_buckets", {})
    keys = [
        exit_bucket_key(event.get("regime"), event.get("cluster")),
        exit_bucket_key(event.get("regime"), event.get("cluster"), event.get("exit_reason")),
    ]
    for key in keys:
        bucket = init_exit_bucket(buckets.setdefault(key, {}))
        bucket["post_3d_n"] += 1
        if ret > 2.0:
            bucket["too_early_n"] += 1
        denom = max(1, bucket["post_3d_n"])
        bucket["too_early_rate_pct"] = round(bucket["too_early_n"] / denom * 100.0, 2)
        bucket["live"] = bucket["n"] >= EXIT_LIVE_N
        bucket["last_updated"] = event.get("ts")
        buckets[key] = bucket
    event.setdefault("post_bucketed", []).append(hkey)


def update_exit_post_outcomes(state, price_lookup, ts=None):
    ensure_attribution_state(state)
    ts = int(ts or now_ts())
    updated = 0
    for event in state.get("exit_attribution_events", []):
        if "3d" in event.setdefault("post_exit_returns", {}):
            continue
        if ts - event.get("ts", 0) < 3 * 86400:
            continue
        start = event.get("price") or 0
        cur = price_lookup(event.get("ticker"))
        if not start or not cur:
            continue
        ret = (cur - start) / start * 100.0
        event["post_exit_returns"]["3d"] = round(ret, 4)
        event["too_early"] = ret > 2.0
        if "3d" not in event.setdefault("post_bucketed", []):
            _update_exit_post_bucket(state, event, "3d", ret)
        updated += 1
    _refresh_status(state)
    return updated


def exit_profile(state, regime, cluster):
    ensure_attribution_state(state)
    key = exit_bucket_key(regime, cluster)
    b = init_exit_bucket((state.get("exit_attribution_buckets") or {}).get(key, {}))
    if b.get("n", 0) < EXIT_LIVE_N:
        return {"live": False, "key": key, "n": b.get("n", 0),
                "stop_mult": 1.0, "trail_mult": 1.0, "aging_mult": 1.0,
                "notes": "shadow"}
    cluster = normalize_cluster(cluster)
    stop_mult = 1.0
    trail_mult = 1.0
    aging_mult = 1.0
    notes = []
    too_late = b.get("too_late_rate_pct", 0) / 100.0
    too_early = b.get("too_early_rate_pct", 0) / 100.0
    capture = b.get("avg_capture_ratio")
    capture = capture if capture is not None else 0
    if cluster == "dip":
        if too_early > 0.20 or capture < 0.25:
            stop_mult *= 1.15; trail_mult *= 1.20; aging_mult *= 1.30
            notes.append("dip wider/slower")
        if too_late > 0.30:
            trail_mult *= 0.90
            notes.append("dip reduce giveback")
    elif cluster in ("breakout", "momentum"):
        if too_late > 0.20:
            stop_mult *= 0.85; trail_mult *= 0.80; aging_mult *= 0.70
            notes.append("momentum faster failure")
        elif too_early > 0.25:
            stop_mult *= 1.10; trail_mult *= 1.10
            notes.append("momentum less early")
    elif cluster == "news_catalyst_confirmed":
        stop_mult *= 0.90; trail_mult *= 0.85; aging_mult *= 0.50
        notes.append("news short timeout")
    elif cluster == "trend_continuation":
        if capture >= 0.40 and too_late < 0.25:
            trail_mult *= 1.20; aging_mult *= 1.30
            notes.append("trend let run")
        elif too_late > 0.30:
            trail_mult *= 0.90; aging_mult *= 0.90
            notes.append("trend reduce giveback")
    stop_mult = round(clamp(stop_mult, 0.75, 1.30), 4)
    trail_mult = round(clamp(trail_mult, 0.70, 1.35), 4)
    aging_mult = round(clamp(aging_mult, 0.50, 1.50), 4)
    return {"live": True, "key": key, "n": b.get("n", 0),
            "stop_mult": stop_mult, "trail_mult": trail_mult,
            "aging_mult": aging_mult, "notes": ", ".join(notes) or "neutral"}


def summarize_attribution(state):
    ensure_attribution_state(state)
    _refresh_status(state)
    entry = []
    for key, bucket in (state.get("attribution_buckets") or {}).items():
        b = init_entry_bucket(bucket)
        if b.get("n", 0) <= 0:
            continue
        entry.append({
            "bucket": key,
            "n": b["n"],
            "live": b["n"] >= ENTRY_LIVE_N,
            "p_success": round(p_success(b), 4),
            "ema_net_ret_pct": b.get("ema_net_ret_pct", 0.0),
            "avg_win_pct": b.get("avg_win_pct", 0.0),
            "avg_loss_pct": b.get("avg_loss_pct", 0.0),
            "payoff_ratio": b.get("payoff_ratio"),
            "multiplier": attribution_multiplier(b),
            "score": round(bucket_score(b), 4),
        })
    entry.sort(key=lambda r: (r["live"], r["ema_net_ret_pct"], r["n"]), reverse=True)
    ranked_by_return = sorted(entry, key=lambda r: (r["ema_net_ret_pct"], r["n"]),
                              reverse=True)
    worst_by_return = sorted(entry, key=lambda r: (r["ema_net_ret_pct"], r["n"]))
    skipped_winners = []
    for event in state.get("attribution_events", []) or []:
        if event.get("decision") == "executed":
            continue
        ret5 = (event.get("forward_returns") or {}).get(MAIN_HORIZON)
        if ret5 is None:
            continue
        net5 = ret5 - float(event.get("friction_pct") or 0.0)
        if net5 > SUCCESS_THRESHOLD_PCT:
            skipped_winners.append({
                "ticker": event.get("ticker"),
                "source": event.get("source"),
                "cluster": event.get("cluster"),
                "regime": event.get("regime"),
                "skip_reason": event.get("skip_reason"),
                "net_5d_pct": round(net5, 4),
                "raw_5d_pct": ret5,
                "confidence": event.get("confidence"),
                "ts": event.get("ts"),
            })
    skipped_winners.sort(key=lambda r: r["net_5d_pct"], reverse=True)
    exits = []
    active_profiles = []
    for key, bucket in (state.get("exit_attribution_buckets") or {}).items():
        b = init_exit_bucket(bucket)
        if b.get("n", 0) <= 0:
            continue
        parts = key.split(":")
        if len(parts) == 2 and b.get("n", 0) >= EXIT_LIVE_N:
            active_profiles.append(exit_profile(state, parts[0], parts[1]))
        exits.append({
            "bucket": key,
            "n": b["n"],
            "capture_n": b.get("capture_n", 0),
            "post_3d_n": b.get("post_3d_n", 0),
            "live": b["n"] >= EXIT_LIVE_N,
            "avg_capture_ratio": b.get("avg_capture_ratio"),
            "avg_realized_pnl_pct": b.get("avg_realized_pnl_pct"),
            "avg_mfe_pct": b.get("avg_mfe_pct"),
            "too_late_rate_pct": b.get("too_late_rate_pct"),
            "too_early_rate_pct": b.get("too_early_rate_pct"),
        })
    exits.sort(key=lambda r: (r["live"], r["n"]), reverse=True)
    return {
        "status": state.get("attribution_status", {}),
        "entry_buckets": entry,
        "best_clusters": ranked_by_return[:10],
        "worst_clusters": worst_by_return[:10],
        "skipped_winners": skipped_winners[:25],
        "exit_buckets": exits,
        "active_exit_profiles": active_profiles,
        "recent_events": state.get("attribution_events", [])[:25],
        "recent_exit_events": state.get("exit_attribution_events", [])[:25],
    }


def _refresh_status(state):
    ensure_attribution_state(state)
    status = state.setdefault("attribution_status", {})
    entry_live = any((b or {}).get("n", 0) >= ENTRY_LIVE_N
                     for b in (state.get("attribution_buckets") or {}).values())
    exit_live = any((b or {}).get("n", 0) >= EXIT_LIVE_N
                    for b in (state.get("exit_attribution_buckets") or {}).values())
    status["entry_live"] = bool(entry_live)
    status["exit_live"] = bool(exit_live)
    status["entry_bucket_count"] = len(state.get("attribution_buckets", {}))
    status["exit_bucket_count"] = len(state.get("exit_attribution_buckets", {}))
    return status
