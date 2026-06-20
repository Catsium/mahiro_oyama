"""Advisory extra-ticker suggestion scoring.

This module ranks display ideas only. It never changes execution EV ranking.
"""
from __future__ import annotations

import math
from typing import Any

from trading.attribution import bucket_feedback_key
from trading.config import DEFAULT_CONFIG
from trading.sizing import sector_rotation_bonus


ISSUER_ALIASES: dict[str, str] = {}
KNIFE_CATALYST_TYPES = {
    "earnings_miss",
    "guidance_cut",
    "regulatory_risk",
    "lawsuit_investigation",
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _suggestion_cfg(config: dict | None) -> dict:
    config = config or DEFAULT_CONFIG
    if isinstance(config, dict) and "suggestion" in config:
        return config.get("suggestion") or {}
    if isinstance(config, dict) and "min_suggestion_score" in config:
        return config
    return {}


def normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def issuer_key(ticker: str) -> str:
    normalized = normalize_ticker(ticker)
    return ISSUER_ALIASES.get(normalized, normalized)


def coarse_scan_score(row: dict) -> float:
    """Cheap scan score from already-present leaderboard fields."""
    direction = 1.0 if _float(row.get("direction")) > 0 else 0.0
    conf = _clamp(_float(row.get("confidence")) / 100.0)
    score = _clamp((_float(row.get("score")) + 5.0) / 10.0)
    pct = _clamp((_float(row.get("pct")) + 5.0) / 15.0)
    return round(direction * (0.45 * conf + 0.35 * score + 0.20 * pct), 4)


def prior_uncertainty_penalty(candidate: dict, *,
                              min_live_samples_soft: int = 20) -> float:
    if candidate.get("edge_source") != "confidence_prior":
        return 0.0
    samples = int(_float(candidate.get("edge_samples"), 0.0))
    trust = _clamp(samples / max(1.0, float(min_live_samples_soft)))
    return round(0.16 * (1.0 - trust) + 0.04, 4)


def freshness_component(candidate: dict, now_ts: int, *,
                        max_age_hours: float = 24.0,
                        half_life_hours: float = 12.0) -> float:
    ages = []
    for key in ("age_hours", "freshness_age_hours", "scan_age_hours", "news_age_hours"):
        if candidate.get(key) is not None:
            ages.append(_float(candidate.get(key), max_age_hours))
    rec = candidate.get("rec") or {}
    catalyst = rec.get("catalyst") or {}
    if catalyst.get("age_hours") is not None:
        ages.append(_float(catalyst.get("age_hours"), max_age_hours))
    if candidate.get("ts"):
        ages.append(max(0.0, (now_ts - _float(candidate.get("ts"))) / 3600.0))
    if not ages:
        return 0.55
    age = min(ages)
    if age > max_age_hours:
        return 0.0
    return round(_clamp(math.pow(0.5, age / max(0.1, half_life_hours))), 4)


def _sector(candidate: dict) -> str | None:
    if candidate.get("sector"):
        return candidate.get("sector")
    try:
        from trading.risk import get_sector
        return get_sector(candidate.get("ticker") or "")
    except Exception:
        return None


def _corr_group(candidate: dict) -> str | None:
    if candidate.get("corr_group"):
        return candidate.get("corr_group")
    try:
        from trading.risk import get_corr_group
        return get_corr_group(candidate.get("ticker") or "")
    except Exception:
        return None


def duplication_penalty(candidate: dict, selected: list[dict], holdings: dict, *,
                        same_sector_limit: int = 1,
                        same_corr_group_limit: int = 1) -> float:
    sector = _sector(candidate)
    corr_group = _corr_group(candidate)
    sector_count = 0
    corr_count = 0
    for row in selected:
        sector_count += 1 if sector and _sector(row) == sector else 0
        corr_count += 1 if corr_group and _corr_group(row) == corr_group else 0
    for ticker, holding in (holdings or {}).items():
        snap = holding.get("entry_snapshot") or {}
        h_sector = snap.get("sector") or holding.get("sector")
        h_group = holding.get("corr_group")
        if not h_sector:
            try:
                from trading.risk import get_sector
                h_sector = get_sector(ticker)
            except Exception:
                h_sector = None
        if not h_group:
            try:
                from trading.risk import get_corr_group
                h_group = get_corr_group(ticker)
            except Exception:
                h_group = None
        sector_count += 1 if sector and h_sector == sector else 0
        corr_count += 1 if corr_group and h_group == corr_group else 0
    sector_excess = max(0, sector_count + (1 if sector else 0) - int(same_sector_limit))
    corr_excess = max(0, corr_count + (1 if corr_group else 0) - int(same_corr_group_limit))
    return round(min(0.35, 0.12 * sector_excess + 0.16 * corr_excess), 4)


def max_holding_corr(candidate: dict, holdings: dict, threshold: float = 0.70) -> float:
    pv = candidate.get("portfolio_variance") or {}
    if pv.get("max_pair_corr") is not None:
        return _float(pv.get("max_pair_corr"), 0.0)
    held = [str(t).upper() for t in (holdings or {}).keys() if t]
    ticker = normalize_ticker(candidate.get("ticker") or "")
    if not ticker or not held:
        return 0.0
    try:
        from trading.portfolio_variance import load_close_history, smoothed_corr
        from trading.risk import get_corr_group, get_sector
        hist = load_close_history([ticker] + held)
        max_corr = 0.0
        for h in held:
            corr = smoothed_corr(ticker, h, hist, candidate.get("regime"),
                                 get_sector, get_corr_group)
            max_corr = max(max_corr, float(corr or 0.0))
            if max_corr > threshold:
                break
        return round(max_corr, 4)
    except Exception:
        return 0.0


def fragility_penalty(candidate: dict, config: dict | None = None) -> float:
    cfg = _suggestion_cfg(config)
    ctx = candidate.get("ctx") or {}
    rec = candidate.get("rec") or {}
    catalyst_type = candidate.get("catalyst_type") or (rec.get("catalyst") or {}).get("type")
    penalty = prior_uncertainty_penalty(
        candidate,
        min_live_samples_soft=cfg.get("min_live_samples_soft", 20),
    )
    if candidate.get("source") == "scan" and _float(candidate.get("scan_age_hours")) > 2.0:
        penalty += 0.06
    if catalyst_type and not candidate.get("catalyst_confirmed"):
        penalty += 0.05
    if candidate.get("vol_regime") == "explosive" and _float(ctx.get("rsi"), 50.0) < 35.0:
        penalty += 0.12
    if catalyst_type in KNIFE_CATALYST_TYPES:
        penalty += 0.30
    if _float(ctx.get("week_chg_pct")) <= -5.0 and _float(ctx.get("dist_from_high_pct")) <= -10.0:
        penalty += 0.18
    return round(min(0.45, penalty), 4)


def feedback_multiplier(candidate: dict, feedback_stats: dict, *,
                        min_samples: int = 15,
                        min_mult: float = 0.90,
                        max_mult: float = 1.10) -> float:
    key = candidate.get("feedback_bucket") or bucket_feedback_key(candidate)
    stats = (feedback_stats or {}).get(key) or {}
    n = int(stats.get("n", 0) or 0)
    if n < min_samples:
        return 1.0
    useful = _float(stats.get("useful"), 0.0)
    weak = _float(stats.get("weak"), 0.0)
    hidden = _float(stats.get("hide"), 0.0)
    alpha = 1.0 + useful
    beta = 1.0 + weak + hidden
    score = alpha / max(1.0, alpha + beta)
    mult = min_mult + (max_mult - min_mult) * score
    return round(_clamp(mult, min_mult, max_mult), 4)


def compute_suggestion_score(candidate: dict, *, holdings: dict,
                             selected: list[dict],
                             top_sectors: list[str] | None,
                             feedback_stats: dict,
                              recent_suggestions: dict[str, int],
                              loss_cooldowns: dict[str, int] | None = None,
                              recent_sells: dict | None = None,
                              now_ts: int,
                              config: dict) -> dict:
    cfg = _suggestion_cfg(config)
    out = dict(candidate)
    rec = out.get("rec") or {}
    ctx = out.get("ctx") or {}
    confidence_component = _clamp(_float(rec.get("confidence")) / 100.0)
    # Net edge gets full credit around 2%+; smaller edges can still pass hard gates.
    net_edge_component = _clamp(_float(out.get("net_edge_pct")) / 2.0)
    adv = _float(ctx.get("avg_dollar_vol_20d"), 0.0)
    liquidity_component = _clamp(math.log10(max(adv, 1.0)) / 9.0) if adv else 0.5
    catalyst_type = out.get("catalyst_type") or (rec.get("catalyst") or {}).get("type")
    catalyst_confirmed = bool(out.get("catalyst_confirmed"))
    catalyst_component = 0.5
    if catalyst_type in KNIFE_CATALYST_TYPES:
        catalyst_component = 0.0
    elif catalyst_confirmed:
        catalyst_component = _clamp(0.65 + _float(out.get("catalyst_score_shadow")) / 2.0)
    regime_fit = _clamp(
        (_float(out.get("regime_risk_mult"), 1.0) * _float(out.get("cluster_regime_mult"), 1.0)) / 1.20
    )
    rotation_bonus = sector_rotation_bonus(out, {}, top_sectors=top_sectors)
    sector_component = _clamp(0.5 + rotation_bonus * 5.0)
    fb_mult = feedback_multiplier(
        out,
        feedback_stats,
        min_samples=int(cfg.get("feedback_min_samples", 15)),
        min_mult=float(cfg.get("feedback_min_mult", 0.90)),
        max_mult=float(cfg.get("feedback_max_mult", 1.10)),
    )
    raw = (
        0.33 * confidence_component
        + 0.28 * net_edge_component
        + 0.12 * liquidity_component
        + 0.12 * catalyst_component
        + 0.08 * regime_fit
        + 0.07 * sector_component
    )
    dup_penalty = duplication_penalty(
        out,
        selected,
        holdings,
        same_sector_limit=int(cfg.get("same_sector_limit", 1)),
        same_corr_group_limit=int(cfg.get("same_corr_group_limit", 1)),
    )
    frag_penalty = fragility_penalty(out, cfg)
    score = _clamp(raw * fb_mult - dup_penalty - frag_penalty)
    holding_corr = max_holding_corr(
        out, holdings, threshold=float(cfg.get("max_holding_corr", 0.70))
    )
    out.update({
        "now_ts": int(now_ts),
        "_recent_suggestions": recent_suggestions or {},
        "_loss_cooldowns": loss_cooldowns or {},
        "_recent_sells": recent_sells or {},
        "sector": _sector(out),
        "corr_group": _corr_group(out),
        "feedback_bucket": bucket_feedback_key(out),
        "suggestion_max_holding_corr": holding_corr,
        "suggestion_score": round(score, 4),
        "suggestion_reasons": [
            f"conf {int(_float(rec.get('confidence')))}",
            f"net {_float(out.get('net_edge_pct')):+.2f}%",
            f"liquidity {liquidity_component:.2f}",
        ],
        "suggestion_fragility_penalty": frag_penalty,
        "suggestion_duplication_penalty": dup_penalty,
        "suggestion_feedback_mult": fb_mult,
    })
    ok, reason = should_emit_suggestion(out, cfg)
    out["showable"] = ok
    out["show_reason"] = reason
    return out


def diversify_suggestions(candidates: list[dict], *, holdings: dict,
                          top_sectors: list[str] | None,
                          feedback_stats: dict,
                          recent_suggestions: dict[str, int],
                          loss_cooldowns: dict[str, int] | None = None,
                          recent_sells: dict | None = None,
                          now_ts: int,
                          config: dict) -> list[dict]:
    cfg = _suggestion_cfg(config)
    selected: list[dict] = []
    used_catalysts: set[str] = set()
    for cand in candidates:
        scored = compute_suggestion_score(
            cand,
            holdings=holdings,
            selected=selected,
            top_sectors=top_sectors,
            feedback_stats=feedback_stats,
            recent_suggestions=recent_suggestions,
            loss_cooldowns=loss_cooldowns,
            recent_sells=recent_sells,
            now_ts=now_ts,
            config=cfg,
        )
        if not scored.get("showable"):
            continue
        ctype = scored.get("catalyst_type") or ((scored.get("rec") or {}).get("catalyst") or {}).get("type")
        if ctype and ctype in used_catalysts:
            continue
        if ctype:
            used_catalysts.add(ctype)
        selected.append(scored)
        if len(selected) >= int(cfg.get("max_extra_tickers", 2)):
            break
    return selected


def rank_suggestion_candidates(candidates: list[dict], *, holdings: dict,
                               top_sectors: list[str] | None,
                                feedback_stats: dict,
                                recent_suggestions: dict[str, int],
                                loss_cooldowns: dict[str, int] | None = None,
                                recent_sells: dict | None = None,
                                now_ts: int,
                                config: dict) -> list[dict]:
    cfg = _suggestion_cfg(config)
    if not cfg.get("enabled", True):
        return []
    prelim = []
    for cand in candidates:
        rec = cand.get("rec") or {}
        if not (cand.get("tradable") or rec.get("cls") in ("buy", "strong-buy")):
            continue
        scored = compute_suggestion_score(
            cand,
            holdings=holdings,
            selected=[],
            top_sectors=top_sectors,
            feedback_stats=feedback_stats,
            recent_suggestions=recent_suggestions,
            loss_cooldowns=loss_cooldowns,
            recent_sells=recent_sells,
            now_ts=now_ts,
            config=cfg,
        )
        prelim.append(scored)
    prelim.sort(
        key=lambda c: (
            c.get("showable", False),
            c.get("suggestion_score", 0.0),
            c.get("net_edge_pct", -999.0),
            (c.get("rec") or {}).get("confidence", 0),
        ),
        reverse=True,
    )
    return diversify_suggestions(
        prelim,
        holdings=holdings,
        top_sectors=top_sectors,
        feedback_stats=feedback_stats,
        recent_suggestions=recent_suggestions,
        loss_cooldowns=loss_cooldowns,
        recent_sells=recent_sells,
        now_ts=now_ts,
        config=cfg,
    )


def should_emit_suggestion(candidate: dict, config: dict) -> tuple[bool, str]:
    cfg = _suggestion_cfg(config)
    if not cfg.get("enabled", True):
        return False, "suggestions disabled"
    ticker = normalize_ticker(candidate.get("ticker") or "")
    rec = candidate.get("rec") or {}
    ctx = candidate.get("ctx") or {}
    sell_info = (candidate.get("_recent_sells") or {}).get(ticker)
    if sell_info:
        if isinstance(sell_info, dict):
            sell_ts = int(sell_info.get("ts", 0) or 0)
            sell_reason = str(sell_info.get("reason", "") or "")
        else:
            sell_ts = int(sell_info or 0)
            sell_reason = ""
        age_sec = int(candidate.get("now_ts", 0) or 0) - sell_ts
        if sell_reason == "loss" and age_sec < int(cfg.get("loss_cooldown_sec", 14 * 86400)):
            return False, "loss_cooldown_14d"
        if age_sec < 2 * 3600:
            return False, "recently_sold_cooldown"
    score = _float(candidate.get("suggestion_score"))
    if score < _float(cfg.get("min_suggestion_score"), 0.72):
        return False, "score below suggestion threshold"
    net = _float(candidate.get("net_edge_pct"))
    gross = _float(candidate.get("gross_edge_pct"))
    if net < _float(cfg.get("min_net_edge_pct"), 0.35):
        return False, "net edge below suggestion threshold"
    if candidate.get("edge_source") == "confidence_prior":
        if gross < _float(cfg.get("min_gross_edge_prior_pct"), 0.90):
            return False, "prior edge too weak"
    elif gross < _float(cfg.get("min_gross_edge_live_pct"), 0.60):
        return False, "live edge too weak"
    if candidate.get("source") == "scan" and _float(rec.get("confidence")) < _float(cfg.get("min_scan_confidence"), 68):
        return False, "scan confidence below suggestion threshold"
    price = _float(candidate.get("price"))
    if price < _float(cfg.get("min_price"), 5.0):
        return False, "price below suggestion threshold"
    adv = _float(ctx.get("avg_dollar_vol_20d"), 0.0)
    if adv and adv < _float(cfg.get("min_adv_usd"), 15_000_000):
        return False, "liquidity below suggestion threshold"
    loss_ts = (candidate.get("_loss_cooldowns") or {}).get(ticker)
    if loss_ts and candidate.get("now_ts", 0):
        age = int(candidate["now_ts"]) - int(loss_ts)
        if age < int(cfg.get("loss_cooldown_sec", 14 * 86400)):
            return False, "recent loss cooldown"
    if not ctx.get("is_dip"):
        min_vol = _float(cfg.get("non_dip_min_vol_ratio"), 1.2)
        if _float(ctx.get("vol_ratio"), 1.0) < min_vol:
            return False, "non-dip volume confirmation missing"
    pv = candidate.get("portfolio_variance") or {}
    max_corr = _float(candidate.get("suggestion_max_holding_corr"),
                      _float(pv.get("max_pair_corr"), 0.0))
    if max_corr > _float(cfg.get("max_holding_corr"), 0.70):
        return False, "correlation to holding too high"
    catalyst_type = candidate.get("catalyst_type") or (rec.get("catalyst") or {}).get("type")
    if catalyst_type in KNIFE_CATALYST_TYPES:
        return False, "negative catalyst"
    if candidate.get("suggestion_duplication_penalty", 0) >= 0.12:
        return False, "sector or correlation duplicate"
    recent = (candidate.get("_recent_suggestions") or {}).get(ticker)
    if recent:
        if isinstance(recent, dict):
            recent_ts = int(recent.get("ts", 0) or 0)
            prev_score = _float(recent.get("score"), score)
        else:
            recent_ts = int(recent or 0)
            prev_score = score
        if recent_ts and candidate.get("now_ts", 0):
            age = int(candidate["now_ts"]) - recent_ts
        else:
            age = 0
        if age < int(cfg.get("ticker_cooldown_sec", 21600)) and score < prev_score + 0.08:
            return False, "recently suggested"
    return True, "show"
