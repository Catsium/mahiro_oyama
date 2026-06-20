"""Catalyst classifier foundation.

Shadow-first: this tags events and confirmation quality. It does not directly
change signal score or sizing unless config flags are enabled later.
"""

import math
import re
import time

from trading.config import CATALYST_CONFIG, CATALYST_SCORE


CATALYST_TYPES = [
    "earnings_beat",
    "earnings_miss",
    "guidance_raise",
    "guidance_cut",
    "analyst_upgrade",
    "analyst_downgrade",
    "insider_buy",
    "insider_sell",
    "regulatory_risk",
    "lawsuit_investigation",
    "m_and_a",
    "product_launch",
    "macro_rates_inflation_jobs",
    "layoffs_restructuring",
    "mixed_or_unclear",
]

KEYWORDS = {
    "guidance_raise": (r"raise[sd]? guidance", r"guidance raise", r"boost[sd]? forecast", r"raises outlook"),
    "guidance_cut": (r"cut[sd]? guidance", r"lower[sd]? forecast", r"guidance cut", r"cuts outlook"),
    "earnings_beat": (r"earnings beat", r"beats estimates", r"tops estimates", r"better-than-expected"),
    "earnings_miss": (r"earnings miss", r"misses estimates", r"missed estimates", r"worse-than-expected"),
    "analyst_upgrade": (r"upgrade[sd]?", r"price target raised", r"initiates buy", r"outperform"),
    "analyst_downgrade": (r"downgrade[sd]?", r"price target cut", r"underperform", r"sell rating"),
    "regulatory_risk": (r"regulator", r"sec\b", r"ftc\b", r"doj\b", r"probe", r"antitrust"),
    "lawsuit_investigation": (r"lawsuit", r"investigation", r"class action", r"subpoena"),
    "m_and_a": (r"acquire[sd]?", r"merger", r"takeover", r"buyout", r"deal"),
    "product_launch": (r"launch", r"unveil", r"announces new product", r"release[sd]?"),
    "macro_rates_inflation_jobs": (r"\brates\b", r"inflation", r"\bcpi\b", r"\bpce\b", r"jobs report", r"fed\b"),
    "layoffs_restructuring": (r"layoff", r"restructur", r"job cuts", r"cost cuts"),
}


def _decay(age_hours, half_life_hours):
    if age_hours is None:
        return 0.5
    if age_hours < 0:
        return 1.0
    return 0.5 ** (age_hours / max(float(half_life_hours or 1), 0.1))


def _article_age_hours(article, now_ts=None):
    now_ts = now_ts or time.time()
    pub_ts = article.get("pub_ts")
    if pub_ts:
        try:
            return max(0.0, (float(now_ts) - float(pub_ts)) / 3600.0)
        except Exception:
            return None
    return None


def _headline_matches(text):
    hits = []
    lower = (text or "").lower()
    for ctype, patterns in KEYWORDS.items():
        if any(re.search(p, lower) for p in patterns):
            hits.append(ctype)
    return hits


def _price_volume_confirmed(ctx, cfg):
    ctx = ctx or {}
    vol_ratio = float(ctx.get("vol_ratio", 0) or 0)
    mom = float(ctx.get("mom_30d_pct", 0) or 0)
    week = float(ctx.get("week_chg_pct", 0) or 0)
    current = float(ctx.get("current", 0) or 0)
    ma30 = float(ctx.get("ma30", 0) or 0)
    macd = float(ctx.get("macd_hist", 0) or 0)
    macd_prev = float(ctx.get("macd_hist_prev", 0) or 0)
    volume_ok = vol_ratio >= float(cfg.get("min_rel_volume_confirm", 1.5))
    price_ok = mom >= 2.0 or week >= 2.0 or (current > 0 and ma30 > 0 and current > ma30)
    trend_ok = macd >= macd_prev
    return bool(volume_ok and price_ok and trend_ok)


def classify_catalyst(articles=None, earnings=None, analyst=None, insider=None,
                      ctx=None, now_ts=None, config=None):
    cfg = dict(CATALYST_CONFIG)
    if config:
        cfg.update(config.get("catalyst", config))
    score_map = CATALYST_SCORE
    if config and "catalyst_score" in config:
        score_map = config["catalyst_score"]

    candidates = []
    max_news_age = float(cfg.get("news_max_age_hours", 48))
    half_life = float(cfg.get("news_half_life_hours", 6))
    for article in articles or []:
        age_h = _article_age_hours(article, now_ts)
        if age_h is not None and age_h > max_news_age:
            continue
        decay = _decay(age_h, half_life)
        for ctype in _headline_matches(article.get("title", "")):
            candidates.append({
                "type": ctype,
                "source": "news",
                "age_hours": round(age_h, 2) if age_h is not None else None,
                "score": float(score_map.get(ctype, 0.0)) * decay,
                "title": article.get("title", ""),
            })

    analyst_age_h = (analyst or {}).get("age_hours")
    analyst_max_h = float(cfg.get("analyst_max_age_days", 14)) * 24
    if analyst and (analyst_age_h is None or analyst_age_h <= analyst_max_h):
        net = float(analyst.get("net", 0) or 0)
        if net >= 0.5:
            candidates.append({"type": "analyst_upgrade", "source": "analyst",
                               "score": score_map["analyst_upgrade"], "age_hours": analyst_age_h})
        elif net <= -0.5:
            candidates.append({"type": "analyst_downgrade", "source": "analyst",
                               "score": score_map["analyst_downgrade"], "age_hours": analyst_age_h})

    insider_age_h = (insider or {}).get("age_hours")
    insider_max_h = float(cfg.get("insider_max_age_days", 30)) * 24
    if insider and (insider_age_h is None or insider_age_h <= insider_max_h):
        sent = float(insider.get("sentiment", 0) or 0)
        if insider.get("samples", 0) > 0 and sent >= 0.15:
            candidates.append({"type": "insider_buy", "source": "insider",
                               "score": score_map["insider_buy"], "age_hours": insider_age_h})
        elif insider.get("samples", 0) > 0 and sent <= -0.15:
            candidates.append({"type": "insider_sell", "source": "insider",
                               "score": score_map["insider_sell"], "age_hours": insider_age_h})

    if earnings and earnings.get("soon"):
        candidates.append({"type": "mixed_or_unclear", "source": "earnings_calendar",
                           "score": 0.0, "date": earnings.get("date")})

    if candidates:
        best = max(candidates, key=lambda c: abs(c.get("score", 0.0)))
        ctype = best["type"]
        raw_score = float(best.get("score", 0.0) or 0.0)
    else:
        best = {}
        ctype = "mixed_or_unclear"
        raw_score = 0.0

    max_contrib = float(cfg.get("max_score_contribution", 0.50))
    shadow_score = max(-max_contrib, min(max_contrib, raw_score))
    confirmed = _price_volume_confirmed(ctx, cfg)
    if ctype == "mixed_or_unclear" and cfg.get("require_price_confirmation_for_ambiguous", True):
        confirmed = False

    return {
        "type": ctype,
        "score_shadow": round(shadow_score, 4),
        "confirmed": bool(confirmed and ctype != "mixed_or_unclear"),
        "price_volume_confirmed": bool(confirmed),
        "source": best.get("source"),
        "all_candidates": candidates[:8],
        "shadow_first": bool(cfg.get("shadow_first", True)),
        "size_effect_enabled": bool(cfg.get("size_effect_enabled", False)),
        "score_effect_enabled": bool(cfg.get("score_effect_enabled", False)),
        "cluster_tagging_enabled": bool(cfg.get("cluster_tagging_enabled", True)),
    }


def catalyst_cluster(catalyst):
    if not catalyst:
        return None
    if (catalyst.get("cluster_tagging_enabled", True)
            and catalyst.get("confirmed")
            and catalyst.get("score_shadow", 0) > 0):
        return "news_catalyst_confirmed"
    return None
