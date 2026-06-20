"""Cluster-specific exit ladder defaults and profile composition."""

CLUSTER_ORDER = (
    "news_catalyst_confirmed",
    "dip",
    "breakout",
    "trend_continuation",
    "momentum",
    "mixed",
)

LEGACY_CLUSTER_ALIASES = {
    "trend": "trend_continuation",
    "news": "mixed",
    "analyst": "mixed",
    "insider": "mixed",
    "news_catalyst": "news_catalyst_confirmed",
    "news_catalyst_unconfirmed": "mixed",
    None: "mixed",
}

EXIT_LADDERS = {
    "news_catalyst_confirmed": {
        "atr_stop_mult": 1.2,
        "partial_take_pct": 0.05,
        "trail_start_pct": 0.03,
        "trail_mult": 0.9,
        "max_hold_days": 7,
        "failure_timeout_days": 2,
    },
    "dip": {
        "atr_stop_mult": 1.5,
        "partial_take_pct": 0.08,
        "trail_start_pct": 0.05,
        "trail_mult": 1.3,
        "max_hold_days": 15,
        "failure_timeout_days": 5,
    },
    "breakout": {
        "atr_stop_mult": 1.0,
        "partial_take_pct": 0.05,
        "trail_start_pct": 0.03,
        "trail_mult": 0.8,
        "max_hold_days": 7,
        "failure_timeout_days": 2,
    },
    "trend_continuation": {
        "atr_stop_mult": 1.2,
        "partial_take_pct": 0.08,
        "trail_start_pct": 0.05,
        "trail_mult": 1.1,
        "max_hold_days": 20,
        "failure_timeout_days": 7,
    },
    "momentum": {
        "atr_stop_mult": 1.0,
        "partial_take_pct": 0.05,
        "trail_start_pct": 0.03,
        "trail_mult": 0.9,
        "max_hold_days": 10,
        "failure_timeout_days": 3,
    },
    "mixed": {
        "atr_stop_mult": 1.2,
        "partial_take_pct": 0.06,
        "trail_start_pct": 0.04,
        "trail_mult": 1.0,
        "max_hold_days": 10,
        "failure_timeout_days": 3,
    },
}

PROFILE_CLAMPS = {
    "atr_stop_mult": (0.80, 1.80),
    "trail_mult": (0.70, 1.50),
    "partial_take_pct": (0.03, 0.12),
    "max_hold_days": (2.0, 25.0),
    "failure_timeout_days": (1.0, 8.0),
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def normalize_cluster(cluster):
    c = str(cluster or "mixed").lower()
    c = LEGACY_CLUSTER_ALIASES.get(c, c)
    return c if c in EXIT_LADDERS else "mixed"


def ladder_for_cluster(cluster):
    c = normalize_cluster(cluster)
    out = dict(EXIT_LADDERS[c])
    out["cluster"] = c
    return out


def compose_exit_profile(cluster, learned=None):
    """Blend default cluster ladder with learned V2 exit profile, tightly clamped."""
    learned = learned or {}
    base = ladder_for_cluster(cluster)
    aging_mult = float(learned.get("aging_mult", 1.0) or 1.0)
    too_late = float(learned.get("too_late_rate_pct", 0.0) or 0.0)
    too_early = float(learned.get("too_early_rate_pct", 0.0) or 0.0)
    if learned.get("live") and too_late > 30.0:
        partial_mult = 0.85
    elif learned.get("live") and too_early > 25.0:
        partial_mult = 1.15
    else:
        partial_mult = 1.0
    profile = {
        "cluster": base["cluster"],
        "base_ladder": dict(base),
        "learned_live": bool(learned.get("live")),
        "learned_key": learned.get("key"),
        "learned_n": learned.get("n", 0),
        "learned_notes": learned.get("notes", "shadow"),
        "atr_stop_mult": clamp(
            base["atr_stop_mult"] * float(learned.get("stop_mult", 1.0) or 1.0),
            *PROFILE_CLAMPS["atr_stop_mult"],
        ),
        "trail_mult": clamp(
            base["trail_mult"] * float(learned.get("trail_mult", 1.0) or 1.0),
            *PROFILE_CLAMPS["trail_mult"],
        ),
        "partial_take_pct": clamp(
            base["partial_take_pct"] * partial_mult, *PROFILE_CLAMPS["partial_take_pct"]
        ),
        "trail_start_pct": base["trail_start_pct"],
        "max_hold_days": clamp(
            base["max_hold_days"] * aging_mult, *PROFILE_CLAMPS["max_hold_days"]
        ),
        "failure_timeout_days": clamp(
            base["failure_timeout_days"] * aging_mult,
            *PROFILE_CLAMPS["failure_timeout_days"],
        ),
    }
    for key in ("atr_stop_mult", "trail_mult", "partial_take_pct",
                "trail_start_pct", "max_hold_days", "failure_timeout_days"):
        profile[key] = round(profile[key], 4)
    profile["notes"] = (
        f"cluster {profile['cluster']} ladder"
        + (f"; learned {profile['learned_notes']}" if profile["learned_live"] else "")
    )
    return profile


def apply_regime_exit_tightening(profile, regime_v3=None):
    """Tighten active ladder in weak regimes; no-op for normal/fallback regimes."""
    if not profile:
        return profile
    label = str(regime_v3 or "").lower()
    if label == "risk_off_neutral":
        mods = {"atr_stop_mult": 0.95, "trail_mult": 0.90,
                "partial_take_pct": 0.85,
                "max_hold_days": 0.90, "failure_timeout_days": 0.85}
    elif label == "bear":
        mods = {"atr_stop_mult": 0.90, "trail_mult": 0.85,
                "partial_take_pct": 0.70,
                "max_hold_days": 0.75, "failure_timeout_days": 0.75}
    elif label == "panic":
        mods = {"atr_stop_mult": 0.80, "trail_mult": 0.75,
                "partial_take_pct": 0.60,
                "max_hold_days": 0.50, "failure_timeout_days": 0.50}
    else:
        return profile
    out = dict(profile)
    for key, mult in mods.items():
        lo, hi = PROFILE_CLAMPS[key]
        out[key] = round(clamp(float(out.get(key, 0) or 0) * mult, lo, hi), 4)
    out["regime_v3_exit_tightening"] = label
    out["notes"] = f"{out.get('notes', '')}; V3 {label} tightened".strip("; ")
    return out
