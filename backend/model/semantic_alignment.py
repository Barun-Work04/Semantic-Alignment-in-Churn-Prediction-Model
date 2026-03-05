"""
semantic_alignment.py – Semantic Alignment Layer for Churn Prediction

How it works
------------
Semantic checks run against raw feature values BEFORE the ML output is trusted.
Each failing check adds a probability boost to the ML's churn probability.
The final prediction is derived from the adjusted probability (threshold 0.50),
so the ML is never blindly ignored — it is corrected when logic demands it.

Rule priority (all rules are evaluated; boosts accumulate):
    1. SESSION_MATH_IMPOSSIBLE  – claimed session load exceeds available time
    2. BURNOUT_RISK_HIGH        – extreme playtime + long sessions
    3. BINGE_SESSION_PATTERN    – very few sessions but very high daily playtime
    4. NEAR_ZERO_ENGAGEMENT     – player barely active (hard floor override)
    5. ML trust zone            – no flags fired; use ML probabilities as-is

Label convention (consistent throughout the codebase):
    final_prediction = 0  →  Player STAYS
    final_prediction = 1  →  Player CHURNS

Feature ranges observed in gaming_churn.csv:
    PlayTimeHours:             0.2 – 10 h/day  (UI allows up to 10)
    SessionsPerWeek:           1   – 7
    AvgSessionDurationMinutes: 5   – 180 min   (UI allows up to 180)
"""

from __future__ import annotations

# ── Thresholds ────────────────────────────────────────────────────────────────
_NEARZERO_PLAYTIME = 0.5    # h/day  – all three must be below their limit
_NEARZERO_SESSIONS = 2      # /week
_NEARZERO_DURATION = 20     # min/session

_BURNOUT_PLAYTIME  = 5.0    # h/day  – both must exceed their limit
_BURNOUT_DURATION  = 90     # min/session

_BINGE_SESSIONS_MAX  = 2    # /week  – sessions low ...
_BINGE_PLAYTIME_MIN  = 4.0  # h/day  # ... but playtime very high

_PROB_CAP   = 0.95   # ceiling for adjusted churn probability
_PROB_FLOOR = 0.05   # floor  for adjusted churn probability


def apply_semantic_alignment(
    raw_input: dict,
    ml_pred: int,
    prob_stay: float,
    prob_churn: float,
) -> dict:
    """
    Adjust ML churn probability using deterministic semantic rules.

    Args:
        raw_input:  dict of real-world feature values (unscaled).
        ml_pred:    raw model output (0 = stay, 1 = churn).
        prob_stay:  model's P(class 0 = stay).
        prob_churn: model's P(class 1 = churn).

    Returns:
        {
          "final_prediction":   int,        0 = stay, 1 = churn
          "label":              str,
          "reason":             str,        human-readable explanation
          "confidence":         float,      probability of predicted class
          "source":             str,        "semantic_rule" | "ml_model"
          "flags":              list[str],  semantic flags that fired
          "prob_churn_adjusted":float,      churn probability after adjustment
        }
    """
    playtime = float(raw_input.get("PlayTimeHours", 2.0))
    sessions = float(raw_input.get("SessionsPerWeek", 3))
    duration = float(raw_input.get("AvgSessionDurationMinutes", 40))

    adj_churn   = prob_churn          # working copy — only ever raised
    flags       = []                  # semantic flags that fired
    adjustments = []                  # human-readable description of each boost

    # ── Rule 1: Session math impossibility ───────────────────────────────────
    # Total minutes of gaming demanded by sessions vs minutes available per week
    weekly_available  = playtime * 7 * 60          # total playtime budget (min)
    weekly_demanded   = sessions * duration        # what sessions actually require

    if weekly_demanded > weekly_available:
        flags.append("SESSION_MATH_IMPOSSIBLE")
        # The claimed engagement is physically impossible. Regardless of ML output,
        # churn probability must be at least 0.65.
        boost = min(0.40, _PROB_CAP - adj_churn)
        adj_churn = max(adj_churn + boost, 0.65)
        adjustments.append(
            f"impossible session math: {weekly_demanded:.0f} min/week demanded "
            f"but only {weekly_available:.0f} min/week available "
            f"(floor 65% churn)"
        )

    # ── Rule 2: Burnout risk ──────────────────────────────────────────────────
    if playtime > _BURNOUT_PLAYTIME and duration > _BURNOUT_DURATION:
        flags.append("BURNOUT_RISK_HIGH")
        # Unsustainable play patterns historically precede disengagement.
        # Churn probability floor: 0.60.
        boost = min(0.35, _PROB_CAP - adj_churn)
        adj_churn = max(adj_churn + boost, 0.60)
        adjustments.append(
            f"burnout risk: {playtime:.1f}h/day at {int(duration)} min/session "
            f"is unsustainable (floor 60% churn)"
        )

    # ── Rule 3: Binge-session pattern ────────────────────────────────────────
    if sessions <= _BINGE_SESSIONS_MAX and playtime >= _BINGE_PLAYTIME_MIN:
        flags.append("BINGE_SESSION_PATTERN")
        # Very few sessions but very high daily playtime = binge-then-quit risk.
        # Churn probability floor: 0.55.
        boost = min(0.30, _PROB_CAP - adj_churn)
        adj_churn = max(adj_churn + boost, 0.55)
        adjustments.append(
            f"binge pattern: {int(sessions)} sessions/week but {playtime:.1f}h/day "
            f"signals erratic play (floor 55% churn)"
        )

    # ── Rule 4: Near-zero engagement (hard floor — overrides accumulated boosts)
    if playtime < _NEARZERO_PLAYTIME and sessions < _NEARZERO_SESSIONS and duration < _NEARZERO_DURATION:
        flags.append("NEAR_ZERO_ENGAGEMENT")
        adj_churn = 0.92          # hard set; player is barely active
        adjustments = [
            f"near-zero engagement: {playtime:.1f}h/day, "
            f"{int(sessions)} sessions/week, {int(duration)} min/session"
        ]

    # ── Clamp and derive final prediction ────────────────────────────────────
    adj_churn  = round(min(_PROB_CAP, max(_PROB_FLOOR, adj_churn)), 4)
    final_pred = 1 if adj_churn >= 0.50 else 0
    source     = "semantic_rule" if flags else "ml_model"

    # ── Build reason string ───────────────────────────────────────────────────
    if adjustments:
        reason_parts = "; ".join(adjustments)
        if "NEAR_ZERO_ENGAGEMENT" not in flags:
            reason_parts += (
                f". ML base: {prob_churn:.1%} churn -> "
                f"adjusted to {adj_churn:.1%}"
            )
        reason = reason_parts
    else:
        # No flags — pure ML zone
        if ml_pred == 0:
            reason = (
                f"Healthy engagement: {playtime:.1f}h/day, "
                f"{int(sessions)} sessions/week — ML predicts player stays."
            )
        else:
            reason = (
                f"Low-to-moderate engagement: {playtime:.1f}h/day, "
                f"{int(sessions)} sessions/week — ML predicts churn."
            )

    label = "Player Will CHURN" if final_pred == 1 else "Player Will STAY"
    confidence = adj_churn if final_pred == 1 else round(1.0 - adj_churn, 4)

    return {
        "final_prediction":    final_pred,
        "label":               label,
        "reason":              reason,
        "confidence":          confidence,
        "source":              source,
        "flags":               flags,
        "prob_churn_adjusted": adj_churn,
    }


# ── Internal sanity checks ────────────────────────────────────────────────────

def _run_self_checks() -> None:
    """
    Run at import time. Raises AssertionError immediately if any rule breaks.
    """
    import sys

    def _check(label, raw, ml_pred, prob_stay, prob_churn, expect_pred, expect_flag=None):
        r = apply_semantic_alignment(raw, ml_pred, prob_stay, prob_churn)
        assert r["final_prediction"] == expect_pred, (
            f"SELF-CHECK FAILED [{label}]: expected pred={expect_pred}, got {r}"
        )
        if expect_flag:
            assert expect_flag in r["flags"], (
                f"SELF-CHECK FAILED [{label}]: expected flag {expect_flag}, got {r['flags']}"
            )
        tag = f"[{','.join(r['flags']) or 'ml_model'}]"
        print(f"[check ok] {label:<34} -> {r['label']}  conf={r['confidence']:.2f}  {tag}",
              file=sys.stderr)

    _check("Near-zero engagement",
           {"PlayTimeHours": 0.2, "SessionsPerWeek": 1, "AvgSessionDurationMinutes": 10},
           ml_pred=0, prob_stay=0.90, prob_churn=0.10,
           expect_pred=1, expect_flag="NEAR_ZERO_ENGAGEMENT")

    _check("Healthy engagement (ML trusted)",
           {"PlayTimeHours": 2.0, "SessionsPerWeek": 4, "AvgSessionDurationMinutes": 45},
           ml_pred=0, prob_stay=0.75, prob_churn=0.25,
           expect_pred=0)

    _check("Burnout (high playtime + long sessions)",
           {"PlayTimeHours": 6.0, "SessionsPerWeek": 7, "AvgSessionDurationMinutes": 110},
           ml_pred=0, prob_stay=0.80, prob_churn=0.20,
           expect_pred=1, expect_flag="BURNOUT_RISK_HIGH")

    _check("Impossible session math",
           {"PlayTimeHours": 1.0, "SessionsPerWeek": 7, "AvgSessionDurationMinutes": 120},
           ml_pred=0, prob_stay=0.85, prob_churn=0.15,
           expect_pred=1, expect_flag="SESSION_MATH_IMPOSSIBLE")

    _check("Binge pattern (2 sessions, 5h/day)",
           {"PlayTimeHours": 5.0, "SessionsPerWeek": 2, "AvgSessionDurationMinutes": 60},
           ml_pred=0, prob_stay=0.80, prob_churn=0.20,
           expect_pred=1, expect_flag="BINGE_SESSION_PATTERN")

    print("[semantic_alignment] All self-checks passed.", file=sys.stderr)


_run_self_checks()
