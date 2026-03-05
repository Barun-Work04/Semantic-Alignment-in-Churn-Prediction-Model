"""
validate.py – Final validation script.
Run from project root: python validate.py

Tests three scenarios and confirms both STAY and CHURN are achievable.
"""

import sys

SEP = "=" * 62

# ── Step 1: Imports ────────────────────────────────────────────────────────────
print(SEP)
print("STEP 1: Import checks (self-checks run inside semantic_alignment)")
print(SEP)

from backend.model.predict import predict_churn, COL_ORDER          # noqa: E402
from backend.model.semantic_alignment import apply_semantic_alignment  # noqa: E402

print("predict.py          OK  –  COL_ORDER:", COL_ORDER)
print("semantic_alignment  OK  –  (self-checks passed above)")

# ── Step 2: Raw ML output ──────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 2: Raw ML output (no alignment)")
print(SEP)

CASES = [
    ("Low engagement",         {"PlayTimeHours": 0.2,  "SessionsPerWeek": 1, "AvgSessionDurationMinutes": 10}),
    ("Normal engagement",      {"PlayTimeHours": 2.0,  "SessionsPerWeek": 4, "AvgSessionDurationMinutes": 45}),
    ("Burnout",                {"PlayTimeHours": 6.0,  "SessionsPerWeek": 7, "AvgSessionDurationMinutes": 110}),
    ("Impossible session math",{"PlayTimeHours": 1.0,  "SessionsPerWeek": 7, "AvgSessionDurationMinutes": 120}),
    ("Binge pattern",          {"PlayTimeHours": 5.0,  "SessionsPerWeek": 2, "AvgSessionDurationMinutes": 60}),
    ("Max inputs",             {"PlayTimeHours": 10.0, "SessionsPerWeek": 7, "AvgSessionDurationMinutes": 180}),
]

for name, raw in CASES:
    r = predict_churn(raw)
    ml_label = "STAY" if r["ml_pred"] == 0 else "CHURN"
    print(
        f"  {name:<22}  ml_pred={r['ml_pred']} ({ml_label})  "
        f"P(stay)={r['prob_stay']:.3f}  P(churn)={r['prob_churn']:.3f}"
    )

# ── Step 3: Full aligned pipeline ─────────────────────────────────────────────
print()
print(SEP)
print("STEP 3: Full pipeline with semantic alignment")
print(SEP)

aligned_results = []
for name, raw in CASES:
    ml = predict_churn(raw)
    aligned = apply_semantic_alignment(
        raw_input=raw,
        ml_pred=ml["ml_pred"],
        prob_stay=ml["prob_stay"],
        prob_churn=ml["prob_churn"],
    )
    aligned_results.append((name, aligned))
    final_label = "CHURN" if aligned["final_prediction"] == 1 else "STAY"
    flags_str   = ",".join(aligned["flags"]) if aligned["flags"] else "ml_model"
    print(f"  {name:<26}  ->  {final_label:<5}  conf={aligned['confidence']:.2f}  [{flags_str}]")
    print(f"    ML churn={ml['prob_churn']:.3f}  ->  adjusted={aligned['prob_churn_adjusted']:.3f}")
    print(f"    reason: {aligned['reason']}")
    print()

# ── Step 4: Edge-case assertions + both-outcome check ─────────────────────────
print(SEP)
print("STEP 4: Edge-case assertions and outcome coverage")
print(SEP)

by_name = {name: result for name, result in aligned_results}
failures = []

def _assert(cond, msg):
    if not cond:
        failures.append(msg)
        print(f"  [FAIL] {msg}")
    else:
        print(f"  [ok]   {msg}")

# Specific flag assertions
_assert("NEAR_ZERO_ENGAGEMENT"   in by_name["Low engagement"]["flags"],
        "Low engagement must trigger NEAR_ZERO_ENGAGEMENT flag")
_assert("SESSION_MATH_IMPOSSIBLE" in by_name["Impossible session math"]["flags"],
        "Impossible session math must trigger SESSION_MATH_IMPOSSIBLE flag")
_assert("BINGE_SESSION_PATTERN"   in by_name["Binge pattern"]["flags"],
        "Binge pattern must trigger BINGE_SESSION_PATTERN flag")
_assert("BURNOUT_RISK_HIGH"       in by_name["Burnout"]["flags"],
        "Burnout must trigger BURNOUT_RISK_HIGH flag")
_assert(by_name["Normal engagement"]["source"] == "ml_model",
        "Normal engagement must use ml_model (no flags)")

# Probability should only increase vs ML base
for name, result in aligned_results:
    ml_raw = predict_churn(CASES[[n for n, _ in CASES].index(name)][1])
    _assert(result["prob_churn_adjusted"] >= ml_raw["prob_churn"] - 0.001,
            f"{name}: adjusted churn prob must be >= ML base prob")

# Both outcomes coverage
final_codes = [r["final_prediction"] for _, r in aligned_results]
_assert(0 in final_codes, "STAY (0) outcome must be achievable")
_assert(1 in final_codes, "CHURN (1) outcome must be achievable")

print()
if not failures:
    print("[PASS] ALL CHECKS PASSED - system produces both STAY and CHURN outputs.")
    print("[PASS] Semantic alignment layer is working correctly.")
    sys.exit(0)
else:
    print(f"[FAIL] {len(failures)} check(s) failed. See above.")
    sys.exit(1)

