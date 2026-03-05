"""
predict.py - Shared inference logic used by both API and UI.

Responsibilities:
- Load model artifacts from a single source of truth
- Prepare input features in the exact order the model expects
- Expose a predict_churn() function that returns raw ML output

Label convention (confirmed from preprocessing):
    pred = 0  → Churn=0 → High engagement → Player STAYS
    pred = 1  → Churn=1 → Low/Medium engagement → Player CHURNS
"""

import os
import pickle
import pandas as pd

# ── Resolve artifact paths relative to THIS file, not the caller's cwd ──────
_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "models")

with open(os.path.join(_BASE, "best_model.pkl"), "rb") as f:
    MODEL = pickle.load(f)

with open(os.path.join(_BASE, "col_order.pkl"), "rb") as f:
    COL_ORDER = pickle.load(f)          # list of 11 feature names, in order

with open(os.path.join(_BASE, "scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)

# ── Safe neutral defaults for features NOT supplied by the user ──────────────
# Values chosen to be median/modal, not extreme, so they don't skew predictions.
_NEUTRAL_DEFAULTS = {
    "Age":                      25,
    "Gender":                    1,   # Male (most common in dataset)
    "Location":                  2,   # Asia / Other
    "GameGenre":                 1,   # RPG
    "InGamePurchases":           1,   # Yes
    "GameDifficulty":            1,   # Medium
    "PlayerLevel":              30,   # Mid-level
    "AchievementsUnlocked":     20,   # Some achievements
    # Engagement inputs below – only used as fallback, UI always supplies them
    "PlayTimeHours":            2.0,
    "SessionsPerWeek":          3,
    "AvgSessionDurationMinutes": 40,
}


def prepare_input(raw_input: dict) -> tuple[pd.DataFrame, any]:
    """
    Build a properly ordered, scaled DataFrame row from raw user inputs.

    Args:
        raw_input: dict of {feature_name: value} in any order, real-world scale.

    Returns:
        (df_raw, X_scaled)
        df_raw   – unscaled DataFrame with all 11 features in COL_ORDER
        X_scaled – numpy array ready for model.predict()

    Raises:
        ValueError if a required column cannot be filled.
    """
    row = {}
    for col in COL_ORDER:
        if col in raw_input:
            row[col] = raw_input[col]
        elif col in _NEUTRAL_DEFAULTS:
            row[col] = _NEUTRAL_DEFAULTS[col]
        else:
            raise ValueError(
                f"Feature '{col}' missing from input and has no default. "
                "Add it to _NEUTRAL_DEFAULTS or supply it explicitly."
            )

    df_raw = pd.DataFrame([row], columns=COL_ORDER)
    # Keep scaled result as a named DataFrame so RandomForest doesn't warn
    # about missing feature names (it was fitted with named columns).
    X_scaled = pd.DataFrame(SCALER.transform(df_raw), columns=COL_ORDER)
    return df_raw, X_scaled


def predict_churn(raw_input: dict) -> dict:
    """
    Run raw ML inference only (no semantic alignment).

    Returns:
        {
          "ml_pred":      int   (0 = stay, 1 = churn),
          "prob_stay":    float (probability player stays),
          "prob_churn":   float (probability player churns),
        }
    """
    _, X_scaled = prepare_input(raw_input)
    pred = MODEL.predict(X_scaled)[0]
    proba = MODEL.predict_proba(X_scaled)[0]  # [P(class 0), P(class 1)]

    return {
        "ml_pred":    int(pred),
        "prob_stay":  round(float(proba[0]), 4),   # class 0 = stay
        "prob_churn": round(float(proba[1]), 4),   # class 1 = churn
    }
