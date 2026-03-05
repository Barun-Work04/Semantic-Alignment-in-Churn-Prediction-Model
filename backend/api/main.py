"""
main.py – FastAPI inference endpoint for Gaming Churn Prediction.

Pipeline per request:
    1. Validate & parse input (Pydantic)
    2. Prepare features via shared predict.prepare_input()
    3. Run raw ML inference via predict.predict_churn()
    4. Apply semantic alignment via semantic_alignment.apply_semantic_alignment()
    5. Return structured JSON response

Run from project root:
    uvicorn backend.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.model.predict import predict_churn
from backend.model.semantic_alignment import apply_semantic_alignment

app = FastAPI(
    title="Gaming Churn Prediction API",
    description=(
        "Hybrid ML + semantic alignment churn predictor. "
        "Returns prediction, reason, and confidence for each player."
    ),
    version="2.0.0",
)


# ── Request schema ────────────────────────────────────────────────────────────
class PlayerInput(BaseModel):
    """
    Minimal engagement features required for inference.
    The four fields below drive churn prediction; everything else
    is filled with neutral defaults internally.
    """
    Age: float = Field(default=25, ge=13, le=80, description="Player age in years")
    PlayTimeHours: float = Field(
        default=2.0, ge=0.1, le=10.0,
        description="Average daily playtime in hours"
    )
    SessionsPerWeek: int = Field(
        default=3, ge=1, le=7,
        description="Number of gaming sessions per week"
    )
    AvgSessionDurationMinutes: float = Field(
        default=40.0, ge=5.0, le=180.0,
        description="Average duration of a single session in minutes"
    )

    @field_validator("PlayTimeHours", "AvgSessionDurationMinutes")
    @classmethod
    def positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Value must be positive")
        return round(v, 2)


# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction_label: str    # "Player Will STAY" or "Player Will CHURN"
    prediction_code:  int    # 0 = stay, 1 = churn
    reason:           str    # plain-English explanation
    confidence:       float  # 0.0 – 1.0
    source:           str    # "semantic_rule" or "ml_model"
    ml_raw:           dict   # raw ML output for transparency


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "Gaming Churn Prediction API v2"}


@app.post("/predict", response_model=PredictionResponse, summary="Predict player churn")
def predict(player: PlayerInput) -> PredictionResponse:
    """
    Predict whether a player will churn or stay.

    - Runs ML model with safe feature defaults for non-supplied columns.
    - Applies semantic alignment rules for edge cases.
    - Returns structured response with explanation.
    """
    raw_input = player.model_dump()

    # Step 1: Raw ML inference (uses shared predict.py logic)
    try:
        ml_result = predict_churn(raw_input)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ML inference failed: {exc}")

    # Step 2: Semantic alignment (correct model when behavioral signal is clear)
    aligned = apply_semantic_alignment(
        raw_input=raw_input,
        ml_pred=ml_result["ml_pred"],
        prob_stay=ml_result["prob_stay"],
        prob_churn=ml_result["prob_churn"],
    )

    return PredictionResponse(
        prediction_label=aligned["label"],
        prediction_code=aligned["final_prediction"],
        reason=aligned["reason"],
        confidence=aligned["confidence"],
        source=aligned["source"],
        ml_raw=ml_result,
    )