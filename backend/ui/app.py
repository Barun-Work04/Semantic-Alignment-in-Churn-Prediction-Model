"""
app.py – Gradio UI for Gaming Churn Prediction.

Minimal 4-input interface. All categorical features are filled with neutral
defaults. Semantic alignment layer is applied before displaying the result.

Run from project root:
    python -m backend.ui.app
"""

import gradio as gr

from backend.model.predict import predict_churn
from backend.model.semantic_alignment import apply_semantic_alignment


# ── Inference callback ────────────────────────────────────────────────────────
def run_prediction(
    Age: float,
    PlayTimeHours: float,
    SessionsPerWeek: int,
    AvgSessionDurationMinutes: float,
) -> tuple[str, str, str]:
    """
    Called by Gradio on every submission.
    Returns (outcome, reason, debug_info).
    """
    raw_input = {
        "Age":                       Age,
        "PlayTimeHours":             PlayTimeHours,
        "SessionsPerWeek":           int(SessionsPerWeek),
        "AvgSessionDurationMinutes": AvgSessionDurationMinutes,
    }

    # Step 1: Raw ML prediction (uses shared predict.py; fills defaults internally)
    ml_result = predict_churn(raw_input)

    # Step 2: Semantic alignment (override when logic is obvious)
    aligned = apply_semantic_alignment(
        raw_input=raw_input,
        ml_pred=ml_result["ml_pred"],
        prob_stay=ml_result["prob_stay"],
        prob_churn=ml_result["prob_churn"],
    )

    # ── Format output ────────────────────────────────────────────────────────
    if aligned["final_prediction"] == 0:
        outcome = f"🟢  {aligned['label']}"
    else:
        outcome = f"🔴  {aligned['label']}"

    confidence_pct = f"{aligned['confidence'] * 100:.1f}%"
    reason_text    = aligned["reason"]

    flags_str = ", ".join(aligned["flags"]) if aligned["flags"] else "none"
    debug_text = (
        f"Source        : {aligned['source']}\n"
        f"Flags         : {flags_str}\n"
        f"Confidence    : {confidence_pct}\n"
        f"Churn prob    : ML {ml_result['prob_churn']:.3f}  "
        f"-> Adjusted {aligned['prob_churn_adjusted']:.3f}"
    )

    return outcome, reason_text, debug_text


# ── Gradio layout ─────────────────────────────────────────────────────────────
with gr.Blocks(title="🎮 Gaming Churn Predictor") as demo:
    gr.Markdown(
        """
        # 🎮 Gaming Churn Prediction Engine
        **Hybrid ML + Semantic Alignment** — a churn prediction case study focused on semantic alignment.

        Adjust the four engagement sliders and click **Predict**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            age_slider = gr.Slider(
                minimum=13, maximum=80, value=25, step=1,
                label="Age",
                info="Player's age in years"
            )
            playtime_slider = gr.Slider(
                minimum=0.1, maximum=10.0, value=2.0, step=0.1,
                label="Play Time (hours / day)",
                info="Average hours played per day"
            )
            sessions_slider = gr.Slider(
                minimum=1, maximum=7, value=3, step=1,
                label="Sessions Per Week",
                info="How many days per week the player games"
            )
            duration_slider = gr.Slider(
                minimum=5, maximum=180, value=40, step=5,
                label="Avg Session Duration (minutes)",
                info="Average length of a single gaming session"
            )
            predict_btn = gr.Button("🔍 Predict", variant="primary")

        with gr.Column(scale=1):
            outcome_box = gr.Textbox(
                label="Prediction",
                lines=1,
                interactive=False,
            )
            reason_box = gr.Textbox(
                label="Reason",
                lines=3,
                interactive=False,
            )
            debug_box = gr.Textbox(
                label="Model Details  (source | flags | confidence | probability adjustment)",
                lines=4,
                interactive=False,
            )

    gr.Markdown(
        """
        ---
        **Label convention:**
        - 🟢 Player Will STAY → high / balanced engagement (model class 0)
        - 🔴 Player Will CHURN → low or unsustainably high engagement (model class 1)

        *Source = `semantic_rule` means the semantic alignment layer adjusted the ML probability;*
        *`ml_model` means the ML prediction was trusted directly (no semantic adjustment).*

        *Flags: `SESSION_MATH_IMPOSSIBLE` | `BURNOUT_RISK_HIGH` | `BINGE_SESSION_PATTERN` | `NEAR_ZERO_ENGAGEMENT`*
        """
    )

    predict_btn.click(
        fn=run_prediction,
        inputs=[age_slider, playtime_slider, sessions_slider, duration_slider],
        outputs=[outcome_box, reason_box, debug_box],
    )

    # Show a default prediction on load
    demo.load(
        fn=run_prediction,
        inputs=[age_slider, playtime_slider, sessions_slider, duration_slider],
        outputs=[outcome_box, reason_box, debug_box],
    )


if __name__ == "__main__":
    demo.launch()