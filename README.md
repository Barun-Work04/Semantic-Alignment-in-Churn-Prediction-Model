<<<<<<< Updated upstream
# Semantic Alignment in Churn Prediction (Case Study)

This project is a productionized case study that shows how to diagnose and correct
semantic misalignment in machine learning systems. The example domain is gaming churn,
but the framework generalizes across industries.

## Problem Statement

Modern ML models often fail not because of weak algorithms, but because training labels
encode outcomes while stakeholders expect predictions about intent or future behavior.

This mismatch is semantic misalignment. It creates models that are highly confident,
technically correct, and practically misleading.

This repository demonstrates how to detect and correct semantic misalignment using a
structured framework and a hybrid ML + rules alignment layer.

## What Is Semantic Misalignment?

Semantic misalignment occurs when:

- Labels represent an observed outcome
- But the prediction task is interpreted as intent or risk

### Example (Gaming Churn)

- Label: Player churned
- Reality: churn can mean burnout, lifecycle completion, or external factors
- Business intent: which players are at risk of disengagement

The model learns what churn looked like in the past, not why a player is likely to leave next.

## Framework: Semantic Alignment (Reusable Across ML Systems)

This project formalizes a 4-step framework that applies to many domains.

### Step 1: Identify Label Origin

Ask:

- How were labels generated?
- What real-world event do they represent?

Examples:

- Churn: account deletion after N days
- Fraud: detected fraud cases (not all fraud)
- Medical: diagnosed cases (not early-stage disease)

### Step 2: Identify Prediction Intent

Ask:

- What does the user actually want to predict?

Examples:

- Disengagement risk
- Financial instability risk
- Health deterioration risk

### Step 3: Detect Semantic Mismatch

Common signals:

- Counter-intuitive predictions
- Edge cases behaving backwards
- Extremely confident outputs in obvious scenarios
- Model predicts churn for high engagement users

This is not overfitting. It is objective misalignment.

### Step 4: Apply a Semantic Alignment Layer

Correction strategies:

- Hybrid rule + ML systems
- Task reframing
- Label reinterpretation
- Post-prediction constraints

This project uses a hybrid alignment layer:

- Rules capture near-certain real-world semantics
- ML handles complex interactions and tradeoffs

## Case Study: Gaming Churn Prediction

### Why Gaming?

Gaming data provides:

- Rich behavioral signals
- Clear semantic ambiguity in churn labels
- Intuitive edge cases that reveal misalignment

### Key Finding

The model learned:

"High engagement -> churn" (many churned users burned out)

But business logic expects:

"Low engagement -> churn risk"

This contradiction exposed semantic bias in the dataset, not a modeling flaw.

### Concrete Misalignment Examples (from this project)

- Burnout pattern: very high playtime + long sessions can indicate churn risk
- Impossible session math: weekly session demand exceeds available time
- Binge pattern: very few sessions but very high daily playtime
- Near-zero engagement: minimal activity should override a confident stay prediction

These cases are encoded as explicit alignment rules layered on top of the ML output.

## Generalization Beyond Gaming

Semantic alignment applies broadly:

- Credit scoring: label = defaulted, intent = risk
- Fraud detection: label = caught fraud, intent = fraud likelihood
- Medical ML: label = diagnosed disease, intent = early detection
- Recommenders: label = clicks, intent = satisfaction

Common pattern: labels describe what happened, not what will happen.

## Why Not Just Retrain?

Retraining without correcting semantic misalignment optimizes the wrong objective faster.
More data amplifies the same misunderstanding. Accuracy may improve while usefulness declines.

## What This Project Contributes

This project does not claim to solve churn universally or replace ML models.
It demonstrates how to identify and correct semantic bias between labels and intent.

Outcomes:

- Explainable behavior in edge cases
- Transferable alignment framework
- Industry-relevant methodology
- Research-oriented case study

## System Overview

- ML inference: RandomForest model with preprocessing and scaling
- Semantic alignment: rule-based probability adjustments
- API: FastAPI inference service
- UI: Gradio interactive interface

## Repository Structure

- backend/
	- api/           FastAPI service
	- ui/            Gradio UI
	- model/         Model loading and semantic alignment logic
- models/          Trained model artifacts
- data/            Training data (not used at runtime)
- notebooks/       Research notebooks (not used at runtime)
- reports/         Figures and outputs (not used at runtime)

## Quickstart (Local)

1) Create environment and install deps

	 pip install -r requirements.txt

2) Run API and UI

	 ./start.sh

3) Validate semantic alignment behavior

	 python validate.py

## Docker

Build:

	docker build -t churn-prediction .

Run:

	docker run -p 8000:8000 -p 7860:7860 --name churn-app churn-prediction

Endpoints:

- API health: http://localhost:8000/
- UI: http://localhost:7860/

## API Example

POST /predict

{
	"Age": 25,
	"PlayTimeHours": 2.0,
	"SessionsPerWeek": 4,
	"AvgSessionDurationMinutes": 45
}

## CI/CD

The CI workflow:

- Runs validation tests
- Performs an API smoke test
- Builds and pushes Docker images


## Known Challenges Encountered

- The model learned churn patterns that contradicted business intent
- High engagement cases were frequently labeled as churn due to burnout
- Raw ML outputs looked correct statistically but failed on real-world semantics

The alignment layer addresses these mismatches without changing the underlying model.


>>>>>>> Stashed changes
