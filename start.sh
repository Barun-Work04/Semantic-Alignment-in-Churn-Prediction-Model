#!/bin/bash
# start.sh - Starts both FastAPI and Gradio inside the single container
# FastAPI runs in background, Gradio runs in foreground (keeps container alive)

echo "Starting FastAPI backend on port 8000..."
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 &

# Small delay so FastAPI is ready before Gradio tries to call it
sleep 3

echo "Starting Gradio UI on port 7860..."
python -m backend.ui.app