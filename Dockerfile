# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Runtime env ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# ── Dependencies ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── App files ─────────────────────────────────────────────────────────────────
COPY backend/ ./backend/
COPY models/ ./models/
COPY start.sh .
RUN chmod +x start.sh

# ── Ports ─────────────────────────────────────────────────────────────────────
EXPOSE 8000
EXPOSE 7860

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["./start.sh"]