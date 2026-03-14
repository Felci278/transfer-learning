# ── Urdu Sentiment Analysis — Inference Container ─────────────────────────────
# Serves 4 trained DistilBERT models via FastAPI on port 8000.
#
# Build:  docker build -t urdu-sentiment .
# Run:    docker run -p 8000:8000 urdu-sentiment
# Test:   curl -X POST http://localhost:8000/predict \
#           -H 'Content-Type: application/json' \
#           -d '{"text": "Drama bohat acha tha", "model": "hf_full"}'
# ───────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# System build deps (needed by some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ────────────────────────────────────────────────
# Install CPU-only PyTorch first (saves ~2 GB vs the CUDA wheel).
# All other packages come from requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy trained model assets ──────────────────────────────────────────────────
# Both notebooks export their artifacts into these directories.
COPY deployment_assets/     deployment_assets/
COPY deployment_assets_hf/  deployment_assets_hf/

# ── Pre-cache the base model for LoRA inference ────────────────────────────────
# The two LoRA adapters were trained on top of distilbert-base-multilingual-cased.
# We download and cache it into /app/hf_cache at build time so the container
# can run fully offline at inference time.
COPY cache_model.py /tmp/cache_model.py
RUN python /tmp/cache_model.py && rm /tmp/cache_model.py

# ── Copy application ───────────────────────────────────────────────────────────
COPY app.py .

# ── Runtime ────────────────────────────────────────────────────────────────────
EXPOSE 8000

# 1 worker is enough for CPU inference; add --workers N for multi-model parallelism
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
