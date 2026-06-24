FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── API image — includes BGE-M3 model baked in ────────────────────────────────
FROM base AS api

ARG EMBEDDING_MODEL=BAAI/bge-m3
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', device='cpu')"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "75"]

# ── Frontend image — no model, stays lightweight ──────────────────────────────
FROM base AS frontend

EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
