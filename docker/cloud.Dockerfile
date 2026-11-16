# syntax=docker/dockerfile:1
#
# One self-contained backend container for the public demo on Azure Container
# Apps (also runs anywhere else Docker does). It answers with hosted Gemini, so
# it needs no Ollama. Embedding and reranker models are baked into the image
# so a cold start does not download them.
#
# Build and run locally exactly as the cloud does:
#   docker build -f docker/cloud.Dockerfile -t docqna-cloud .
#   docker run --rm -p 7860:7860 -e GEMINI_API_KEY=... docqna-cloud
#
# Required secret: GEMINI_API_KEY. Set CORS_ALLOWED_ORIGINS to the website's
# address. Uploaded documents live in the container and are lost on restart.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Tesseract OCR for scanned PDF pages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch: the default wheel bundles CUDA and is several GB larger.
RUN python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Pinned runtime dependencies from pyproject.toml, in their own layer.
COPY pyproject.toml ./
RUN python - <<'EOF'
import subprocess, sys, tomllib
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
EOF

COPY README.md ./
COPY src ./src
RUN python -m pip install --no-deps .

# Run as an unprivileged user (UID 1000).
RUN useradd --create-home --uid 1000 user \
    && mkdir -p /app/data /app/logs /app/.cache/huggingface /app/samples \
    && chown -R user:user /app
USER user

ENV HOME=/home/user \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ANONYMIZED_TELEMETRY=false

# Bake the evaluated embedding and reranker models into the image.
RUN python - <<'EOF'
from sentence_transformers import CrossEncoder, SentenceTransformer
from docqna.settings import Settings
settings = Settings(_env_file=None)
SentenceTransformer(settings.embedding_model, device="cpu")
CrossEncoder(settings.reranker_model, device="cpu")
EOF

# Fictional demo documents for the "load sample documents" button.
COPY --chown=user:user data/evaluation/public/corpus/ /app/samples/

ENV APP_ENV=production \
    PORT=7860 \
    LLM_PROVIDER=gemini \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    LOG_TO_FILE=false \
    SAMPLE_DOCUMENTS_DIR=/app/samples \
    TRUSTED_PROXY_HOPS=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"7860\")}/health', timeout=5)"

# One worker: rate limits and per-session search indexes live in process
# memory. No --proxy-headers: with --forwarded-allow-ips='*' uvicorn trusts the
# leftmost X-Forwarded-For entry, which a client can fake. The rate limiter
# reads the entry the platform proxy appended instead (TRUSTED_PROXY_HOPS=1).
CMD ["sh", "-c", "exec uvicorn docqna.api.main:app --host 0.0.0.0 --port \"${PORT}\" --workers 1 --no-proxy-headers --no-access-log"]
