# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Tesseract OCR for scanned PDF pages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch: the default wheel bundles CUDA and is several GB larger.
RUN python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

# Install pinned runtime dependencies from pyproject.toml in their own layer,
# so source-only changes do not reinstall the dependency stack.
COPY pyproject.toml ./
RUN python - <<'EOF'
import subprocess, sys, tomllib
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
EOF

COPY README.md ./
COPY src ./src
RUN python -m pip install --no-deps . \
    && useradd --create-home --uid 10001 docqna \
    && mkdir -p /app/data/indexes /app/data/metadata /app/logs /app/.cache/huggingface \
    && chown -R docqna:docqna /app

USER docqna

EXPOSE 8000

# The app logs every request itself (with request IDs); uvicorn's access log
# would duplicate each line.
CMD ["uvicorn", "docqna.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
