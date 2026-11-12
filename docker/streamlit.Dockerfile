# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# The UI talks to the API over HTTP only; it needs Streamlit and nothing else.
COPY pyproject.toml ./
RUN python - <<'EOF'
import subprocess, sys, tomllib
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["optional-dependencies"]["ui"]
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
EOF

COPY src/docqna/ui/streamlit_app.py /app/streamlit_app.py

RUN useradd --create-home --uid 10001 docqna
USER docqna

EXPOSE 8501

CMD ["streamlit", "run", "/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
