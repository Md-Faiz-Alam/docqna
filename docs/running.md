# Running DocQ&A

How to run DocQ&A on your own machine, with Docker, or as a single cloud container, and how to run the checks.

## 1. Requirements

### Docker path

- Docker Engine / Docker Desktop
- Docker Compose

The Docker stack provides FastAPI and Ollama; run the website from `web/` against it. The configured Ollama model is pulled into a persistent Docker volume by the one-shot initialization service.

### Local Python path

- Python 3.11
- Ollama
- `phi3:mini`

---

## 2. Local setup

```bash
git clone https://github.com/Md-Faiz-Alam/docqna.git
cd docqna
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the package with development tooling:

```bash
pip install -e ".[dev]"
pre-commit install
```

Runtime only (API): `pip install .`

---

## 3. Configuration

Pull the Ollama model:

```bash
ollama pull phi3:mini
```

Create `.env` from `.env.example`.

Typical local values:

```env
APP_ENV=development

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=phi3:mini

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32

MAX_UPLOAD_SIZE_MB=20
MAX_QUESTION_CHARS=2000
MAX_DOCUMENTS_PER_SESSION=3
MAX_ACTIVE_DOCUMENTS=25
MAX_CHUNKS_PER_DOCUMENT=1000
DOCUMENT_RETENTION_HOURS=24
LOG_LEVEL=INFO
```

To answer with hosted Google Gemini instead of local Ollama, create a key in Google AI Studio and set:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here   # only in .env, which Git ignores
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_THINKING_LEVEL=minimal
```

Gemini mode sends the question and the retrieved document passages to Google. On the free tier, Google may use that data to improve its products, so use Ollama for private documents. Gemini has its own measured grounding threshold (`GEMINI_GROUNDING_MIN_SCORE=1.5`); see [`docs/evaluation.md`](evaluation.md#14-gemini-recalibration-roadmap-step-3).

---

## 4. Run locally

FastAPI:

```bash
uvicorn docqna.api.main:app --reload --host 0.0.0.0 --port 8000
```

Website (needs Node.js 20.9+; set `CORS_ALLOWED_ORIGINS=http://localhost:3000` on the API):

```bash
cd web
npm ci
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000. See [`web/README.md`](../web/README.md).


## 5. Run with Docker

Build and start the complete stack:

```bash
docker compose up -d --build
```

Check service state:

```bash
docker compose ps
```

Local endpoints:

```text
FastAPI:   http://127.0.0.1:8000
Health:    http://127.0.0.1:8000/health
```

The Compose topology binds FastAPI to the Docker host loopback address (`127.0.0.1`) and allows the local website (`http://localhost:3000`) through CORS. Start the website from `web/` as shown above. Ollama is reachable only over the internal Compose network. In public deployment, a reverse proxy will be the internet-facing entry point.

The Docker stack uses persistent named volumes for:

```text
ollama_data
docqna_indexes
docqna_metadata
huggingface_cache
```

This preserves the Ollama model, Chroma index, document registry, and downloaded Hugging Face models across container recreation.

Stop the stack without deleting data:

```bash
docker compose down
```

Do **not** use `docker compose down -v` unless you intentionally want to delete the named volumes and their persisted data.

The API container runs as a non-root user (`uid 10001`). Volumes created by older root-based images keep root ownership. To reuse them, fix ownership once:

```bash
docker compose run --rm --user root api chown -R 10001:10001 /app/data /app/.cache/huggingface
```

The first document upload or first QA request can be slower because the embedding and cross-encoder models may need to download into the persistent Hugging Face cache.

### Cloud container (Azure Container Apps or any Docker host)

[`docker/cloud.Dockerfile`](../docker/cloud.Dockerfile) builds one self-contained backend for any Docker host. The live demo runs it on Azure Container Apps, pulling `ghcr.io/md-faiz-alam/docqna-api:latest`, which [`publish-image.yml`](../.github/workflows/publish-image.yml) rebuilds on every backend change. It answers with Gemini, so it needs no Ollama. The embedding and reranker models are baked into the image, so a cold start downloads nothing. It runs as UID 1000 on `$PORT` (default 7860).

```bash
docker build -f docker/cloud.Dockerfile -t docqna-cloud .
docker run --rm -p 7860:7860 -e GEMINI_API_KEY -e CORS_ALLOWED_ORIGINS=http://localhost:3000 docqna-cloud
```

`-e GEMINI_API_KEY` without a value passes the key from your shell, so it does not appear in the command. On the host, store `GEMINI_API_KEY` as a **secret** and set `CORS_ALLOWED_ORIGINS` to the website's address, for example `https://docqna-lake.vercel.app,http://localhost:3000`. Uploaded documents live inside the container and are lost when it restarts or scales to zero; that is intended for a public demo.

---

## 6. Validated Docker behavior

The local containerized stack has been validated for:

- healthy Ollama and FastAPI services
- PDF upload and indexing
- grounded answers with citations
- exact abstention for an unrelated question
- Chroma and registry persistence across API-container recreation
- browser-refresh continuity for the same anonymous session
- QA after API-container recreation without re-uploading the document

These checks validate local packaging and persistence. They do not represent a large-scale load test or a completed public cloud deployment.

---

## 7. Tests and quality checks

Unit tests are hermetic: they use fakes, download no models, and need no Ollama. They run in about 3 seconds:

```bash
pytest
```

Integration tests build a temporary index from `tests/fixtures/` using the real embedding and cross-encoder models. The Ollama answer test is skipped when Ollama is not reachable:

```bash
pytest -m integration
```

Quality gates (also enforced in CI):

```bash
ruff check src tests experiments scripts
ruff format --check src tests experiments scripts
mypy
pytest --cov --cov-fail-under=75
```
