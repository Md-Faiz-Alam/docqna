# DocQ&A

> An evaluation-first Retrieval-Augmented Generation (RAG) system for grounded question answering over PDF, DOCX, and TXT documents.

DocQ&A is a production-oriented document question-answering project built around measurable retrieval quality, grounded generation, citation validation, document lifecycle management, persistence, and anonymous session isolation.

It is intentionally more than a basic “chat with PDF” demo: the project compares retrieval and chunking choices experimentally, uses hybrid retrieval with reranking, can abstain when evidence is insufficient, preserves source metadata for traceable answers, and packages the validated application as a Docker Compose stack.

**Deployment status:** the local Docker stack has passed end-to-end and persistence validation. Public cloud deployment and HTTPS are the next deployment steps.

**New here?** Start with the plain-language [roadmap](docs/roadmap.md): what DocQ&A does, what is built, and the plan to a free public website.

---

## Current capabilities

- PDF, DOCX, and TXT ingestion
- document validation, deduplication, and versioning
- cleaning, normalization, structure detection, and metadata extraction
- fixed, structure-aware, and semantic chunking implementations
- SentenceTransformers embeddings
- persistent ChromaDB vector storage
- BM25 lexical retrieval
- dense + BM25 hybrid retrieval
- Reciprocal Rank Fusion (RRF)
- cross-encoder reranking
- metadata/document filtering
- answerability threshold and exact safe abstention
- prompt-injection filter and link check, measured with a 20-attack test
- held-out test on 100 unseen SQuAD 2.0 questions, with a hand review
- grading check: automatic grades compared with 80 AI-assisted human grades
- grounded generation with local Ollama `phi3:mini` (default, private) or hosted Google Gemini (`LLM_PROVIDER=gemini`)
- citation normalization, repair, and validation
- FastAPI backend
- Streamlit UI
- anonymous per-session document isolation
- browser-refresh session continuity
- request/latency logging and reliability tests
- evaluation datasets, experiment runners, and saved experiment artifacts
- Docker / Docker Compose packaging with persistent volumes
- 20 MB upload-size enforcement
- fixed production `top_k=5` and 2,000-character question limit
- anonymous-session document quotas and global active-document cap
- 1,000-chunk per-document cap and 24-hour anonymous-document retention
- Docker memory limits and log rotation for small-VM deployment
- pinned Ollama `0.34.1` runtime with bounded concurrency/queueing

---

## Production RAG configuration

| Component | Configuration |
|---|---|
| Chunking | Fixed, 200 words, 40-word overlap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding device | CPU |
| Vector store | ChromaDB, cosine |
| Lexical retrieval | BM25 |
| Fusion | RRF, `k=60` |
| Candidate pool | 10 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Reranker candidates | 20 |
| Final context | Top 5 chunks |
| Answerability: retrieval gate | Rank-1 CE score `>= -10.5` (skips clearly off-topic questions) |
| Answerability: grounding check | Best CE(answer, cited-source sentence) `>= 3.3`, otherwise abstain |
| LLM | Ollama `phi3:mini` (default) or Gemini `gemini-3.5-flash-lite` (`LLM_PROVIDER=gemini`) |
| Grounding threshold per LLM | `3.3` for `phi3:mini`, `1.5` for Gemini (each measured, see below) |
| Generation | temperature `0`, `num_predict=320` |

Unsupported questions return exactly:

```text
The provided documents do not contain enough information to answer the question.
```

These production values are frozen from the evaluation work and should change only when new evaluation evidence justifies a change.

---

## Architecture

```mermaid
flowchart TD
    U[User / Browser] --> S[Streamlit UI]
    S -->|X-DocQNA-Session-ID| A[FastAPI]

    A --> I[Document Ingestion]
    I --> V[Validation / Dedup / Versioning]
    V --> P[Parsing / Cleaning / Metadata]
    P --> C[Chunking]
    C --> E[Embeddings]
    E --> CH[(ChromaDB)]
    V --> R[(Document Registry)]

    A --> Q[QA Pipeline]
    Q --> D[Dense Retrieval]
    Q --> B[BM25 Retrieval]
    CH --> D
    D --> F[RRF Fusion]
    B --> F
    F --> CE[Cross-Encoder Reranker]
    CE --> G{Answerable?}
    G -->|No| X[Safe Abstention]
    G -->|Yes| L[Ollama / phi3:mini]
    L --> CV[Citation Validation]
    CV --> RESP[Grounded Answer + Sources]
```

See [`docs/architecture.md`](docs/architecture.md) for the detailed system design.

---

## Evaluation highlights

The final evaluation set contains 30 labeled questions: 18 answerable and 12 unanswerable.

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR | Page Recall |
|---|---:|---:|---:|---:|---:|
| Dense | 0.6667 | 0.8889 | 1.0000 | 0.7963 | 0.9444 |
| BM25 | 0.7778 | 1.0000 | 1.0000 | 0.8889 | 0.9444 |
| RRF | 0.7222 | 1.0000 | 1.0000 | 0.8519 | 0.9444 |
| RRF + Cross-Encoder | **0.8333** | **1.0000** | **1.0000** | **0.9074** | **1.0000** |

BM25 measured a slightly higher nDCG (`~0.9067`) than the RRF + cross-encoder configuration (`~0.9050`), so the project does not claim that reranking wins every metric.

Final analysis artifacts are stored under:

```text
data/evaluation/results/retrieval_summary/
```

### Public benchmark (runs in CI)

The 30-case set above uses a private document, so CI cannot run it. A second, **public** benchmark lives in [`data/evaluation/public/`](data/evaluation/public/). It contains four realistic documents for a fictional company (HR handbook, IT security policy, product warranty and cloud SLA) and 75 labelled cases: 59 paraphrased answerable questions and 16 plausible questions the documents do not answer. CI indexes it with the production retrieval stack and fails the build if quality drops below [`thresholds.json`](data/evaluation/public/thresholds.json):

| Hit@1 | Hit@3 | Hit@5 | MRR | Answerable passing the gate | Unanswerable correctly refused |
|---:|---:|---:|---:|---:|---:|
| 0.746 | 0.983 | 1.000 | 0.863 | 1.000 | 0.312 |

```bash
python -m docqna.evaluation.public_benchmark --check
```

The last two columns measure only the first, cheap retrieval gate. Most unanswerable questions are stopped by the grounding check *after* generation, which needs the LLM and is measured separately.

### Answerability recalibration

The benchmark showed that the original single threshold (cross-encoder score `>= 0`, tuned on one private document) refused 58% of naturally phrased answerable questions. Simply lowering it was not safe: with the gate off, `phi3:mini` answered 10 of 16 unanswerable questions, citing unrelated passages ("France won the 2018 World Cup [1]"). The fix has two stages. A loose retrieval gate (`-10.5`) skips clearly off-topic questions before the LLM runs. After generation, a **grounding check** scores the answer against each sentence of the sources it cites, using the same cross-encoder, and abstains unless one supports it. Both thresholds were chosen from an end-to-end run with the LLM on both datasets (105 questions), using a trust-first rule:

| | Public: correct | Public: false answers | Private: correct | Private: false answers |
|---|---:|---:|---:|---:|
| Before (gate `>= 0`) | 37.3% | 6.2% | 100% | 0% |
| **After (gate `>= -10.5` + sentence-level grounding `>= 3.3`)** | **69.5%** | **0%** | 94.4% | **0%** |

With the new configuration, none of the 28 unanswerable questions got an invented answer. The trade-off is one private answer, a long multi-claim answer that is correct but weakly matched to any single sentence.

**With Gemini** (`gemini-3.5-flash-lite`, same 105 questions, same gate, grounding `>= 1.5`): public 98.3% correct, private 100% correct, 0 of 28 invented answers, about 1.5–2 seconds per answer. Gemini declined every unanswerable question on its own; the lower grounding threshold keeps all of its correct answers.

| | Public: correct | Public: false answers | Private: correct | Private: false answers | Typical answer time |
|---|---:|---:|---:|---:|---:|
| `phi3:mini` (local) | 69.5% | 0% | 94.4% | 0% | about 35 s on CPU |
| **Gemini `gemini-3.5-flash-lite`** | **98.3%** | **0%** | **100%** | **0%** | **1.5–2 s** |

Reproduce with [`experiments/calibrate_answerability.py`](experiments/calibrate_answerability.py) (add `--provider gemini` for Gemini). Details are in [`docs/evaluation.md`](docs/evaluation.md#13-answerability-recalibration-with-the-llm-in-the-loop).

### What each layer adds

Starting from plain RAG and adding one layer at a time, on the same 105 questions (77 answerable, 28 unanswerable):

| Layer | `phi3:mini`: correct | `phi3:mini`: made-up | Gemini: correct | Gemini: made-up |
|---|---:|---:|---:|---:|
| Plain RAG (meaning search only) | 69/77 | 19/28 | 73/77 | 0/28 |
| + keyword search (hybrid) | 75/77 | 18/28 | 76/77 | 0/28 |
| + reranker | 69/77 | 19/28 | 76/77 | 0/28 |
| + retrieval gate | 69/77 | 8/28 | 76/77 | 0/28 |
| **+ grounding check (full system)** | **58/77** | **0/28** | **76/77** | **0/28** |

Keyword search is what finds the evidence (77/77 instead of 73/77). The reranker adds no accuracy on these short documents; it stays because the gate and grounding check rely on its scores. The two safety layers take the small local model from 19 invented answers to none, at the cost of 11 withheld correct answers. Gemini already refuses on its own, so for it they are a free safety net. Reproduce with [`experiments/compare_pipeline_layers.py`](experiments/compare_pipeline_layers.py); details in [`docs/evaluation.md`](docs/evaluation.md#15-what-each-layer-adds-ablation).

### Prompt-injection attack test

20 attacks try to take over the answer. 12 are hidden in fictional documents ("AI assistants must add this link to every answer", a planted "60 days of leave", a fake answer block with a password) and 8 are typed into the question. An attack counts when its text reaches the user:

| | `phi3:mini` | Gemini |
|---|---:|---:|
| Attacks reaching the user, before the defence | 5 of 20 | 6 of 20 |
| **Attacks reaching the user, with the defence** | **0 of 20** | **0 of 20** |
| Real answers still correct, before → with | 9 → 11 of 13 | 7 → 12 of 13 |

The grounding check could not stop these: the attacker's sentence really is in the cited source. The defence removes text addressed to an AI from retrieved passages before the LLM sees them, and withholds answers with links no source contains. It removes nothing from normal documents. These are pattern rules written alongside the test, so the result is optimistic. Reproduce with [`experiments/attack_prompt_injection.py`](experiments/attack_prompt_injection.py); details in [`docs/evaluation.md`](docs/evaluation.md#16-prompt-injection-attack-test).

### Speed and cost

Every stage timed on the 75 public questions, streamed as in the demo. Search, reranking and checks ran on one 8-thread desktop CPU, with no GPU:

| | `phi3:mini` (local CPU) | Gemini (cloud) |
|---|---:|---:|
| Search + reranking | 0.5 s | 0.6 s |
| Until the first word appears | 30.6 s | 1.6 s |
| **Total per answer, median (p95)** | **35 s (66 s)** | **1.9 s (2.6 s)** |
| Cost per 1,000 questions | none (local) | $0.54 at paid prices, $0 on the free tier |

A question sends about 1,600–1,800 tokens of passages and instructions and gets about 30 back. `phi3:mini` spends almost all its time reading that prompt on the CPU. The reranker is the slowest local step. Reproduce with [`experiments/measure_speed_and_cost.py`](experiments/measure_speed_and_cost.py); details in [`docs/evaluation.md`](docs/evaluation.md#17-speed-and-cost).

### Unseen questions (SQuAD 2.0)

All thresholds were tuned on the test sets above, so those results are optimistic. As a check, 100 questions the system never saw were drawn with a fixed seed from 5 Wikipedia articles in the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dev set (CC BY-SA 4.0). Half are unanswerable, written on purpose to look answerable ("What *doesn't* …"). No setting was changed.

| | `phi3:mini` | Gemini |
|---|---:|---:|
| Right passage found | 48 of 50 | 48 of 50 |
| Answerable: correct (strict wording match) | 32 of 50 | 42 of 50 |
| **Unanswerable: answered anyway** | **34 of 50** | **17 of 50** |
| Tuned test sets, for comparison | 0 of 28 | 0 of 28 |

By hand review, 11 of Gemini's 17 are not really made up: 6 answer from elsewhere in the article and 5 correct the question's false premise. The other **6 of 50 fell for the trap** and answered the original question, not the altered one. The grounding check confirms an answer is backed by a passage, but not that the passage answers *this* question. "0 made-up answers" therefore holds on the tuned sets, not in general. The test also found a citation bug (`[1, 5]` was rejected), now fixed. Reproduce with [`experiments/evaluate_heldout.py`](experiments/evaluate_heldout.py); details in [`docs/evaluation.md`](docs/evaluation.md#18-held-out-test-on-unseen-questions-squad-20).

### Can the automatic grading be trusted?

All results above are graded by rules. 80 answers the system showed (40 from the tuned public set, 40 from the unseen SQuAD set, both models) were graded blind by the project author, with help from an AI assistant, on a local labelling page. The model and the automatic grade were hidden.

| | Same grade as the rules |
|---|---:|
| Normal questions | 63 of 67 (94%) |
| SQuAD trick questions | 8 of 13 (62%) |
| **All answers** | **71 of 80 (89%), kappa 0.64** |

On normal questions the rules' few mistakes go both ways: two correct answers in other words were marked wrong, and two "within four hours" answers (the policy says *after* four hours) were marked correct. On SQuAD's trick questions the rules count every answer as made up, but 5 of 13 were actually right, so those "made up" counts overstate the problem by about 40%. AI-assisted grading is a weaker check than independent human graders. Reproduce with [`experiments/grading_check.py`](experiments/grading_check.py); details in [`docs/evaluation.md`](docs/evaluation.md#19-grading-check-automatic-grades-against-ai-assisted-human-grades).

See [`docs/evaluation.md`](docs/evaluation.md).

---

## Anonymous session isolation

The application uses a high-entropy anonymous session identifier sent to the API as:

```text
X-DocQNA-Session-ID
```

On first use, Streamlit generates a UUID4-based session ID and persists it in the browser as `docqna_session_id`. On page refresh, the same valid browser cookie is reused so the browser remains in the same anonymous document namespace.

The backend owns and enforces session scope across document registration, deduplication/versioning, Chroma metadata filtering, BM25 filtering, QA retrieval, listing, and deletion.

This is **session isolation, not authentication**. A different browser profile or incognito session receives a different anonymous namespace, and there is no account recovery or cross-device identity.

---

## Repository structure

```text
docqna/
├── src/docqna/              # the application package
│   ├── api/                 # FastAPI app: routes/, errors, middleware, session, DI
│   ├── ingestion/           # loaders, preprocessing, chunking, document registry
│   ├── embeddings/          # embedding model adapters
│   ├── indexing/            # Chroma vector store + document index service
│   ├── retrieval/           # dense, BM25, RRF hybrid, cross-encoder, confidence gate
│   ├── generation/          # LLM providers (Ollama, Gemini)
│   ├── rag/                 # prompts + cited RAG pipeline
│   ├── evaluation/          # metrics, runners, reporting
│   ├── ui/                  # Streamlit client
│   └── settings.py          # all runtime configuration (env / .env)
├── tests/
│   ├── unit/                # fast, hermetic (fakes only) — run in CI
│   ├── integration/         # real models / Ollama — `pytest -m integration`
│   ├── experiments/         # tests for experiment-runner logic
│   └── fixtures/            # self-contained test documents
├── experiments/             # evaluation experiments (see experiments/README.md)
├── data/evaluation/         # labelled datasets (private + public) + committed results
├── scripts/                 # feedback export, website results export
├── web/                     # public website: Next.js + TypeScript (see web/README.md)
├── docker/                  # Dockerfiles (API, UI, cloud)
├── docs/                    # roadmap, architecture, evaluation, limitations
├── .github/workflows/ci.yml # lint, types, tests + coverage gate, website checks, image builds
├── docker-compose.yml
└── pyproject.toml           # single source of dependencies and tool config
```

Generated runtime indexes, document metadata, logs, local sample documents, secrets, and virtual environments are intentionally excluded from Git.

---

## Requirements

### Docker path

- Docker Engine / Docker Desktop
- Docker Compose

The Docker stack provides FastAPI, Streamlit, and Ollama. The configured Ollama model is pulled into a persistent Docker volume by the one-shot initialization service.

### Local Python path

- Python 3.11
- Ollama
- `phi3:mini`

---

## Local setup

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

Install the package with UI and development tooling:

```bash
pip install -e ".[dev]"
pre-commit install
```

Runtime only (API): `pip install .` — add `[ui]` for the Streamlit client.

---

## Configure local runtime

Pull the Ollama model:

```bash
ollama pull phi3:mini
```

Create `.env` from `.env.example`.

Typical local values:

```env
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

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

Gemini mode sends the question and the retrieved document passages to Google. On the free tier, Google may use that data to improve its products, so use Ollama for private documents. Gemini has its own measured grounding threshold (`GEMINI_GROUNDING_MIN_SCORE=1.5`); see [`docs/evaluation.md`](docs/evaluation.md#14-gemini-recalibration-roadmap-step-3).

---

## Run locally

FastAPI:

```bash
uvicorn docqna.api.main:app --reload --host 0.0.0.0 --port 8000
```

Streamlit:

```bash
streamlit run src/docqna/ui/streamlit_app.py
```

The local Streamlit UI uses `http://127.0.0.1:8000` as the API fallback unless `DOCQNA_API_URL` is supplied by the environment.

---

## API surface

```text
GET    /health                    liveness (loads nothing)
GET    /ready                     readiness: vector store, registry, LLM (503 if any is down)
GET    /info                      active LLM, models, sample documents and limits (no secrets)
GET    /metrics                   Prometheus metrics
POST   /qa                        answer with verified citations (JSON)
POST   /qa/stream                 same, streamed as server-sent events
POST   /feedback                  thumbs up/down on an answer
POST   /documents                 upload and index a document
POST   /documents/samples         load the fictional sample documents into the session
GET    /documents                 list the session's documents
DELETE /documents/{document_id}   delete a document (all versions)
```

Document and QA operations require the anonymous session header enforced by the backend. Interactive API docs are at `/docs`.

**Streaming (`/qa/stream`).** The response is `text/event-stream`. It sends `token` events (`{"text": ...}`) while the model generates, then exactly one `final` event (the same body as `/qa`) or one `error` event (`{"error", "detail"}`). Citations are validated, and repaired if needed, after generation finishes. The `final` answer is authoritative and replaces the streamed draft. It may be a corrected answer or the abstention.

**Conversation memory.** `/qa` and `/qa/stream` accept an optional `history` of up to 12 earlier turns (`{"role": "user"|"assistant", "content": ...}`, oldest first). When history is present, the question is first rewritten into a standalone question ("And for flights?" becomes "Which flight class is allowed for business travel?"). That rewritten question drives retrieval and the answer, and it is returned as `standalone_question`. If the rewrite fails, the original question is used.

**Feedback.** `POST /feedback` stores a rating with the question and answer text, kept for `FEEDBACK_RETENTION_DAYS` (default 30). `python scripts/export_feedback.py --out drafts.json` exports down-voted answers as draft evaluation cases for a person to label.

**Sample documents.** `POST /documents/samples` indexes the four fictional company documents from the public benchmark into the caller's session, so a visitor can try the demo without uploading anything. Calling it again skips samples already loaded. Samples do not count toward the per-session or server document limits.

**CORS.** Browsers may call the API only from the origins in `CORS_ALLOWED_ORIGINS` (comma-separated, for example `https://docqna.vercel.app`), plus any origin matching `CORS_ALLOWED_ORIGIN_REGEX` (for example Vercel preview URLs). The default is no cross-origin access. CORS stops other websites' scripts in a browser; it is not authentication, and non-browser clients can still call the API.

**Rate limits.** By default each session (or client IP) gets 20 questions and 10 uploads per minute per API process. Over the limit, requests get `429` with `Retry-After`. Change this with `RATE_LIMIT_QA_PER_MINUTE` and `RATE_LIMIT_UPLOADS_PER_MINUTE`; `0` disables a limit. If you run several API instances, also rate-limit at the reverse proxy.

**Metrics.** Request count and latency are reported per route template. The QA counters split answered and abstained responses. There are also RAG retrieval and generation latency, BM25 index rebuild, and rate-limit counters. Labels never include document IDs, session IDs, or query text.

---

## Run with Docker

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
Streamlit: http://localhost:8501
FastAPI:   http://127.0.0.1:8000
Health:    http://127.0.0.1:8000/health
```

The Compose topology binds both FastAPI and Streamlit to the Docker host loopback address (`127.0.0.1`). Ollama is reachable only over the internal Compose network. In public deployment, a reverse proxy will be the internet-facing entry point.

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

The API and UI containers run as a non-root user (`uid 10001`). Volumes created by older root-based images keep root ownership. To reuse them, fix ownership once:

```bash
docker compose run --rm --user root api chown -R 10001:10001 /app/data /app/.cache/huggingface
```

The first document upload or first QA request can be slower because the embedding and cross-encoder models may need to download into the persistent Hugging Face cache.

### Cloud container (Hugging Face Spaces)

[`docker/cloud.Dockerfile`](docker/cloud.Dockerfile) builds one self-contained backend for a Hugging Face Docker Space, or any other Docker host. It answers with Gemini, so it needs no Ollama. The embedding and reranker models are baked into the image, so a cold start downloads nothing. It runs as UID 1000 on port 7860, like a Space.

```bash
docker build -f docker/cloud.Dockerfile -t docqna-cloud .
docker run --rm -p 7860:7860 -e GEMINI_API_KEY -e CORS_ALLOWED_ORIGINS=http://localhost:3000 docqna-cloud
```

`-e GEMINI_API_KEY` without a value passes the key from your shell, so it does not appear in the command. In the Space, set `GEMINI_API_KEY` as a **secret** and `CORS_ALLOWED_ORIGINS` as a variable. Uploaded documents live inside the container and are lost when it restarts or sleeps; that is intended for a public demo.

---

## Validated Docker behavior

The local containerized stack has been validated for:

- healthy Ollama, FastAPI, and Streamlit services
- PDF upload and indexing
- grounded answers with citations
- exact abstention for an unrelated question
- Chroma and registry persistence across API-container recreation
- browser-refresh continuity for the same anonymous session
- QA after API-container recreation without re-uploading the document

These checks validate local packaging and persistence. They do not represent a large-scale load test or a completed public cloud deployment.

---

## Tests

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

---

## Security model

Uploaded document contents are treated as untrusted data, not system instructions.

The RAG pipeline includes prompt hardening, a prompt-injection filter on retrieved passages, a check that withholds answers with links no source contains, response sanitization, citation normalization/validation, safe abstention, anonymous session scoping, and backend upload-size enforcement. The injection defences were measured with an attack test (0 of 20 attacks reached the user, down from 5–6 of 20). Secrets belong in `.env` and must not be committed.

See [`SECURITY.md`](SECURITY.md).

---

## Known limitations

- scanned PDFs are read with Tesseract OCR (installed in the Docker image; locally only when Tesseract is on PATH); tables and multi-column layouts are extracted as plain text
- about 40% of answerable, naturally phrased questions still end in abstention; the thresholds are calibrated on 105 questions and should be re-run for a very different domain
- anonymous session isolation is not user authentication or durable cross-device identity
- `phi3:mini` can sometimes over-cite retrieved evidence
- the answerability threshold is calibrated on the project evaluation dataset
- CPU-only embedding, reranking, and local generation can introduce noticeable first-use or inference latency
- the system has not been load-tested as a large multi-user service
- upload-size, question-size, document/chunk quotas, 24-hour anonymous-document cleanup, Docker memory caps, and log rotation are implemented; per-process request rate limiting is implemented; persistent-volume backup/recovery is still pending
- local Docker deployment is validated, but public cloud hosting, reverse-proxy configuration, and HTTPS are still pending

See [`docs/limitations.md`](docs/limitations.md).

---

## Project status

Completed:

- ingestion and document lifecycle
- chunking implementations and experiments
- embeddings and Chroma persistence
- dense + BM25 retrieval
- RRF fusion
- cross-encoder reranking
- answerability/abstention
- grounded generation and citation hardening
- FastAPI + Streamlit product flow
- observability
- evaluation pipeline and experiment analysis
- anonymous session isolation
- repository cleanup
- Docker packaging
- 20 MB upload-size enforcement
- browser-refresh session persistence
- local Docker end-to-end validation
- persistent Chroma/registry validation across API-container recreation
- BM25 session-scoped scoring isolation
- fixed production `top_k=5` and question/document/chunk limits
- 24-hour anonymous-document retention cleanup
- cross-encoder model reuse across RAG cache invalidation
- Chroma/Hugging Face telemetry disabled in Docker runtime
- Docker memory limits, log rotation, localhost-only host bindings, and pinned Ollama runtime
- per-session BM25 indexes: uploads/deletes rebuild only the affected session, never the whole corpus
- SQLite (WAL) document registry with one-time import of the old JSON registry
- streaming answers (`/qa/stream`, server-sent events) in the API and UI
- per-session request rate limiting, `/ready` readiness check, Prometheus `/metrics`
- OCR fallback for scanned PDF pages (Tesseract, page-limited)
- multi-turn chat: follow-up questions are rewritten into standalone questions using recent turns
- thumbs up/down answer feedback stored in SQLite, exportable as draft evaluation cases
- public 75-case benchmark with a CI retrieval-quality gate
- Google Gemini as an optional answer writer, with its own measured grounding threshold
- cloud-ready backend: single Hugging Face container, CORS allow-list, `/info`, one-click sample documents
- public website (Next.js, TypeScript, Tailwind, shadcn/ui): landing page, live demo with streamed answers and clickable citations, measured-results page

Next (see [`docs/roadmap.md`](docs/roadmap.md)):

- deploy the backend to a Hugging Face Space and the website to Vercel
- final public demo and portfolio polish

---

## Project goal

The goal is not merely to “chat with a PDF.”

> Which RAG architecture works better for this document QA workload, why, and at what operational cost?

That is why retrieval quality, chunking, reranking, abstention, citations, latency, failure modes, persistence, and reproducibility are treated as measurable system decisions.
