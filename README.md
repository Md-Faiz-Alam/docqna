# DocQ&A

> An evaluated Retrieval-Augmented Generation system for grounded question answering over PDF, DOCX, and TXT documents.

DocQ&A is a modular document-intelligence system that lets users upload documents, ask natural-language questions, and receive answers grounded in retrieved evidence with source citations.

Unlike a basic "chat with PDF" application, DocQ&A is built around an **evaluation-first RAG workflow**. It supports multiple chunking strategies, dense and lexical retrieval, Reciprocal Rank Fusion, cross-encoder reranking, answerability detection, citation validation, and reproducible experiments for comparing architectural choices.

---

## Features

- PDF, DOCX, and TXT ingestion
- File validation, deduplication, and version tracking
- Text cleaning and structure detection
- Fixed, recursive, structure-aware, and semantic chunking
- Metadata-rich document chunks
- Sentence-transformer embeddings
- Persistent vector storage with ChromaDB
- BM25 lexical indexing
- Dense retrieval
- Hybrid Dense + BM25 retrieval
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Optional document metadata filtering
- Answerability / abstention layer
- Grounded LLM generation
- Ollama, Gemini, and Groq generation providers
- Source citations with page and section metadata
- Citation validation
- FastAPI backend
- Streamlit frontend
- Retrieval and RAG evaluation pipeline
- Reproducible experiment configurations
- Latency and error analysis
- Docker support
- Unit and integration tests

---

## Architecture

![DocQ&A Architecture](docs/images/architecture.png)

The operational RAG pipeline is:

```text
Documents
   │
   ▼
Validation
   │
   ▼
Parsing & Cleaning
   │
   ▼
Chunking & Metadata
   │
   ├──────────────────────┐
   │                      │
   ▼                      ▼
Embeddings             BM25 Index
   │                      │
   ▼                      │
ChromaDB                  │
   │                      │
   └──────────┬───────────┘
              │
              ▼
          User Query
              │
      ┌───────┴───────┐
      ▼               ▼
Dense Retrieval   BM25 Retrieval
      │               │
      └───────┬───────┘
              ▼
          RRF Fusion
              │
              ▼
      Candidate Chunks
              │
              ▼
    Cross-Encoder Reranker
              │
              ▼
       Best Evidence
              │
              ▼
      Answerability Check
          ┌───┴───┐
          │       │
          ▼       ▼
       Abstain    LLM
                    │
                    ▼
             Grounded Answer
                    │
                    ▼
            Citation Validation
                    │
                    ▼
               Response
```

The evaluation pipeline is kept separate from the online query path:

```text
Evaluation Dataset
        │
        ▼
Experiment Configuration
        │
        ▼
Run RAG Configuration
        │
        ▼
┌─────────────────────────────┐
│ Retrieval Metrics           │
│ Generation Metrics          │
│ Citation Metrics            │
│ Abstention Metrics          │
│ Latency                     │
└─────────────────────────────┘
        │
        ▼
Baseline vs Improved Systems
        │
        ▼
Error Analysis
```

---

## Why Hybrid Retrieval?

Document questions can require different kinds of matching.

Dense retrieval works well for semantic questions such as:

```text
"What percentage attendance do students need?"
```

when the document says:

```text
"Students must maintain a minimum attendance of 75%."
```

BM25 is useful when exact terms matter:

```text
"What does section 4.2.7 say?"
```

DocQ&A can retrieve from both systems and combine their rankings using **Reciprocal Rank Fusion (RRF)**.

A cross-encoder can then rerank the fused candidate set before context is passed to the LLM.

---

## Grounded Generation

The LLM does not search the entire document collection directly.

Instead:

1. relevant chunks are retrieved,
2. candidates are reranked,
3. evidence quality is checked,
4. selected chunks are provided as context,
5. the LLM generates an answer from that evidence,
6. citations are mapped back to document metadata.

If the retrieved evidence is insufficient, the system can abstain instead of forcing an answer.

Example:

```text
I don't have enough information in the uploaded documents
to answer this question.
```

---

## Citations

Each indexed chunk stores source metadata such as:

```text
document_id
document_version
filename
file_type
page_start
page_end
section
chunk_id
chunk_index
content_hash
```

Generated answers reference retrieved context IDs internally.

The backend maps those references back to their original source information before returning the response.

Example:

```text
Answer:
Students must maintain at least 75% attendance.

Source:
student_handbook.pdf
Page 42
Section: Academic Rules
```

---

# Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| Frontend | Streamlit |
| Vector Store | ChromaDB |
| Dense Embeddings | Sentence Transformers |
| Lexical Retrieval | BM25 |
| Fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| LLM Providers | Ollama / Gemini / Groq |
| Configuration | YAML |
| Evaluation | Custom evaluation framework |
| Testing | pytest |
| Containers | Docker / Docker Compose |

---

# Project Structure

```text
docqna/
│
├── README.md
├── LICENSE
├── .env.example
├── pyproject.toml
├── requirements.txt
├── Makefile
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
│
├── configs/
│   ├── app.yaml
│   ├── logging.yaml
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   └── experiments/
│
├── src/
│   └── docqna/
│       ├── api/
│       ├── core/
│       ├── ingestion/
│       ├── embeddings/
│       ├── indexing/
│       ├── retrieval/
│       ├── reranking/
│       ├── answerability/
│       ├── generation/
│       ├── citations/
│       ├── pipelines/
│       ├── services/
│       └── repositories/
│
├── frontend/
│   ├── pages/
│   ├── components/
│   ├── api/
│   └── utils/
│
├── evaluation/
│   ├── datasets/
│   ├── metrics/
│   ├── judges/
│   ├── runners/
│   └── analysis/
│
├── experiments/
│   ├── results/
│   ├── artifacts/
│   └── reports/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── indexes/
│   ├── metadata/
│   └── samples/
│
├── scripts/
├── tests/
├── notebooks/
├── docs/
├── logs/
│
└── .github/
    └── workflows/
```

The repository separates the interactive application from the evaluation framework so experiments can be run independently of the frontend.

---

# Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/docqna.git
cd docqna
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

Using `requirements.txt`:

```bash
pip install -r requirements.txt
```

or, when using the package configuration:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

---

## 4. Configure Environment Variables

Copy the example file:

```bash
cp .env.example .env
```

Configure the providers you intend to use.

Example:

```env
GEMINI_API_KEY=
GROQ_API_KEY=

OLLAMA_BASE_URL=http://localhost:11434
```

You do not need every provider configured.

The active generation provider is selected through the application configuration.

---

# Configuration

DocQ&A keeps runtime and experiment configuration outside the application code.

```text
configs/
├── app.yaml
├── logging.yaml
│
├── ingestion/
│   ├── default.yaml
│   └── semantic_chunking.yaml
│
├── retrieval/
│   ├── dense.yaml
│   ├── bm25.yaml
│   ├── hybrid_rrf.yaml
│   └── hybrid_rerank.yaml
│
├── generation/
│   ├── ollama.yaml
│   ├── gemini.yaml
│   └── groq.yaml
│
└── experiments/
    ├── baseline.yaml
    ├── hybrid.yaml
    ├── hybrid_rerank.yaml
    └── final.yaml
```

This allows retrieval and generation strategies to be changed without modifying pipeline code.

---

# Running the Application

## Start the FastAPI Backend

```bash
uvicorn docqna.main:app --reload
```

The API will typically be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## Start the Streamlit Frontend

In another terminal:

```bash
streamlit run frontend/app.py
```

The interface provides pages for:

```text
Chat
Documents
Evaluation
About
```

---

# API

The backend exposes document, query, experiment, and health functionality.

Main route groups are implemented under:

```text
src/docqna/api/routers/
```

including:

```text
health.py
documents.py
query.py
experiments.py
```

Typical endpoints include:

```text
GET    /health

POST   /documents
GET    /documents
GET    /documents/{id}
DELETE /documents/{id}

POST   /query
```

See:

```text
docs/api.md
```

for API documentation.

---

# Document Ingestion

The ingestion pipeline is responsible for turning uploaded documents into retrieval-ready chunks.

```text
Upload
  │
  ▼
File Validation
  │
  ▼
Duplicate / Version Check
  │
  ▼
Document Loader
  │
  ▼
Cleaning & Normalization
  │
  ▼
Structure Detection
  │
  ▼
Chunking
  │
  ▼
Metadata Creation
  │
  ├──────────────► Embeddings ──► ChromaDB
  │
  └──────────────► BM25 Index
```

Supported loaders are located under:

```text
src/docqna/ingestion/loaders/
```

Current document types:

```text
PDF
DOCX
TXT
```

---

# Chunking Strategies

DocQ&A supports multiple chunking implementations:

```text
fixed.py
recursive.py
structure_aware.py
semantic.py
```

These are intentionally configurable because chunking is treated as an experimental variable rather than a fixed implementation detail.

Example parameters include:

```yaml
strategy: fixed
chunk_size: 500
overlap: 50
```

---

# Retrieval Strategies

Several retrieval configurations can be evaluated.

## Dense

```text
Query
  ↓
Embedding
  ↓
ChromaDB
  ↓
Top-K chunks
```

## BM25

```text
Query
  ↓
Tokenization
  ↓
BM25 Index
  ↓
Top-K chunks
```

## Hybrid RRF

```text
Dense ──┐
        ├── RRF ──► Candidates
BM25  ──┘
```

## Hybrid + Reranking

```text
Dense + BM25
      ↓
   RRF Fusion
      ↓
Top-N Candidates
      ↓
Cross-Encoder
      ↓
Top-K Context
```

---

# Answerability and Abstention

Retrieval systems can always return the closest available chunks even when those chunks do not actually answer the question.

DocQ&A therefore includes an answerability layer:

```text
Reranked Evidence
       │
       ▼
Evidence Check
   ┌───┴────┐
   │        │
Enough    Insufficient
   │        │
   ▼        ▼
  LLM     Abstain
```

The initial implementation can use a reranker-score threshold.

Additional signals can be added later, including:

```text
top relevance score
score margin
number of strong passages
dense retrieval score
retrieval agreement
```

The threshold is treated as an experiment parameter rather than a hard-coded constant.

---

# Evaluation

Evaluation is a first-class part of DocQ&A.

The evaluation package is located at:

```text
evaluation/
```

and contains datasets, metrics, experiment runners, judges, comparison utilities, latency analysis, and error analysis.

---

## Evaluation Dataset

Evaluation examples are stored as JSONL.

Example:

```json
{
  "question": "What is the minimum attendance requirement?",
  "relevant_chunk_ids": ["student_handbook_0047"],
  "expected_answer": "75%",
  "answerable": true,
  "category": "paraphrase"
}
```

The repository separates:

```text
development.jsonl
test.jsonl
```

The development set can be used for architecture and threshold tuning.

The held-out test set is reserved for final evaluation.

---

# Retrieval Metrics

Implemented retrieval metrics include:

- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- nDCG@K

These are implemented under:

```text
evaluation/metrics/retrieval/
```

---

# Generation Metrics

The generation evaluation layer includes:

- Faithfulness
- Answer relevance
- Context relevance

Located under:

```text
evaluation/metrics/generation/
```

---

# Citation Evaluation

Citation evaluation measures whether generated citations correctly support the associated answer.

```text
evaluation/metrics/citation/
└── correctness.py
```

Runtime citation validation and evaluation-time citation correctness are intentionally separate concepts.

---

# Abstention Evaluation

The abstention system can be evaluated using:

- accuracy,
- answer coverage,
- confusion analysis,
- correct abstentions,
- false abstentions,
- unsupported answers.

Implementation:

```text
evaluation/metrics/abstention/
```

---

# Experiments

Experiment definitions live under:

```text
configs/experiments/
```

Current configurations include:

```text
baseline.yaml
hybrid.yaml
hybrid_rerank.yaml
final.yaml
```

Conceptually:

```text
Baseline
Fixed Chunking
    ↓
Dense Retrieval
    ↓
LLM
```

```text
Hybrid
Dense + BM25
    ↓
RRF
    ↓
LLM
```

```text
Hybrid + Reranker
Dense + BM25
    ↓
RRF
    ↓
Cross-Encoder
    ↓
LLM
```

```text
Final
Optimized Chunking
    ↓
Hybrid Retrieval
    ↓
RRF
    ↓
Reranking
    ↓
Answerability
    ↓
Grounded Generation
    ↓
Citation Validation
```

Run experiments with:

```bash
python scripts/run_experiments.py
```

or:

```bash
python -m evaluation.run
```

Experiment outputs are stored under:

```text
experiments/
├── results/
├── artifacts/
└── reports/
```

---

# Experiment Results

Results should only be added here after running the evaluation pipeline.

A final comparison can be reported in this format:

| Configuration | Recall@5 | MRR | nDCG@5 | Faithfulness | Latency |
|---|---:|---:|---:|---:|---:|
| Dense baseline | TBD | TBD | TBD | TBD | TBD |
| BM25 | TBD | TBD | TBD | TBD | TBD |
| Hybrid RRF | TBD | TBD | TBD | TBD | TBD |
| Hybrid + reranker | TBD | TBD | TBD | TBD | TBD |

No benchmark results are claimed until they have been measured on the held-out evaluation set.

---

# Running Evaluation

Run the evaluation pipeline:

```bash
python scripts/run_evaluation.py
```

Run experiment comparisons:

```bash
python scripts/run_experiments.py
```

Export results:

```bash
python scripts/export_results.py
```

---

# Command-Line Utilities

Utility scripts are available under:

```text
scripts/
```

### Ingest documents

```bash
python scripts/ingest.py
```

### Query the system

```bash
python scripts/query.py
```

### Rebuild indexes

```bash
python scripts/rebuild_indexes.py
```

### Create an evaluation dataset

```bash
python scripts/create_eval_dataset.py
```

### Run evaluation

```bash
python scripts/run_evaluation.py
```

### Run experiments

```bash
python scripts/run_experiments.py
```

### Export results

```bash
python scripts/export_results.py
```

---

# Testing

Tests are divided into unit and integration suites.

```text
tests/
├── unit/
├── integration/
└── fixtures/
```

Run all tests:

```bash
pytest
```

Run unit tests:

```bash
pytest tests/unit
```

Run integration tests:

```bash
pytest tests/integration
```

Run with coverage:

```bash
pytest --cov=docqna --cov-report=term-missing
```

---

# Docker

DocQ&A includes separate containers for the backend and frontend.

```text
Dockerfile.backend
Dockerfile.frontend
docker-compose.yml
```

Build and start:

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

---

# Local LLM Support

DocQ&A can use Ollama as the generation backend.

Start Ollama and make sure the configured model is available.

For example:

```bash
ollama pull <model-name>
```

Then select the Ollama generation configuration:

```text
configs/generation/ollama.yaml
```

The embedding and reranking components can also run locally.

---

# Observability

The system is structured to record information useful for debugging and experimentation, including:

```text
request ID
retrieval strategy
dense candidates
BM25 candidates
RRF ranking
reranker scores
selected context chunks
abstention decision
LLM latency
total latency
```

Logging configuration lives in:

```text
configs/logging.yaml
```

Application logs are written under:

```text
logs/
```

depending on the active configuration.

---

# Security Considerations

Uploaded documents are treated as **untrusted data**.

Retrieved document text is evidence for answering a question; it is not treated as system instruction.

The generation prompt should explicitly prevent instructions contained inside uploaded documents from overriding application-level instructions.

The system should also avoid logging sensitive source-document contents unnecessarily in deployed environments.

---

# Documentation

Detailed documentation is available under:

```text
docs/
├── architecture.md
├── system_design.md
├── evaluation.md
├── experiments.md
├── api.md
├── deployment.md
└── limitations.md
```

Use the root README for orientation and setup.

Use the documents above for deeper architectural and implementation details.

---

# Notebooks

Exploratory analysis is kept separate from production code.

```text
notebooks/
├── 01_document_exploration.ipynb
├── 02_chunking_experiments.ipynb
├── 03_embedding_analysis.ipynb
├── 04_retrieval_analysis.ipynb
└── 05_error_analysis.ipynb
```

The notebooks are intended for investigation and visualization rather than serving as the primary implementation of the system.

---

# Development Philosophy

DocQ&A follows a few core principles:

**Retrieval and generation are evaluated separately.**

A poor answer may come from retrieval failure even when the LLM behaves correctly.

**Complexity must justify itself.**

Hybrid retrieval, semantic chunking, reranking, and additional models should remain in the final system only when evaluation demonstrates useful improvement.

**Answers should be traceable.**

Source metadata is preserved throughout ingestion, retrieval, generation, and citation formatting.

**Experiments must be reproducible.**

Important model and retrieval choices belong in configuration files rather than scattered constants.

**Unsupported questions should not force an answer.**

Abstention is explicitly evaluated as part of the system.

---

# Project Goal

The goal of DocQ&A is not simply:

> Upload a PDF and ask an LLM questions.

The goal is to build a **measurable, modular document-intelligence system** and answer:

> Which RAG architecture performs better, why, and at what cost?

The project therefore treats chunking, dense retrieval, BM25, hybrid fusion, reranking, answerability, citations, generation quality, latency, and failure analysis as measurable engineering decisions.

---

# Roadmap

- [ ] Core document ingestion
- [ ] Fixed chunking baseline
- [ ] ChromaDB indexing
- [ ] Dense retrieval
- [ ] Grounded generation
- [ ] Basic source citations
- [ ] Evaluation dataset
- [ ] Retrieval metrics
- [ ] BM25 retrieval
- [ ] RRF hybrid retrieval
- [ ] Cross-encoder reranking
- [ ] Answerability / abstention
- [ ] Citation validation
- [ ] Generation-quality evaluation
- [ ] Chunking experiments
- [ ] Embedding experiments
- [ ] Streamlit interface
- [ ] FastAPI refinement
- [ ] Unit and integration testing
- [ ] Docker deployment
- [ ] Error analysis
- [ ] Held-out benchmark
- [ ] Final experiment report

Update the checklist as implementation progresses.

---

# License

This project is licensed under the terms defined in [`LICENSE`](LICENSE).

---

## Acknowledgements

DocQ&A builds on concepts and tools from information retrieval, semantic search, Retrieval-Augmented Generation, sentence-transformer models, BM25, cross-encoder reranking, ChromaDB, FastAPI, and Streamlit.