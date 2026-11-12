"""Prometheus metrics for the API and RAG pipeline.

Labels are deliberately low-cardinality: routes use their template path
(`/documents/{document_id}`), never raw URLs, session IDs, or query text.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HTTP_REQUESTS = Counter(
    "docqna_http_requests_total",
    "HTTP requests by route template, method, and status code.",
    ["method", "route", "status"],
)

HTTP_LATENCY = Histogram(
    "docqna_http_request_duration_seconds",
    "HTTP request latency by route template.",
    ["method", "route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160),
)

QA_OUTCOMES = Counter(
    "docqna_qa_responses_total",
    "Question-answering responses by outcome (answered or abstained).",
    ["outcome"],
)

RAG_STAGE_LATENCY = Histogram(
    "docqna_rag_stage_duration_seconds",
    "RAG pipeline stage latency.",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160),
)

BM25_INDEX_BUILDS = Counter("docqna_bm25_index_builds_total", "Per-session BM25 index (re)builds.")

RATE_LIMITED = Counter(
    "docqna_rate_limited_total", "Requests rejected by the rate limiter.", ["bucket"]
)


def record_http_request(*, method: str, route: str, status: int, seconds: float) -> None:
    HTTP_REQUESTS.labels(method=method, route=route, status=str(status)).inc()
    HTTP_LATENCY.labels(method=method, route=route).observe(seconds)


def record_qa_outcome(*, abstained: bool) -> None:
    QA_OUTCOMES.labels(outcome="abstained" if abstained else "answered").inc()


def record_rag_timings(*, retrieval_ms: float, generation_ms: float) -> None:
    RAG_STAGE_LATENCY.labels(stage="retrieval").observe(retrieval_ms / 1000.0)
    if generation_ms > 0:
        RAG_STAGE_LATENCY.labels(stage="generation").observe(generation_ms / 1000.0)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
