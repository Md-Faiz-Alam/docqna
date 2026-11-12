from unittest.mock import patch

from fastapi.testclient import TestClient

from docqna.api.dependencies import get_rag_pipeline_provider
from docqna.api.main import SESSION_HEADER_NAME, app
from docqna.api.rate_limit import RateLimitRule, SlidingWindowLimiter
from docqna.rag import ABSTENTION_ANSWER, RAGResponse


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


RULE = RateLimitRule(bucket="question", methods=frozenset({"POST"}), path_prefix="/qa", limit=2)


def test_sliding_window_allows_limit_then_blocks_until_window_passes() -> None:
    clock = Clock()
    limiter = SlidingWindowLimiter(clock=clock)

    assert limiter.hit(RULE, "k") is None
    assert limiter.hit(RULE, "k") is None
    assert limiter.hit(RULE, "k") == 60.0
    assert limiter.hit(RULE, "other-client") is None

    clock.now += 60.1
    assert limiter.hit(RULE, "k") is None


class AbstainingPipeline:
    def ask_with_citations(
        self, query, top_k=5, metadata_filters=None, history=None
    ) -> RAGResponse:
        return RAGResponse(answer=ABSTENTION_ANSWER, sources=[], cited_source_ids=[])


def test_question_endpoint_returns_429_with_retry_after(client: TestClient) -> None:
    app.dependency_overrides[get_rag_pipeline_provider] = lambda: AbstainingPipeline
    limit = 20  # settings default

    statuses = [
        client.post("/qa", json={"question": "Hello?"}).status_code for _ in range(limit + 1)
    ]

    assert statuses[:limit] == [200] * limit
    assert statuses[-1] == 429
    rejected = client.post("/qa", json={"question": "Hello?"})
    assert rejected.json()["error"] == "rate_limited"
    assert int(rejected.headers["Retry-After"]) >= 1
    assert "X-Request-ID" in rejected.headers

    # A different session has its own budget.
    other = client.post("/qa", json={"question": "Hello?"}, headers={SESSION_HEADER_NAME: "z" * 32})
    assert other.status_code == 200


def test_ready_reports_each_dependency(client: TestClient) -> None:
    class Healthy:
        def count(self) -> int:
            return 0

        def list_records(self, session_id=None):
            return []

        def health_check(self) -> bool:
            return True

    class Down(Healthy):
        def health_check(self) -> bool:
            return False

    target = "docqna.api.routes.system"
    with (
        patch(f"{target}.get_vector_store", return_value=Healthy()),
        patch(f"{target}.get_document_registry", return_value=Healthy()),
        patch(f"{target}.get_llm", return_value=Healthy()),
    ):
        ready = client.get("/ready")
    with (
        patch(f"{target}.get_vector_store", return_value=Healthy()),
        patch(f"{target}.get_document_registry", return_value=Healthy()),
        patch(f"{target}.get_llm", return_value=Down()),
    ):
        not_ready = client.get("/ready")

    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "checks": {"vector_store": True, "registry": True, "llm": True},
    }
    assert not_ready.status_code == 503
    assert not_ready.json()["checks"]["llm"] is False


def test_metrics_use_route_templates_not_raw_paths(client: TestClient) -> None:
    client.get("/health")
    client.delete("/documents/doc_secret_id_123")

    body = client.get("/metrics").text

    assert 'docqna_http_requests_total{method="GET",route="/health",status="200"}' in body
    assert 'route="/documents/{document_id}"' in body
    assert "doc_secret_id_123" not in body
