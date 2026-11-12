import pytest
from fastapi.testclient import TestClient

from docqna.api.dependencies import get_rag_pipeline_provider
from docqna.api.main import SESSION_HEADER_NAME, app
from docqna.constants import APP_NAME, APP_VERSION
from docqna.exceptions import CitationError, LLMConnectionError, RetrievalError
from docqna.rag.models import CitationSource, RAGResponse
from docqna.rag.pipeline import ABSTENTION_ANSWER


class FakePipeline:
    def __init__(self, response: RAGResponse | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def ask_with_citations(self, query: str, top_k: int = 5, **kwargs) -> RAGResponse:
        self.calls.append({"query": query, "top_k": top_k, **kwargs})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def use_pipeline(pipeline: FakePipeline) -> None:
    app.dependency_overrides[get_rag_pipeline_provider] = lambda: (lambda: pipeline)


def make_source(citation_id: int, text: str) -> CitationSource:
    return CitationSource(
        citation_id=citation_id,
        chunk_id=f"chunk_test_{citation_id}",
        rank=citation_id,
        filename="sample.pdf",
        file_type="pdf",
        page_start=89,
        page_end=90,
        section=None,
        distance=-7.0,
        text=text,
    )


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app": APP_NAME, "version": APP_VERSION}
    assert "X-Request-ID" in response.headers


def test_blank_question_is_rejected(client: TestClient) -> None:
    response = client.post("/qa", json={"question": "   "})

    assert response.status_code == 422


@pytest.mark.parametrize("header", [None, "short", "has spaces " + "x" * 40])
def test_qa_requires_valid_session_header(header: str | None) -> None:
    headers = {SESSION_HEADER_NAME: header} if header is not None else {}
    with TestClient(app) as raw_client:
        response = raw_client.post("/qa", json={"question": "Anything?"}, headers=headers)

    assert response.status_code == 400


def test_answer_returns_only_cited_sources(client: TestClient, session_id: str) -> None:
    pipeline = FakePipeline(
        RAGResponse(
            answer="Risk identification uses brainstorming [1].",
            cited_source_ids=[1],
            sources=[
                make_source(1, "Brainstorming can be used to identify risks."),
                make_source(2, "Another retrieved source."),
            ],
        )
    )
    use_pipeline(pipeline)

    response = client.post("/qa", json={"question": "How are project risks identified?"})

    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is False
    assert data["cited_source_ids"] == [1]
    assert [source["citation_id"] for source in data["sources"]] == [1]
    # The server, not the client, decides the ownership filter.
    assert pipeline.calls[0]["metadata_filters"]["session_id"] == session_id


def test_abstention(client: TestClient) -> None:
    use_pipeline(
        FakePipeline(RAGResponse(answer=ABSTENTION_ANSWER, cited_source_ids=[], sources=[]))
    )

    response = client.post("/qa", json={"question": "What is the capital city of Japan?"})

    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is True
    assert data["cited_source_ids"] == []
    assert data["sources"] == []


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (LLMConnectionError("Ollama is not reachable."), 503, "llm_connection_error"),
        (RetrievalError("Retrieval failed."), 500, "retrieval_error"),
        (CitationError("Citation validation failed."), 500, "citation_error"),
    ],
)
def test_pipeline_errors_map_to_http_errors(
    client: TestClient, error: Exception, status_code: int, code: str
) -> None:
    use_pipeline(FakePipeline(error=error))

    response = client.post("/qa", json={"question": "Test question"})

    assert response.status_code == status_code
    assert response.json() == {"error": code, "detail": str(error)}
