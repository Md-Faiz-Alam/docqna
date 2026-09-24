from fastapi.testclient import TestClient

from docqna.api.dependencies import get_rag_pipeline_provider
from docqna.api.main import app
from docqna.exceptions import (
    CitationError,
    LLMConnectionError,
    RetrievalError,
)
from docqna.rag.models import (
    CitationSource,
    RAGResponse,
)


class FakePipeline:
    def __init__(
        self,
        response: RAGResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error

    def ask_with_citations(
        self,
        query: str,
        top_k: int = 5,
    ) -> RAGResponse:
        if self.error is not None:
            raise self.error

        if self.response is None:
            raise RuntimeError(
                "FakePipeline requires a response or error."
            )

        return self.response


def set_pipeline(
    pipeline: FakePipeline,
) -> None:
    app.dependency_overrides[
        get_rag_pipeline_provider
    ] = lambda: (
        lambda: pipeline
    )


def clear_overrides() -> None:
    app.dependency_overrides.clear()


def test_health(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["app"] == "DocQ&A"
    assert data["version"] == "0.1.0"


def test_invalid_question(
    client: TestClient,
) -> None:
    response = client.post(
        "/qa",
        json={
            "question": "   ",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_answerable_question(
    client: TestClient,
) -> None:
    rag_response = RAGResponse(
        answer="Risk identification uses brainstorming [1].",
        cited_source_ids=[1],
        sources=[
            CitationSource(
                citation_id=1,
                chunk_id="chunk_test_1",
                rank=1,
                filename="sample.pdf",
                file_type="pdf",
                page_start=89,
                page_end=90,
                section=None,
                distance=-7.0,
                text="Brainstorming can be used to identify risks.",
            ),
            CitationSource(
                citation_id=2,
                chunk_id="chunk_test_2",
                rank=2,
                filename="sample.pdf",
                file_type="pdf",
                page_start=90,
                page_end=91,
                section=None,
                distance=-6.0,
                text="Another retrieved source.",
            ),
        ],
    )

    set_pipeline(
        FakePipeline(
            response=rag_response
        )
    )

    try:
        response = client.post(
            "/qa",
            json={
                "question": (
                    "How are project risks identified?"
                ),
                "top_k": 5,
            },
        )
    finally:
        clear_overrides()

    assert response.status_code == 200

    data = response.json()

    assert data["abstained"] is False
    assert data["cited_source_ids"] == [1]

    assert len(data["sources"]) == 1

    assert (
        data["sources"][0]["citation_id"]
        == 1
    )


def test_abstention(
    client: TestClient,
) -> None:
    rag_response = RAGResponse(
        answer=(
            "The provided documents do not contain "
            "enough information to answer the question."
        ),
        cited_source_ids=[],
        sources=[],
    )

    set_pipeline(
        FakePipeline(
            response=rag_response
        )
    )

    try:
        response = client.post(
            "/qa",
            json={
                "question": (
                    "What is the capital city of Japan?"
                ),
                "top_k": 5,
            },
        )
    finally:
        clear_overrides()

    assert response.status_code == 200

    data = response.json()

    assert data["abstained"] is True
    assert data["cited_source_ids"] == []
    assert data["sources"] == []


def test_llm_connection_error(
    client: TestClient,
) -> None:
    set_pipeline(
        FakePipeline(
            error=LLMConnectionError(
                "Ollama is not reachable."
            )
        )
    )

    try:
        response = client.post(
            "/qa",
            json={
                "question": "Test question",
                "top_k": 5,
            },
        )
    finally:
        clear_overrides()

    assert response.status_code == 503

    assert response.json() == {
        "error": "llm_connection_error",
        "detail": "Ollama is not reachable.",
    }


def test_retrieval_error(
    client: TestClient,
) -> None:
    set_pipeline(
        FakePipeline(
            error=RetrievalError(
                "Retrieval failed."
            )
        )
    )

    try:
        response = client.post(
            "/qa",
            json={
                "question": "Test question",
                "top_k": 5,
            },
        )
    finally:
        clear_overrides()

    assert response.status_code == 500

    assert response.json() == {
        "error": "retrieval_error",
        "detail": "Retrieval failed.",
    }


def test_citation_error(
    client: TestClient,
) -> None:
    set_pipeline(
        FakePipeline(
            error=CitationError(
                "Citation validation failed."
            )
        )
    )

    try:
        response = client.post(
            "/qa",
            json={
                "question": "Test question",
                "top_k": 5,
            },
        )
    finally:
        clear_overrides()

    assert response.status_code == 500

    assert response.json() == {
        "error": "citation_error",
        "detail": (
            "Citation validation failed."
        ),
    }


def main() -> None:
    client = TestClient(app)

    tests = [
        test_health,
        test_invalid_question,
        test_answerable_question,
        test_abstention,
        test_llm_connection_error,
        test_retrieval_error,
        test_citation_error,
    ]

    for test in tests:
        clear_overrides()

        test(client)

        print(
            f"PASS: {test.__name__}"
        )

    clear_overrides()

    print()
    print("All API tests passed.")


if __name__ == "__main__":
    main()