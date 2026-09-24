from fastapi.testclient import TestClient

from docqna.api.dependencies import (
    get_rag_pipeline_provider,
)
from docqna.api.main import app


class FakeSource:
    def __init__(
        self,
        citation_id: int = 1,
    ) -> None:
        self.citation_id = citation_id

    def model_dump(
        self,
    ) -> dict[str, object]:
        return {
            "citation_id": self.citation_id,
            "chunk_id": "chunk_1",
            "rank": 1,
            "filename": "beta.txt",
            "file_type": "txt",
            "page_start": 1,
            "page_end": 1,
            "section": None,
            "distance": 0.1,
            "text": "Filtered evidence.",
        }


class FakeResponse:
    answer = "Filtered answer [1]"
    cited_source_ids = [1]
    sources = [
        FakeSource()
    ]


class FakePipeline:
    def __init__(
        self,
    ) -> None:
        self.last_metadata_filters: (
            dict[str, str | int] | None
        ) = None

    def ask_with_citations(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: (
            dict[str, str | int] | None
        ) = None,
    ) -> FakeResponse:
        self.last_metadata_filters = (
            metadata_filters
        )

        return FakeResponse()


def main() -> None:
    fake_pipeline = FakePipeline()

    app.dependency_overrides[
        get_rag_pipeline_provider
    ] = lambda: (
        lambda: fake_pipeline
    )

    try:
        client = TestClient(app)

        # ---------------------------------
        # Filtered QA request
        # ---------------------------------

        response = client.post(
            "/qa",
            json={
                "question": "Finance guidance?",
                "top_k": 5,
                "filters": {
                    "document_id": "doc_beta",
                    "file_type": ".TXT",
                    "document_version": 2,
                },
            },
        )

        assert response.status_code == 200

        assert (
            fake_pipeline.last_metadata_filters
            == {
                "document_id": "doc_beta",
                "file_type": "txt",
                "document_version": 2,
            }
        )

        body = response.json()

        assert body["answer"] == (
            "Filtered answer [1]"
        )

        assert (
            body["sources"][0]["filename"]
            == "beta.txt"
        )

        print(
            "Filtered /qa request: PASS"
        )

        # ---------------------------------
        # Invalid filter
        # ---------------------------------

        response = client.post(
            "/qa",
            json={
                "question": "Finance guidance?",
                "filters": {
                    "document_version": 0,
                },
            },
        )

        assert response.status_code == 422

        print(
            "Invalid filter validation: PASS"
        )

        # ---------------------------------
        # Original unfiltered call shape
        # ---------------------------------

        fake_pipeline.last_metadata_filters = (
            {"sentinel": "value"}
        )

        response = client.post(
            "/qa",
            json={
                "question": "General question?",
                "top_k": 3,
            },
        )

        assert response.status_code == 200

        # FakePipeline's default is None only because
        # main.py used the original unfiltered call shape.
        assert (
            fake_pipeline.last_metadata_filters
            is None
        )

        print(
            "Unfiltered /qa compatibility: PASS"
        )

        print()
        print(
            "QA metadata filter API tests passed."
        )

    finally:
        app.dependency_overrides.clear()


if __name__ == "__main__":
    main()
