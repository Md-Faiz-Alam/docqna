from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient
from pydantic import ValidationError

from docqna.api.main import SESSION_HEADER_NAME, app
from docqna.api.schemas import QARequest
from docqna.embeddings import BaseEmbeddingModel
from docqna.exceptions import ResourceLimitError
from docqna.indexing.chroma import ChromaVectorStore
from docqna.indexing.service import DocumentIndexService
from docqna.ingestion.models import (
    DocumentChunk,
    DocumentRecord,
    LoadedDocument,
)
from docqna.ingestion.registry import DocumentRegistry
from docqna.ingestion.service import (
    DocumentIngestionService,
    DocumentProcessingResult,
)


class FakeRegistry:
    def __init__(self) -> None:
        self.session_records: dict[str, list[DocumentRecord]] = {}

    def list_latest_records(
        self,
        session_id: str | None = None,
    ) -> list[DocumentRecord]:
        if session_id is None:
            return [
                record
                for records in self.session_records.values()
                for record in records
            ]

        return list(
            self.session_records.get(
                session_id,
                [],
            )
        )

    def prepare(
        self,
        path: str | Path,
        session_id: str | None = None,
    ) -> DocumentRecord:
        resolved_session_id = session_id or "legacy_local_session"
        return record(
            "prepared",
            Path(path).name,
            resolved_session_id,
        )


class TooManyChunksIngestionService:
    def process(
        self,
        path: str | Path,
    ) -> DocumentProcessingResult:
        file_path = Path(path)

        document = LoadedDocument(
            source_path=file_path,
            filename=file_path.name,
            file_type="txt",
            segments=[],
        )

        chunks = [
            DocumentChunk(
                chunk_id=f"chunk_{index}",
                chunk_index=index,
                text="chunk text",
                filename=file_path.name,
                file_type="txt",
                source_segment_indices=[0],
                metadata={
                    "chunking_strategy": "fixed",
                },
            )
            for index in range(1001)
        ]

        return DocumentProcessingResult(
            document=document,
            chunks=chunks,
        )


class FailIfEmbedded:
    def embed_documents(self, texts):
        raise AssertionError(
            "Chunk limit must be enforced before embedding."
        )


def record(
    document_id: str,
    filename: str,
    session_id: str,
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        session_id=session_id,
        filename=filename,
        file_type=Path(filename).suffix.lstrip("."),
        version=1,
        content_hash=(document_id * 64)[:64],
        size_bytes=1,
        registered_at=datetime.now(timezone.utc),
    )


def make_service(
    registry: FakeRegistry,
    *,
    per_session: int = 3,
    active: int = 25,
) -> DocumentIndexService:
    return DocumentIndexService(
        ingestion_service=cast(
            DocumentIngestionService,
            object(),
        ),
        registry=cast(DocumentRegistry, registry),
        embedding_model=cast(
            BaseEmbeddingModel,
            object(),
        ),
        vector_store=cast(
            ChromaVectorStore,
            object(),
        ),
        max_documents_per_session=per_session,
        max_active_documents=active,
        max_chunks_per_document=1000,
    )


def main() -> None:
    assert QARequest(question="hello").top_k == 5

    try:
        QARequest.model_validate(
            {"question": "hello", "top_k": 10}
        )
    except ValidationError:
        pass
    else:
        raise AssertionError(
            "top_k other than 5 must be rejected"
        )

    print("Fixed production top_k=5: PASS")

    client = TestClient(app)
    headers = {
        SESSION_HEADER_NAME: "a" * 32,
    }

    response = client.post(
        "/qa",
        headers=headers,
        json={
            "question": "x" * 2001,
            "top_k": 5,
        },
    )

    assert response.status_code == 422
    assert "2000-character limit" in response.json()["detail"]

    print("Question length limit: PASS")

    registry = FakeRegistry()
    registry.session_records["session_a"] = [
        record("a1", "one.pdf", "session_a"),
        record("a2", "two.pdf", "session_a"),
        record("a3", "three.pdf", "session_a"),
    ]

    service = make_service(registry)

    try:
        service._enforce_document_limits(  # pyright: ignore[reportPrivateUsage]
            "four.pdf",
            "session_a",
        )
    except ResourceLimitError:
        pass
    else:
        raise AssertionError(
            "session document quota was not enforced"
        )

    # Changed content under the same filename is a new version of the
    # same logical document, so it does not consume another quota slot.
    service._enforce_document_limits(  # pyright: ignore[reportPrivateUsage]
        "one.pdf",
        "session_a",
    )

    print("Per-session logical-document limit: PASS")
    print("Existing-document version allowance: PASS")

    registry = FakeRegistry()
    registry.session_records["session_a"] = [
        record("a1", "one.pdf", "session_a"),
    ]
    registry.session_records["session_b"] = [
        record("b1", "two.pdf", "session_b"),
    ]

    service = make_service(
        registry,
        per_session=3,
        active=2,
    )

    try:
        service._enforce_document_limits(  # pyright: ignore[reportPrivateUsage]
            "three.pdf",
            "session_c",
        )
    except ResourceLimitError:
        pass
    else:
        raise AssertionError(
            "global active-document quota was not enforced"
        )

    print("Global active-document limit: PASS")

    chunk_limited_service = DocumentIndexService(
        ingestion_service=cast(
            DocumentIngestionService,
            TooManyChunksIngestionService(),
        ),
        registry=cast(DocumentRegistry, FakeRegistry()),
        embedding_model=cast(
            BaseEmbeddingModel,
            FailIfEmbedded(),
        ),
        vector_store=cast(
            ChromaVectorStore,
            object(),
        ),
        max_chunks_per_document=1000,
    )

    try:
        chunk_limited_service.index_document(
            "large.txt",
            session_id="session_chunks",
        )
    except ResourceLimitError:
        pass
    else:
        raise AssertionError(
            "chunk-count limit was not enforced"
        )

    print("Per-document chunk limit: PASS")
    print()
    print("Public resource-limit tests passed.")


if __name__ == "__main__":
    main()
