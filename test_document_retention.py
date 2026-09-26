from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from fastapi.testclient import TestClient

from docqna.api.dependencies import (
    get_document_index_service_provider,
)
from docqna.api.main import SESSION_HEADER_NAME, app
from docqna.embeddings import BaseEmbeddingModel
from docqna.indexing.chroma import ChromaVectorStore
from docqna.indexing.service import DocumentIndexService
from docqna.ingestion.models import DocumentRecord
from docqna.ingestion.registry import DocumentRegistry
from docqna.ingestion.service import DocumentIngestionService


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str | None]] = []

    def delete_document_chunks(
        self,
        document_id: str,
        version: int | None = None,
        session_id: str | None = None,
    ) -> int:
        self.deleted.append((document_id, session_id))
        return 1


class FakeAPIService:
    def __init__(self) -> None:
        self.purge_calls = 0

    def purge_expired_documents(self) -> int:
        self.purge_calls += 1
        return 0

    def list_documents(
        self,
        session_id: str | None = None,
    ) -> list[DocumentRecord]:
        return []


def record(
    document_id: str,
    filename: str,
    session_id: str,
    version: int,
    registered_at: datetime,
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        session_id=session_id,
        filename=filename,
        file_type=Path(filename).suffix.lstrip("."),
        version=version,
        content_hash=(f"{document_id}-{version}" * 64)[:64],
        size_bytes=1,
        registered_at=registered_at,
    )


def main() -> None:
    now = datetime(2026, 9, 25, 18, 0, tzinfo=timezone.utc)

    with TemporaryDirectory() as temp_dir:
        registry = DocumentRegistry(
            registry_path=Path(temp_dir) / "documents.json"
        )

        expired = record(
            "doc_expired",
            "expired.pdf",
            "session_a",
            1,
            now - timedelta(hours=25),
        )
        recent = record(
            "doc_recent",
            "recent.pdf",
            "session_b",
            1,
            now - timedelta(hours=2),
        )
        old_version = record(
            "doc_versioned",
            "versioned.pdf",
            "session_c",
            1,
            now - timedelta(hours=30),
        )
        current_version = record(
            "doc_versioned",
            "versioned.pdf",
            "session_c",
            2,
            now - timedelta(hours=1),
        )

        for item in (
            expired,
            recent,
            old_version,
            current_version,
        ):
            registry.commit(item)

        vector_store = FakeVectorStore()

        service = DocumentIndexService(
            ingestion_service=cast(
                DocumentIngestionService,
                object(),
            ),
            registry=registry,
            embedding_model=cast(
                BaseEmbeddingModel,
                object(),
            ),
            vector_store=cast(
                ChromaVectorStore,
                vector_store,
            ),
            max_documents_per_session=3,
            max_active_documents=3,
            max_chunks_per_document=1000,
            document_retention_hours=24,
        )

        deleted = service.purge_expired_documents(now=now)

        assert deleted == 1
        assert registry.list_records(session_id="session_a") == []
        assert {
            item.document_id
            for item in registry.list_latest_records()
        } == {"doc_recent", "doc_versioned"}
        assert len(
            registry.list_records(session_id="session_c")
        ) == 2
        assert vector_store.deleted == [
            ("doc_expired", "session_a")
        ]

        print("24-hour expired-document cleanup: PASS")
        print("Latest-version retention semantics: PASS")

        # The expired document no longer consumes a global active-document
        # quota slot, so a third active logical document can be admitted.
        service._enforce_document_limits(  # pyright: ignore[reportPrivateUsage]
            "new.pdf",
            "session_new",
        )
        print("Expired documents release capacity: PASS")

        disabled_service = DocumentIndexService(
            ingestion_service=cast(
                DocumentIngestionService,
                object(),
            ),
            registry=registry,
            embedding_model=cast(
                BaseEmbeddingModel,
                object(),
            ),
            vector_store=cast(
                ChromaVectorStore,
                vector_store,
            ),
            document_retention_hours=0,
        )
        assert disabled_service.purge_expired_documents(now=now) == 0
        print("Retention disable behavior: PASS")

    fake_api_service = FakeAPIService()
    fake_service = cast(DocumentIndexService, fake_api_service)

    app.dependency_overrides[
        get_document_index_service_provider
    ] = lambda: lambda: fake_service

    try:
        response = TestClient(app).get(
            "/documents",
            headers={
                SESSION_HEADER_NAME: "a" * 32,
            },
        )

        assert response.status_code == 200
        assert response.json()["count"] == 0
        assert fake_api_service.purge_calls == 1

    finally:
        app.dependency_overrides.clear()

    print("API request-triggered retention cleanup: PASS")
    print()
    print("Document retention tests passed.")


if __name__ == "__main__":
    main()
