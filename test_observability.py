import logging
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast

from fastapi.testclient import TestClient

from docqna.embeddings.base import BaseEmbeddingModel
from docqna.filters import MetadataFilters
from docqna.generation.providers.base import BaseLLMProvider
from docqna.indexing.chroma import ChromaVectorStore
from docqna.indexing.service import DocumentIndexService
from docqna.ingestion.models import (
    DocumentChunk,
    DocumentRecord,
)
from docqna.ingestion.registry import DocumentRegistry
from docqna.ingestion.service import DocumentIngestionService
from docqna.logging_config import (
    RequestIdFilter,
    configure_logging,
    shutdown_logging,
)
from docqna.rag.pipeline import RAGPipeline
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult


SECRET_QUERY = "DO_NOT_LOG_THIS_QUERY_TEXT"
SECRET_DOCUMENT_TEXT = (
    "DO_NOT_LOG_THIS_DOCUMENT_TEXT"
)


class CaptureHandler(logging.Handler):
    """Capture formatted application log messages in memory."""

    def __init__(self) -> None:
        super().__init__(
            level=logging.INFO
        )
        self.messages: list[str] = []

    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:
        self.messages.append(
            record.getMessage()
        )


class FakeRetriever(BaseRetriever):
    """Return one deterministic retrieval result."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="chunk_test",
                text="Safe evidence text.",
                rank=1,
                distance=0.1,
                score=1.0,
                retrieval_method="fake",
                filename="safe.txt",
                file_type="txt",
                chunk_index=0,
                page_start=None,
                page_end=None,
                section=None,
                source_segment_indices=[0],
                sections=[],
                document_id="doc_safe",
                document_version=1,
            )
        ]


class FakeLLM:
    """Minimal LLM test double."""

    def generate(
        self,
        prompt: str,
    ) -> str:
        return "Grounded answer [1]"


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Deterministic embedding model for indexing logs."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeIngestionService:
    """Return one already-prepared chunk."""

    def __init__(
        self,
        chunk: DocumentChunk,
    ) -> None:
        self.chunk = chunk

    def process(
        self,
        path: str | Path,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            chunks=[self.chunk]
        )


class FakeRegistry:
    """Minimal registry used by DocumentIndexService."""

    def __init__(
        self,
        record: DocumentRecord,
    ) -> None:
        self.record = record
        self.committed = False

    def prepare(
        self,
        path: str | Path,
    ) -> DocumentRecord:
        return self.record

    def commit(
        self,
        record: DocumentRecord,
    ) -> None:
        self.committed = True

    def list_records(
        self,
    ) -> list[DocumentRecord]:
        return (
            [self.record]
            if self.committed
            else []
        )

    def list_latest_records(
        self,
    ) -> list[DocumentRecord]:
        return self.list_records()

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        if (
            self.committed
            and document_id
            == self.record.document_id
        ):
            self.committed = False
            return 1

        return 0


class FakeVectorStore:
    """Minimal vector-store test double."""

    def __init__(self) -> None:
        self.chunks: list[
            DocumentChunk
        ] = []

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        self.chunks.extend(
            chunks
        )

    def delete_document_chunks(
        self,
        document_id: str,
        version: int | None = None,
    ) -> int:
        before = len(
            self.chunks
        )

        self.chunks = [
            chunk
            for chunk in self.chunks
            if not (
                chunk.metadata.get(
                    "document_id"
                )
                == document_id
                and (
                    version is None
                    or chunk.metadata.get(
                        "document_version"
                    )
                    == version
                )
            )
        ]

        return before - len(
            self.chunks
        )


def test_rag_observability(
    messages: list[str],
) -> None:
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        llm=cast(
            BaseLLMProvider,
            FakeLLM(),
        ),
    )

    response = (
        pipeline.ask_with_citations(
            query=SECRET_QUERY,
            top_k=5,
            metadata_filters={
                "document_id": "doc_safe",
            },
        )
    )

    assert response.answer == (
        "Grounded answer [1]"
    )

    joined = "\n".join(
        messages
    )

    assert "rag_complete mode=cited" in joined
    assert "retrieval_ms=" in joined
    assert "generation_ms=" in joined
    assert SECRET_QUERY not in joined

    print(
        "RAG timing/privacy logging: PASS"
    )


def test_index_observability(
    root: Path,
    messages: list[str],
) -> None:
    path = root / "private.txt"

    path.write_text(
        SECRET_DOCUMENT_TEXT,
        encoding="utf-8",
    )

    record = DocumentRecord(
        document_id="doc_observe",
        filename="private.txt",
        file_type="txt",
        version=1,
        content_hash="a" * 64,
        size_bytes=path.stat().st_size,
        registered_at=datetime.now(
            timezone.utc
        ),
    )

    chunk = DocumentChunk(
        chunk_id="chunk_original",
        chunk_index=0,
        text=SECRET_DOCUMENT_TEXT,
        filename="private.txt",
        file_type="txt",
        source_segment_indices=[0],
        page_start=None,
        page_end=None,
        section=None,
        metadata={
            "chunking_strategy": "fixed",
            "word_count": 1,
            "sections": [],
        },
    )

    fake_registry = FakeRegistry(
        record
    )

    fake_store = FakeVectorStore()

    service = DocumentIndexService(
        ingestion_service=cast(
            DocumentIngestionService,
            FakeIngestionService(
                chunk
            ),
        ),
        registry=cast(
            DocumentRegistry,
            fake_registry,
        ),
        embedding_model=(
            FakeEmbeddingModel()
        ),
        vector_store=cast(
            ChromaVectorStore,
            fake_store,
        ),
    )

    result = service.index_document(
        path
    )

    assert result.chunk_count == 1
    assert fake_registry.committed

    joined = "\n".join(
        messages
    )

    assert (
        "document_index_complete"
        in joined
    )

    assert (
        "processing_ms="
        in joined
    )

    assert (
        "embedding_ms="
        in joined
    )

    assert (
        SECRET_DOCUMENT_TEXT
        not in joined
    )

    print(
        "Document indexing logging: PASS"
    )


def test_http_observability(
    messages: list[str],
) -> None:
    # Import only after configure_logging() has installed
    # the temporary test log handlers.
    from docqna.api.main import app

    with TestClient(
        app
    ) as client:
        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        request_id = response.headers.get(
            "X-Request-ID"
        )

        assert request_id
        assert len(request_id) == 32

        assert (
            response.headers.get(
                "X-Content-Type-Options"
            )
            == "nosniff"
        )

        assert (
            response.headers.get(
                "Referrer-Policy"
            )
            == "no-referrer"
        )

    joined = "\n".join(
        messages
    )

    assert "app_start" in joined
    assert (
        "http_request method=GET "
        "path=/health status=200"
        in joined
    )

    assert "duration_ms=" in joined
    assert "app_stop" in joined

    print(
        "HTTP request observability: PASS"
    )
    print(
        "Security response headers: PASS"
    )
    print(
        "Startup/shutdown diagnostics: PASS"
    )


def main() -> None:
    with TemporaryDirectory(
        ignore_cleanup_errors=True,
    ) as temp_dir:
        root = Path(
            temp_dir
        )

        log_dir = (
            root / "logs"
        )

        logger = configure_logging(
            force=True,
            log_dir=log_dir,
        )

        capture = CaptureHandler()
        capture.addFilter(
            RequestIdFilter()
        )

        logger.addHandler(
            capture
        )

        try:
            test_rag_observability(
                capture.messages
            )

            test_index_observability(
                root,
                capture.messages,
            )

            test_http_observability(
                capture.messages
            )

            log_file = (
                log_dir / "docqna.log"
            )

            assert log_file.exists()

            log_text = log_file.read_text(
                encoding="utf-8"
            )

            assert "rag_complete" in log_text
            assert (
                "document_index_complete"
                in log_text
            )
            assert "http_request" in log_text

            assert (
                SECRET_QUERY
                not in log_text
            )

            assert (
                SECRET_DOCUMENT_TEXT
                not in log_text
            )

            print(
                "Rotating file logging: PASS"
            )
            print(
                "Sensitive-content log check: PASS"
            )

        finally:
            logger.removeHandler(
                capture
            )
            capture.close()
            shutdown_logging()

    print()
    print(
        "Observability tests passed."
    )


if __name__ == "__main__":
    main()
