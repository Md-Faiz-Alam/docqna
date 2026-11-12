"""Per-session BM25 indexing: isolation, caching, and targeted invalidation."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from docqna.api.dependencies import get_document_index_service_provider, make_chunk_loader
from docqna.api.main import app
from docqna.exceptions import IndexingError
from docqna.filters import MetadataFilters
from docqna.indexing import DocumentIndexingResult
from docqna.ingestion.models import DocumentChunk, DocumentRecord
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult
from docqna.retrieval.session_index import SessionScopedHybridRetriever

SESSION_A = "a" * 32
SESSION_B = "b" * 32


def chunk(chunk_id: str, text: str, session_id: str, strategy: str = "fixed") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        filename=f"{chunk_id}.txt",
        file_type="txt",
        source_segment_indices=[0],
        metadata={
            "chunking_strategy": strategy,
            "session_id": session_id,
            "document_id": f"doc_{chunk_id}",
            "document_version": 1,
        },
    )


class EmptyDense(BaseRetriever):
    def retrieve(
        self, query: str, top_k: int = 5, metadata_filters: MetadataFilters | None = None
    ) -> list[RetrievalResult]:
        return []


class CountingLoader:
    def __init__(self, corpus: dict[str, list[DocumentChunk]]) -> None:
        self.corpus = corpus
        self.calls: list[str | None] = []

    def __call__(self, session_id: str | None) -> list[DocumentChunk]:
        self.calls.append(session_id)
        if session_id is None:
            return [c for chunks in self.corpus.values() for c in chunks]
        return list(self.corpus.get(session_id, []))


@pytest.fixture
def loader() -> CountingLoader:
    return CountingLoader(
        {
            SESSION_A: [chunk("a1", "alpha budget approval process", SESSION_A)],
            SESSION_B: [chunk("b1", "alpha travel expense policy", SESSION_B)],
        }
    )


def make_retriever(loader: CountingLoader, **kwargs) -> SessionScopedHybridRetriever:
    return SessionScopedHybridRetriever(EmptyDense(), loader, **kwargs)


def ask(retriever: SessionScopedHybridRetriever, session_id: str) -> list[str]:
    results = retriever.retrieve("alpha", top_k=5, metadata_filters={"session_id": session_id})
    return [r.chunk_id for r in results]


def test_each_session_only_sees_and_loads_its_own_chunks(loader: CountingLoader) -> None:
    retriever = make_retriever(loader)

    assert ask(retriever, SESSION_A) == ["a1"]
    assert ask(retriever, SESSION_B) == ["b1"]
    assert loader.calls == [SESSION_A, SESSION_B]


def test_index_is_cached_until_that_session_is_invalidated(loader: CountingLoader) -> None:
    retriever = make_retriever(loader)
    ask(retriever, SESSION_A)
    ask(retriever, SESSION_B)
    ask(retriever, SESSION_A)
    assert loader.calls == [SESSION_A, SESSION_B]

    loader.corpus[SESSION_A].append(chunk("a2", "alpha new upload", SESSION_A))
    retriever.invalidate(SESSION_A)

    assert sorted(ask(retriever, SESSION_A)) == ["a1", "a2"]
    ask(retriever, SESSION_B)
    # Only session A was rebuilt; B kept its cached index.
    assert loader.calls == [SESSION_A, SESSION_B, SESSION_A]


def test_invalidate_all_drops_every_session(loader: CountingLoader) -> None:
    retriever = make_retriever(loader)
    ask(retriever, SESSION_A)
    ask(retriever, SESSION_B)

    retriever.invalidate()

    assert retriever.cached_sessions() == 0


def test_session_without_documents_returns_nothing(loader: CountingLoader) -> None:
    retriever = make_retriever(loader)

    assert ask(retriever, "c" * 32) == []


def test_lru_bounds_memory(loader: CountingLoader) -> None:
    retriever = make_retriever(loader, max_cached_sessions=1)
    ask(retriever, SESSION_A)
    ask(retriever, SESSION_B)

    assert retriever.cached_sessions() == 1
    ask(retriever, SESSION_A)
    assert loader.calls == [SESSION_A, SESSION_B, SESSION_A]


def test_index_built_during_invalidation_is_not_cached() -> None:
    retriever: SessionScopedHybridRetriever

    def racing_loader(session_id: str | None) -> list[DocumentChunk]:
        # An upload lands while this (now stale) index is being built.
        retriever.invalidate(session_id)
        return [chunk("a1", "alpha", SESSION_A)]

    retriever = SessionScopedHybridRetriever(EmptyDense(), racing_loader)
    ask(retriever, SESSION_A)

    assert retriever.cached_sessions() == 0


def test_chunk_loader_rejects_unevaluated_chunking_strategy() -> None:
    class Store:
        def get_all_chunks(self, session_id: str | None = None) -> list[DocumentChunk]:
            return [chunk("x", "text", SESSION_A, strategy="semantic")]

    with pytest.raises(IndexingError):
        make_chunk_loader(Store())(SESSION_A)  # type: ignore[arg-type]


class FakeDocumentIndexService:
    def purge_expired_documents(self) -> int:
        return 0

    def index_document(self, path: Path, session_id: str | None = None) -> DocumentIndexingResult:
        record = DocumentRecord(
            document_id="doc_sync",
            session_id=session_id or "",
            filename=Path(path).name,
            file_type="txt",
            version=1,
            content_hash="a" * 64,
            size_bytes=Path(path).stat().st_size,
            registered_at=datetime.now(UTC),
        )
        return DocumentIndexingResult(record=record, chunk_count=1)

    def delete_document(self, document_id: str, session_id: str | None = None) -> bool:
        return document_id == "doc_sync"


def test_upload_and_delete_invalidate_only_the_callers_session(
    client: TestClient, session_id: str
) -> None:
    service = FakeDocumentIndexService()
    app.dependency_overrides[get_document_index_service_provider] = lambda: (lambda: service)

    with patch("docqna.api.routes.documents.invalidate_session_index") as invalidate:
        upload = client.post("/documents", files={"file": ("sync.txt", b"content", "text/plain")})
        delete = client.delete("/documents/doc_sync")

    assert upload.status_code == 201
    assert delete.status_code == 200
    assert [call.args for call in invalidate.call_args_list] == [(session_id,), (session_id,)]
