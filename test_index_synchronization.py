from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

from fastapi.testclient import TestClient

from docqna.api import dependencies
from docqna.api.dependencies import (
    get_document_index_service_provider,
)
from docqna.api.main import app
from docqna.embeddings.base import BaseEmbeddingModel
from docqna.filters import MetadataFilters
from docqna.indexing import (
    ChromaVectorStore,
    DocumentIndexingResult,
)
from docqna.ingestion.models import (
    DocumentChunk,
    DocumentRecord,
)
from docqna.retrieval import (
    BM25Retriever,
    RRFHybridRetriever,
)
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Small deterministic embedding model for synchronization tests."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(
        text: str,
    ) -> list[float]:
        lower = text.lower()

        return [
            1.0,
            float(lower.count("alpha")),
            float(lower.count("beta")),
        ]


class FakeReranker(BaseRetriever):
    """
    Stand-in for CrossEncoderReranker.

    It preserves the base retrieval stack so this test can
    inspect the BM25 corpus without loading a real model.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        model_name: str,
        candidate_pool_size: int,
        device: str,
    ) -> None:
        self.base_retriever = base_retriever
        self.model_name = model_name
        self.candidate_pool_size = candidate_pool_size
        self.device = device

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        return self.base_retriever.retrieve(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )


class FakeOllamaProvider:
    """Minimal Ollama stand-in used while building the RAG pipeline."""

    def __init__(
        self,
        base_url: str,
        model: str,
    ) -> None:
        self.base_url = base_url
        self.model = model

    def health_check(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
    ) -> str:
        return "Unused in synchronization test."


class FakeDocumentIndexService:
    """Minimal service used to verify API cache invalidation hooks."""

    def index_document(
        self,
        path: str | Path,
    ) -> DocumentIndexingResult:
        file_path = Path(path)

        record = DocumentRecord(
            document_id="doc_api_sync",
            filename=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            version=1,
            content_hash="a" * 64,
            size_bytes=file_path.stat().st_size,
            registered_at=datetime.now(
                timezone.utc
            ),
        )

        return DocumentIndexingResult(
            record=record,
            chunk_count=1,
        )

    def list_documents(
        self,
    ) -> list[DocumentRecord]:
        return []

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        return document_id == "doc_api_sync"


def build_chunk(
    *,
    chunk_id: str,
    text: str,
    filename: str,
    document_id: str,
) -> DocumentChunk:
    """Create one fixed-strategy managed chunk."""

    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        filename=filename,
        file_type="txt",
        source_segment_indices=[0],
        page_start=None,
        page_end=None,
        section=None,
        metadata={
            "chunking_strategy": "fixed",
            "word_count": len(
                text.split()
            ),
            "sections": [],
            "document_id": document_id,
            "document_version": 1,
            "content_hash": (
                document_id.replace(
                    "doc_",
                    "",
                )
                * 8
            )[:64],
            "original_chunk_id": chunk_id,
        },
    )


def get_bm25_chunk_ids(
    pipeline: object,
) -> list[str]:
    """Read the BM25 corpus IDs from a built production stack."""

    rag_pipeline = cast(
        object,
        pipeline,
    )

    retriever = cast(
        FakeReranker,
        getattr(
            rag_pipeline,
            "retriever",
        ),
    )

    hybrid = cast(
        RRFHybridRetriever,
        retriever.base_retriever,
    )

    bm25 = cast(
        BM25Retriever,
        hybrid.sparse_retriever,
    )

    return [
        chunk.chunk_id
        for chunk in bm25.chunks
    ]


def test_rag_cache_and_bm25_rebuild() -> None:
    """
    Verify that BM25 is a derived index built from the current
    Chroma corpus whenever the cached RAG stack is invalidated.
    """

    embedding_model = FakeEmbeddingModel()

    chunk_alpha = build_chunk(
        chunk_id="chunk_alpha",
        text="Alpha project information.",
        filename="alpha.txt",
        document_id="doc_alpha",
    )

    chunk_beta = build_chunk(
        chunk_id="chunk_beta",
        text="Beta project information.",
        filename="beta.txt",
        document_id="doc_beta",
    )

    with TemporaryDirectory(
        ignore_cleanup_errors=True,
    ) as temp_dir:
        vector_store = ChromaVectorStore(
            persist_directory=(
                Path(temp_dir) / "chroma"
            ),
            collection_name=(
                "index_sync_test"
            ),
        )

        vector_store.add_chunks(
            chunks=[chunk_alpha],
            embeddings=(
                embedding_model.embed_documents(
                    [chunk_alpha.text]
                )
            ),
        )

        dependencies.get_rag_pipeline.cache_clear()

        try:
            with (
                patch.object(
                    dependencies,
                    "get_vector_store",
                    return_value=vector_store,
                ),
                patch.object(
                    dependencies,
                    "get_embedding_model",
                    return_value=embedding_model,
                ),
                patch.object(
                    dependencies,
                    "CrossEncoderReranker",
                    FakeReranker,
                ),
                patch.object(
                    dependencies,
                    "OllamaProvider",
                    FakeOllamaProvider,
                ),
            ):
                # ---------------------------------
                # Initial build
                # ---------------------------------

                pipeline_1 = (
                    dependencies.get_rag_pipeline()
                )

                assert get_bm25_chunk_ids(
                    pipeline_1
                ) == [
                    "chunk_alpha"
                ]

                print(
                    "Initial BM25 from Chroma: PASS"
                )

                # ---------------------------------
                # Cache reuse
                # ---------------------------------

                pipeline_2 = (
                    dependencies.get_rag_pipeline()
                )

                assert pipeline_2 is pipeline_1

                print(
                    "Unchanged corpus cache reuse: PASS"
                )

                # ---------------------------------
                # Chroma changes, cache not yet
                # invalidated
                # ---------------------------------

                vector_store.add_chunks(
                    chunks=[chunk_beta],
                    embeddings=(
                        embedding_model.embed_documents(
                            [chunk_beta.text]
                        )
                    ),
                )

                cached_pipeline = (
                    dependencies.get_rag_pipeline()
                )

                assert cached_pipeline is pipeline_1

                assert get_bm25_chunk_ids(
                    cached_pipeline
                ) == [
                    "chunk_alpha"
                ]

                print(
                    "Cached BM25 remains stable before "
                    "invalidation: PASS"
                )

                # ---------------------------------
                # Upload-style invalidation/rebuild
                # ---------------------------------

                dependencies.clear_rag_pipeline_cache()

                rebuilt_pipeline = (
                    dependencies.get_rag_pipeline()
                )

                assert rebuilt_pipeline is not pipeline_1

                assert set(
                    get_bm25_chunk_ids(
                        rebuilt_pipeline
                    )
                ) == {
                    "chunk_alpha",
                    "chunk_beta",
                }

                assert vector_store.count() == 2

                print(
                    "BM25 rebuild after index growth: PASS"
                )

                # ---------------------------------
                # Delete a logical document
                # ---------------------------------

                removed = (
                    vector_store.delete_document_chunks(
                        document_id="doc_alpha"
                    )
                )

                assert removed == 1
                assert vector_store.count() == 1

                dependencies.clear_rag_pipeline_cache()

                after_delete_pipeline = (
                    dependencies.get_rag_pipeline()
                )

                assert get_bm25_chunk_ids(
                    after_delete_pipeline
                ) == [
                    "chunk_beta"
                ]

                print(
                    "BM25 rebuild after deletion: PASS"
                )

        finally:
            dependencies.get_rag_pipeline.cache_clear()


def test_document_routes_invalidate_rag_cache() -> None:
    """
    Verify that successful document upload and deletion both
    call the production RAG cache invalidation hook.
    """

    fake_service = FakeDocumentIndexService()

    app.dependency_overrides[
        get_document_index_service_provider
    ] = lambda: (
        lambda: fake_service
    )

    try:
        with patch(
            "docqna.api.main.clear_rag_pipeline_cache"
        ) as clear_cache:
            client = TestClient(app)

            upload_response = client.post(
                "/documents",
                files={
                    "file": (
                        "sync.txt",
                        b"Synchronization test document.",
                        "text/plain",
                    )
                },
            )

            assert upload_response.status_code == 201
            assert clear_cache.call_count == 1

            print(
                "Upload invalidates RAG cache: PASS"
            )

            delete_response = client.delete(
                "/documents/doc_api_sync"
            )

            assert delete_response.status_code == 200
            assert clear_cache.call_count == 2

            print(
                "Delete invalidates RAG cache: PASS"
            )

    finally:
        app.dependency_overrides.clear()


def main() -> None:
    test_rag_cache_and_bm25_rebuild()
    test_document_routes_invalidate_rag_cache()

    print()
    print(
        "Index synchronization tests passed."
    )


if __name__ == "__main__":
    main()
