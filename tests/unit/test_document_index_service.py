from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from docqna.embeddings import BaseEmbeddingModel
from docqna.exceptions import DuplicateDocumentError
from docqna.indexing import ChromaVectorStore, DocumentIndexService
from docqna.ingestion import DocumentIngestionService, DocumentRegistry


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Small deterministic embedding model for testing."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        length = float(len(text))

        word_count = float(len(text.split()))

        checksum = float(sum(ord(char) for char in text) % 1000)

        return [length, word_count, checksum]


def test_document_index_service() -> None:
    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        root = Path(temp_dir)

        document_path = root / "project.txt"

        registry = DocumentRegistry(registry_path=(root / "documents.json"))

        vector_store = ChromaVectorStore(
            persist_directory=(root / "chroma"), collection_name=("document_index_test")
        )

        service = DocumentIndexService(
            ingestion_service=(DocumentIngestionService()),
            registry=registry,
            embedding_model=(FakeEmbeddingModel()),
            vector_store=vector_store,
        )

        # ---------------------------------
        # Version 1
        # ---------------------------------

        document_path.write_text(
            "This is the first version of the project document.", encoding="utf-8"
        )

        result_1 = service.index_document(document_path)

        assert result_1.record.version == 1
        assert result_1.chunk_count == 1
        assert vector_store.count() == 1

        chunks_v1 = vector_store.get_all_chunks()

        assert len(chunks_v1) == 1

        assert chunks_v1[0].metadata["document_id"] == result_1.record.document_id

        assert chunks_v1[0].metadata["document_version"] == 1

        # ---------------------------------
        # Duplicate
        # ---------------------------------

        with pytest.raises(DuplicateDocumentError):
            service.index_document(document_path)

        assert vector_store.count() == 1

        # ---------------------------------
        # Version 2
        # ---------------------------------

        document_path.write_text(
            "This is the second version of the project document with changed content.",
            encoding="utf-8",
        )

        result_2 = service.index_document(document_path)

        assert result_2.record.document_id == result_1.record.document_id

        assert result_2.record.version == 2

        # Old version must no longer be active.
        stored_chunks = vector_store.get_all_chunks()

        assert len(stored_chunks) == 1

        assert stored_chunks[0].metadata["document_version"] == 2

        assert stored_chunks[0].metadata["document_id"] == result_2.record.document_id

        records = registry.list_records()

        assert len(records) == 2

        # ---------------------------------
        # Active document listing
        # ---------------------------------

        latest_documents = service.list_documents()

        assert len(latest_documents) == 1

        assert latest_documents[0].document_id == result_2.record.document_id

        assert latest_documents[0].version == 2

        # ---------------------------------
        # Document deletion
        # ---------------------------------

        deleted = service.delete_document(result_2.record.document_id)

        assert deleted is True

        assert vector_store.count() == 0

        assert registry.list_records() == []

        assert service.list_documents() == []

        deleted_again = service.delete_document(result_2.record.document_id)

        assert deleted_again is False
