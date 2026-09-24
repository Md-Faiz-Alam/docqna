from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException

from docqna.api.main import require_session_id
from docqna.exceptions import DuplicateDocumentError
from docqna.filters import normalize_metadata_filters
from docqna.indexing.chroma import ChromaVectorStore
from docqna.ingestion.models import (
    LEGACY_SESSION_ID,
    DocumentChunk,
    DocumentRecord,
)
from docqna.ingestion.registry import DocumentRegistry
from docqna.retrieval.bm25 import BM25Retriever


SESSION_A = "a" * 32
SESSION_B = "b" * 32


def make_chunk(*, chunk_id: str, text: str, session_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        filename="shared.txt",
        file_type="txt",
        source_segment_indices=[0],
        metadata={
            "chunking_strategy": "fixed",
            "word_count": len(text.split()),
            "sections": [],
            "document_id": f"doc_{session_id[:4]}",
            "session_id": session_id,
            "document_version": 1,
            "content_hash": session_id,
            "original_chunk_id": chunk_id,
        },
    )


def main() -> None:
    # Old registry JSON records stay readable.
    legacy = DocumentRecord.model_validate(
        {
            "document_id": "doc_legacy",
            "filename": "legacy.txt",
            "file_type": "txt",
            "version": 1,
            "content_hash": "abc",
            "size_bytes": 3,
            "registered_at": "2026-01-01T00:00:00Z",
        }
    )
    assert legacy.session_id == LEGACY_SESSION_ID
    print("Legacy registry compatibility: PASS")

    assert normalize_metadata_filters(
        {"session_id": SESSION_A}
    ) == {"session_id": SESSION_A}
    print("Session metadata filter: PASS")

    assert require_session_id(SESSION_A) == SESSION_A
    try:
        require_session_id("too-short")
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("Invalid session header was accepted.")
    print("API session header validation: PASS")

    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        root = Path(temp_dir)
        registry = DocumentRegistry(
            registry_path=root / "documents.json"
        )
        file_path = root / "same.txt"
        file_path.write_text("same content", encoding="utf-8")

        record_a = registry.register(
            file_path,
            session_id=SESSION_A,
        )
        record_b = registry.register(
            file_path,
            session_id=SESSION_B,
        )

        assert record_a.document_id != record_b.document_id
        assert len(registry.list_latest_records(SESSION_A)) == 1
        assert len(registry.list_latest_records(SESSION_B)) == 1

        try:
            registry.register(
                file_path,
                session_id=SESSION_A,
            )
        except DuplicateDocumentError:
            pass
        else:
            raise AssertionError(
                "Same-session duplicate was not rejected."
            )

        assert registry.delete_document(
            record_a.document_id,
            session_id=SESSION_B,
        ) == 0
        assert len(registry.list_latest_records(SESSION_A)) == 1
        print("Registry ownership isolation: PASS")

    chunk_a = make_chunk(
        chunk_id="a0",
        text="alpha private evidence",
        session_id=SESSION_A,
    )
    chunk_b = make_chunk(
        chunk_id="b0",
        text="beta private evidence",
        session_id=SESSION_B,
    )

    bm25 = BM25Retriever([chunk_a, chunk_b])
    results_a = bm25.retrieve(
        "private evidence",
        top_k=5,
        metadata_filters={"session_id": SESSION_A},
    )
    assert results_a
    assert all(
        result.document_id == "doc_aaaa"
        for result in results_a
    )
    print("BM25 session isolation: PASS")

    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        store = ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="session_isolation_test",
        )
        store.add_chunks(
            chunks=[chunk_a, chunk_b],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )

        reconstructed = store.get_all_chunks()
        by_id = {chunk.chunk_id: chunk for chunk in reconstructed}
        assert by_id["a0"].metadata["session_id"] == SESSION_A
        assert by_id["b0"].metadata["session_id"] == SESSION_B

        result = store.query_by_embedding(
            embedding=[1.0, 0.0],
            top_k=1,
            metadata_filters={"session_id": SESSION_A},
        )
        assert result["ids"][0] == ["a0"]

        deleted_wrong_owner = store.delete_document_chunks(
            document_id="doc_aaaa",
            session_id=SESSION_B,
        )
        assert deleted_wrong_owner == 0

        deleted_owner = store.delete_document_chunks(
            document_id="doc_aaaa",
            session_id=SESSION_A,
        )
        assert deleted_owner == 1
        print("Chroma session isolation: PASS")

    print("\nSession isolation regression tests passed.")


if __name__ == "__main__":
    main()
