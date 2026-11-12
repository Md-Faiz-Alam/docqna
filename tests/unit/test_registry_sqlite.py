import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from docqna.exceptions import DocumentError, DuplicateDocumentError
from docqna.ingestion.registry import DocumentRegistry

SESSION = "s" * 32


def _legacy_record(document_id: str, version: int, content_hash: str) -> dict:
    return {
        "document_id": document_id,
        "session_id": SESSION,
        "filename": "policy.txt",
        "file_type": "txt",
        "version": version,
        "content_hash": content_hash,
        "size_bytes": 10,
        "registered_at": "2026-01-01T00:00:00+00:00",
    }


def test_legacy_json_registry_is_imported_once(tmp_path: Path) -> None:
    legacy = tmp_path / "documents.json"
    legacy.write_text(
        json.dumps([_legacy_record("doc_a", 1, "h1"), _legacy_record("doc_a", 2, "h2")]),
        encoding="utf-8",
    )

    registry = DocumentRegistry(legacy)
    assert [r.version for r in registry.list_records(session_id=SESSION)] == [1, 2]
    assert registry.db_path == tmp_path / "documents.db"

    # Deleting everything must not resurrect legacy rows on restart.
    assert registry.delete_document("doc_a", session_id=SESSION) == 2
    assert DocumentRegistry(legacy).list_records() == []


def test_corrupt_legacy_json_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "documents.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(DocumentError):
        DocumentRegistry(tmp_path / "documents.db")


def test_concurrent_identical_uploads_commit_exactly_once(tmp_path: Path) -> None:
    registry = DocumentRegistry(tmp_path / "documents.db")
    document = tmp_path / "report.txt"
    document.write_text("same bytes", encoding="utf-8")

    # Every worker prepares before any commits, mimicking parallel requests
    # that all passed the pre-indexing duplicate check.
    prepared = [registry.prepare(document, session_id=SESSION) for _ in range(8)]

    def commit(record):
        try:
            DocumentRegistry(tmp_path / "documents.db").commit(record)
            return "ok"
        except DuplicateDocumentError:
            return "duplicate"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(commit, prepared))

    assert outcomes.count("ok") == 1
    assert outcomes.count("duplicate") == 7
    assert len(registry.list_records(session_id=SESSION)) == 1


def test_sessions_are_isolated(tmp_path: Path) -> None:
    registry = DocumentRegistry(tmp_path / "documents.db")
    document = tmp_path / "shared.txt"
    document.write_text("shared content", encoding="utf-8")

    first = registry.register(document, session_id="a" * 32)
    second = registry.register(document, session_id="b" * 32)

    assert first.document_id != second.document_id
    assert registry.delete_document(first.document_id, session_id="b" * 32) == 0
    assert len(registry.list_records(session_id="a" * 32)) == 1
