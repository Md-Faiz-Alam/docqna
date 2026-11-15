from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docqna.exceptions import DuplicateDocumentError
from docqna.ingestion.registry import DocumentRegistry

SESSION = "s" * 32


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
