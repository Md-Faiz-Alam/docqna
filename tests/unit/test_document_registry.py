from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from docqna.exceptions import DuplicateDocumentError
from docqna.ingestion import DocumentRegistry


def test_document_registry() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        registry_path = root / "documents.json"

        document_path = root / "project.txt"

        registry = DocumentRegistry(registry_path=registry_path)

        # ---------------------------------
        # Version 1
        # ---------------------------------

        document_path.write_text("Project document version one.", encoding="utf-8")

        version_1 = registry.register(document_path)

        assert version_1.version == 1

        assert version_1.document_id.startswith("doc_")

        # ---------------------------------
        # Exact duplicate
        # ---------------------------------

        with pytest.raises(DuplicateDocumentError):
            registry.register(document_path)

        # ---------------------------------
        # Same logical document, new content
        # ---------------------------------

        document_path.write_text(
            "Project document version two with changed content.", encoding="utf-8"
        )

        version_2 = registry.register(document_path)

        assert version_2.document_id == version_1.document_id

        assert version_2.version == 2

        assert version_2.content_hash != version_1.content_hash

        # ---------------------------------
        # Persistence check
        # ---------------------------------

        reloaded_registry = DocumentRegistry(registry_path=registry_path)

        records = reloaded_registry.list_records()

        assert len(records) == 2

        assert records[0].version == 1
        assert records[1].version == 2
