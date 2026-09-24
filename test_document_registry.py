from pathlib import Path
from tempfile import TemporaryDirectory

from docqna.exceptions import (
    DuplicateDocumentError,
)
from docqna.ingestion import (
    DocumentRegistry,
)


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        registry_path = (
            root / "documents.json"
        )

        document_path = (
            root / "project.txt"
        )

        registry = DocumentRegistry(
            registry_path=registry_path
        )

        # ---------------------------------
        # Version 1
        # ---------------------------------

        document_path.write_text(
            "Project document version one.",
            encoding="utf-8",
        )

        version_1 = registry.register(
            document_path
        )

        assert version_1.version == 1

        assert version_1.document_id.startswith(
            "doc_"
        )

        print(
            "Version 1:",
            version_1.document_id,
            version_1.version,
        )

        # ---------------------------------
        # Exact duplicate
        # ---------------------------------

        try:
            registry.register(
                document_path
            )

        except DuplicateDocumentError:
            print(
                "Duplicate detection: PASS"
            )

        else:
            raise AssertionError(
                "Exact duplicate was not rejected."
            )

        # ---------------------------------
        # Same logical document, new content
        # ---------------------------------

        document_path.write_text(
            "Project document version two "
            "with changed content.",
            encoding="utf-8",
        )

        version_2 = registry.register(
            document_path
        )

        assert (
            version_2.document_id
            == version_1.document_id
        )

        assert version_2.version == 2

        assert (
            version_2.content_hash
            != version_1.content_hash
        )

        print(
            "Version 2:",
            version_2.document_id,
            version_2.version,
        )

        # ---------------------------------
        # Persistence check
        # ---------------------------------

        reloaded_registry = (
            DocumentRegistry(
                registry_path=registry_path
            )
        )

        records = (
            reloaded_registry.list_records()
        )

        assert len(records) == 2

        assert records[0].version == 1
        assert records[1].version == 2

        print(
            "Persistent records:",
            len(records),
        )

        print()
        print(
            "Document registry test passed."
        )


if __name__ == "__main__":
    main()