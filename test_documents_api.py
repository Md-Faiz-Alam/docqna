from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from docqna.api.dependencies import (
    get_document_index_service_provider,
)
from docqna.api.main import app
from docqna.exceptions import DuplicateDocumentError
from docqna.indexing import DocumentIndexingResult
from docqna.ingestion import DocumentRecord


class FakeDocumentIndexService:
    """In-memory fake used only for API route tests."""

    def __init__(self) -> None:
        self.records: dict[str, DocumentRecord] = {}
        self._filename_to_id: dict[str, str] = {}

    def index_document(
        self,
        path: str | Path,
    ) -> DocumentIndexingResult:
        file_path = Path(path)
        filename = file_path.name
        key = filename.casefold()

        if key in self._filename_to_id:
            raise DuplicateDocumentError(
                "An identical document has already "
                "been registered."
            )

        document_id = "doc_test1234567890"

        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            file_type=file_path.suffix.lower().lstrip("."),
            version=1,
            content_hash="a" * 64,
            size_bytes=file_path.stat().st_size,
            registered_at=datetime.now(
                timezone.utc
            ),
        )

        self.records[document_id] = record
        self._filename_to_id[key] = document_id

        return DocumentIndexingResult(
            record=record,
            chunk_count=1,
        )

    def list_documents(
        self,
    ) -> list[DocumentRecord]:
        return list(
            self.records.values()
        )

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        record = self.records.pop(
            document_id,
            None,
        )

        if record is None:
            return False

        self._filename_to_id.pop(
            record.filename.casefold(),
            None,
        )

        return True


def main() -> None:
    fake_service = FakeDocumentIndexService()

    app.dependency_overrides[
        get_document_index_service_provider
    ] = lambda: (
        lambda: fake_service
    )

    try:
        client = TestClient(app)

        # ---------------------------------
        # Initially empty
        # ---------------------------------

        response = client.get(
            "/documents"
        )

        assert response.status_code == 200

        body = response.json()

        assert body["count"] == 0
        assert body["documents"] == []

        print("Empty document list: PASS")

        # ---------------------------------
        # Upload
        # ---------------------------------

        response = client.post(
            "/documents",
            files={
                "file": (
                    "project.txt",
                    b"Project documentation.",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 201

        body = response.json()

        assert (
            body["document"]["document_id"]
            == "doc_test1234567890"
        )

        assert (
            body["document"]["filename"]
            == "project.txt"
        )

        assert (
            body["document"]["version"]
            == 1
        )

        assert body["chunk_count"] == 1

        print("Document upload: PASS")

        # ---------------------------------
        # Listing after upload
        # ---------------------------------

        response = client.get(
            "/documents"
        )

        assert response.status_code == 200

        body = response.json()

        assert body["count"] == 1

        assert (
            body["documents"][0]["filename"]
            == "project.txt"
        )

        print("Document listing: PASS")

        # ---------------------------------
        # Duplicate
        # ---------------------------------

        response = client.post(
            "/documents",
            files={
                "file": (
                    "project.txt",
                    b"Project documentation.",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 409
        assert (
            response.json()["error"]
            == "duplicate_document"
        )

        print("Duplicate upload rejection: PASS")

        # ---------------------------------
        # Unsupported type
        # ---------------------------------

        response = client.post(
            "/documents",
            files={
                "file": (
                    "image.png",
                    b"not-an-image",
                    "image/png",
                )
            },
        )

        assert response.status_code == 415
        assert (
            response.json()["error"]
            == "unsupported_file_type"
        )

        print("Unsupported type rejection: PASS")

        # ---------------------------------
        # Empty supported document
        # ---------------------------------

        response = client.post(
            "/documents",
            files={
                "file": (
                    "empty.txt",
                    b"",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 400
        assert (
            response.json()["error"]
            == "document_validation_error"
        )

        print("Empty document rejection: PASS")

        # ---------------------------------
        # Delete existing
        # ---------------------------------

        response = client.delete(
            "/documents/doc_test1234567890"
        )

        assert response.status_code == 200

        assert response.json() == {
            "document_id": "doc_test1234567890",
            "deleted": True,
        }

        print("Document deletion: PASS")

        # ---------------------------------
        # Delete missing
        # ---------------------------------

        response = client.delete(
            "/documents/doc_test1234567890"
        )

        assert response.status_code == 404

        print("Missing document deletion: PASS")

        # ---------------------------------
        # Final list
        # ---------------------------------

        response = client.get(
            "/documents"
        )

        assert response.status_code == 200
        assert response.json()["count"] == 0
        assert response.json()["documents"] == []

        print("Final document list: PASS")

        print()
        print(
            "Document API route tests passed."
        )

    finally:
        app.dependency_overrides.clear()


if __name__ == "__main__":
    main()
