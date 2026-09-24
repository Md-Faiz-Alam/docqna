import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from docqna.constants import METADATA_DIR
from docqna.exceptions import (
    DocumentError,
    DocumentValidationError,
    DuplicateDocumentError,
)
from docqna.ingestion.models import (
    LEGACY_SESSION_ID,
    DocumentRecord,
)


DEFAULT_DOCUMENT_REGISTRY = (
    METADATA_DIR / "documents.json"
)


class DocumentRegistry:
    """
    Persistent registry for document identity,
    deduplication, and version tracking.

    Document identity rules:

    Identity is scoped to one anonymous session:

    - identical file content in the same session is a duplicate
    - the same filename with changed content in the same session
      becomes a new version of the same logical document
    - the same content may be uploaded independently by another
      session
    - a new filename receives a new document_id
    """

    def __init__(
        self,
        registry_path: str | Path = (
            DEFAULT_DOCUMENT_REGISTRY
        ),
    ) -> None:
        self.registry_path = Path(
            registry_path
        )

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # FastAPI sync routes may execute concurrently in worker
        # threads. Protect registry read-modify-write operations
        # from in-process lost updates.
        self._write_lock = RLock()

    @staticmethod
    def _normalize_session_id(
        session_id: str | None,
    ) -> str:
        """Return a non-empty internal session namespace."""

        if session_id is None:
            return LEGACY_SESSION_ID

        value = session_id.strip()

        if not value:
            raise ValueError(
                "session_id cannot be empty."
            )

        return value

    @staticmethod
    def compute_content_hash(
        path: str | Path,
    ) -> str:
        """
        Compute a SHA-256 hash from the raw file bytes.
        """

        file_path = Path(path)

        if not file_path.exists():
            raise DocumentValidationError(
                f"Document does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise DocumentValidationError(
                f"Path is not a file: {file_path}"
            )

        digest = hashlib.sha256()

        try:
            with file_path.open("rb") as file:
                while True:
                    block = file.read(
                        1024 * 1024
                    )

                    if not block:
                        break

                    digest.update(block)

        except OSError as exc:
            raise DocumentError(
                "Could not read document while "
                f"computing hash: {file_path}"
            ) from exc

        return digest.hexdigest()

    def _load_records(
        self,
    ) -> list[DocumentRecord]:
        """
        Load all persisted document records.
        """

        if not self.registry_path.exists():
            return []

        try:
            raw_data = json.loads(
                self.registry_path.read_text(
                    encoding="utf-8"
                )
            )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise DocumentError(
                "Could not read document registry: "
                f"{self.registry_path}"
            ) from exc

        if not isinstance(raw_data, list):
            raise DocumentError(
                "Document registry must contain "
                "a JSON list."
            )

        try:
            return [
                DocumentRecord.model_validate(
                    item
                )
                for item in raw_data
            ]

        except Exception as exc:
            raise DocumentError(
                "Document registry contains "
                "invalid records."
            ) from exc

    def _save_records(
        self,
        records: list[DocumentRecord],
    ) -> None:
        """
        Persist records using an atomic file replacement.
        """

        payload = [
            record.model_dump(
                mode="json"
            )
            for record in records
        ]

        temporary_path = (
            self.registry_path.with_suffix(
                self.registry_path.suffix
                + ".tmp"
            )
        )

        try:
            temporary_path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                ),
                encoding="utf-8",
            )

            temporary_path.replace(
                self.registry_path
            )

        except OSError as exc:
            raise DocumentError(
                "Could not write document registry: "
                f"{self.registry_path}"
            ) from exc

    def list_records(
        self,
        session_id: str | None = None,
    ) -> list[DocumentRecord]:
        """
        Return registered document versions.

        When session_id is provided, only records owned by that
        anonymous session are returned. None preserves the legacy
        internal/admin call shape and returns all records.
        """

        records = self._load_records()

        if session_id is None:
            return records

        normalized_session_id = (
            self._normalize_session_id(session_id)
        )

        return [
            record
            for record in records
            if record.session_id == normalized_session_id
        ]

    def list_latest_records(
        self,
        session_id: str | None = None,
    ) -> list[DocumentRecord]:
        """
        Return the latest registered version of each
        logical document, optionally scoped to one session.
        """

        records = self.list_records(
            session_id=session_id
        )

        latest_by_document: dict[
            str,
            DocumentRecord,
        ] = {}

        for record in records:
            current = latest_by_document.get(
                record.document_id
            )

            if (
                current is None
                or record.version > current.version
            ):
                latest_by_document[
                    record.document_id
                ] = record

        return sorted(
            latest_by_document.values(),
            key=lambda record: (
                record.filename.casefold(),
                record.document_id,
            ),
        )

    def delete_document(
        self,
        document_id: str,
        session_id: str | None = None,
    ) -> int:
        """
        Delete registry versions for one logical document.

        When session_id is provided, ownership is enforced.
        Returns the number of removed records.
        """

        document_id = document_id.strip()

        if not document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        normalized_session_id = (
            self._normalize_session_id(session_id)
            if session_id is not None
            else None
        )

        with self._write_lock:
            records = self._load_records()

            remaining = [
                record
                for record in records
                if not (
                    record.document_id == document_id
                    and (
                        normalized_session_id is None
                        or record.session_id
                        == normalized_session_id
                    )
                )
            ]

            removed_count = (
                len(records) - len(remaining)
            )

            if removed_count == 0:
                return 0

            self._save_records(
                remaining
            )

        return removed_count

    def prepare(
        self,
        path: str | Path,
        session_id: str | None = None,
    ) -> DocumentRecord:
        """
        Validate document identity and prepare the next
        version record without persisting it.

        This lets the indexing service process and index
        the document before committing the registry entry.
        """

        normalized_session_id = (
            self._normalize_session_id(session_id)
        )

        file_path = Path(path)

        if not file_path.exists():
            raise DocumentValidationError(
                f"Document does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise DocumentValidationError(
                f"Path is not a file: {file_path}"
            )

        content_hash = (
            self.compute_content_hash(
                file_path
            )
        )

        records = [
            record
            for record in self._load_records()
            if record.session_id == normalized_session_id
        ]

        duplicate = next(
            (
                record
                for record in records
                if record.content_hash
                == content_hash
            ),
            None,
        )

        if duplicate is not None:
            raise DuplicateDocumentError(
                "An identical document has already "
                "been registered as "
                f"'{duplicate.filename}' "
                f"(document_id="
                f"{duplicate.document_id}, "
                f"version={duplicate.version})."
            )

        same_filename = [
            record
            for record in records
            if record.filename.casefold()
            == file_path.name.casefold()
        ]

        if same_filename:
            latest = max(
                same_filename,
                key=lambda record: (
                    record.version
                ),
            )

            document_id = (
                latest.document_id
            )

            version = (
                latest.version + 1
            )

        else:
            document_id = (
                f"doc_{uuid4().hex[:16]}"
            )

            version = 1

        file_type = (
            file_path.suffix.lower()
            .lstrip(".")
        )

        return DocumentRecord(
            document_id=document_id,
            session_id=normalized_session_id,
            filename=file_path.name,
            file_type=file_type,
            version=version,
            content_hash=content_hash,
            size_bytes=(
                file_path.stat().st_size
            ),
            registered_at=datetime.now(
                timezone.utc
            ),
        )

    def commit(
        self,
        record: DocumentRecord,
    ) -> None:
        """
        Persist a prepared document record.

        Duplicate checks are repeated under the registry write
        lock so concurrent requests cannot overwrite one another.
        Duplicate/version checks are scoped to record.session_id.
        """

        with self._write_lock:
            records = self._load_records()

            duplicate = next(
                (
                    existing
                    for existing in records
                    if (
                        existing.session_id
                        == record.session_id
                        and existing.content_hash
                        == record.content_hash
                    )
                ),
                None,
            )

            if duplicate is not None:
                raise DuplicateDocumentError(
                    "An identical document has already "
                    "been registered as "
                    f"'{duplicate.filename}' "
                    f"(document_id="
                    f"{duplicate.document_id}, "
                    f"version={duplicate.version})."
                )

            conflicting_version = next(
                (
                    existing
                    for existing in records
                    if (
                        existing.session_id
                        == record.session_id
                        and existing.document_id
                        == record.document_id
                        and existing.version
                        == record.version
                    )
                ),
                None,
            )

            if conflicting_version is not None:
                raise DocumentError(
                    "Document registry already contains "
                    f"document_id={record.document_id}, "
                    f"version={record.version}."
                )

            records.append(
                record
            )

            self._save_records(
                records
            )

    def register(
        self,
        path: str | Path,
        session_id: str | None = None,
    ) -> DocumentRecord:
        """
        Prepare and immediately persist a document.

        The future indexing service will instead use
        prepare() -> index -> commit().
        """

        record = self.prepare(
            path,
            session_id=session_id,
        )

        self.commit(
            record
        )

        return record