"""Persistent document registry backed by SQLite.

SQLite (WAL mode) replaces the original JSON file so concurrent API
workers cannot lose updates: duplicate/version checks and inserts run in
one `BEGIN IMMEDIATE` transaction, and uniqueness is enforced by the
schema itself. An existing `documents.json` registry is imported once on
first start.
"""

import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from docqna.exceptions import DocumentError, DocumentValidationError, DuplicateDocumentError
from docqna.ingestion.models import LEGACY_SESSION_ID, DocumentRecord
from docqna.settings import get_settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    session_id    TEXT    NOT NULL,
    document_id   TEXT    NOT NULL,
    version       INTEGER NOT NULL CHECK (version >= 1),
    filename      TEXT    NOT NULL,
    file_type     TEXT    NOT NULL,
    content_hash  TEXT    NOT NULL,
    size_bytes    INTEGER NOT NULL CHECK (size_bytes >= 0),
    registered_at TEXT    NOT NULL,
    PRIMARY KEY (session_id, document_id, version),
    UNIQUE (session_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_documents_filename
    ON documents (session_id, filename COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS registry_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_COLUMNS = (
    "document_id, session_id, filename, file_type, version, content_hash, size_bytes, registered_at"
)

_LEGACY_IMPORTED_KEY = "legacy_json_imported"


def _duplicate_error(record: DocumentRecord) -> DuplicateDocumentError:
    return DuplicateDocumentError(
        f"An identical document has already been registered as '{record.filename}' "
        f"(document_id={record.document_id}, version={record.version})."
    )


class DocumentRegistry:
    """Document identity, deduplication, and version tracking.

    Identity is scoped to one anonymous session:

    - identical file content in the same session is a duplicate
    - the same filename with changed content in the same session becomes a
      new version of the same logical document
    - the same content may be uploaded independently by another session
    - a new filename receives a new document_id
    """

    def __init__(self, registry_path: str | Path | None = None) -> None:
        path = Path(registry_path) if registry_path is not None else get_settings().registry_path

        # Accept the historical `.json` path and keep the database beside it.
        self.db_path = path.with_suffix(".db") if path.suffix == ".json" else path
        self.legacy_json_path = self.db_path.with_suffix(".json")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.executescript(_SCHEMA)
        self._import_legacy_json()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived connection; commit on success, roll back on error."""

        try:
            raw = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        except sqlite3.Error as exc:
            raise DocumentError(f"Could not open document registry: {self.db_path}") from exc

        with closing(raw) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            try:
                yield connection
            except sqlite3.Error as exc:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise DocumentError("Document registry operation failed.") from exc

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialize writers across threads *and* processes."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    @staticmethod
    def _to_record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord.model_validate(dict(row))

    @staticmethod
    def _insert(connection: sqlite3.Connection, record: DocumentRecord) -> None:
        connection.execute(
            f"INSERT INTO documents ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.document_id,
                record.session_id,
                record.filename,
                record.file_type,
                record.version,
                record.content_hash,
                record.size_bytes,
                record.registered_at.isoformat(),
            ),
        )

    def _import_legacy_json(self) -> None:
        """One-time import of a pre-SQLite `documents.json` registry."""

        with self._write_transaction() as connection:
            already = connection.execute(
                "SELECT 1 FROM registry_meta WHERE key = ?", (_LEGACY_IMPORTED_KEY,)
            ).fetchone()
            if already is not None:
                return

            imported = 0
            if self.legacy_json_path.exists():
                try:
                    raw = json.loads(self.legacy_json_path.read_text(encoding="utf-8"))
                    records = [DocumentRecord.model_validate(item) for item in raw]
                except (OSError, ValueError, TypeError) as exc:
                    raise DocumentError(
                        f"Could not import legacy registry: {self.legacy_json_path}"
                    ) from exc

                for record in records:
                    try:
                        self._insert(connection, record)
                        imported += 1
                    except sqlite3.IntegrityError:
                        logger.warning(
                            "registry_legacy_import_skipped document_id=%s version=%d",
                            record.document_id,
                            record.version,
                        )

            connection.execute(
                "INSERT INTO registry_meta (key, value) VALUES (?, ?)",
                (_LEGACY_IMPORTED_KEY, datetime.now(UTC).isoformat()),
            )

        if imported:
            logger.info("registry_legacy_import_complete records=%d", imported)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_session_id(session_id: str | None) -> str:
        """Return a non-empty internal session namespace."""

        if session_id is None:
            return LEGACY_SESSION_ID

        value = session_id.strip()
        if not value:
            raise ValueError("session_id cannot be empty.")
        return value

    @staticmethod
    def compute_content_hash(path: str | Path) -> str:
        """SHA-256 of the raw file bytes."""

        file_path = Path(path)
        if not file_path.exists():
            raise DocumentValidationError(f"Document does not exist: {file_path}")
        if not file_path.is_file():
            raise DocumentValidationError(f"Path is not a file: {file_path}")

        digest = hashlib.sha256()
        try:
            with file_path.open("rb") as file:
                while block := file.read(1024 * 1024):
                    digest.update(block)
        except OSError as exc:
            raise DocumentError(
                f"Could not read document while computing hash: {file_path}"
            ) from exc

        return digest.hexdigest()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_records(self, session_id: str | None = None) -> list[DocumentRecord]:
        """Return registered versions in registration order.

        None returns every session's records (internal/admin use).
        """

        with self._connect() as connection:
            if session_id is None:
                rows = connection.execute(
                    f"SELECT {_COLUMNS} FROM documents ORDER BY rowid"
                ).fetchall()
            else:
                rows = connection.execute(
                    f"SELECT {_COLUMNS} FROM documents WHERE session_id = ? ORDER BY rowid",
                    (self._normalize_session_id(session_id),),
                ).fetchall()

        return [self._to_record(row) for row in rows]

    def list_latest_records(self, session_id: str | None = None) -> list[DocumentRecord]:
        """Return the latest version of each logical document."""

        latest_by_document: dict[tuple[str, str], DocumentRecord] = {}
        for record in self.list_records(session_id=session_id):
            key = (record.session_id, record.document_id)
            current = latest_by_document.get(key)
            if current is None or record.version > current.version:
                latest_by_document[key] = record

        return sorted(
            latest_by_document.values(),
            key=lambda record: (record.filename.casefold(), record.document_id),
        )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str, session_id: str | None = None) -> int:
        """Delete every version of a logical document; returns rows removed.

        When session_id is provided, ownership is enforced.
        """

        document_id = document_id.strip()
        if not document_id:
            raise ValueError("document_id cannot be empty.")

        with self._write_transaction() as connection:
            if session_id is None:
                cursor = connection.execute(
                    "DELETE FROM documents WHERE document_id = ?", (document_id,)
                )
            else:
                cursor = connection.execute(
                    "DELETE FROM documents WHERE document_id = ? AND session_id = ?",
                    (document_id, self._normalize_session_id(session_id)),
                )
            return cursor.rowcount

    def prepare(self, path: str | Path, session_id: str | None = None) -> DocumentRecord:
        """Validate identity and build the next version record without persisting it.

        The indexing service processes and indexes the document before
        calling commit().
        """

        normalized_session_id = self._normalize_session_id(session_id)
        file_path = Path(path)
        content_hash = self.compute_content_hash(file_path)

        with self._connect() as connection:
            duplicate = connection.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE session_id = ? AND content_hash = ?",
                (normalized_session_id, content_hash),
            ).fetchone()
            if duplicate is not None:
                raise _duplicate_error(self._to_record(duplicate))

            latest = connection.execute(
                "SELECT document_id, MAX(version) AS version FROM documents "
                "WHERE session_id = ? AND filename = ? COLLATE NOCASE "
                "GROUP BY document_id ORDER BY version DESC LIMIT 1",
                (normalized_session_id, file_path.name),
            ).fetchone()

        if latest is not None:
            document_id, version = latest["document_id"], latest["version"] + 1
        else:
            document_id, version = f"doc_{uuid4().hex[:16]}", 1

        return DocumentRecord(
            document_id=document_id,
            session_id=normalized_session_id,
            filename=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            version=version,
            content_hash=content_hash,
            size_bytes=file_path.stat().st_size,
            registered_at=datetime.now(UTC),
        )

    def commit(self, record: DocumentRecord) -> None:
        """Persist a prepared record.

        Checks run inside one write transaction, so concurrent uploads in
        any worker process cannot both commit the same content or version.
        """

        with self._write_transaction() as connection:
            duplicate = connection.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE session_id = ? AND content_hash = ?",
                (record.session_id, record.content_hash),
            ).fetchone()
            if duplicate is not None:
                raise _duplicate_error(self._to_record(duplicate))

            conflict = connection.execute(
                "SELECT 1 FROM documents WHERE session_id = ? AND document_id = ? AND version = ?",
                (record.session_id, record.document_id, record.version),
            ).fetchone()
            if conflict is not None:
                raise DocumentError(
                    "Document registry already contains "
                    f"document_id={record.document_id}, version={record.version}."
                )

            self._insert(connection, record)

    def register(self, path: str | Path, session_id: str | None = None) -> DocumentRecord:
        """Prepare and immediately persist a document (no indexing)."""

        record = self.prepare(path, session_id=session_id)
        self.commit(record)
        return record
