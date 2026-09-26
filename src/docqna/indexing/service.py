import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import perf_counter
from pathlib import Path

from docqna.embeddings import BaseEmbeddingModel
from docqna.exceptions import (
    IndexingError,
    ResourceLimitError,
)
from docqna.indexing.chroma import ChromaVectorStore
from docqna.ingestion import (
    DocumentIngestionService,
    DocumentRecord,
    DocumentRegistry,
)
from docqna.ingestion.models import DocumentChunk


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DocumentIndexingResult:
    """Result of successfully indexing one document."""

    record: DocumentRecord
    chunk_count: int


class DocumentIndexService:
    """
    Production document indexing workflow.

    Flow:

    registry.prepare()
        -> load / preprocess / chunk
        -> attach document identity
        -> embed
        -> index new version
        -> registry.commit()
        -> remove older indexed versions
    """

    def __init__(
        self,
        ingestion_service: DocumentIngestionService,
        registry: DocumentRegistry,
        embedding_model: BaseEmbeddingModel,
        vector_store: ChromaVectorStore,
        max_documents_per_session: int | None = None,
        max_active_documents: int | None = None,
        max_chunks_per_document: int | None = None,
        document_retention_hours: int | None = None,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.registry = registry
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.max_documents_per_session = (
            max_documents_per_session
        )
        self.max_active_documents = max_active_documents
        self.max_chunks_per_document = max_chunks_per_document
        self.document_retention_hours = document_retention_hours

    def purge_expired_documents(
        self,
        *,
        now: datetime | None = None,
    ) -> int:
        """Delete logical documents older than the retention window.

        Expiration is based on the latest registered version of each
        logical document. A recent version keeps that logical document
        active even when older version records are older than the cutoff.

        Returns the number of logical documents deleted.
        """

        retention_hours = self.document_retention_hours

        if retention_hours is None or retention_hours <= 0:
            return 0

        current_time = now or datetime.now(timezone.utc)

        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)

        cutoff = current_time - timedelta(hours=retention_hours)

        expired_records: list[DocumentRecord] = []

        for record in self.registry.list_latest_records():
            registered_at = record.registered_at

            if registered_at.tzinfo is None:
                registered_at = registered_at.replace(tzinfo=timezone.utc)
            else:
                registered_at = registered_at.astimezone(timezone.utc)

            if registered_at <= cutoff:
                expired_records.append(record)

        deleted_count = 0

        for record in expired_records:
            if self.delete_document(
                record.document_id,
                session_id=record.session_id,
            ):
                deleted_count += 1

        if expired_records:
            logger.info(
                (
                    "document_retention_cleanup candidates=%d "
                    "deleted=%d retention_hours=%d"
                ),
                len(expired_records),
                deleted_count,
                retention_hours,
            )

        return deleted_count


    def _enforce_document_limits(
        self,
        path: str | Path,
        session_id: str | None,
    ) -> None:
        """Enforce logical-document quotas before expensive processing."""

        filename = Path(path).name

        session_records = self.registry.list_latest_records(
            session_id=session_id
        )

        # Uploading changed content under an existing filename creates
        # a new version of the same logical document and does not consume
        # another active-document quota slot.
        is_existing_logical_document = any(
            record.filename.casefold() == filename.casefold()
            for record in session_records
        )

        if is_existing_logical_document:
            return

        if (
            self.max_documents_per_session is not None
            and len(session_records)
            >= self.max_documents_per_session
        ):
            raise ResourceLimitError(
                "Anonymous session document limit reached "
                f"({self.max_documents_per_session}). "
                "Delete an existing document before uploading another."
            )

        if self.max_active_documents is not None:
            active_records = self.registry.list_latest_records()

            if len(active_records) >= self.max_active_documents:
                raise ResourceLimitError(
                    "Server active-document capacity has been reached. "
                    "Please try again later."
                )

    @staticmethod
    def _attach_document_identity(
        chunks: list[DocumentChunk],
        record: DocumentRecord,
    ) -> list[DocumentChunk]:
        """
        Attach document-level identity to every chunk.

        Managed chunk IDs include document/version identity
        so two versions cannot collide during Chroma upsert.
        """

        enriched_chunks: list[DocumentChunk] = []

        for chunk in chunks:
            managed_chunk_id = (
                f"{record.document_id}"
                f"_v{record.version}_"
                f"{chunk.chunk_id}"
            )

            metadata = {
                **chunk.metadata,
                "document_id": record.document_id,
                "session_id": record.session_id,
                "document_version": record.version,
                "content_hash": record.content_hash,
                "original_chunk_id": chunk.chunk_id,
            }

            enriched_chunks.append(
                chunk.model_copy(
                    update={
                        "chunk_id": managed_chunk_id,
                        "metadata": metadata,
                    }
                )
            )

        return enriched_chunks

    def index_document(
        self,
        path: str | Path,
        session_id: str | None = None,
    ) -> DocumentIndexingResult:
        """
        Process and index one document.

        Exact duplicates are rejected by DocumentRegistry
        before preprocessing or embedding begins.
        """

        total_started = perf_counter()
        filename = Path(path).name

        logger.info(
            "document_index_start filename=%s",
            filename,
        )

        self._enforce_document_limits(
            path=path,
            session_id=session_id,
        )

        record = self.registry.prepare(
            path,
            session_id=session_id,
        )

        processing_started = perf_counter()

        processing_result = (
            self.ingestion_service.process(path)
        )

        processing_ms = (
            perf_counter() - processing_started
        ) * 1000.0

        if (
            self.max_chunks_per_document is not None
            and len(processing_result.chunks)
            > self.max_chunks_per_document
        ):
            raise ResourceLimitError(
                "Document produces too many chunks "
                f"({len(processing_result.chunks)}). "
                "The public deployment limit is "
                f"{self.max_chunks_per_document} chunks per document."
            )

        chunks = self._attach_document_identity(
            chunks=processing_result.chunks,
            record=record,
        )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embedding_started = perf_counter()

        embeddings = (
            self.embedding_model.embed_documents(
                texts
            )
        )

        embedding_ms = (
            perf_counter() - embedding_started
        ) * 1000.0

        if len(embeddings) != len(chunks):
            raise IndexingError(
                "Embedding count does not match "
                "document chunk count."
            )

        indexing_started = perf_counter()

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        indexing_ms = (
            perf_counter() - indexing_started
        ) * 1000.0

        try:
            self.registry.commit(record)

        except Exception:
            self.vector_store.delete_document_chunks(
                document_id=record.document_id,
                version=record.version,
                session_id=record.session_id,
            )

            logger.exception(
                (
                    "document_registry_commit_failed "
                    "filename=%s document_id=%s version=%d"
                ),
                filename,
                record.document_id,
                record.version,
            )

            raise

        removed_old_chunks = 0

        # After the new version is safely registered,
        # remove older versions from the active index.
        if record.version > 1:
            for previous_version in range(
                1,
                record.version,
            ):
                removed_old_chunks += (
                    self.vector_store.delete_document_chunks(
                        document_id=record.document_id,
                        version=previous_version,
                        session_id=record.session_id,
                    )
                )

        total_ms = (
            perf_counter() - total_started
        ) * 1000.0

        logger.info(
            (
                "document_index_complete filename=%s "
                "document_id=%s version=%d chunks=%d "
                "removed_old_chunks=%d processing_ms=%.2f "
                "embedding_ms=%.2f indexing_ms=%.2f "
                "total_ms=%.2f"
            ),
            filename,
            record.document_id,
            record.version,
            len(chunks),
            removed_old_chunks,
            processing_ms,
            embedding_ms,
            indexing_ms,
            total_ms,
        )

        return DocumentIndexingResult(
            record=record,
            chunk_count=len(chunks),
        )


    def list_documents(
        self,
        session_id: str | None = None,
    ) -> list[DocumentRecord]:
        """
        Return the latest version of every registered
        logical document.
        """

        return self.registry.list_latest_records(
            session_id=session_id
        )

    def delete_document(
        self,
        document_id: str,
        session_id: str | None = None,
    ) -> bool:
        """
        Remove a logical document from the active vector
        index and document registry.

        All indexed versions are removed.
        """

        document_id = document_id.strip()

        if not document_id:
            raise ValueError(
                "document_id cannot be empty."
            )

        started = perf_counter()

        records = self.registry.list_records(
            session_id=session_id
        )

        exists = any(
            record.document_id == document_id
            for record in records
        )

        if not exists:
            logger.info(
                "document_delete_missing document_id=%s",
                document_id,
            )
            return False

        deleted_chunks = (
            self.vector_store.delete_document_chunks(
                document_id=document_id,
                session_id=session_id,
            )
        )

        removed_records = (
            self.registry.delete_document(
                document_id=document_id,
                session_id=session_id,
            )
        )

        deleted = removed_records > 0

        logger.info(
            (
                "document_delete_complete document_id=%s "
                "deleted=%s chunks=%d registry_records=%d "
                "total_ms=%.2f"
            ),
            document_id,
            deleted,
            deleted_chunks,
            removed_records,
            (
                perf_counter() - started
            ) * 1000.0,
        )

        return deleted
