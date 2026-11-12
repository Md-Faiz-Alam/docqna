import json
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api.types import Metadata, OneOrMany, PyEmbedding, QueryResult
from chromadb.config import Settings

from docqna.exceptions import IndexingError
from docqna.filters import MetadataFilters, normalize_metadata_filters
from docqna.indexing.base import BaseVectorStore
from docqna.ingestion.models import DocumentChunk


class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB vector store."""

    def __init__(self, persist_directory: str | Path, collection_name: str) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory), settings=Settings(anonymized_telemetry=False)
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

        except Exception as exc:
            raise IndexingError(
                f"Could not initialize Chroma collection '{self.collection_name}'."
            ) from exc

    @staticmethod
    def _parse_required_int(value: Any, field_name: str, chunk_id: str) -> int:
        """Safely parse required integer metadata."""

        if value is None:
            raise IndexingError(f"Chunk '{chunk_id}' has no {field_name} metadata.")

        if isinstance(value, bool):
            raise IndexingError(f"Chunk '{chunk_id}' has invalid {field_name} metadata.")

        try:
            return int(value)

        except (TypeError, ValueError) as exc:
            raise IndexingError(
                f"Chunk '{chunk_id}' has invalid {field_name} metadata: {value!r}."
            ) from exc

    @staticmethod
    def _parse_optional_int(value: Any, field_name: str, chunk_id: str) -> int | None:
        """Safely parse optional integer metadata."""

        if value is None:
            return None

        return ChromaVectorStore._parse_required_int(
            value=value, field_name=field_name, chunk_id=chunk_id
        )

    def _build_metadata(self, chunk: DocumentChunk) -> Metadata:
        """
        Convert chunk metadata into Chroma-safe metadata.

        Complex values such as lists are serialized
        into JSON strings.
        """

        metadata: dict[str, Any] = {
            "chunk_index": chunk.chunk_index,
            "filename": chunk.filename,
            "file_type": chunk.file_type,
            "source_segment_indices": json.dumps(chunk.source_segment_indices),
        }

        if chunk.page_start is not None:
            metadata["page_start"] = chunk.page_start

        if chunk.page_end is not None:
            metadata["page_end"] = chunk.page_end

        if chunk.section is not None:
            metadata["section"] = chunk.section

        metadata["chunking_strategy"] = str(chunk.metadata.get("chunking_strategy", "unknown"))

        metadata["word_count"] = int(chunk.metadata.get("word_count", 0))

        metadata["sections"] = json.dumps(chunk.metadata.get("sections", []))

        document_id = chunk.metadata.get("document_id")

        if document_id is not None:
            metadata["document_id"] = str(document_id)

        session_id = chunk.metadata.get("session_id")

        if session_id is not None:
            metadata["session_id"] = str(session_id)

        document_version = chunk.metadata.get("document_version")

        if document_version is not None:
            metadata["document_version"] = int(document_version)

        content_hash = chunk.metadata.get("content_hash")

        if content_hash is not None:
            metadata["content_hash"] = str(content_hash)

        original_chunk_id = chunk.metadata.get("original_chunk_id")

        if original_chunk_id is not None:
            metadata["original_chunk_id"] = str(original_chunk_id)

        return metadata

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Store chunks and embeddings in ChromaDB."""

        if not chunks:
            raise ValueError("chunks cannot be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings.")

        if any(not embedding for embedding in embeddings):
            raise ValueError("Embeddings cannot be empty.")

        ids = [chunk.chunk_id for chunk in chunks]

        documents = [chunk.text for chunk in chunks]

        metadatas = [self._build_metadata(chunk) for chunk in chunks]

        chroma_embeddings = cast(OneOrMany[PyEmbedding], embeddings)

        try:
            self.collection.upsert(
                ids=ids, documents=documents, embeddings=chroma_embeddings, metadatas=metadatas
            )

        except Exception as exc:
            raise IndexingError("Failed to store chunks in ChromaDB.") from exc

    def query_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> QueryResult:
        """Query nearest chunks by vector similarity."""

        if not embedding:
            raise ValueError("Query embedding cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        metadata_filters = normalize_metadata_filters(metadata_filters)

        where_clause: dict[str, Any] | None = None

        if metadata_filters:
            clauses: list[dict[str, Any]] = [
                {field_name: field_value} for field_name, field_value in metadata_filters.items()
            ]

            where_clause = clauses[0] if len(clauses) == 1 else {"$and": clauses}

        query_embedding = cast(PyEmbedding, embedding)

        try:
            if where_clause is None:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )

            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=cast(Any, where_clause),
                    include=["documents", "metadatas", "distances"],
                )

        except Exception as exc:
            raise IndexingError("ChromaDB query failed.") from exc

        return results

    def get_all_chunks(self, session_id: str | None = None) -> list[DocumentChunk]:
        """
        Reconstruct stored DocumentChunk objects, optionally for one session.

        This allows sparse retrievers such as BM25
        to operate on exactly the same corpus that
        is already stored in ChromaDB.
        """

        try:
            if session_id is None:
                results = self.collection.get(include=["documents", "metadatas"])
            else:
                results = self.collection.get(
                    where={"session_id": session_id}, include=["documents", "metadatas"]
                )

        except Exception as exc:
            raise IndexingError("Could not load chunks from ChromaDB.") from exc

        ids = results.get("ids") or []

        documents = results.get("documents") or []

        metadatas = results.get("metadatas") or []

        if not (len(ids) == len(documents) == len(metadatas)):
            raise IndexingError("ChromaDB returned inconsistent chunk data.")

        chunks: list[DocumentChunk] = []

        for chunk_id, document, raw_metadata in zip(ids, documents, metadatas, strict=True):
            if document is None:
                raise IndexingError(f"Chunk '{chunk_id}' has no document text.")

            metadata = dict(raw_metadata or {})

            # -----------------------------
            # source_segment_indices
            # -----------------------------

            source_segment_indices: list[int] = []

            raw_segment_indices = metadata.get("source_segment_indices")

            if isinstance(raw_segment_indices, str):
                try:
                    parsed_indices = json.loads(raw_segment_indices)

                    if isinstance(parsed_indices, list):
                        source_segment_indices = [int(value) for value in parsed_indices]

                except (json.JSONDecodeError, TypeError, ValueError):
                    source_segment_indices = []

            # -----------------------------
            # sections
            # -----------------------------

            sections: list[str] = []

            raw_sections = metadata.get("sections")

            if isinstance(raw_sections, str):
                try:
                    parsed_sections = json.loads(raw_sections)

                    if isinstance(parsed_sections, list):
                        sections = [str(value) for value in parsed_sections]

                except (json.JSONDecodeError, TypeError, ValueError):
                    sections = []

            # -----------------------------
            # Required metadata
            # -----------------------------

            chunk_index_value = metadata.get("chunk_index")

            if chunk_index_value is None:
                raise IndexingError(f"Chunk '{chunk_id}' has no chunk_index metadata.")

            filename_value = metadata.get("filename")

            if filename_value is None:
                raise IndexingError(f"Chunk '{chunk_id}' has no filename metadata.")

            file_type_value = metadata.get("file_type")

            if file_type_value is None:
                raise IndexingError(f"Chunk '{chunk_id}' has no file_type metadata.")

            # -----------------------------
            # Optional metadata
            # -----------------------------

            page_start_value = metadata.get("page_start")

            page_end_value = metadata.get("page_end")

            section_value = metadata.get("section")

            word_count_value = metadata.get("word_count", 0)

            # -----------------------------
            # Reconstruct DocumentChunk
            # -----------------------------

            chunk = DocumentChunk(
                chunk_id=str(chunk_id),
                chunk_index=self._parse_required_int(
                    value=chunk_index_value, field_name="chunk_index", chunk_id=str(chunk_id)
                ),
                text=str(document),
                filename=str(filename_value),
                file_type=str(file_type_value),
                source_segment_indices=(source_segment_indices),
                page_start=self._parse_optional_int(
                    value=page_start_value, field_name="page_start", chunk_id=str(chunk_id)
                ),
                page_end=self._parse_optional_int(
                    value=page_end_value, field_name="page_end", chunk_id=str(chunk_id)
                ),
                section=(str(section_value) if section_value is not None else None),
                metadata={
                    "chunking_strategy": str(metadata.get("chunking_strategy", "unknown")),
                    "word_count": self._parse_required_int(
                        value=word_count_value, field_name="word_count", chunk_id=str(chunk_id)
                    ),
                    "sections": sections,
                    **(
                        {"document_id": str(metadata["document_id"])}
                        if metadata.get("document_id") is not None
                        else {}
                    ),
                    **(
                        {"session_id": str(metadata["session_id"])}
                        if metadata.get("session_id") is not None
                        else {}
                    ),
                    **(
                        {
                            "document_version": self._parse_required_int(
                                value=metadata["document_version"],
                                field_name=("document_version"),
                                chunk_id=str(chunk_id),
                            )
                        }
                        if metadata.get("document_version") is not None
                        else {}
                    ),
                    **(
                        {"content_hash": str(metadata["content_hash"])}
                        if metadata.get("content_hash") is not None
                        else {}
                    ),
                    **(
                        {"original_chunk_id": str(metadata["original_chunk_id"])}
                        if metadata.get("original_chunk_id") is not None
                        else {}
                    ),
                },
            )

            chunks.append(chunk)

        # Keep multi-document corpus ordering deterministic.
        chunks.sort(
            key=lambda chunk: (
                chunk.filename.casefold(),
                str(chunk.metadata.get("document_id", "")),
                int(chunk.metadata.get("document_version", 0)),
                chunk.chunk_index,
                chunk.chunk_id,
            )
        )

        return chunks

    def delete_document_chunks(
        self, document_id: str, version: int | None = None, session_id: str | None = None
    ) -> int:
        """
        Delete indexed chunks belonging to one logical
        document.

        When version is provided, only that document
        version is removed. When session_id is provided,
        only chunks owned by that session are eligible.

        Returns the number of deleted chunks.
        """

        document_id = document_id.strip()

        if not document_id:
            raise ValueError("document_id cannot be empty.")

        if version is not None and version <= 0:
            raise ValueError("version must be greater than 0.")

        if session_id is not None:
            session_id = session_id.strip()

            if not session_id:
                raise ValueError("session_id cannot be empty.")

        try:
            results = self.collection.get(include=["metadatas"])

        except Exception as exc:
            raise IndexingError("Could not inspect ChromaDB document metadata.") from exc

        ids = results.get("ids") or []

        metadatas = results.get("metadatas") or []

        if len(ids) != len(metadatas):
            raise IndexingError("ChromaDB returned inconsistent document metadata.")

        matching_ids: list[str] = []

        for chunk_id, raw_metadata in zip(ids, metadatas, strict=True):
            metadata = dict(raw_metadata or {})

            stored_document_id = metadata.get("document_id")

            if stored_document_id != document_id:
                continue

            if session_id is not None and metadata.get("session_id") != session_id:
                continue

            if version is not None:
                stored_version = metadata.get("document_version")

                if stored_version is None:
                    continue

                try:
                    parsed_version = self._parse_required_int(
                        value=stored_version,
                        field_name=("document_version"),
                        chunk_id=str(chunk_id),
                    )

                except IndexingError:
                    continue

                if parsed_version != version:
                    continue

            matching_ids.append(str(chunk_id))

        if not matching_ids:
            return 0

        try:
            self.collection.delete(ids=matching_ids)

        except Exception as exc:
            raise IndexingError("Failed to delete document chunks from ChromaDB.") from exc

        return len(matching_ids)

    def count(self) -> int:
        """Return number of stored chunks."""

        try:
            return self.collection.count()

        except Exception as exc:
            raise IndexingError("Could not count ChromaDB records.") from exc
