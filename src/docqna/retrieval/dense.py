import json
from typing import Any

from docqna.embeddings.base import BaseEmbeddingModel
from docqna.exceptions import RetrievalError
from docqna.filters import (
    MetadataFilters,
    normalize_metadata_filters,
)
from docqna.indexing.base import BaseVectorStore
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult


class DenseRetriever(BaseRetriever):
    """
    Dense semantic retriever backed by an embedding model
    and vector store.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    @staticmethod
    def _parse_json_list(
        value: Any,
    ) -> list[Any]:
        """Convert serialized JSON metadata back into a list."""

        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []

            if isinstance(parsed, list):
                return parsed

        return []

    @staticmethod
    def _parse_optional_int(
        value: Any,
    ) -> int | None:
        """Convert a metadata value to an integer when possible."""

        if value is None:
            return None

        # bool is a subclass of int, so handle it first.
        if isinstance(value, bool):
            return None

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None

        return None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve chunks using dense vector similarity."""

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        metadata_filters = normalize_metadata_filters(
            metadata_filters
        )

        try:
            query_embedding = (
                self.embedding_model.embed_query(query)
            )

            if metadata_filters:
                raw_results = (
                    self.vector_store.query_by_embedding(
                        embedding=query_embedding,
                        top_k=top_k,
                        metadata_filters=metadata_filters,
                    )
                )

            else:
                raw_results = (
                    self.vector_store.query_by_embedding(
                        embedding=query_embedding,
                        top_k=top_k,
                    )
                )

        except ValueError:
            raise

        except Exception as exc:
            raise RetrievalError(
                "Dense retrieval failed."
            ) from exc

        ids = raw_results["ids"]

        documents = raw_results.get("documents")
        metadatas = raw_results.get("metadatas")
        distances = raw_results.get("distances")

        if (
            documents is None
            or metadatas is None
            or distances is None
        ):
            raise RetrievalError(
                "Vector store did not return all "
                "required retrieval fields."
            )

        if not ids:
            return []

        result_ids = ids[0]
        result_documents = documents[0]
        result_metadatas = metadatas[0]
        result_distances = distances[0]

        results: list[RetrievalResult] = []

        for index, chunk_id in enumerate(result_ids):
            document_text = result_documents[index]
            metadata = result_metadatas[index]
            distance = result_distances[index]

            if document_text is None:
                continue

            if metadata is None:
                metadata = {}

            source_segment_indices = (
                self._parse_json_list(
                    metadata.get(
                        "source_segment_indices"
                    )
                )
            )

            sections = self._parse_json_list(
                metadata.get("sections")
            )

            chunk_index = self._parse_optional_int(
                metadata.get("chunk_index")
            )

            page_start = self._parse_optional_int(
                metadata.get("page_start")
            )

            page_end = self._parse_optional_int(
                metadata.get("page_end")
            )

            document_version = (
                self._parse_optional_int(
                    metadata.get(
                        "document_version"
                    )
                )
            )

            document_id_value = metadata.get(
                "document_id"
            )

            document_id = (
                str(document_id_value)
                if document_id_value is not None
                else None
            )

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=document_text,
                    rank=index + 1,
                    distance=float(distance),

                    filename=str(
                        metadata.get(
                            "filename",
                            "",
                        )
                    ),

                    file_type=str(
                        metadata.get(
                            "file_type",
                            "",
                        )
                    ),

                    chunk_index=(
                        chunk_index
                        if chunk_index is not None
                        else 0
                    ),

                    document_id=document_id,
                    document_version=document_version,

                    page_start=page_start,
                    page_end=page_end,

                    section=(
                        str(metadata["section"])
                        if "section" in metadata
                        else None
                    ),

                    source_segment_indices=[
                        int(value)
                        for value in source_segment_indices
                    ],

                    sections=[
                        str(value)
                        for value in sections
                    ],
                )
            )

        return results