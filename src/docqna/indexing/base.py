from abc import ABC, abstractmethod

from chromadb.api.types import QueryResult

from docqna.filters import MetadataFilters
from docqna.ingestion.models import DocumentChunk


class BaseVectorStore(ABC):
    """Base interface for persistent vector stores."""

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Store chunks and their embeddings."""
        raise NotImplementedError

    @abstractmethod
    def query_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> QueryResult:
        """Return nearest stored chunks."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""
        raise NotImplementedError
