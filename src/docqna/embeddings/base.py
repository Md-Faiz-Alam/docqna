from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Base interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for document texts."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a search query."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimensionality."""
        raise NotImplementedError
