from abc import ABC, abstractmethod

from docqna.ingestion.models import (
    DocumentChunk,
    LoadedDocument,
)


class BaseChunker(ABC):
    """Base interface for all chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        document: LoadedDocument,
    ) -> list[DocumentChunk]:
        """Split a document into retrieval chunks."""
        raise NotImplementedError