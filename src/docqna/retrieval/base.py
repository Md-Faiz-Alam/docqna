from abc import ABC, abstractmethod

from docqna.filters import MetadataFilters
from docqna.retrieval.models import RetrievalResult


class BaseRetriever(ABC):
    """Base interface for retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query."""
        raise NotImplementedError
