import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping

from docqna.exceptions import RetrievalError
from docqna.filters import MetadataFilters, normalize_metadata_filters
from docqna.ingestion.models import DocumentChunk
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult


class BM25Retriever(BaseRetriever):
    """
    BM25 sparse lexical retriever.

    Uses the standard BM25 ranking formula with configurable
    k1 and b parameters.

    No external BM25 dependency is required.
    """

    def __init__(self, chunks: list[DocumentChunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25 chunks cannot be empty.")

        if k1 <= 0:
            raise ValueError("k1 must be greater than 0.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.tokenized_documents = [self._tokenize(chunk.text) for chunk in chunks]

        self.document_lengths = [len(tokens) for tokens in self.tokenized_documents]

        self.num_documents = len(self.tokenized_documents)

        self.average_document_length = sum(self.document_lengths) / self.num_documents

        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized_documents]

        self.document_frequencies: dict[str, int] = defaultdict(int)

        for tokens in self.tokenized_documents:
            for token in set(tokens):
                self.document_frequencies[token] += 1

        self._session_statistics_cache: dict[str, tuple[int, float, dict[str, int]]] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Perform lightweight lexical tokenization.

        Lowercases text and extracts alphanumeric word tokens.
        """

        return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _build_statistics(
        *,
        document_indices: list[int],
        tokenized_documents: list[list[str]],
        document_lengths: list[int],
    ) -> tuple[int, float, dict[str, int]]:
        """Build BM25 corpus statistics for a document subset."""

        num_documents = len(document_indices)

        if num_documents == 0:
            return (0, 0.0, {})

        average_document_length = (
            sum(document_lengths[document_index] for document_index in document_indices)
            / num_documents
        )

        document_frequencies: dict[str, int] = defaultdict(int)

        for document_index in document_indices:
            for token in set(tokenized_documents[document_index]):
                document_frequencies[token] += 1

        return (num_documents, average_document_length, dict(document_frequencies))

    def _scoring_statistics(
        self, metadata_filters: MetadataFilters | None
    ) -> tuple[int, float, Mapping[str, int]]:
        """Return BM25 statistics for the active session scope."""

        session_id = metadata_filters.get("session_id") if metadata_filters else None

        if session_id is None:
            return (self.num_documents, self.average_document_length, self.document_frequencies)

        if not isinstance(session_id, str):
            raise ValueError("session_id filter must be a string.")

        cached_statistics = self._session_statistics_cache.get(session_id)

        if cached_statistics is not None:
            return cached_statistics

        session_document_indices = [
            document_index
            for document_index, chunk in enumerate(self.chunks)
            if chunk.metadata.get("session_id") == session_id
        ]

        statistics = self._build_statistics(
            document_indices=(session_document_indices),
            tokenized_documents=(self.tokenized_documents),
            document_lengths=(self.document_lengths),
        )

        self._session_statistics_cache[session_id] = statistics

        return statistics

    @staticmethod
    def _idf(term: str, *, num_documents: int, document_frequencies: Mapping[str, int]) -> float:
        """Calculate BM25 inverse document frequency."""

        document_frequency = document_frequencies.get(term, 0)

        return math.log(
            1.0 + (num_documents - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def _score_document(
        self,
        query_terms: list[str],
        document_index: int,
        *,
        num_documents: int,
        average_document_length: float,
        document_frequencies: Mapping[str, int],
    ) -> float:
        """Calculate BM25 score for one document."""

        frequencies = self.term_frequencies[document_index]

        document_length = self.document_lengths[document_index]

        score = 0.0

        for term in query_terms:
            term_frequency = frequencies.get(term, 0)

            if term_frequency == 0:
                continue

            idf = self._idf(
                term, num_documents=(num_documents), document_frequencies=(document_frequencies)
            )

            numerator = term_frequency * (self.k1 + 1.0)

            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * (document_length / average_document_length)
            )

            score += idf * numerator / denominator

        return score

    @staticmethod
    def _chunk_matches_filters(
        chunk: DocumentChunk, metadata_filters: MetadataFilters | None
    ) -> bool:
        """Return whether a chunk satisfies all filters."""

        if metadata_filters is None:
            return True

        for field_name, expected_value in metadata_filters.items():
            if field_name == "document_id":
                actual_value = chunk.metadata.get("document_id")

            elif field_name == "session_id":
                actual_value = chunk.metadata.get("session_id")

            elif field_name == "document_version":
                actual_value = chunk.metadata.get("document_version")

            elif field_name == "filename":
                actual_value = chunk.filename

            elif field_name == "file_type":
                actual_value = chunk.file_type

            else:
                raise ValueError(f"Unsupported metadata filter: {field_name!r}.")

            if actual_value != expected_value:
                return False

        return True

    def retrieve(
        self, query: str, top_k: int = 5, metadata_filters: MetadataFilters | None = None
    ) -> list[RetrievalResult]:
        """Retrieve chunks using BM25 lexical ranking."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        metadata_filters = normalize_metadata_filters(metadata_filters)

        try:
            query_terms = self._tokenize(query)

            if not query_terms:
                return []

            (
                scoring_num_documents,
                scoring_average_document_length,
                scoring_document_frequencies,
            ) = self._scoring_statistics(metadata_filters)

            if scoring_num_documents == 0:
                return []

            scored_documents: list[tuple[int, float]] = []

            for document_index in range(self.num_documents):
                chunk = self.chunks[document_index]

                if not self._chunk_matches_filters(chunk=chunk, metadata_filters=metadata_filters):
                    continue

                score = self._score_document(
                    query_terms=query_terms,
                    document_index=document_index,
                    num_documents=(scoring_num_documents),
                    average_document_length=(scoring_average_document_length),
                    document_frequencies=(scoring_document_frequencies),
                )

                if score > 0:
                    scored_documents.append((document_index, score))

            scored_documents.sort(
                key=lambda item: (
                    -item[1],
                    self.chunks[item[0]].filename.casefold(),
                    str(self.chunks[item[0]].metadata.get("document_id", "")),
                    self.chunks[item[0]].chunk_index,
                    self.chunks[item[0]].chunk_id,
                )
            )

            results: list[RetrievalResult] = []

            for rank, (document_index, score) in enumerate(scored_documents[:top_k], start=1):
                chunk = self.chunks[document_index]

                results.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        rank=rank,
                        # BaseRetriever currently expects the
                        # shared RetrievalResult schema where
                        # distance is required.
                        #
                        # Negative BM25 score preserves the
                        # convention that "smaller is better".
                        distance=-score,
                        score=score,
                        retrieval_method="bm25",
                        filename=chunk.filename,
                        file_type=chunk.file_type,
                        chunk_index=chunk.chunk_index,
                        document_id=(
                            str(chunk.metadata["document_id"])
                            if chunk.metadata.get("document_id") is not None
                            else None
                        ),
                        document_version=(
                            int(chunk.metadata["document_version"])
                            if chunk.metadata.get("document_version") is not None
                            else None
                        ),
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section=chunk.section,
                        source_segment_indices=(chunk.source_segment_indices),
                        sections=(
                            chunk.metadata.get("sections", [])
                            if isinstance(chunk.metadata.get("sections", []), list)
                            else []
                        ),
                    )
                )

            return results

        except ValueError:
            raise

        except Exception as exc:
            raise RetrievalError("BM25 retrieval failed.") from exc
