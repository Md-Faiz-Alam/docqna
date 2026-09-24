import logging
import re
from time import perf_counter

from docqna.filters import (
    MetadataFilters,
    normalize_metadata_filters,
)
from docqna.generation.providers.base import BaseLLMProvider
from docqna.rag.models import (
    CitationSource,
    RAGResponse,
)
from docqna.rag.prompt import (
    build_citation_repair_prompt,
    build_cited_rag_prompt,
    build_rag_prompt,
)
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.confidence import RetrievalConfidenceGate
from docqna.retrieval.models import RetrievalResult


logger = logging.getLogger(__name__)


ABSTENTION_ANSWER = (
    "The provided documents do not contain enough "
    "information to answer the question."
)


class RAGPipeline:
    """Retrieval-augmented generation pipeline."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLMProvider,
        confidence_gate: RetrievalConfidenceGate | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.confidence_gate = confidence_gate

    @staticmethod
    def _validate_request(
        query: str,
        top_k: int,
    ) -> str:
        """Validate and normalize a RAG request."""

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        return query

    @staticmethod
    def _normalize_citations(
        answer: str,
    ) -> str:
        """
        Normalize common model-generated citation formats
        into canonical forms such as [1], [2], and [3].
        """

        # Example:
        # [Source [3], Pages 23-24] -> [3]
        answer = re.sub(
            r"\[\s*Source\s*\[(\d+)\]"
            r"\s*(?:,\s*Pages?\s*[^\]]+)?\]",
            r"[\1]",
            answer,
            flags=re.IGNORECASE,
        )

        # Example:
        # [Source 3, Pages 23-24] -> [3]
        answer = re.sub(
            r"\[\s*Source\s+(\d+)"
            r"\s*(?:,\s*Pages?\s*[^\]]+)?\]",
            r"[\1]",
            answer,
            flags=re.IGNORECASE,
        )

        # Example:
        # Source [3] -> [3]
        answer = re.sub(
            r"\bSource\s*\[(\d+)\]",
            r"[\1]",
            answer,
            flags=re.IGNORECASE,
        )

        # Example:
        # (Source 3) or (Source: 3) -> [3]
        answer = re.sub(
            r"\(\s*Source\s*:?\s*(\d+)\s*\)",
            r"[\1]",
            answer,
            flags=re.IGNORECASE,
        )

        # Example:
        # [Citation 3] or [Citation: 3] -> [3]
        answer = re.sub(
            r"\[\s*Citation\s*:?\s*(\d+)\s*\]",
            r"[\1]",
            answer,
            flags=re.IGNORECASE,
        )

        # Example:
        # Source 3 -> [3]
        answer = re.sub(
            r"\bSource\s*:?\s*(\d+)\b",
            r"[\1]",
            answer,
            flags=re.IGNORECASE,
        )

        return answer

    @staticmethod
    def _sanitize_generated_answer(
        answer: str,
    ) -> str:
        """
        Remove accidental prompt/instruction leakage from an LLM answer.

        The function does not alter normal answer content. It strips a
        leading answer label and truncates output if an internal prompt
        section is echoed after the real answer.
        """

        answer = answer.strip()

        leading_labels = (
            "FINAL ANSWER:",
            "CORRECTED FINAL ANSWER:",
            "ANSWER:",
        )

        for label in leading_labels:
            if answer.upper().startswith(
                label
            ):
                answer = answer[
                    len(label):
                ].lstrip()
                break

        internal_markers = (
            "\nINSTRUCTIONS:",
            "\nGROUNDING AND CITATION RULES:",
            "\nSTRICT GROUNDING RULES:",
            "\nDOCUMENT SOURCES:",
            "\nUSER QUESTION:",
            "\nCITATION REPAIR TASK:",
            "\nPREVIOUS CANDIDATE ANSWER:",
            "\nFINAL ANSWER:",
            "\nCORRECTED FINAL ANSWER:",
            "\nANSWER:",
        )

        upper_answer = answer.upper()

        positions = [
            upper_answer.find(
                marker
            )
            for marker in internal_markers
            if upper_answer.find(
                marker
            ) > 0
        ]

        if positions:
            answer = answer[
                :min(positions)
            ].rstrip()

        return answer

    @staticmethod
    def _extract_citation_ids(
        answer: str,
    ) -> list[int]:
        """Extract citation numbers such as [1] and [2]."""

        matches = re.findall(
            r"\[(\d+)\]",
            answer,
        )

        citation_ids: list[int] = []
        seen: set[int] = set()

        for match in matches:
            citation_id = int(match)

            if citation_id not in seen:
                citation_ids.append(
                    citation_id
                )
                seen.add(
                    citation_id
                )

        return citation_ids

    @staticmethod
    def _is_abstention_answer(
        answer: str,
    ) -> bool:
        """Return True for the canonical abstention response."""

        return (
            answer.strip()
            == ABSTENTION_ANSWER
        )

    @classmethod
    def _validate_citations(
        cls,
        *,
        answer: str,
        result_count: int,
    ) -> tuple[
        list[int],
        str | None,
    ]:
        """
        Validate citation presence and citation-ID range.

        Returns the unique citation IDs plus an error code.
        The method does not raise because citation-format
        failures are recoverable at runtime.
        """

        citation_ids = (
            cls._extract_citation_ids(
                answer
            )
        )

        if not citation_ids:
            return (
                [],
                "missing_citations",
            )

        valid_ids = set(
            range(
                1,
                result_count + 1,
            )
        )

        invalid_ids = [
            citation_id
            for citation_id in citation_ids
            if citation_id not in valid_ids
        ]

        if invalid_ids:
            return (
                citation_ids,
                (
                    "invalid_citation_ids:"
                    + ",".join(
                        str(
                            citation_id
                        )
                        for citation_id
                        in invalid_ids
                    )
                ),
            )

        return (
            citation_ids,
            None,
        )

    @staticmethod
    def _build_strict_cited_prompt(
        *,
        query: str,
        results: list[RetrievalResult],
    ) -> str:
        """
        Build the cited prompt with grounding rules placed before
        the final answer marker.
        """

        return build_cited_rag_prompt(
            query=query,
            results=results,
        )

    @classmethod
    def _build_citation_repair_prompt(
        cls,
        *,
        query: str,
        results: list[RetrievalResult],
        candidate_answer: str,
    ) -> str:
        """Build one constrained citation-repair prompt."""

        return build_citation_repair_prompt(
            query=query,
            results=results,
            candidate_answer=candidate_answer,
        )

    @staticmethod
    def _abstention_response(
    ) -> RAGResponse:
        """Build the canonical cited-mode abstention response."""

        return RAGResponse(
            answer=ABSTENTION_ANSWER,
            sources=[],
            cited_source_ids=[],
        )

    @staticmethod
    def _build_sources(
        results: list[RetrievalResult],
    ) -> list[CitationSource]:
        """Convert retrieval results into citation sources."""

        sources: list[CitationSource] = []

        for citation_id, result in enumerate(
            results,
            start=1,
        ):
            sources.append(
                CitationSource(
                    citation_id=citation_id,
                    chunk_id=result.chunk_id,
                    rank=result.rank,
                    filename=result.filename,
                    file_type=result.file_type,
                    page_start=result.page_start,
                    page_end=result.page_end,
                    section=result.section,
                    distance=result.distance,
                    text=result.text,
                )
            )

        return sources

    def _should_abstain(
        self,
        results: list[RetrievalResult],
    ) -> bool:
        """
        Return True when retrieval evidence is not strong
        enough to support answer generation.

        If no confidence gate is configured, existing
        Phase 9/10 behavior is preserved.
        """

        if not results:
            return True

        if self.confidence_gate is None:
            return False

        decision = self.confidence_gate.evaluate(
            results
        )

        return not decision.is_confident

    def _retrieve(
        self,
        *,
        query: str,
        top_k: int,
        metadata_filters: MetadataFilters | None,
    ) -> tuple[list[RetrievalResult], float]:
        """
        Retrieve evidence and return results plus elapsed
        retrieval time in milliseconds.
        """

        started = perf_counter()

        if metadata_filters:
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                metadata_filters=metadata_filters,
            )

        else:
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
            )

        elapsed_ms = (
            perf_counter() - started
        ) * 1000.0

        return results, elapsed_ms

    @staticmethod
    def _log_completion(
        *,
        mode: str,
        query: str,
        top_k: int,
        metadata_filters: MetadataFilters | None,
        result_count: int,
        abstained: bool,
        retrieval_ms: float,
        generation_ms: float,
        total_ms: float,
    ) -> None:
        """
        Log RAG timing without logging the query text or
        retrieved document contents.
        """

        filter_keys = (
            ",".join(
                sorted(metadata_filters)
            )
            if metadata_filters
            else "-"
        )

        logger.info(
            (
                "rag_complete mode=%s abstained=%s "
                "query_chars=%d top_k=%d results=%d "
                "filter_keys=%s retrieval_ms=%.2f "
                "generation_ms=%.2f total_ms=%.2f"
            ),
            mode,
            abstained,
            len(query),
            top_k,
            result_count,
            filter_keys,
            retrieval_ms,
            generation_ms,
            total_ms,
        )

    def ask(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> str:
        """
        Retrieve relevant chunks and generate a basic answer.

        This preserves the Phase 9 interface.
        """

        total_started = perf_counter()

        query = self._validate_request(
            query=query,
            top_k=top_k,
        )

        metadata_filters = normalize_metadata_filters(
            metadata_filters
        )

        results, retrieval_ms = self._retrieve(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )

        if self._should_abstain(
            results
        ):
            total_ms = (
                perf_counter() - total_started
            ) * 1000.0

            self._log_completion(
                mode="basic",
                query=query,
                top_k=top_k,
                metadata_filters=metadata_filters,
                result_count=len(results),
                abstained=True,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                total_ms=total_ms,
            )

            return ABSTENTION_ANSWER

        prompt = build_rag_prompt(
            query=query,
            results=results,
        )

        generation_started = perf_counter()

        answer = self.llm.generate(
            prompt
        )

        generation_ms = (
            perf_counter() - generation_started
        ) * 1000.0

        total_ms = (
            perf_counter() - total_started
        ) * 1000.0

        self._log_completion(
            mode="basic",
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            result_count=len(results),
            abstained=False,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

        return answer

    def ask_with_citations(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> RAGResponse:
        """
        Retrieve relevant chunks and generate a cited answer.
        """

        total_started = perf_counter()

        query = self._validate_request(
            query=query,
            top_k=top_k,
        )

        metadata_filters = normalize_metadata_filters(
            metadata_filters
        )

        results, retrieval_ms = self._retrieve(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )

        if self._should_abstain(
            results
        ):
            total_ms = (
                perf_counter() - total_started
            ) * 1000.0

            self._log_completion(
                mode="cited",
                query=query,
                top_k=top_k,
                metadata_filters=metadata_filters,
                result_count=len(results),
                abstained=True,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                total_ms=total_ms,
            )

            return self._abstention_response()

        prompt = self._build_strict_cited_prompt(
            query=query,
            results=results,
        )

        generation_started = perf_counter()

        answer = self.llm.generate(
            prompt
        ).strip()

        answer = self._sanitize_generated_answer(
            answer
        )

        answer = self._normalize_citations(
            answer
        )

        if self._is_abstention_answer(
            answer
        ):
            generation_ms = (
                perf_counter()
                - generation_started
            ) * 1000.0

            total_ms = (
                perf_counter()
                - total_started
            ) * 1000.0

            self._log_completion(
                mode="cited",
                query=query,
                top_k=top_k,
                metadata_filters=metadata_filters,
                result_count=len(results),
                abstained=True,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            )

            return self._abstention_response()

        (
            citation_ids,
            citation_error,
        ) = self._validate_citations(
            answer=answer,
            result_count=len(results),
        )

        if citation_error is not None:
            logger.warning(
                (
                    "citation_validation_failed "
                    "stage=initial reason=%s "
                    "query_chars=%d results=%d"
                ),
                citation_error,
                len(query),
                len(results),
            )

            repair_prompt = (
                self._build_citation_repair_prompt(
                    query=query,
                    results=results,
                    candidate_answer=answer,
                )
            )

            repaired_answer = (
                self.llm.generate(
                    repair_prompt
                ).strip()
            )

            repaired_answer = (
                self._sanitize_generated_answer(
                    repaired_answer
                )
            )

            answer = (
                self._normalize_citations(
                    repaired_answer
                )
            )

            if self._is_abstention_answer(
                answer
            ):
                generation_ms = (
                    perf_counter()
                    - generation_started
                ) * 1000.0

                total_ms = (
                    perf_counter()
                    - total_started
                ) * 1000.0

                self._log_completion(
                    mode="cited",
                    query=query,
                    top_k=top_k,
                    metadata_filters=metadata_filters,
                    result_count=len(results),
                    abstained=True,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    total_ms=total_ms,
                )

                return (
                    self._abstention_response()
                )

            (
                citation_ids,
                citation_error,
            ) = self._validate_citations(
                answer=answer,
                result_count=len(results),
            )

            if citation_error is not None:
                logger.warning(
                    (
                        "citation_validation_failed "
                        "stage=repair reason=%s "
                        "query_chars=%d results=%d "
                        "action=abstain"
                    ),
                    citation_error,
                    len(query),
                    len(results),
                )

                generation_ms = (
                    perf_counter()
                    - generation_started
                ) * 1000.0

                total_ms = (
                    perf_counter()
                    - total_started
                ) * 1000.0

                self._log_completion(
                    mode="cited",
                    query=query,
                    top_k=top_k,
                    metadata_filters=metadata_filters,
                    result_count=len(results),
                    abstained=True,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    total_ms=total_ms,
                )

                return (
                    self._abstention_response()
                )

        generation_ms = (
            perf_counter()
            - generation_started
        ) * 1000.0

        sources = self._build_sources(
            results
        )

        total_ms = (
            perf_counter() - total_started
        ) * 1000.0

        self._log_completion(
            mode="cited",
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            result_count=len(results),
            abstained=False,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            cited_source_ids=citation_ids,
        )
