"""Cited RAG pipeline: retrieve -> gate -> generate -> validate/repair citations.

Any answer whose citations cannot be validated (after one repair attempt)
is replaced by the canonical abstention, so the API never returns an
answer that points at sources it did not receive.
"""

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from time import perf_counter

from docqna.exceptions import GenerationError
from docqna.filters import MetadataFilters, normalize_metadata_filters
from docqna.generation.providers.base import BaseLLMProvider
from docqna.observability import GROUNDING_REJECTIONS, record_rag_timings
from docqna.rag.grounding import GroundingVerifier
from docqna.rag.models import CitationSource, RAGResponse
from docqna.rag.prompt import (
    ABSTENTION_ANSWER,
    build_citation_repair_prompt,
    build_cited_rag_prompt,
    build_condense_question_prompt,
)
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.confidence import RetrievalConfidenceGate
from docqna.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

__all__ = ["ABSTENTION_ANSWER", "ChatHistory", "RAGPipeline"]

# Earlier conversation turns as (role, text), oldest first; role is "user" or "assistant".
ChatHistory = Sequence[tuple[str, str]]

# Quotes and backticks small models sometimes wrap a rewritten question in.
_REWRITE_STRIP_CHARS = "\"'` "

# (pattern, replacement) pairs that rewrite model citation variants to "[n]".
_CITATION_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), r"[\1]")
    for pattern in (
        # [Source [3], Pages 23-24] -> [3]
        r"\[\s*Source\s*\[(\d+)\]\s*(?:,\s*Pages?\s*[^\]]+)?\]",
        # [Source 3, Pages 23-24] -> [3]
        r"\[\s*Source\s+(\d+)\s*(?:,\s*Pages?\s*[^\]]+)?\]",
        # Source [3] -> [3]
        r"\bSource\s*\[(\d+)\]",
        # (Source 3) / (Source: 3) -> [3]
        r"\(\s*Source\s*:?\s*(\d+)\s*\)",
        # [Citation 3] / [Citation: 3] -> [3]
        r"\[\s*Citation\s*:?\s*(\d+)\s*\]",
        # Source 3 -> [3]
        r"\bSource\s*:?\s*(\d+)\b",
    )
)

_LEADING_LABELS = ("FINAL ANSWER:", "CORRECTED FINAL ANSWER:", "ANSWER:")

# Prompt sections that must never appear in a user-facing answer.
_INTERNAL_MARKERS = (
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

_CITATION_ID = re.compile(r"\[(\d+)\]")


@dataclass
class _Timer:
    started: float = field(default_factory=perf_counter)
    rewrite_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_started: float | None = None

    def mark_rewrite_done(self) -> None:
        self.rewrite_ms = (perf_counter() - self.started) * 1000.0

    def mark_retrieval_done(self) -> None:
        self.retrieval_ms = (perf_counter() - self.started) * 1000.0 - self.rewrite_ms

    def start_generation(self) -> None:
        self.generation_started = perf_counter()

    @property
    def generation_ms(self) -> float:
        if self.generation_started is None:
            return 0.0
        return (perf_counter() - self.generation_started) * 1000.0

    @property
    def total_ms(self) -> float:
        return (perf_counter() - self.started) * 1000.0


class RAGPipeline:
    """Retrieval-augmented generation with verified citations."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLMProvider,
        confidence_gate: RetrievalConfidenceGate | None = None,
        grounding_verifier: GroundingVerifier | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.confidence_gate = confidence_gate
        self.grounding_verifier = grounding_verifier

    # ------------------------------------------------------------------
    # Answer post-processing (pure functions, unit-tested directly)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_citations(answer: str) -> str:
        """Rewrite common model citation formats into canonical `[n]`."""

        for pattern, replacement in _CITATION_REWRITES:
            answer = pattern.sub(replacement, answer)
        return answer

    @staticmethod
    def _sanitize_generated_answer(answer: str) -> str:
        """Strip a leading answer label and truncate echoed prompt sections."""

        answer = answer.strip()

        for label in _LEADING_LABELS:
            if answer.upper().startswith(label):
                answer = answer[len(label) :].lstrip()
                break

        upper_answer = answer.upper()
        positions = [
            position for marker in _INTERNAL_MARKERS if (position := upper_answer.find(marker)) > 0
        ]
        if positions:
            answer = answer[: min(positions)].rstrip()

        return answer

    @staticmethod
    def _extract_citation_ids(answer: str) -> list[int]:
        """Return unique citation numbers in order of first appearance."""

        return list(dict.fromkeys(int(match) for match in _CITATION_ID.findall(answer)))

    @staticmethod
    def _is_abstention_answer(answer: str) -> bool:
        return answer.strip() == ABSTENTION_ANSWER

    @classmethod
    def _validate_citations(cls, *, answer: str, result_count: int) -> tuple[list[int], str | None]:
        """Return (citation_ids, error_code); error_code is None when valid."""

        citation_ids = cls._extract_citation_ids(answer)
        if not citation_ids:
            return [], "missing_citations"

        invalid_ids = [cid for cid in citation_ids if not 1 <= cid <= result_count]
        if invalid_ids:
            return citation_ids, "invalid_citation_ids:" + ",".join(map(str, invalid_ids))

        return citation_ids, None

    @staticmethod
    def _build_sources(results: list[RetrievalResult]) -> list[CitationSource]:
        return [
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
            for citation_id, result in enumerate(results, start=1)
        ]

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(query: str, top_k: int) -> str:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        return query

    def _should_abstain(self, results: list[RetrievalResult]) -> bool:
        """True when retrieval evidence is too weak to support generation."""

        if not results:
            return True
        if self.confidence_gate is None:
            return False
        return not self.confidence_gate.evaluate(results).is_confident

    def _retrieve(
        self, *, query: str, top_k: int, metadata_filters: MetadataFilters | None
    ) -> list[RetrievalResult]:
        if metadata_filters:
            return self.retriever.retrieve(
                query=query, top_k=top_k, metadata_filters=metadata_filters
            )
        return self.retriever.retrieve(query=query, top_k=top_k)

    def _clean(self, raw_answer: str) -> str:
        return self._normalize_citations(self._sanitize_generated_answer(raw_answer.strip()))

    def _generate_clean(self, prompt: str) -> str:
        return self._clean(self.llm.generate(prompt))

    def _check_grounding(
        self, results: list[RetrievalResult], generated: tuple[str, list[int]] | None
    ) -> tuple[str, list[int]] | None:
        """Abstain when no cited source actually supports the answer text."""

        if generated is None or self.grounding_verifier is None:
            return generated

        answer, citation_ids = generated
        cited_texts = [results[cid - 1].text for cid in citation_ids if 1 <= cid <= len(results)]
        decision = self.grounding_verifier.evaluate(answer, cited_texts)
        if decision.is_grounded:
            return generated

        GROUNDING_REJECTIONS.inc()
        logger.warning(
            "grounding_check_failed score=%s min_score=%.2f action=abstain",
            "none" if decision.score is None else f"{decision.score:.2f}",
            self.grounding_verifier.min_score,
        )
        return None

    def _finalize_answer(
        self, query: str, results: list[RetrievalResult], answer: str
    ) -> tuple[str, list[int]] | None:
        """Validate citations (repairing once), or return None to abstain."""

        if self._is_abstention_answer(answer):
            return None

        citation_ids, error = self._validate_citations(answer=answer, result_count=len(results))
        if error is None:
            return answer, citation_ids

        logger.warning(
            "citation_validation_failed stage=initial reason=%s query_chars=%d results=%d",
            error,
            len(query),
            len(results),
        )

        answer = self._generate_clean(
            build_citation_repair_prompt(query=query, results=results, candidate_answer=answer)
        )
        if self._is_abstention_answer(answer):
            return None

        citation_ids, error = self._validate_citations(answer=answer, result_count=len(results))
        if error is None:
            return answer, citation_ids

        logger.warning(
            "citation_validation_failed stage=repair reason=%s query_chars=%d results=%d "
            "action=abstain",
            error,
            len(query),
            len(results),
        )
        return None

    @staticmethod
    def _log_completion(
        *,
        query: str,
        top_k: int,
        metadata_filters: MetadataFilters | None,
        result_count: int,
        abstained: bool,
        timer: _Timer,
    ) -> None:
        """Log timings without the query text or retrieved content."""

        record_rag_timings(retrieval_ms=timer.retrieval_ms, generation_ms=timer.generation_ms)
        logger.info(
            "rag_complete mode=cited abstained=%s query_chars=%d top_k=%d results=%d "
            "filter_keys=%s rewrite_ms=%.2f retrieval_ms=%.2f generation_ms=%.2f total_ms=%.2f",
            abstained,
            len(query),
            top_k,
            result_count,
            ",".join(sorted(metadata_filters)) if metadata_filters else "-",
            timer.rewrite_ms,
            timer.retrieval_ms,
            timer.generation_ms,
            timer.total_ms,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _condense_question(self, query: str, history: ChatHistory | None) -> str:
        """Rewrite a follow-up into a standalone question using recent turns.

        Falls back to the original question if rewriting fails or produces
        something implausible, so memory can only help retrieval, never block it.
        """

        if not history:
            return query

        try:
            raw = self.llm.generate(build_condense_question_prompt(query, list(history)))
        except GenerationError:
            logger.warning("question_rewrite_failed action=use_original")
            return query

        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        candidate = lines[0] if lines else ""
        for label in ("STANDALONE QUESTION:", "QUESTION:"):
            if candidate.upper().startswith(label):
                candidate = candidate[len(label) :].strip()
        candidate = candidate.strip(_REWRITE_STRIP_CHARS)

        if not candidate or len(candidate) > max(3 * len(query), len(query) + 300):
            logger.warning("question_rewrite_rejected action=use_original")
            return query
        return candidate

    def _prepare(
        self,
        query: str,
        top_k: int,
        metadata_filters: MetadataFilters | None,
        history: ChatHistory | None,
    ) -> tuple[str, MetadataFilters | None, list[RetrievalResult], _Timer]:
        timer = _Timer()
        query = self._validate_request(query, top_k)
        query = self._condense_question(query, history)
        timer.mark_rewrite_done()
        metadata_filters = normalize_metadata_filters(metadata_filters)
        results = self._retrieve(query=query, top_k=top_k, metadata_filters=metadata_filters)
        timer.mark_retrieval_done()
        return query, metadata_filters, results, timer

    def _respond(
        self,
        *,
        query: str,
        top_k: int,
        metadata_filters: MetadataFilters | None,
        results: list[RetrievalResult],
        timer: _Timer,
        generated: tuple[str, list[int]] | None,
        rewritten: bool,
    ) -> RAGResponse:
        self._log_completion(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            result_count=len(results),
            abstained=generated is None,
            timer=timer,
        )

        standalone_question = query if rewritten else None
        if generated is None:
            return RAGResponse(
                answer=ABSTENTION_ANSWER,
                sources=[],
                cited_source_ids=[],
                standalone_question=standalone_question,
            )

        answer, citation_ids = generated
        return RAGResponse(
            answer=answer,
            sources=self._build_sources(results),
            cited_source_ids=citation_ids,
            standalone_question=standalone_question,
        )

    def ask_with_citations(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
        history: ChatHistory | None = None,
    ) -> RAGResponse:
        """Answer `query` from retrieved evidence, or abstain.

        `history` holds earlier (role, text) turns, oldest first; when given,
        a follow-up question is rewritten into a standalone one first.
        """

        query, metadata_filters, results, timer = self._prepare(
            query, top_k, metadata_filters, history
        )

        generated: tuple[str, list[int]] | None = None
        if not self._should_abstain(results):
            timer.start_generation()
            answer = self._generate_clean(build_cited_rag_prompt(query=query, results=results))
            generated = self._check_grounding(
                results, self._finalize_answer(query, results, answer)
            )

        return self._respond(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            results=results,
            timer=timer,
            generated=generated,
            rewritten=bool(history),
        )

    def stream_with_citations(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
        history: ChatHistory | None = None,
    ) -> Iterator[str | RAGResponse]:
        """Yield draft answer text as it is generated, then the final RAGResponse.

        The final response is authoritative: citation validation/repair run
        after generation, so clients must replace the draft with it (it may
        be a corrected answer or the abstention).
        """

        query, metadata_filters, results, timer = self._prepare(
            query, top_k, metadata_filters, history
        )

        generated: tuple[str, list[int]] | None = None
        if not self._should_abstain(results):
            timer.start_generation()
            parts: list[str] = []
            for delta in self.llm.generate_stream(
                build_cited_rag_prompt(query=query, results=results)
            ):
                parts.append(delta)
                yield delta
            generated = self._check_grounding(
                results, self._finalize_answer(query, results, self._clean("".join(parts)))
            )

        yield self._respond(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            results=results,
            timer=timer,
            generated=generated,
            rewritten=bool(history),
        )
