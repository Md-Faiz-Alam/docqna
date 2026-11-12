import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

from docqna.exceptions import ChunkingError
from docqna.ingestion.chunking.base import BaseChunker
from docqna.ingestion.models import DocumentChunk, DocumentSegment, LoadedDocument


class _EmbeddingModelProtocol(Protocol):
    """
    Structural interface required by SemanticChunker.

    This intentionally matches the project's existing embedding
    model interface without coupling the chunker to a concrete
    embedding implementation.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        ...

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...


@dataclass(frozen=True, slots=True)
class _SemanticUnit:
    """Sentence/line-aware unit with its embedding and metadata."""

    text: str
    segment_index: int
    page_number: int | None
    section: str | None
    embedding: list[float]

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class SemanticChunker(BaseChunker):
    """
    Split documents using adjacent semantic similarity.

    The algorithm:
    - splits source segments on sentence/line boundaries
    - embeds all units in one batch
    - starts a semantic boundary when adjacent-unit cosine
      similarity falls below ``similarity_threshold``
    - avoids tiny chunks with ``min_chunk_size``
    - always enforces ``chunk_size`` as a hard maximum
    - supports configurable word overlap
    - preserves page, section, and source-segment metadata
    - never carries overlap across explicit section boundaries

    Production defaults are intentionally aligned with the existing
    200-word / 40-word evaluation configuration.
    """

    def __init__(
        self,
        embedding_model: _EmbeddingModelProtocol,
        chunk_size: int = 200,
        overlap: int = 40,
        min_chunk_size: int = 80,
        similarity_threshold: float = 0.45,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if overlap < 0:
            raise ValueError("overlap cannot be negative.")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be greater than 0.")

        if min_chunk_size > chunk_size:
            raise ValueError("min_chunk_size cannot exceed chunk_size.")

        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1.")

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _unique_in_order(values: list[str]) -> list[str]:
        """Remove duplicates while preserving original order."""

        return list(dict.fromkeys(values))

    @staticmethod
    def _build_chunk_id(filename: str, chunk_index: int, text: str) -> str:
        """
        Create a deterministic semantic-chunk identifier.

        The strategy name prevents accidental collisions with fixed
        or structure-aware chunks containing identical text.
        """

        value = f"semantic|{filename}|{chunk_index}|{text}"

        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

        return f"chunk_{digest}"

    def _split_segment(
        self, segment: DocumentSegment
    ) -> list[tuple[str, int, int | None, str | None]]:
        """
        Split one source segment into sentence/line-aware text units.

        Oversized individual units are hard-split by words so no
        semantic unit can exceed ``chunk_size``.
        """

        text = segment.text.strip()

        if not text:
            return []

        pieces = re.split(r"(?<=[.!?])\s+|\n+", text)

        units: list[tuple[str, int, int | None, str | None]] = []

        for piece in pieces:
            piece = piece.strip()

            if not piece:
                continue

            words = piece.split()

            if len(words) <= self.chunk_size:
                units.append(
                    (" ".join(words), segment.segment_index, segment.page_number, segment.section)
                )
                continue

            start = 0

            while start < len(words):
                end = min(start + self.chunk_size, len(words))

                units.append(
                    (
                        " ".join(words[start:end]),
                        segment.segment_index,
                        segment.page_number,
                        segment.section,
                    )
                )

                start = end

        return units

    def _build_units(self, document: LoadedDocument) -> list[_SemanticUnit]:
        """Create semantic units and embed them in one batch."""

        raw_units: list[tuple[str, int, int | None, str | None]] = []

        for segment in document.segments:
            raw_units.extend(self._split_segment(segment))

        if not raw_units:
            return []

        texts = [unit[0] for unit in raw_units]

        try:
            embeddings = self.embedding_model.embed_documents(texts)

        except Exception as exc:
            raise ChunkingError(
                f"Semantic chunking could not embed document '{document.filename}'."
            ) from exc

        if len(embeddings) != len(raw_units):
            raise ChunkingError("Semantic chunking received an unexpected number of embeddings.")

        units: list[_SemanticUnit] = []

        expected_dimension: int | None = None

        for raw_unit, embedding in zip(raw_units, embeddings, strict=True):
            if not embedding:
                raise ChunkingError("Semantic chunking received an empty embedding vector.")

            if expected_dimension is None:
                expected_dimension = len(embedding)

            elif len(embedding) != expected_dimension:
                raise ChunkingError("Semantic chunking received inconsistent embedding dimensions.")

            units.append(
                _SemanticUnit(
                    text=raw_unit[0],
                    segment_index=raw_unit[1],
                    page_number=raw_unit[2],
                    section=raw_unit[3],
                    embedding=embedding,
                )
            )

        return units

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        """Return cosine similarity for two embedding vectors."""

        if len(left) != len(right):
            raise ChunkingError("Cannot compare embeddings with different dimensions.")

        dot_product = sum(
            left_value * right_value for left_value, right_value in zip(left, right, strict=True)
        )

        left_norm = math.sqrt(sum(value * value for value in left))

        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        return dot_product / (left_norm * right_norm)

    @staticmethod
    def _word_count(units: list[_SemanticUnit]) -> int:
        """Return total words across semantic units."""

        return sum(unit.word_count for unit in units)

    @staticmethod
    def _sections(units: list[_SemanticUnit]) -> list[str]:
        """Return explicit sections represented by semantic units."""

        return list(dict.fromkeys(unit.section for unit in units if unit.section))

    def _section_conflict(self, current: list[_SemanticUnit], incoming: _SemanticUnit) -> bool:
        """Return True when adding a unit crosses a known section."""

        if incoming.section is None:
            return False

        current_sections = self._sections(current)

        if not current_sections:
            return False

        return incoming.section not in current_sections

    def _semantic_conflict(self, current: list[_SemanticUnit], incoming: _SemanticUnit) -> bool:
        """
        Return True when the incoming unit represents a semantic shift.

        A semantic split is allowed only after the current chunk has
        reached ``min_chunk_size`` so isolated short sentences do not
        create excessive fragmentation.
        """

        if not current:
            return False

        if self._word_count(current) < self.min_chunk_size:
            return False

        similarity = self._cosine_similarity(current[-1].embedding, incoming.embedding)

        return similarity < self.similarity_threshold

    def _build_overlap(self, units: list[_SemanticUnit], max_words: int) -> list[_SemanticUnit]:
        """
        Build metadata-aware word overlap from a previous chunk tail.

        The overlap never exceeds either the configured overlap or
        available capacity in the next chunk.
        """

        overlap_words = min(self.overlap, max_words)

        if overlap_words <= 0:
            return []

        remaining = overlap_words

        reversed_overlap: list[_SemanticUnit] = []

        for unit in reversed(units):
            if remaining <= 0:
                break

            words = unit.text.split()

            take = min(remaining, len(words))

            selected_words = words[-take:]

            reversed_overlap.append(
                _SemanticUnit(
                    text=" ".join(selected_words),
                    segment_index=(unit.segment_index),
                    page_number=(unit.page_number),
                    section=(unit.section),
                    embedding=(unit.embedding),
                )
            )

            remaining -= take

        reversed_overlap.reverse()

        return reversed_overlap

    def _create_chunk(
        self, document: LoadedDocument, units: list[_SemanticUnit], chunk_index: int
    ) -> DocumentChunk:
        """Build one DocumentChunk from semantic units."""

        text = " ".join(unit.text for unit in units).strip()

        segment_indices = list(dict.fromkeys(unit.segment_index for unit in units))

        pages = [unit.page_number for unit in units if unit.page_number is not None]

        sections = self._sections(units)

        page_start = min(pages) if pages else None

        page_end = max(pages) if pages else None

        section = sections[0] if len(sections) == 1 else None

        chunk_id = self._build_chunk_id(
            filename=document.filename, chunk_index=chunk_index, text=text
        )

        return DocumentChunk(
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            text=text,
            filename=document.filename,
            file_type=document.file_type,
            source_segment_indices=(segment_indices),
            page_start=page_start,
            page_end=page_end,
            section=section,
            metadata={
                "chunking_strategy": "semantic",
                "word_count": len(text.split()),
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "min_chunk_size": (self.min_chunk_size),
                "similarity_threshold": (self.similarity_threshold),
                "embedding_dimension": len(units[0].embedding),
                "source_segment_count": len(segment_indices),
                "sections": sections,
            },
        )

    def _flush_with_overlap(
        self,
        *,
        document: LoadedDocument,
        chunks: list[DocumentChunk],
        current: list[_SemanticUnit],
        incoming: _SemanticUnit,
    ) -> list[_SemanticUnit]:
        """Create a chunk and return overlap reserved for the next."""

        chunks.append(self._create_chunk(document=document, units=current, chunk_index=len(chunks)))

        available_for_overlap = max(0, self.chunk_size - incoming.word_count)

        return self._build_overlap(units=current, max_words=(available_for_overlap))

    def chunk(self, document: LoadedDocument) -> list[DocumentChunk]:
        """Split a document into semantically coherent chunks."""

        units = self._build_units(document)

        if not units:
            raise ChunkingError(
                f"Document '{document.filename}' contains no text that can be chunked."
            )

        chunks: list[DocumentChunk] = []

        current: list[_SemanticUnit] = []

        for unit in units:
            # Explicit document structure remains a hard boundary.
            if current and self._section_conflict(current=current, incoming=unit):
                chunks.append(
                    self._create_chunk(document=document, units=current, chunk_index=len(chunks))
                )

                # Do not leak overlap across explicit section changes.
                current = []

            # Hard size boundary always takes precedence.
            if (current and (self._word_count(current) + unit.word_count > self.chunk_size)) or (
                current and self._semantic_conflict(current=current, incoming=unit)
            ):
                current = self._flush_with_overlap(
                    document=document, chunks=chunks, current=current, incoming=unit
                )

            current.append(unit)

        if current:
            chunks.append(
                self._create_chunk(document=document, units=current, chunk_index=len(chunks))
            )

        if not chunks:
            raise ChunkingError(f"Semantic chunking produced no chunks for '{document.filename}'.")

        return chunks
