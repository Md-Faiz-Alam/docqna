import hashlib
import re
from dataclasses import dataclass

from docqna.exceptions import ChunkingError
from docqna.ingestion.chunking.base import BaseChunker
from docqna.ingestion.models import DocumentChunk, DocumentSegment, LoadedDocument


@dataclass(frozen=True, slots=True)
class _StructureUnit:
    """
    Internal sentence/line-aware unit.

    Each unit retains the source segment, page,
    and section from which it originated.
    """

    text: str
    segment_index: int
    page_number: int | None
    section: str | None

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class StructureAwareChunker(BaseChunker):
    """
    Structure-aware document chunker.

    Compared with FixedChunker, this strategy:

    - prefers sentence and line boundaries
    - preserves source segment metadata
    - avoids crossing explicit section changes
    - uses controlled word overlap
    - hard-splits only when a structural unit itself
      exceeds the configured chunk size
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if overlap < 0:
            raise ValueError("overlap cannot be negative.")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def _unique_in_order(values: list[str]) -> list[str]:
        """Remove duplicates while preserving order."""

        return list(dict.fromkeys(values))

    @staticmethod
    def _build_chunk_id(filename: str, chunk_index: int, text: str) -> str:
        """
        Create a deterministic structure-aware chunk ID.

        The strategy name is included so structure-aware
        chunks cannot accidentally collide with fixed
        chunks that happen to contain identical text.
        """

        value = f"structure_aware|{filename}|{chunk_index}|{text}"

        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

        return f"chunk_{digest}"

    def _split_segment(self, segment: DocumentSegment) -> list[_StructureUnit]:
        """
        Split one document segment into sentence/line-aware
        structural units.

        A unit larger than chunk_size is hard-split by words.
        """

        text = segment.text.strip()

        if not text:
            return []

        # Prefer both sentence boundaries and source line
        # boundaries from the preprocessing stage.
        pieces = re.split(r"(?<=[.!?])\s+|\n+", text)

        units: list[_StructureUnit] = []

        for piece in pieces:
            piece = piece.strip()

            if not piece:
                continue

            words = piece.split()

            # Normal structural unit.
            if len(words) <= self.chunk_size:
                units.append(
                    _StructureUnit(
                        text=" ".join(words),
                        segment_index=(segment.segment_index),
                        page_number=(segment.page_number),
                        section=(segment.section),
                    )
                )

                continue

            # Fallback for unusually long sentences/lines.
            start = 0

            while start < len(words):
                end = min(start + self.chunk_size, len(words))

                units.append(
                    _StructureUnit(
                        text=" ".join(words[start:end]),
                        segment_index=(segment.segment_index),
                        page_number=(segment.page_number),
                        section=(segment.section),
                    )
                )

                start = end

        return units

    def _build_units(self, document: LoadedDocument) -> list[_StructureUnit]:
        """Convert a document into structural units."""

        units: list[_StructureUnit] = []

        for segment in document.segments:
            units.extend(self._split_segment(segment))

        return units

    @staticmethod
    def _word_count(units: list[_StructureUnit]) -> int:
        """Return total words across structural units."""

        return sum(unit.word_count for unit in units)

    @staticmethod
    def _sections(units: list[_StructureUnit]) -> list[str]:
        """Return explicit sections represented by units."""

        return list(dict.fromkeys(unit.section for unit in units if unit.section))

    def _section_conflict(self, current: list[_StructureUnit], incoming: _StructureUnit) -> bool:
        """
        Return True when adding the incoming unit would
        cross an explicit section boundary.

        Missing section labels do not force a boundary.
        """

        if incoming.section is None:
            return False

        current_sections = self._sections(current)

        if not current_sections:
            return False

        return incoming.section not in current_sections

    def _build_overlap(self, units: list[_StructureUnit], max_words: int) -> list[_StructureUnit]:
        """
        Build metadata-aware overlap from the tail of
        the previous chunk.

        Overlap never exceeds the configured overlap
        or the available capacity of the next chunk.
        """

        overlap_words = min(self.overlap, max_words)

        if overlap_words <= 0:
            return []

        remaining = overlap_words

        reversed_overlap: list[_StructureUnit] = []

        for unit in reversed(units):
            if remaining <= 0:
                break

            words = unit.text.split()

            take = min(remaining, len(words))

            selected_words = words[-take:]

            reversed_overlap.append(
                _StructureUnit(
                    text=" ".join(selected_words),
                    segment_index=(unit.segment_index),
                    page_number=(unit.page_number),
                    section=(unit.section),
                )
            )

            remaining -= take

        reversed_overlap.reverse()

        return reversed_overlap

    def _create_chunk(
        self, document: LoadedDocument, units: list[_StructureUnit], chunk_index: int
    ) -> DocumentChunk:
        """Build one DocumentChunk from structural units."""

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
                "chunking_strategy": ("structure_aware"),
                "word_count": len(text.split()),
                "chunk_size": (self.chunk_size),
                "overlap": (self.overlap),
                "source_segment_count": len(segment_indices),
                "sections": sections,
            },
        )

    def chunk(self, document: LoadedDocument) -> list[DocumentChunk]:
        """
        Split a document into structure-aware chunks.
        """

        units = self._build_units(document)

        if not units:
            raise ChunkingError(
                f"Document '{document.filename}' contains no text that can be chunked."
            )

        chunks: list[DocumentChunk] = []

        current: list[_StructureUnit] = []

        current_word_count = 0

        for unit in units:
            # ---------------------------------
            # Explicit section boundary
            # ---------------------------------

            if current and self._section_conflict(current=current, incoming=unit):
                chunks.append(
                    self._create_chunk(document=document, units=current, chunk_index=len(chunks))
                )

                # Do not carry overlap across
                # explicit section boundaries.
                current = []
                current_word_count = 0

            # ---------------------------------
            # Size boundary
            # ---------------------------------

            if current and (current_word_count + unit.word_count > self.chunk_size):
                chunks.append(
                    self._create_chunk(document=document, units=current, chunk_index=len(chunks))
                )

                # Reserve enough capacity for
                # the incoming structural unit.
                available_for_overlap = max(0, self.chunk_size - unit.word_count)

                current = self._build_overlap(units=current, max_words=(available_for_overlap))

                current_word_count = self._word_count(current)

            current.append(unit)

            current_word_count += unit.word_count

        if current:
            chunks.append(
                self._create_chunk(document=document, units=current, chunk_index=len(chunks))
            )

        if not chunks:
            raise ChunkingError(
                f"Structure-aware chunking produced no chunks for '{document.filename}'."
            )

        return chunks
