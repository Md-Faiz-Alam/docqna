import hashlib
from dataclasses import dataclass

from docqna.exceptions import ChunkingError
from docqna.ingestion.chunking.base import BaseChunker
from docqna.ingestion.models import DocumentChunk, LoadedDocument


@dataclass(frozen=True, slots=True)
class _WordUnit:
    """Internal word representation used during fixed chunking."""

    text: str
    segment_index: int
    page_number: int | None
    section: str | None


class FixedChunker(BaseChunker):
    """
    Split documents into fixed-size overlapping word windows.

    This is the baseline chunking strategy.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if overlap < 0:
            raise ValueError("overlap cannot be negative.")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def _flatten_document(self, document: LoadedDocument) -> list[_WordUnit]:
        """Convert document segments into metadata-aware words."""

        units: list[_WordUnit] = []

        for segment in document.segments:
            words = segment.text.split()

            for word in words:
                units.append(
                    _WordUnit(
                        text=word,
                        segment_index=segment.segment_index,
                        page_number=segment.page_number,
                        section=segment.section,
                    )
                )

        return units

    @staticmethod
    def _unique_in_order(values: list[str]) -> list[str]:
        """Remove duplicates while preserving original order."""

        return list(dict.fromkeys(values))

    @staticmethod
    def _build_chunk_id(filename: str, chunk_index: int, text: str) -> str:
        """Create a deterministic chunk identifier."""

        value = f"{filename}|{chunk_index}|{text}"

        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

        return f"chunk_{digest}"

    def chunk(self, document: LoadedDocument) -> list[DocumentChunk]:
        """Split a document into overlapping fixed-size chunks."""

        units = self._flatten_document(document)

        if not units:
            raise ChunkingError(
                f"Document '{document.filename}' contains no text that can be chunked."
            )

        chunks: list[DocumentChunk] = []

        step = self.chunk_size - self.overlap

        start = 0

        while start < len(units):
            end = min(start + self.chunk_size, len(units))

            window = units[start:end]

            text = " ".join(unit.text for unit in window).strip()

            if not text:
                break

            chunk_index = len(chunks)

            segment_indices = list(dict.fromkeys(unit.segment_index for unit in window))

            pages = [unit.page_number for unit in window if unit.page_number is not None]

            raw_sections = [unit.section for unit in window if unit.section]

            sections = self._unique_in_order(raw_sections)

            page_start = min(pages) if pages else None
            page_end = max(pages) if pages else None

            # Only expose a single section when the whole
            # chunk belongs to the same section.
            section = sections[0] if len(sections) == 1 else None

            chunk_id = self._build_chunk_id(
                filename=document.filename, chunk_index=chunk_index, text=text
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    text=text,
                    filename=document.filename,
                    file_type=document.file_type,
                    source_segment_indices=segment_indices,
                    page_start=page_start,
                    page_end=page_end,
                    section=section,
                    metadata={
                        "chunking_strategy": "fixed",
                        "word_count": len(window),
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                        "source_segment_count": len(segment_indices),
                        "sections": sections,
                    },
                )
            )

            if end >= len(units):
                break

            start += step

        if not chunks:
            raise ChunkingError(f"Chunking produced no chunks for '{document.filename}'.")

        return chunks
