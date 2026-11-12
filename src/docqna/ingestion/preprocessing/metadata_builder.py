import re
from typing import Any

from docqna.ingestion.models import DocumentSegment, LoadedDocument


class MetadataBuilder:
    """Build derived metadata for loaded documents and segments."""

    _WORD_PATTERN = re.compile(r"\b[\w'-]+\b")

    def count_words(self, text: str) -> int:
        """Count approximate words in text."""

        return len(self._WORD_PATTERN.findall(text))

    def build(self, document: LoadedDocument) -> LoadedDocument:
        """Add useful derived metadata."""

        enriched_segments: list[DocumentSegment] = []

        total_characters = 0
        total_words = 0

        sections: set[str] = set()

        for segment in document.segments:
            text = segment.text

            character_count = len(text)
            word_count = self.count_words(text)

            total_characters += character_count
            total_words += word_count

            if segment.section:
                sections.add(segment.section)

            metadata: dict[str, Any] = {
                **segment.metadata,
                "character_count": character_count,
                "word_count": word_count,
            }

            enriched_segments.append(segment.model_copy(update={"metadata": metadata}))

        document_metadata: dict[str, Any] = {
            **document.metadata,
            "segment_count": len(enriched_segments),
            "character_count": total_characters,
            "word_count": total_words,
        }

        if sections:
            document_metadata["sections"] = sorted(sections)

        return document.model_copy(
            update={"segments": enriched_segments, "metadata": document_metadata}
        )
