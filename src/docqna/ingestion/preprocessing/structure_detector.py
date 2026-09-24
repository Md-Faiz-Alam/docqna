import re

from docqna.ingestion.models import (
    DocumentSegment,
    LoadedDocument,
)


class StructureDetector:
    """
    Detect simple document structure while preserving existing metadata.

    This first version is intentionally conservative.
    """

    _SECTION_PREFIX = re.compile(
        r"^(?:"
        r"\d+(?:\.\d+)*"
        r"|[A-Z]\."
        r"|[IVXLCDM]+\."
        r")\s+\S+",
        re.IGNORECASE,
    )

    def looks_like_heading(self, text: str) -> bool:
        """Return True when a short text segment resembles a heading."""

        text = text.strip()

        if not text:
            return False

        # Multiline segments are unlikely to be simple headings.
        if "\n" in text:
            return False

        if len(text) > 120:
            return False

        words = text.split()

        if not 1 <= len(words) <= 12:
            return False

        # Normal prose usually ends with these.
        if text.endswith((".", "?", "!", ";", ",")):
            return False

        # SECTION TITLE
        if text.isupper():
            return True

        # 4.2 Academic Rules
        if self._SECTION_PREFIX.match(text):
            return True

        alphabetic_words = [
            word
            for word in words
            if any(character.isalpha() for character in word)
        ]

        if not alphabetic_words:
            return False

        capitalized = sum(
            1
            for word in alphabetic_words
            if word[0].isupper()
        )

        ratio = capitalized / len(alphabetic_words)

        return ratio >= 0.7

    def detect(
        self,
        document: LoadedDocument,
    ) -> LoadedDocument:
        """Detect and propagate basic section information."""

        detected_segments: list[DocumentSegment] = []

        current_section: str | None = None

        for segment in document.segments:
            metadata = dict(segment.metadata)

            section = segment.section

            # DOCX loader may already provide section information.
            if section:
                current_section = section

                if section.strip() == segment.text.strip():
                    metadata["is_heading"] = True

            # Only infer headings when the loader did not already
            # provide structure.
            elif self.looks_like_heading(segment.text):
                current_section = segment.text.strip()
                section = current_section
                metadata["is_heading"] = True

            elif current_section:
                section = current_section

            detected_segments.append(
                segment.model_copy(
                    update={
                        "section": section,
                        "metadata": metadata,
                    }
                )
            )

        return document.model_copy(
            update={
                "segments": detected_segments,
            }
        )