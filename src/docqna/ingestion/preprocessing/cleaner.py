import re

from docqna.exceptions import DocumentLoadError
from docqna.ingestion.models import (
    DocumentSegment,
    LoadedDocument,
)


class TextCleaner:
    """
    Remove extraction artifacts while keeping source text recognizable.
    """

    # Control characters except newline.
    _CONTROL_CHARACTERS = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
    )

    # More than two consecutive newlines.
    _EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")

    def clean_text(self, text: str) -> str:
        """Clean common extraction artifacts."""

        text = self._CONTROL_CHARACTERS.sub("", text)

        text = self._EXCESSIVE_NEWLINES.sub(
            "\n\n",
            text,
        )

        return text.strip()

    def clean_document(
        self,
        document: LoadedDocument,
    ) -> LoadedDocument:
        """Clean every segment and remove empty segments."""

        cleaned_segments: list[DocumentSegment] = []

        for segment in document.segments:
            cleaned_text = self.clean_text(segment.text)

            if not cleaned_text:
                continue

            cleaned_segments.append(
                segment.model_copy(
                    update={
                        "text": cleaned_text,
                        "segment_index": len(cleaned_segments),
                    }
                )
            )

        if not cleaned_segments:
            raise DocumentLoadError(
                f"No usable text remained after cleaning "
                f"document '{document.filename}'."
            )

        return document.model_copy(
            update={
                "segments": cleaned_segments,
            }
        )