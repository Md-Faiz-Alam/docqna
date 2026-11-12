import re
import unicodedata

from docqna.ingestion.models import DocumentSegment, LoadedDocument


class TextNormalizer:
    """
    Normalize extracted text while preserving document meaning.
    """

    _MULTIPLE_SPACES = re.compile(r"[ \t]+")

    def normalize_text(self, text: str) -> str:
        """
        Normalize Unicode, line endings, and unnecessary whitespace.
        """

        # Normalize Unicode representations.
        text = unicodedata.normalize("NFKC", text)

        # Normalize line endings.
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Replace non-breaking spaces.
        text = text.replace("\u00a0", " ")

        normalized_lines: list[str] = []

        for line in text.splitlines():
            line = self._MULTIPLE_SPACES.sub(" ", line)
            normalized_lines.append(line.strip())

        return "\n".join(normalized_lines).strip()

    def normalize_document(self, document: LoadedDocument) -> LoadedDocument:
        """Normalize every segment in a document."""

        segments: list[DocumentSegment] = []

        for segment in document.segments:
            normalized_text = self.normalize_text(segment.text)

            segments.append(segment.model_copy(update={"text": normalized_text}))

        return document.model_copy(update={"segments": segments})
