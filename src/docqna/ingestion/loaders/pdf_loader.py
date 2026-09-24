from pathlib import Path

from pypdf import PdfReader

from docqna.exceptions import DocumentLoadError
from docqna.ingestion.loaders.base import BaseDocumentLoader
from docqna.ingestion.models import (
    DocumentSegment,
    LoadedDocument,
)


class PdfLoader(BaseDocumentLoader):
    """Loader for text-based PDF documents."""

    supported_extensions = frozenset({".pdf"})

    def load(self, path: str | Path) -> LoadedDocument:
        file_path = self.validate_path(path)

        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not open PDF: {file_path}"
            ) from exc

        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
            except Exception as exc:
                raise DocumentLoadError(
                    f"Could not decrypt PDF: {file_path}"
                ) from exc

            if decrypt_result == 0:
                raise DocumentLoadError(
                    "Password-protected PDFs are not supported."
                )

        segments: list[DocumentSegment] = []

        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1

            try:
                text = page.extract_text() or ""
            except Exception as exc:
                raise DocumentLoadError(
                    f"Failed to extract text from page "
                    f"{page_number} of {file_path.name}."
                ) from exc

            text = text.strip()

            if not text:
                continue

            segments.append(
                DocumentSegment(
                    text=text,
                    segment_index=len(segments),
                    page_number=page_number,
                    metadata={
                        "source_page_index": page_index,
                    },
                )
            )

        if not segments:
            raise DocumentLoadError(
                "No extractable text was found in the PDF. "
                "The document may be scanned and require OCR."
            )

        return LoadedDocument(
            source_path=file_path,
            filename=file_path.name,
            file_type="pdf",
            segments=segments,
            metadata={
                "total_pages": len(reader.pages),
                "size_bytes": file_path.stat().st_size,
            },
        )