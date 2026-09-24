from pathlib import Path
from typing import Any

from docx import Document

from docqna.exceptions import DocumentLoadError
from docqna.ingestion.loaders.base import BaseDocumentLoader
from docqna.ingestion.models import (
    DocumentSegment,
    LoadedDocument,
)


class DocxLoader(BaseDocumentLoader):
    """Loader for Microsoft Word DOCX documents."""

    supported_extensions = frozenset({".docx"})

    def load(self, path: str | Path) -> LoadedDocument:
        file_path = self.validate_path(path)

        try:
            document = Document(str(file_path))
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not open DOCX document: {file_path}"
            ) from exc

        segments: list[DocumentSegment] = []

        current_section: str | None = None

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if not text:
                continue

            style_name = (
                paragraph.style.name
                if paragraph.style is not None
                else None
            )

            if (
                style_name
                and style_name.lower().startswith("heading")
            ):
                current_section = text

            segments.append(
                DocumentSegment(
                    text=text,
                    segment_index=len(segments),
                    section=current_section,
                    metadata={
                        "style": style_name,
                    },
                )
            )

        if not segments:
            raise DocumentLoadError(
                f"DOCX document contains no usable text: {file_path}"
            )

        core_properties = document.core_properties

        metadata: dict[str, Any] = {
            "paragraph_count": len(document.paragraphs),
            "size_bytes": file_path.stat().st_size,
        }

        if core_properties.title:
            metadata["title"] = core_properties.title

        if core_properties.author:
            metadata["author"] = core_properties.author

        return LoadedDocument(
            source_path=file_path,
            filename=file_path.name,
            file_type="docx",
            segments=segments,
            metadata=metadata,
        )