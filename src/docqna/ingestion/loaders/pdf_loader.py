import logging
from pathlib import Path

from pypdf import PdfReader

from docqna.exceptions import DocumentLoadError, ResourceLimitError
from docqna.ingestion.loaders.base import BaseDocumentLoader
from docqna.ingestion.models import DocumentSegment, LoadedDocument
from docqna.ingestion.ocr import OcrEngine, default_ocr_engine

logger = logging.getLogger(__name__)

# Pages with fewer extracted characters than this are treated as image-only.
MIN_TEXT_LAYER_CHARS = 20
OCR_RENDER_DPI = 300

_DEFAULT = object()


class PdfLoader(BaseDocumentLoader):
    """PDF loader: text layer via pypdf, OCR fallback for scanned pages."""

    supported_extensions = frozenset({".pdf"})

    def __init__(
        self, ocr_engine: OcrEngine | None | object = _DEFAULT, max_ocr_pages: int | None = None
    ) -> None:
        if ocr_engine is _DEFAULT:
            ocr_engine = default_ocr_engine()
        if max_ocr_pages is None:
            from docqna.settings import get_settings

            max_ocr_pages = get_settings().max_ocr_pages

        self.ocr_engine: OcrEngine | None = ocr_engine  # type: ignore[assignment]
        self.max_ocr_pages = max_ocr_pages

    def _ocr_pages(self, file_path: Path, page_indices: list[int]) -> dict[int, str]:
        """Render and OCR the given zero-based pages."""

        assert self.ocr_engine is not None
        import pypdfium2 as pdfium

        texts: dict[int, str] = {}
        try:
            pdf = pdfium.PdfDocument(str(file_path))
        except Exception as exc:
            raise DocumentLoadError(f"Could not render PDF for OCR: {file_path.name}") from exc

        try:
            for page_index in page_indices:
                page = pdf[page_index]
                try:
                    image = page.render(scale=OCR_RENDER_DPI / 72).to_pil()
                    texts[page_index] = self.ocr_engine.image_to_text(image).strip()
                except Exception as exc:
                    raise DocumentLoadError(
                        f"OCR failed on page {page_index + 1} of {file_path.name}."
                    ) from exc
                finally:
                    page.close()
        finally:
            pdf.close()

        return texts

    def load(self, path: str | Path) -> LoadedDocument:
        file_path = self.validate_path(path)

        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            raise DocumentLoadError(f"Could not open PDF: {file_path}") from exc

        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
            except Exception as exc:
                raise DocumentLoadError(f"Could not decrypt PDF: {file_path}") from exc

            if decrypt_result == 0:
                raise DocumentLoadError("Password-protected PDFs are not supported.")

        page_texts: list[str] = []
        for page_index, page in enumerate(reader.pages):
            try:
                page_texts.append((page.extract_text() or "").strip())
            except Exception as exc:
                raise DocumentLoadError(
                    f"Failed to extract text from page {page_index + 1} of {file_path.name}."
                ) from exc

        image_only = [i for i, text in enumerate(page_texts) if len(text) < MIN_TEXT_LAYER_CHARS]
        ocr_texts: dict[int, str] = {}

        if image_only and self.ocr_engine is not None:
            if len(image_only) > self.max_ocr_pages:
                raise ResourceLimitError(
                    f"This PDF has {len(image_only)} scanned pages; OCR is limited to "
                    f"{self.max_ocr_pages} pages per document."
                )
            ocr_texts = self._ocr_pages(file_path, image_only)
            logger.info(
                "pdf_ocr_complete filename=%s ocr_pages=%d total_pages=%d",
                file_path.name,
                len(image_only),
                len(page_texts),
            )

        segments: list[DocumentSegment] = []
        for page_index, text in enumerate(page_texts):
            extraction = "text_layer"
            if page_index in ocr_texts and len(ocr_texts[page_index]) > len(text):
                text, extraction = ocr_texts[page_index], "ocr"
            if not text:
                continue

            segments.append(
                DocumentSegment(
                    text=text,
                    segment_index=len(segments),
                    page_number=page_index + 1,
                    metadata={"source_page_index": page_index, "extraction": extraction},
                )
            )

        if not segments:
            hint = (
                "OCR found no readable text."
                if self.ocr_engine is not None
                else "The document appears to be scanned and OCR is not available on this server."
            )
            raise DocumentLoadError(f"No extractable text was found in the PDF. {hint}")

        return LoadedDocument(
            source_path=file_path,
            filename=file_path.name,
            file_type="pdf",
            segments=segments,
            metadata={
                "total_pages": len(reader.pages),
                "ocr_pages": len(ocr_texts),
                "size_bytes": file_path.stat().st_size,
            },
        )
