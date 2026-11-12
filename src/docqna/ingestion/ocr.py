"""OCR for scanned / image-only PDF pages.

Pages with no extractable text layer are rendered with pdfium and read with
Tesseract. Tesseract is a system binary: it is installed in the Docker image;
locally, OCR is simply unavailable unless Tesseract is on PATH.
"""

import logging
import shutil
from functools import lru_cache
from typing import Protocol

from PIL.Image import Image

logger = logging.getLogger(__name__)


class OcrEngine(Protocol):
    def image_to_text(self, image: Image) -> str: ...


class TesseractOcr:
    """pytesseract wrapper. `languages` uses Tesseract codes, e.g. "eng+deu"."""

    def __init__(self, languages: str = "eng", timeout_seconds: float = 60.0) -> None:
        self.languages = languages
        self.timeout_seconds = timeout_seconds

    def image_to_text(self, image: Image) -> str:
        import pytesseract

        return str(
            pytesseract.image_to_string(image, lang=self.languages, timeout=self.timeout_seconds)
        )


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def default_ocr_engine() -> OcrEngine | None:
    """Return the configured OCR engine, or None when OCR is disabled/unavailable."""

    from docqna.settings import get_settings

    settings = get_settings()
    if not settings.ocr_enabled:
        return None
    if not tesseract_available():
        logger.info("ocr_unavailable reason=tesseract_not_installed")
        return None
    return TesseractOcr(languages=settings.ocr_languages)
