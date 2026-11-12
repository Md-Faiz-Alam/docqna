"""Real Tesseract OCR on a generated scanned PDF (skipped if Tesseract is absent)."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from docqna.ingestion.loaders.pdf_loader import PdfLoader
from docqna.ingestion.ocr import TesseractOcr, tesseract_available

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not tesseract_available(), reason="Tesseract is not installed"),
]


def test_scanned_pdf_text_is_recovered(tmp_path: Path) -> None:
    image = Image.new("RGB", (1700, 2200), "white")
    font = ImageFont.load_default(size=48)
    ImageDraw.Draw(image).text(
        (120, 200), "Meal expenses are capped at 60 euros per day.", fill="black", font=font
    )
    pdf = tmp_path / "scan.pdf"
    image.save(pdf, resolution=200)

    document = PdfLoader(ocr_engine=TesseractOcr(), max_ocr_pages=5).load(pdf)

    text = document.segments[0].text.lower()
    assert "meal expenses" in text
    assert "60 euros" in text
    assert document.segments[0].metadata["extraction"] == "ocr"
