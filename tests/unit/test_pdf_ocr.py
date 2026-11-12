from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from docqna.exceptions import DocumentLoadError, ResourceLimitError
from docqna.ingestion.loaders.pdf_loader import PdfLoader


def make_scanned_pdf(path: Path, pages: int = 2) -> Path:
    """Write an image-only PDF (no text layer), like a scanner produces."""

    images = []
    for number in range(1, pages + 1):
        image = Image.new("RGB", (600, 800), "white")
        ImageDraw.Draw(image).text((50, 50), f"Scanned page {number}", fill="black")
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:])
    return path


class FakeOcr:
    def __init__(self) -> None:
        self.calls = 0

    def image_to_text(self, image) -> str:
        self.calls += 1
        assert image.size[0] > 1000, "pages should be rendered at OCR resolution"
        return f"Recognised text from scanned page number {self.calls}."


def test_scanned_pages_are_read_with_ocr(tmp_path: Path) -> None:
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=2)
    ocr = FakeOcr()

    document = PdfLoader(ocr_engine=ocr, max_ocr_pages=10).load(pdf)

    assert ocr.calls == 2
    assert [s.page_number for s in document.segments] == [1, 2]
    assert document.segments[0].text == "Recognised text from scanned page number 1."
    assert document.segments[0].metadata["extraction"] == "ocr"
    assert document.metadata["ocr_pages"] == 2


def test_scanned_pdf_without_ocr_explains_why(tmp_path: Path) -> None:
    pdf = make_scanned_pdf(tmp_path / "scan.pdf")

    with pytest.raises(DocumentLoadError, match="OCR is not available"):
        PdfLoader(ocr_engine=None, max_ocr_pages=10).load(pdf)


def test_ocr_page_limit_protects_the_server(tmp_path: Path) -> None:
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=3)

    with pytest.raises(ResourceLimitError, match="limited to 2 pages"):
        PdfLoader(ocr_engine=FakeOcr(), max_ocr_pages=2).load(pdf)
