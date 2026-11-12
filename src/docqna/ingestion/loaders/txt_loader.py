from pathlib import Path

from docqna.constants import DEFAULT_TEXT_ENCODING
from docqna.exceptions import DocumentLoadError
from docqna.ingestion.loaders.base import BaseDocumentLoader
from docqna.ingestion.models import DocumentSegment, LoadedDocument


class TxtLoader(BaseDocumentLoader):
    """Loader for plain-text documents."""

    supported_extensions = frozenset({".txt"})

    def load(self, path: str | Path) -> LoadedDocument:
        file_path = self.validate_path(path)

        try:
            text = file_path.read_text(encoding=DEFAULT_TEXT_ENCODING)
        except UnicodeDecodeError as exc:
            raise DocumentLoadError(
                f"Could not decode text file as {DEFAULT_TEXT_ENCODING}: {file_path}"
            ) from exc
        except OSError as exc:
            raise DocumentLoadError(f"Could not read text file: {file_path}") from exc

        text = text.strip()

        if not text:
            raise DocumentLoadError(f"Text document contains no usable text: {file_path}")

        segment = DocumentSegment(
            text=text, segment_index=0, metadata={"line_count": len(text.splitlines())}
        )

        return LoadedDocument(
            source_path=file_path,
            filename=file_path.name,
            file_type="txt",
            segments=[segment],
            metadata={"size_bytes": file_path.stat().st_size},
        )
