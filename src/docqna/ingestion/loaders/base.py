from abc import ABC, abstractmethod
from pathlib import Path

from docqna.exceptions import DocumentLoadError, UnsupportedFileTypeError
from docqna.ingestion.models import LoadedDocument


class BaseDocumentLoader(ABC):
    """Base interface for document loaders."""

    supported_extensions: frozenset[str] = frozenset()

    def can_load(self, path: str | Path) -> bool:
        """Return True when this loader supports the file extension."""

        file_path = Path(path)

        return file_path.suffix.lower() in self.supported_extensions

    def validate_path(self, path: str | Path) -> Path:
        """Validate a path before attempting document extraction."""

        file_path = Path(path)

        if not file_path.exists():
            raise DocumentLoadError(f"Document does not exist: {file_path}")

        if not file_path.is_file():
            raise DocumentLoadError(f"Path is not a file: {file_path}")

        if not self.can_load(file_path):
            raise UnsupportedFileTypeError(
                f"{self.__class__.__name__} does not support '{file_path.suffix}'."
            )

        return file_path

    @abstractmethod
    def load(self, path: str | Path) -> LoadedDocument:
        """Load a document and return its extracted representation."""
        raise NotImplementedError
