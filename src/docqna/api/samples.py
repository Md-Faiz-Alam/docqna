"""Bundled sample documents a visitor can load with one click.

They are the fictional Northwind/CloudBox documents from the public
benchmark, so the demo works without uploading anything private.
"""

from functools import lru_cache
from pathlib import Path

from docqna.constants import SUPPORTED_FILE_EXTENSIONS
from docqna.ingestion.registry import DocumentRegistry


def sample_document_paths(directory: Path) -> list[Path]:
    """Supported files in `directory`, sorted by name; empty if it is missing."""

    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS
    )


@lru_cache(maxsize=4)
def sample_content_hashes(directory: Path) -> frozenset[str]:
    """Content hashes of the sample documents (exempt from upload quotas)."""

    return frozenset(
        DocumentRegistry.compute_content_hash(path) for path in sample_document_paths(directory)
    )
