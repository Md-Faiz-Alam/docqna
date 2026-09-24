from pathlib import Path
from typing import Any
from datetime import datetime

from pydantic import BaseModel, Field


LEGACY_SESSION_ID = "legacy_local_session"


class DocumentSegment(BaseModel):
    """
    A logical piece of text extracted from a source document.

    A segment is not yet a RAG chunk.

    Examples:
        PDF  -> one segment may represent one page
        DOCX -> one segment may represent one paragraph
        TXT  -> one segment may represent the whole text file
    """

    text: str
    segment_index: int

    page_number: int | None = None
    section: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadedDocument(BaseModel):
    """
    Standard representation returned by every document loader.
    """

    source_path: Path
    filename: str
    file_type: str

    segments: list[DocumentSegment]

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Return all document text joined together."""
        return "\n\n".join(
            segment.text
            for segment in self.segments
            if segment.text.strip()
        )

    @property
    def segment_count(self) -> int:
        """Return the number of extracted segments."""
        return len(self.segments)


class DocumentChunk(BaseModel):
    """
    Retrieval unit produced by the chunking layer.
    """

    chunk_id: str
    chunk_index: int

    text: str

    filename: str
    file_type: str

    source_segment_indices: list[int]

    page_start: int | None = None
    page_end: int | None = None

    section: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    """
    Persistent identity and version information for
    one successfully registered document version.
    """

    document_id: str

    # Pre-session-isolation registry records do not contain
    # this field. The default keeps those records readable
    # without exposing them to new anonymous sessions.
    session_id: str = Field(
        default=LEGACY_SESSION_ID,
        min_length=1,
    )

    filename: str
    file_type: str

    version: int = Field(
        ge=1,
    )

    content_hash: str

    size_bytes: int = Field(
        ge=0,
    )

    registered_at: datetime