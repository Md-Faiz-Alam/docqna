from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """One ranked retrieval result."""

    chunk_id: str
    text: str
    rank: int

    # Dense retrieval currently uses cosine distance.
    # BM25 will also expose its native score separately.
    distance: float

    score: float | None = None
    retrieval_method: str = "dense"

    filename: str
    file_type: str
    chunk_index: int

    document_id: str | None = None
    document_version: int | None = None

    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None

    source_segment_indices: list[int] = Field(
        default_factory=list
    )

    sections: list[str] = Field(
        default_factory=list
    )