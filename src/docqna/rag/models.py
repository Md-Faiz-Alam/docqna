from pydantic import BaseModel, Field


class CitationSource(BaseModel):
    """Source information associated with a RAG citation."""

    citation_id: int = Field(ge=1)

    chunk_id: str
    rank: int = Field(ge=1)

    filename: str
    file_type: str

    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None

    distance: float
    text: str


class RAGResponse(BaseModel):
    """Structured RAG response with source citations."""

    answer: str

    sources: list[CitationSource] = Field(default_factory=list)

    cited_source_ids: list[int] = Field(default_factory=list)

    # Set when a follow-up was rewritten using conversation history.
    standalone_question: str | None = None
