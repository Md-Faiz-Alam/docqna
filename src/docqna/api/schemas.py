from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from docqna.constants import SUPPORTED_FILE_TYPES


class HealthResponse(BaseModel):
    """Health response returned by the API."""

    status: str
    app: str
    version: str


class QAMetadataFilters(BaseModel):
    """Optional metadata filters applied during retrieval."""

    document_id: str | None = None
    filename: str | None = None
    file_type: str | None = None
    document_version: int | None = Field(
        default=None,
        ge=1,
    )

    @field_validator(
        "document_id",
        "filename",
    )
    @classmethod
    def validate_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError(
                "Metadata filter values cannot be empty."
            )

        return value

    @field_validator("file_type")
    @classmethod
    def validate_file_type(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip().lower().lstrip(".")

        if value not in SUPPORTED_FILE_TYPES:
            raise ValueError(
                "Unsupported file_type metadata filter."
            )

        return value

    def to_retrieval_filters(
        self,
    ) -> dict[str, str | int]:
        """Return only filters explicitly provided by the client."""

        filters: dict[
            str,
            str | int,
        ] = {}

        if self.document_id is not None:
            filters["document_id"] = (
                self.document_id
            )

        if self.filename is not None:
            filters["filename"] = self.filename

        if self.file_type is not None:
            filters["file_type"] = self.file_type

        if self.document_version is not None:
            filters["document_version"] = (
                self.document_version
            )

        return filters


class QARequest(BaseModel):
    """Question-answering API request."""

    question: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    filters: QAMetadataFilters | None = None

    @field_validator("question")
    @classmethod
    def validate_question(
        cls,
        value: str,
    ) -> str:
        """Reject empty or whitespace-only questions."""

        value = value.strip()

        if not value:
            raise ValueError(
                "Question cannot be empty."
            )

        return value


class CitationResponse(BaseModel):
    """Citation information exposed by the API."""

    citation_id: int = Field(
        ge=1
    )

    chunk_id: str

    rank: int = Field(
        ge=1
    )

    filename: str
    file_type: str

    page_start: int | None = None
    page_end: int | None = None

    section: str | None = None

    distance: float

    text: str


class QAResponse(BaseModel):
    """Question-answering API response."""

    answer: str

    abstained: bool

    cited_source_ids: list[int] = Field(
        default_factory=list
    )

    sources: list[CitationResponse] = Field(
        default_factory=list
    )


class DocumentResponse(BaseModel):
    """Document identity and version exposed by the API."""

    document_id: str
    filename: str
    file_type: str
    version: int = Field(ge=1)
    content_hash: str
    size_bytes: int = Field(ge=0)
    registered_at: datetime


class DocumentUploadResponse(BaseModel):
    """Response returned after indexing an uploaded document."""

    document: DocumentResponse
    chunk_count: int = Field(ge=1)


class DocumentListResponse(BaseModel):
    """Active logical documents currently registered."""

    count: int = Field(ge=0)
    documents: list[DocumentResponse] = Field(
        default_factory=list
    )


class DocumentDeleteResponse(BaseModel):
    """Response returned after deleting a logical document."""

    document_id: str
    deleted: bool


class ErrorResponse(BaseModel):
    """Standard API error response."""

    error: str
    detail: str
