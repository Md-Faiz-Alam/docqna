"""Map DocQ&A domain exceptions to stable HTTP error responses."""

import logging
from dataclasses import dataclass

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from docqna.exceptions import (
    CitationError,
    ConfigurationError,
    DocQNAError,
    DocumentLoadError,
    DocumentValidationError,
    DuplicateDocumentError,
    EmbeddingError,
    GenerationError,
    IndexingError,
    LLMConnectionError,
    LLMModelError,
    ResourceLimitError,
    RetrievalError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ErrorSpec:
    status_code: int
    code: str
    log_level: int


# Client errors log at WARNING, server/dependency failures at ERROR.
ERROR_SPECS: dict[type[DocQNAError], ErrorSpec] = {
    DuplicateDocumentError: ErrorSpec(
        status.HTTP_409_CONFLICT, "duplicate_document", logging.WARNING
    ),
    UnsupportedFileTypeError: ErrorSpec(
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "unsupported_file_type", logging.WARNING
    ),
    DocumentValidationError: ErrorSpec(
        status.HTTP_400_BAD_REQUEST, "document_validation_error", logging.WARNING
    ),
    ResourceLimitError: ErrorSpec(
        status.HTTP_429_TOO_MANY_REQUESTS, "resource_limit", logging.WARNING
    ),
    DocumentLoadError: ErrorSpec(
        status.HTTP_422_UNPROCESSABLE_CONTENT, "document_load_error", logging.WARNING
    ),
    LLMConnectionError: ErrorSpec(
        status.HTTP_503_SERVICE_UNAVAILABLE, "llm_connection_error", logging.ERROR
    ),
    LLMModelError: ErrorSpec(status.HTTP_503_SERVICE_UNAVAILABLE, "llm_model_error", logging.ERROR),
    RetrievalError: ErrorSpec(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "retrieval_error", logging.ERROR
    ),
    CitationError: ErrorSpec(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "citation_error", logging.ERROR
    ),
    GenerationError: ErrorSpec(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "generation_error", logging.ERROR
    ),
    EmbeddingError: ErrorSpec(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "embedding_error", logging.ERROR
    ),
    IndexingError: ErrorSpec(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "indexing_error", logging.ERROR
    ),
    ConfigurationError: ErrorSpec(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "configuration_error", logging.ERROR
    ),
    # Fallback for any other domain error.
    DocQNAError: ErrorSpec(status.HTTP_500_INTERNAL_SERVER_ERROR, "docqna_error", logging.ERROR),
}


def _resolve_spec(exc: DocQNAError) -> ErrorSpec:
    for exc_type in type(exc).__mro__:
        spec = ERROR_SPECS.get(exc_type)
        if spec is not None:
            return spec
    return ERROR_SPECS[DocQNAError]


async def docqna_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return `{error, detail}` without logging messages, queries, or document text."""

    assert isinstance(exc, DocQNAError)
    spec = _resolve_spec(exc)

    logger.log(
        spec.log_level,
        "handled_error code=%s exception=%s method=%s path=%s",
        spec.code,
        type(exc).__name__,
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=spec.status_code, content={"error": spec.code, "detail": str(exc)}
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DocQNAError, docqna_exception_handler)
