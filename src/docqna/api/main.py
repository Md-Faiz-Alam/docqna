import logging
import re
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from docqna.api.dependencies import (
    clear_rag_pipeline_cache,
    get_document_index_service_provider,
    get_rag_pipeline_provider,
)
from docqna.api.schemas import (
    CitationResponse,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    ErrorResponse,
    HealthResponse,
    QARequest,
    QAResponse,
)
from docqna.constants import (
    APP_NAME,
    APP_VERSION,
    SUPPORTED_FILE_EXTENSIONS,
)
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
    RetrievalError,
    UnsupportedFileTypeError,
)
from docqna.indexing import DocumentIndexService
from docqna.logging_config import (
    configure_logging,
    reset_request_id,
    set_request_id,
)
from docqna.rag import RAGPipeline
from docqna.settings import get_settings


configure_logging()

logger = logging.getLogger(__name__)


SESSION_HEADER_NAME = "X-DocQNA-Session-ID"
_SESSION_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{32,128}$"
)


def require_session_id(
    x_docqna_session_id: str | None = Header(
        default=None,
        alias=SESSION_HEADER_NAME,
    ),
) -> str:
    """Validate the anonymous high-entropy session namespace."""

    if x_docqna_session_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Missing required {SESSION_HEADER_NAME} header."
            ),
        )

    session_id = x_docqna_session_id.strip()

    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid {SESSION_HEADER_NAME} header."
            ),
        )

    return session_id


@asynccontextmanager
async def lifespan(
    _: FastAPI,
):
    """Log lightweight application startup/shutdown diagnostics."""

    settings = get_settings()

    logger.info(
        (
            "app_start app=%s version=%s env=%s "
            "llm_provider=%s llm_model=%s "
            "embedding_model=%s embedding_device=%s"
        ),
        APP_NAME,
        APP_VERSION,
        settings.app_env,
        settings.llm_provider,
        settings.ollama_model,
        settings.embedding_model,
        settings.embedding_device,
    )

    try:
        yield

    finally:
        logger.info(
            "app_stop app=%s version=%s",
            APP_NAME,
            APP_VERSION,
        )


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Document question-answering API "
        "with hybrid retrieval, reranking, "
        "citations, confidence-based abstention, "
        "and managed document indexing."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_observability(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """
    Add request IDs, latency logging, and basic security
    headers without logging request bodies or query text.
    """

    request_id = uuid4().hex
    token = set_request_id(
        request_id
    )

    started = perf_counter()
    response: Response | None = None
    status_code = (
        status.HTTP_500_INTERNAL_SERVER_ERROR
    )

    try:
        response = await call_next(
            request
        )

        status_code = response.status_code

        response.headers[
            "X-Request-ID"
        ] = request_id

        response.headers[
            "X-Content-Type-Options"
        ] = "nosniff"

        response.headers[
            "Referrer-Policy"
        ] = "no-referrer"

        return response

    except Exception:
        logger.exception(
            (
                "http_request_unhandled method=%s "
                "path=%s"
            ),
            request.method,
            request.url.path,
        )
        raise

    finally:
        logger.info(
            (
                "http_request method=%s path=%s "
                "status=%d duration_ms=%.2f"
            ),
            request.method,
            request.url.path,
            status_code,
            (
                perf_counter() - started
            ) * 1000.0,
        )

        reset_request_id(
            token
        )


def _log_handled_error(
    request: Request,
    exc: Exception,
    *,
    error_code: str,
    level: int,
) -> None:
    """
    Log handled API errors without logging exception
    messages, request bodies, queries, or document text.
    """

    logger.log(
        level,
        (
            "handled_error code=%s exception=%s "
            "method=%s path=%s"
        ),
        error_code,
        type(exc).__name__,
        request.method,
        request.url.path,
    )


# ---------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------


@app.exception_handler(DuplicateDocumentError)
async def duplicate_document_error_handler(
    request: Request,
    exc: DuplicateDocumentError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="duplicate_document",
        level=logging.WARNING,
    )

    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "error": "duplicate_document",
            "detail": str(exc),
        },
    )


@app.exception_handler(UnsupportedFileTypeError)
async def unsupported_file_type_error_handler(
    request: Request,
    exc: UnsupportedFileTypeError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="unsupported_file_type",
        level=logging.WARNING,
    )

    return JSONResponse(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        content={
            "error": "unsupported_file_type",
            "detail": str(exc),
        },
    )


@app.exception_handler(DocumentValidationError)
async def document_validation_error_handler(
    request: Request,
    exc: DocumentValidationError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="document_validation_error",
        level=logging.WARNING,
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "document_validation_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(DocumentLoadError)
async def document_load_error_handler(
    request: Request,
    exc: DocumentLoadError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="document_load_error",
        level=logging.WARNING,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "document_load_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(LLMConnectionError)
async def llm_connection_error_handler(
    request: Request,
    exc: LLMConnectionError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="llm_connection_error",
        level=logging.ERROR,
    )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "llm_connection_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(LLMModelError)
async def llm_model_error_handler(
    request: Request,
    exc: LLMModelError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="llm_model_error",
        level=logging.ERROR,
    )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "llm_model_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(
    request: Request,
    exc: RetrievalError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="retrieval_error",
        level=logging.ERROR,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "retrieval_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(CitationError)
async def citation_error_handler(
    request: Request,
    exc: CitationError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="citation_error",
        level=logging.ERROR,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "citation_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(GenerationError)
async def generation_error_handler(
    request: Request,
    exc: GenerationError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="generation_error",
        level=logging.ERROR,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "generation_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(
    request: Request,
    exc: EmbeddingError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="embedding_error",
        level=logging.ERROR,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "embedding_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(IndexingError)
async def indexing_error_handler(
    request: Request,
    exc: IndexingError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="indexing_error",
        level=logging.ERROR,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "indexing_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(
    request: Request,
    exc: ConfigurationError,
) -> JSONResponse:
    _log_handled_error(
        request,
        exc,
        error_code="configuration_error",
        level=logging.ERROR,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "configuration_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(DocQNAError)
async def docqna_error_handler(
    request: Request,
    exc: DocQNAError,
) -> JSONResponse:
    """
    Fallback handler for any DocQ&A-specific exception
    not handled more specifically above.
    """

    _log_handled_error(
        request,
        exc,
        error_code="docqna_error",
        level=logging.ERROR,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "docqna_error",
            "detail": str(exc),
        },
    )


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
def health_check() -> HealthResponse:
    """
    Return basic API health information.

    This endpoint intentionally does not load the
    embedding model, reranker, Chroma corpus, or LLM.
    """

    return HealthResponse(
        status="ok",
        app=APP_NAME,
        version=APP_VERSION,
    )


@app.post(
    "/qa",
    response_model=QAResponse,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "DocQ&A processing error",
        },
        503: {
            "model": ErrorResponse,
            "description": "LLM service unavailable",
        },
    },
    tags=["qa"],
)
def ask_question(
    request: QARequest,
    session_id: str = Depends(
        require_session_id
    ),
    pipeline_provider: Callable[
        [], RAGPipeline
    ] = Depends(
        get_rag_pipeline_provider
    ),
) -> QAResponse:
    """
    Answer a question using the production RAG pipeline.
    """

    pipeline = pipeline_provider()

    metadata_filters = (
        request.filters.to_retrieval_filters()
        if request.filters is not None
        else {}
    )

    # Server-enforced ownership filter. The client cannot choose
    # a different session through the request body.
    metadata_filters["session_id"] = session_id

    response = pipeline.ask_with_citations(
        query=request.question,
        top_k=request.top_k,
        metadata_filters=metadata_filters,
    )

    abstained = (
        not response.sources
        and not response.cited_source_ids
    )

    cited_ids = set(
        response.cited_source_ids
    )

    sources = [
        CitationResponse.model_validate(
            source.model_dump()
        )
        for source in response.sources
        if source.citation_id in cited_ids
    ]

    logger.info(
        (
            "qa_response abstained=%s citations=%d "
            "top_k=%d filter_keys=%s"
        ),
        abstained,
        len(response.cited_source_ids),
        request.top_k,
        (
            ",".join(
                sorted(metadata_filters)
            )
            if metadata_filters
            else "-"
        ),
    )

    return QAResponse(
        answer=response.answer,
        abstained=abstained,
        cited_source_ids=(
            response.cited_source_ids
        ),
        sources=sources,
    )


@app.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid document",
        },
        409: {
            "model": ErrorResponse,
            "description": "Duplicate document",
        },
        415: {
            "model": ErrorResponse,
            "description": "Unsupported document type",
        },
        413: {
            "model": ErrorResponse,
            "description": "Uploaded document is too large",
        },
        422: {
            "model": ErrorResponse,
            "description": "Document could not be processed",
        },
        500: {
            "model": ErrorResponse,
            "description": "Indexing error",
        },
    },
    tags=["documents"],
)
def upload_document(
    file: UploadFile = File(...),
    session_id: str = Depends(
        require_session_id
    ),
    service_provider: Callable[
        [], DocumentIndexService
    ] = Depends(
        get_document_index_service_provider
    ),
) -> DocumentUploadResponse:
    """
    Upload, process, embed, and index one supported document.
    """

    filename = Path(
        file.filename or ""
    ).name

    if not filename:
        raise DocumentValidationError(
            "Uploaded document must have a filename."
        )

    extension = (
        Path(filename).suffix.lower()
    )

    if extension not in SUPPORTED_FILE_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported document type: '{extension or '<none>'}'."
        )

    settings = get_settings()
    max_upload_bytes = (
        settings.max_upload_size_mb
        * 1024
        * 1024
    )

    if (
        file.size is not None
        and file.size > max_upload_bytes
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            ),
            detail=(
                "Uploaded document exceeds the "
                f"{settings.max_upload_size_mb} MB "
                "file-size limit."
            ),
        )

    with TemporaryDirectory(
        ignore_cleanup_errors=True,
    ) as temp_dir:
        temp_path = (
            Path(temp_dir) / filename
        )

        try:
            file.file.seek(0)
            total_bytes = 0
            read_size = 1024 * 1024

            with temp_path.open("wb") as target:
                while True:
                    chunk = file.file.read(
                        read_size
                    )

                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    if total_bytes > max_upload_bytes:
                        raise HTTPException(
                            status_code=(
                                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                            ),
                            detail=(
                                "Uploaded document exceeds the "
                                f"{settings.max_upload_size_mb} MB "
                                "file-size limit."
                            ),
                        )

                    target.write(chunk)

        except HTTPException:
            raise

        except OSError as exc:
            raise DocumentLoadError(
                "Could not read the uploaded document."
            ) from exc

        if temp_path.stat().st_size <= 0:
            raise DocumentValidationError(
                "Uploaded document is empty."
            )

        service = service_provider()

        result = service.index_document(
            temp_path,
            session_id=session_id,
        )

    clear_rag_pipeline_cache()

    logger.info(
        (
            "document_upload_response filename=%s "
            "document_id=%s version=%d chunks=%d "
            "size_bytes=%d"
        ),
        result.record.filename,
        result.record.document_id,
        result.record.version,
        result.chunk_count,
        result.record.size_bytes,
    )

    return DocumentUploadResponse(
        document=DocumentResponse.model_validate(
            result.record.model_dump()
        ),
        chunk_count=result.chunk_count,
    )


@app.get(
    "/documents",
    response_model=DocumentListResponse,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Document registry error",
        },
    },
    tags=["documents"],
)
def list_documents(
    session_id: str = Depends(
        require_session_id
    ),
    service_provider: Callable[
        [], DocumentIndexService
    ] = Depends(
        get_document_index_service_provider
    ),
) -> DocumentListResponse:
    """
    Return the latest active version of every managed
    logical document.
    """

    service = service_provider()

    records = service.list_documents(
        session_id=session_id
    )

    documents = [
        DocumentResponse.model_validate(
            record.model_dump()
        )
        for record in records
    ]

    logger.info(
        "document_list count=%d",
        len(documents),
    )

    return DocumentListResponse(
        count=len(documents),
        documents=documents,
    )


@app.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Document not found",
        },
        500: {
            "model": ErrorResponse,
            "description": "Document deletion error",
        },
    },
    tags=["documents"],
)
def delete_document(
    document_id: str,
    session_id: str = Depends(
        require_session_id
    ),
    service_provider: Callable[
        [], DocumentIndexService
    ] = Depends(
        get_document_index_service_provider
    ),
) -> DocumentDeleteResponse:
    """
    Delete a logical document from Chroma and the
    persistent document registry.
    """

    service = service_provider()

    deleted = service.delete_document(
        document_id,
        session_id=session_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Document '{document_id}' "
                "was not found."
            ),
        )

    clear_rag_pipeline_cache()

    logger.info(
        "document_delete_response document_id=%s",
        document_id,
    )

    return DocumentDeleteResponse(
        document_id=document_id,
        deleted=True,
    )
