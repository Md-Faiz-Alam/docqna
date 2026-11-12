import logging
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from docqna.api.dependencies import clear_rag_pipeline_cache, get_document_index_service_provider
from docqna.api.retention import purge_expired_documents
from docqna.api.schemas import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    ErrorResponse,
)
from docqna.api.session import require_session_id
from docqna.constants import SUPPORTED_FILE_EXTENSIONS
from docqna.exceptions import DocumentLoadError, DocumentValidationError, UnsupportedFileTypeError
from docqna.indexing import DocumentIndexService
from docqna.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

ServiceProvider = Callable[[], DocumentIndexService]

_READ_CHUNK_BYTES = 1024 * 1024


def _too_large(limit_mb: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"Uploaded document exceeds the {limit_mb} MB file-size limit.",
    )


def _save_upload(file: UploadFile, target: Path, max_bytes: int, limit_mb: int) -> None:
    """Stream the upload to disk, enforcing the size limit as bytes arrive."""

    try:
        file.file.seek(0)
        total_bytes = 0
        with target.open("wb") as out:
            while chunk := file.file.read(_READ_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise _too_large(limit_mb)
                out.write(chunk)
    except OSError as exc:
        raise DocumentLoadError("Could not read the uploaded document.") from exc

    if target.stat().st_size <= 0:
        raise DocumentValidationError("Uploaded document is empty.")


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid document"},
        409: {"model": ErrorResponse, "description": "Duplicate document"},
        413: {"model": ErrorResponse, "description": "Uploaded document is too large"},
        415: {"model": ErrorResponse, "description": "Unsupported document type"},
        422: {"model": ErrorResponse, "description": "Document could not be processed"},
        429: {"model": ErrorResponse, "description": "Resource limit reached"},
        500: {"model": ErrorResponse, "description": "Indexing error"},
    },
)
def upload_document(
    file: UploadFile = File(...),
    session_id: str = Depends(require_session_id),
    service_provider: ServiceProvider = Depends(get_document_index_service_provider),
) -> DocumentUploadResponse:
    """Upload, process, embed, and index one supported document."""

    filename = Path(file.filename or "").name
    if not filename:
        raise DocumentValidationError("Uploaded document must have a filename.")

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_FILE_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported document type: '{extension or '<none>'}'.")

    settings = get_settings()
    limit_mb = settings.max_upload_size_mb
    max_bytes = limit_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise _too_large(limit_mb)

    service = service_provider()
    purge_expired_documents(service)

    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir) / filename
        _save_upload(file, temp_path, max_bytes, limit_mb)
        result = service.index_document(temp_path, session_id=session_id)

    clear_rag_pipeline_cache()

    logger.info(
        "document_upload_response filename=%s document_id=%s version=%d chunks=%d size_bytes=%d",
        result.record.filename,
        result.record.document_id,
        result.record.version,
        result.chunk_count,
        result.record.size_bytes,
    )

    return DocumentUploadResponse(
        document=DocumentResponse.model_validate(result.record.model_dump()),
        chunk_count=result.chunk_count,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    responses={500: {"model": ErrorResponse, "description": "Document registry error"}},
)
def list_documents(
    session_id: str = Depends(require_session_id),
    service_provider: ServiceProvider = Depends(get_document_index_service_provider),
) -> DocumentListResponse:
    """Return the latest active version of each of the caller's documents."""

    service = service_provider()
    purge_expired_documents(service)

    documents = [
        DocumentResponse.model_validate(record.model_dump())
        for record in service.list_documents(session_id=session_id)
    ]
    logger.info("document_list count=%d", len(documents))

    return DocumentListResponse(count=len(documents), documents=documents)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Document deletion error"},
    },
)
def delete_document(
    document_id: str,
    session_id: str = Depends(require_session_id),
    service_provider: ServiceProvider = Depends(get_document_index_service_provider),
) -> DocumentDeleteResponse:
    """Delete a document (all versions) from Chroma and the registry."""

    service = service_provider()
    purge_expired_documents(service)

    if not service.delete_document(document_id, session_id=session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' was not found."
        )

    clear_rag_pipeline_cache()
    logger.info("document_delete_response document_id=%s", document_id)

    return DocumentDeleteResponse(document_id=document_id, deleted=True)
