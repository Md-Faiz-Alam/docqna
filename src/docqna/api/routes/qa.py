import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, status

from docqna.api.dependencies import get_document_index_service_provider, get_rag_pipeline_provider
from docqna.api.retention import purge_expired_documents
from docqna.api.schemas import CitationResponse, ErrorResponse, QARequest, QAResponse
from docqna.api.session import require_session_id
from docqna.indexing import DocumentIndexService
from docqna.rag import RAGPipeline
from docqna.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["qa"])


@router.post(
    "/qa",
    response_model=QAResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid question"},
        500: {"model": ErrorResponse, "description": "DocQ&A processing error"},
        503: {"model": ErrorResponse, "description": "LLM service unavailable"},
    },
)
def ask_question(
    request: QARequest,
    session_id: str = Depends(require_session_id),
    service_provider: Callable[[], DocumentIndexService] = Depends(
        get_document_index_service_provider
    ),
    pipeline_provider: Callable[[], RAGPipeline] = Depends(get_rag_pipeline_provider),
) -> QAResponse:
    """Answer a question from the caller's documents with verified citations."""

    settings = get_settings()
    if len(request.question) > settings.max_question_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Question exceeds the {settings.max_question_chars}-character limit.",
        )

    purge_expired_documents(service_provider())
    pipeline = pipeline_provider()

    metadata_filters = request.filters.to_retrieval_filters() if request.filters is not None else {}
    # Server-enforced ownership filter: the client cannot pick another session.
    metadata_filters["session_id"] = session_id

    response = pipeline.ask_with_citations(
        query=request.question, top_k=request.top_k, metadata_filters=metadata_filters
    )

    abstained = not response.sources and not response.cited_source_ids
    cited_ids = set(response.cited_source_ids)
    sources = [
        CitationResponse.model_validate(source.model_dump())
        for source in response.sources
        if source.citation_id in cited_ids
    ]

    logger.info(
        "qa_response abstained=%s citations=%d top_k=%d filter_keys=%s",
        abstained,
        len(response.cited_source_ids),
        request.top_k,
        ",".join(sorted(metadata_filters)) or "-",
    )

    return QAResponse(
        answer=response.answer,
        abstained=abstained,
        cited_source_ids=response.cited_source_ids,
        sources=sources,
    )
