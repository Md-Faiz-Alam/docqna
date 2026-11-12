import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from docqna.api.dependencies import get_document_index_service_provider, get_rag_pipeline_provider
from docqna.api.errors import resolve_error_spec
from docqna.api.retention import purge_expired_documents
from docqna.api.schemas import CitationResponse, ErrorResponse, QARequest, QAResponse
from docqna.api.session import require_session_id
from docqna.exceptions import DocQNAError
from docqna.filters import MetadataFilters
from docqna.indexing import DocumentIndexService
from docqna.observability import record_qa_outcome
from docqna.rag import RAGPipeline, RAGResponse
from docqna.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["qa"])

ServiceProvider = Callable[[], DocumentIndexService]
PipelineProvider = Callable[[], RAGPipeline]

_QA_ERRORS: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse, "description": "Invalid question"},
    500: {"model": ErrorResponse, "description": "DocQ&A processing error"},
    503: {"model": ErrorResponse, "description": "LLM service unavailable"},
}


def _prepare(
    request: QARequest,
    session_id: str,
    service_provider: ServiceProvider,
    pipeline_provider: PipelineProvider,
) -> tuple[RAGPipeline, MetadataFilters]:
    settings = get_settings()
    if len(request.question) > settings.max_question_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Question exceeds the {settings.max_question_chars}-character limit.",
        )

    purge_expired_documents(service_provider())
    pipeline = pipeline_provider()

    metadata_filters: MetadataFilters = (
        request.filters.to_retrieval_filters() if request.filters is not None else {}
    )
    # Server-enforced ownership filter: the client cannot pick another session.
    metadata_filters["session_id"] = session_id
    return pipeline, metadata_filters


def _history(request: QARequest) -> list[tuple[str, str]] | None:
    return [(turn.role, turn.content) for turn in request.history] or None


def _to_qa_response(response: RAGResponse, *, top_k: int, filter_keys: str) -> QAResponse:
    abstained = not response.sources and not response.cited_source_ids
    cited_ids = set(response.cited_source_ids)
    sources = [
        CitationResponse.model_validate(source.model_dump())
        for source in response.sources
        if source.citation_id in cited_ids
    ]

    record_qa_outcome(abstained=abstained)
    logger.info(
        "qa_response abstained=%s citations=%d top_k=%d filter_keys=%s",
        abstained,
        len(response.cited_source_ids),
        top_k,
        filter_keys,
    )

    return QAResponse(
        answer=response.answer,
        abstained=abstained,
        cited_source_ids=response.cited_source_ids,
        sources=sources,
        standalone_question=response.standalone_question,
    )


@router.post("/qa", response_model=QAResponse, responses=_QA_ERRORS)
def ask_question(
    request: QARequest,
    session_id: str = Depends(require_session_id),
    service_provider: ServiceProvider = Depends(get_document_index_service_provider),
    pipeline_provider: PipelineProvider = Depends(get_rag_pipeline_provider),
) -> QAResponse:
    """Answer a question from the caller's documents with verified citations."""

    pipeline, metadata_filters = _prepare(request, session_id, service_provider, pipeline_provider)
    response = pipeline.ask_with_citations(
        query=request.question,
        top_k=request.top_k,
        metadata_filters=metadata_filters,
        history=_history(request),
    )
    return _to_qa_response(
        response, top_k=request.top_k, filter_keys=",".join(sorted(metadata_filters))
    )


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/qa/stream",
    response_class=StreamingResponse,
    responses={
        **_QA_ERRORS,
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "Server-sent events: `token` ({text}) while generating, then exactly one "
                "`final` (QAResponse) or `error` ({error, detail}). The `final` answer is "
                "authoritative and replaces the streamed draft."
            ),
        },
    },
)
def ask_question_stream(
    request: QARequest,
    session_id: str = Depends(require_session_id),
    service_provider: ServiceProvider = Depends(get_document_index_service_provider),
    pipeline_provider: PipelineProvider = Depends(get_rag_pipeline_provider),
) -> StreamingResponse:
    """Stream the answer as it is generated (SSE), then send the verified result."""

    # Validation, quota/retention, and LLM availability errors happen here,
    # before streaming starts, so they return normal JSON error responses.
    pipeline, metadata_filters = _prepare(request, session_id, service_provider, pipeline_provider)
    filter_keys = ",".join(sorted(metadata_filters))

    def events() -> Iterator[str]:
        try:
            for item in pipeline.stream_with_citations(
                query=request.question,
                top_k=request.top_k,
                metadata_filters=metadata_filters,
                history=_history(request),
            ):
                if isinstance(item, str):
                    yield _sse("token", {"text": item})
                else:
                    final = _to_qa_response(item, top_k=request.top_k, filter_keys=filter_keys)
                    yield _sse("final", final.model_dump())
        except DocQNAError as exc:
            spec = resolve_error_spec(exc)
            logger.log(spec.log_level, "qa_stream_error code=%s", spec.code)
            yield _sse("error", {"error": spec.code, "detail": str(exc)})
        except Exception:
            logger.exception("qa_stream_unhandled_error")
            yield _sse("error", {"error": "internal_error", "detail": "Unexpected server error."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
