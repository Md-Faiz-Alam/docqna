import logging
from collections.abc import Callable

from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse

from docqna.api.dependencies import get_document_registry, get_llm, get_vector_store
from docqna.api.samples import sample_document_paths
from docqna.api.schemas import HealthResponse, InfoLimits, InfoResponse
from docqna.constants import APP_NAME, APP_VERSION, SUPPORTED_FILE_TYPES
from docqna.observability import render_metrics
from docqna.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Tell a person who opens the API address in a browser where to go."""

    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "message": "This is the DocQ&A API. The website is a separate app (see web/).",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
    }


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness check. Deliberately loads no models, index, or LLM."""

    return HealthResponse(status="ok", app=APP_NAME, version=APP_VERSION)


@router.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Which models and limits this deployment uses. Contains no secrets.

    Like /health, it loads no models and calls no LLM.
    """

    settings = get_settings()
    return InfoResponse(
        app=APP_NAME,
        version=APP_VERSION,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        external_llm=settings.llm_provider != "ollama",
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        sample_documents=[p.name for p in sample_document_paths(settings.sample_documents_dir)],
        limits=InfoLimits(
            max_upload_size_mb=settings.max_upload_size_mb,
            max_question_chars=settings.max_question_chars,
            max_documents_per_session=settings.max_documents_per_session,
            document_retention_hours=settings.document_retention_hours,
            supported_file_types=sorted(SUPPORTED_FILE_TYPES),
        ),
    )


def _check(name: str, probe: Callable[[], object]) -> bool:
    try:
        return bool(probe())
    except Exception:
        logger.warning("readiness_check_failed check=%s", name, exc_info=True)
        return False


@router.get("/ready", responses={503: {"description": "A dependency is unavailable"}})
def readiness_check() -> JSONResponse:
    """Readiness check: vector store, document registry, and LLM are reachable.

    Use this (not /health) for load-balancer or orchestrator readiness.
    """

    checks = {
        "vector_store": _check("vector_store", lambda: get_vector_store().count() >= 0),
        "registry": _check(
            "registry",
            lambda: get_document_registry().list_records(session_id="__readiness_probe__")
            is not None,
        ),
        "llm": _check("llm", lambda: get_llm().health_check()),
    }
    ready = all(checks.values())

    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus metrics in text exposition format."""

    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
