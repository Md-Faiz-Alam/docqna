import logging
from collections.abc import Callable

from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse

from docqna.api.dependencies import get_document_registry, get_llm, get_vector_store
from docqna.api.schemas import HealthResponse
from docqna.constants import APP_NAME, APP_VERSION
from docqna.observability import render_metrics

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness check. Deliberately loads no models, index, or LLM."""

    return HealthResponse(status="ok", app=APP_NAME, version=APP_VERSION)


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
