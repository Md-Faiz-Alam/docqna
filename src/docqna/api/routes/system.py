from fastapi import APIRouter

from docqna.api.schemas import HealthResponse
from docqna.constants import APP_NAME, APP_VERSION

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness check. Deliberately loads no models, index, or LLM."""

    return HealthResponse(status="ok", app=APP_NAME, version=APP_VERSION)
