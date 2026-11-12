"""FastAPI application entry point: `uvicorn docqna.api.main:app`."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from docqna.api.errors import register_exception_handlers
from docqna.api.middleware import request_observability
from docqna.api.routes import documents, qa, system
from docqna.api.session import SESSION_HEADER_NAME, require_session_id
from docqna.constants import APP_NAME, APP_VERSION
from docqna.logging_config import configure_logging
from docqna.settings import get_settings

__all__ = ["SESSION_HEADER_NAME", "app", "create_app", "require_session_id"]

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "app_start app=%s version=%s env=%s llm_provider=%s llm_model=%s "
        "embedding_model=%s embedding_device=%s",
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
        logger.info("app_stop app=%s version=%s", APP_NAME, APP_VERSION)


def create_app() -> FastAPI:
    application = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=(
            "Document question-answering API with hybrid retrieval, reranking, "
            "verified citations, confidence-based abstention, and managed "
            "document indexing."
        ),
        lifespan=lifespan,
    )
    application.middleware("http")(request_observability)
    register_exception_handlers(application)

    application.include_router(system.router)
    application.include_router(qa.router)
    application.include_router(documents.router)
    return application


app = create_app()
