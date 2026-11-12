"""FastAPI application entry point: `uvicorn docqna.api.main:app`."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from docqna.api.errors import register_exception_handlers
from docqna.api.middleware import request_observability
from docqna.api.rate_limit import RateLimitRule, SlidingWindowLimiter, make_rate_limit_middleware
from docqna.api.routes import documents, feedback, qa, system
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
    settings = get_settings()
    application.state.rate_limiter = SlidingWindowLimiter()
    # Registered first so it runs inside request_observability: rejected
    # requests still get request IDs, logs, and metrics.
    application.middleware("http")(
        make_rate_limit_middleware(
            [
                RateLimitRule(
                    bucket="question",
                    methods=frozenset({"POST"}),
                    path_prefix="/qa",
                    limit=settings.rate_limit_qa_per_minute,
                ),
                RateLimitRule(
                    bucket="upload",
                    methods=frozenset({"POST"}),
                    path_prefix="/documents",
                    limit=settings.rate_limit_uploads_per_minute,
                ),
                RateLimitRule(
                    bucket="feedback",
                    methods=frozenset({"POST"}),
                    path_prefix="/feedback",
                    limit=settings.rate_limit_feedback_per_minute,
                ),
            ],
            limiter=application.state.rate_limiter,
        )
    )
    application.middleware("http")(request_observability)
    register_exception_handlers(application)

    application.include_router(system.router)
    application.include_router(qa.router)
    application.include_router(documents.router)
    application.include_router(feedback.router)
    return application


app = create_app()
