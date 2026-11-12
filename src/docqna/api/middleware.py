"""HTTP middleware: request IDs, latency logging, and security headers."""

import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response, status

from docqna.logging_config import reset_request_id, set_request_id
from docqna.observability import record_http_request

logger = logging.getLogger(__name__)

SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}


async def request_observability(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Tag each request with an ID and log method/path/status/latency.

    Request bodies and query text are never logged.
    """

    request_id = uuid4().hex
    token = set_request_id(request_id)
    started = perf_counter()
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        response.headers.update(SECURITY_HEADERS)
        return response

    except Exception:
        logger.exception(
            "http_request_unhandled method=%s path=%s", request.method, request.url.path
        )
        raise

    finally:
        elapsed = perf_counter() - started
        route = request.scope.get("route")
        record_http_request(
            method=request.method,
            route=getattr(route, "path", "unmatched"),
            status=status_code,
            seconds=elapsed,
        )
        logger.info(
            "http_request method=%s path=%s status=%d duration_ms=%.2f",
            request.method,
            request.url.path,
            status_code,
            elapsed * 1000.0,
        )
        reset_request_id(token)
