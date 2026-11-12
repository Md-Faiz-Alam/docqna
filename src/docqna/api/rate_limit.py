"""In-process sliding-window rate limiting for expensive endpoints.

This protects a single API process (e.g. a small VM or free-tier host)
from bursts. Multi-instance deployments should also limit at the reverse
proxy, since each process keeps its own counters.
"""

import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Lock

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from docqna.api.session import SESSION_HEADER_NAME
from docqna.observability import RATE_LIMITED


@dataclass(frozen=True)
class RateLimitRule:
    bucket: str
    methods: frozenset[str]
    path_prefix: str
    limit: int
    window_seconds: float = 60.0

    def matches(self, method: str, path: str) -> bool:
        return method in self.methods and path.startswith(self.path_prefix)


class SlidingWindowLimiter:
    """Allow at most `limit` hits per key within `window_seconds`."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._lock = Lock()

    def hit(self, rule: RateLimitRule, key: str) -> float | None:
        """Record a hit. Returns None if allowed, else seconds until retry."""

        now = self._clock()
        window_start = now - rule.window_seconds
        with self._lock:
            hits = self._hits.setdefault((rule.bucket, key), deque())
            while hits and hits[0] <= window_start:
                hits.popleft()
            if len(hits) >= rule.limit:
                return max(0.0, hits[0] + rule.window_seconds - now)
            hits.append(now)
            if len(self._hits) > 50_000:
                self._evict_idle(window_start)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def _evict_idle(self, window_start: float) -> None:
        idle = [k for k, hits in self._hits.items() if not hits or hits[-1] <= window_start]
        for key in idle:
            del self._hits[key]


def _client_key(request: Request) -> str:
    # The session header is the per-browser namespace; fall back to IP.
    session_id = request.headers.get(SESSION_HEADER_NAME)
    if session_id:
        return f"s:{session_id.strip()[:128]}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def make_rate_limit_middleware(
    rules: list[RateLimitRule], limiter: SlidingWindowLimiter | None = None
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    limiter = limiter or SlidingWindowLimiter()
    active_rules = [rule for rule in rules if rule.limit > 0]

    async def rate_limit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        for rule in active_rules:
            if not rule.matches(request.method, request.url.path):
                continue
            retry_after = limiter.hit(rule, _client_key(request))
            if retry_after is not None:
                RATE_LIMITED.labels(bucket=rule.bucket).inc()
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "rate_limited",
                        "detail": f"Too many {rule.bucket} requests. Please retry shortly.",
                    },
                    headers={"Retry-After": str(max(1, round(retry_after)))},
                )
        return await call_next(request)

    return rate_limit
