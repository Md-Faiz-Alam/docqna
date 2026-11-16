"""In-process sliding-window rate limiting for expensive endpoints.

This protects a single API process (e.g. a small VM or free-tier host)
from bursts. Multi-instance deployments should also limit at the reverse
proxy, since each process keeps its own counters.
"""

import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
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


def client_ip(request: Request, trusted_proxy_hops: int = 0) -> str:
    """The caller's IP address, as far as it can be trusted.

    With no trusted proxy, this is the socket peer address. Behind
    `trusted_proxy_hops` proxies that each append to X-Forwarded-For, it is
    the entry the outermost trusted proxy appended, counted from the right.
    Entries further left were sent by the client and can be anything, so
    they are never used (uvicorn's `--forwarded-allow-ips='*'` takes the
    leftmost one, which a client can fake).
    """

    peer = request.client.host if request.client else "unknown"
    if trusted_proxy_hops <= 0:
        return peer

    forwarded = [
        host.strip()
        for header in request.headers.getlist("x-forwarded-for")
        for host in header.split(",")
        if host.strip()
    ]
    if not forwarded:
        return peer
    return forwarded[-min(trusted_proxy_hops, len(forwarded))]


def _client_keys(request: Request, trusted_proxy_hops: int) -> tuple[str, str | None]:
    """(IP key, session key or None). The session ID is chosen by the client."""

    session_id = request.headers.get(SESSION_HEADER_NAME, "").strip()
    session_key = f"s:{session_id[:128]}" if session_id else None
    return f"ip:{client_ip(request, trusted_proxy_hops)}", session_key


def make_rate_limit_middleware(
    rules: list[RateLimitRule],
    limiter: SlidingWindowLimiter | None = None,
    *,
    ip_limit_factor: int = 3,
    trusted_proxy_hops: int = 0,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Limit each session to `rule.limit`, and each IP to `rule.limit * ip_limit_factor`.

    The session limit keeps one browser from using up a shared quota. The
    IP limit stops a script that sends a fresh session ID with every
    request; it is larger so that several people behind one address (an
    office or mobile network) can still use the service. A factor of 0
    turns the IP limit off.
    """

    limiter = limiter or SlidingWindowLimiter()
    active_rules = [rule for rule in rules if rule.limit > 0]
    ip_rules = {
        rule.bucket: replace(rule, bucket=f"{rule.bucket}:ip", limit=rule.limit * ip_limit_factor)
        for rule in active_rules
        if ip_limit_factor > 0
    }

    async def rate_limit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        for rule in active_rules:
            if not rule.matches(request.method, request.url.path):
                continue
            ip_key, session_key = _client_keys(request, trusted_proxy_hops)
            checks: list[tuple[RateLimitRule, str]] = []
            if rule.bucket in ip_rules:
                checks.append((ip_rules[rule.bucket], ip_key))
            # Without a session header, the IP gets the per-session limit too.
            checks.append((rule, session_key or ip_key))

            retry_after = None
            for check_rule, key in checks:
                retry_after = limiter.hit(check_rule, key)
                if retry_after is not None:
                    break
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
