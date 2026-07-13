"""Security middleware: hardening headers and a lightweight rate limiter.

The rate limiter is in-process and per-instance; it is a sane baseline for a
single container. For horizontally-scaled production, back it with Redis.
"""

import os
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
IS_PRODUCTION = os.environ.get("ENV", "development").lower() in ("production", "prod")

# Paths exempt from rate limiting (health checks, docs).
_EXEMPT_PREFIXES = ("/health", "/status", "/docs", "/redoc", "/openapi.json")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach standard hardening headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        if IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window-ish sliding limiter keyed on client IP."""

    def __init__(self, app, per_minute: int = RATE_LIMIT_PER_MINUTE):
        super().__init__(app)
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)

    def _client_ip(self, request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_developer_request(self, request: Request) -> bool:
        """Developer traffic is exempt. The `dev` claim is baked into the JWT
        at issue time and the decode verifies the signature, so this costs no
        database round-trip and cannot be forged."""
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return False
        from api.auth import decode_token
        payload = decode_token(auth[7:].strip())
        return bool(payload and payload.get("dev"))

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS" or path.startswith(_EXEMPT_PREFIXES):
            return await call_next(request)
        if self._is_developer_request(request):
            return await call_next(request)

        now = time.time()
        window_start = now - 60
        ip = self._client_ip(request)
        hits = self._hits[ip]

        while hits and hits[0] < window_start:
            hits.popleft()

        if len(hits) >= self.per_minute:
            retry = int(60 - (now - hits[0])) + 1
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Slow down and retry shortly."},
                headers={"Retry-After": str(retry)},
            )

        hits.append(now)
        return await call_next(request)
