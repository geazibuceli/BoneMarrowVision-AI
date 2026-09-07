from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import HTTPException, Request, status

from app.core.config import settings


class InMemoryRateLimiter:
    def __init__(self, *, max_keys: int = 10_000) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def allow(self, key: str, *, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            if key not in self._events and len(self._events) >= self._max_keys:
                self._events.pop(next(iter(self._events)))
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                retry_after = max(1, int(events[0] + window_seconds - now))
                return False, retry_after
            events.append(now)
            return True, 0

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


rate_limiter = InMemoryRateLimiter()


def _client_key(request: Request, scope: str) -> str:
    client_host = request.client.host if request.client else "unknown"
    if settings.trust_proxy_headers:
        client_host = request.headers.get("x-real-ip", client_host)
    return f"{scope}:{client_host}"


def _dependency(
    scope: str,
    limit: Callable[[], int],
    window_seconds: Callable[[], int],
):
    def enforce(request: Request) -> None:
        allowed, retry_after = rate_limiter.allow(
            _client_key(request, scope),
            limit=limit(),
            window_seconds=window_seconds(),
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests",
                headers={"Retry-After": str(retry_after)},
            )

    return enforce


enforce_auth_rate_limit = _dependency(
    "auth",
    lambda: settings.auth_rate_limit,
    lambda: settings.auth_rate_window_seconds,
)
enforce_prediction_rate_limit = _dependency(
    "prediction",
    lambda: settings.prediction_rate_limit,
    lambda: settings.prediction_rate_window_seconds,
)