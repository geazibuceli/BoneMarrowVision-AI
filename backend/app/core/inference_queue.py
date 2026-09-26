"""Bound single-process admission without occupying worker threads while waiting."""

import asyncio
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

from fastapi import HTTPException

from app.core.config import settings

LOGGER = logging.getLogger(__name__)


class InferenceQueue:
    def __init__(self):
        self.lock = threading.Lock()
        self.waiters = deque()
        self.active = False
        self.completed = 0
        self.failed = 0
        self.rejected = 0
        self.total_seconds = 0.0

    def snapshot(self):
        with self.lock:
            return {
                "active": self.active, "waiting": len(self.waiters),
                "capacity": settings.inference_queue_size, "completed": self.completed,
                "failed": self.failed, "rejected": self.rejected,
                "total_processing_seconds": round(self.total_seconds, 3),
            }

    @asynccontextmanager
    async def admit(self):
        ticket = object()
        started = time.monotonic()
        with self.lock:
            if len(self.waiters) >= settings.inference_queue_size:
                self.rejected += 1
                raise HTTPException(503, "Analysis queue is full. Please try again shortly.",
                                    headers={"Retry-After": "5"})
            self.waiters.append(ticket)
        acquired = False
        try:
            while not acquired:
                with self.lock:
                    if not self.active and self.waiters[0] is ticket:
                        self.waiters.popleft()
                        self.active = acquired = True
                if not acquired:
                    if time.monotonic() - started >= settings.inference_queue_timeout_seconds:
                        with self.lock:
                            self.rejected += 1
                        raise HTTPException(503, "Analysis queue wait expired. Please try again shortly.",
                                            headers={"Retry-After": "5"})
                    await asyncio.sleep(0.05)
            running = time.monotonic()
            succeeded = False
            try:
                yield
                succeeded = True
            finally:
                elapsed = time.monotonic() - running
                with self.lock:
                    self.completed += int(succeeded)
                    self.failed += int(not succeeded)
                    self.total_seconds += elapsed
                LOGGER.info("Analysis request finished success=%s wait_seconds=%.3f processing_seconds=%.3f",
                            succeeded, running - started, elapsed)
        finally:
            with self.lock:
                if acquired:
                    self.active = False
                elif ticket in self.waiters:
                    self.waiters.remove(ticket)


inference_queue = InferenceQueue()


async def inference_admission():
    async with inference_queue.admit():
        yield
