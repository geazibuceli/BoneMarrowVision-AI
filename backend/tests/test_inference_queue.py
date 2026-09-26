import asyncio

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.core.inference_queue import InferenceQueue


def test_queue_bounds_waiters_releases_cancelled_and_failed_requests(monkeypatch):
    monkeypatch.setattr(settings, "inference_queue_size", 1)
    queue = InferenceQueue()

    async def waiting():
        async with queue.admit():
            pass

    async def scenario():
        async with queue.admit():
            task = asyncio.create_task(waiting())
            await asyncio.sleep(0.01)
            assert queue.snapshot()["waiting"] == 1
            with pytest.raises(HTTPException) as error:
                async with queue.admit():
                    pytest.fail("Queue must reject excess requests")
            assert error.value.status_code == 503
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert queue.snapshot()["waiting"] == 0
        with pytest.raises(RuntimeError):
            async with queue.admit():
                raise RuntimeError("inference failed")
        async with queue.admit():
            pass
    asyncio.run(scenario())
    assert queue.snapshot()["active"] is False
    assert queue.snapshot()["completed"] == 2
    assert queue.snapshot()["failed"] == 1
    assert queue.snapshot()["rejected"] == 1


def test_queue_wait_times_out(monkeypatch):
    monkeypatch.setattr(settings, "inference_queue_timeout_seconds", 0.01)
    queue = InferenceQueue()

    async def scenario():
        async with queue.admit():
            with pytest.raises(HTTPException, match="wait expired"):
                async with queue.admit():
                    pytest.fail("Queue must time out")
        assert queue.snapshot()["waiting"] == 0
    asyncio.run(scenario())
