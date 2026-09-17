from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import anyio

LOGGER = logging.getLogger(__name__)


async def run_periodic_cleanup(cleanup: Callable[[], dict[str, int]], interval_seconds: float) -> None:
    """Run serially off the event loop and wait for in-flight cleanup on shutdown."""
    while True:
        await anyio.sleep(interval_seconds)
        try:
            await anyio.to_thread.run_sync(cleanup)
        except Exception:
            LOGGER.exception("Scheduled artifact cleanup failed; retrying at the next interval")


def _remove_older_than(directory: Path, pattern: str, cutoff: float) -> int:
    removed = 0
    if not directory.is_dir():
        return removed
    for candidate in directory.glob(pattern):
        try:
            if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                removed += 1
        except OSError:
            LOGGER.warning("Could not remove expired artifact path=%s", candidate, exc_info=True)
    return removed


def cleanup_expired_artifacts(
    *,
    upload_dir: Path,
    inference_temp_dir: Path,
    gradcam_retention_days: int,
    temp_retention_hours: int,
    now: float | None = None,
) -> dict[str, int]:
    current_time = time.time() if now is None else now
    removed_gradcams = _remove_older_than(
        upload_dir / "gradcam",
        "bone-marrow-input-*.png",
        current_time - gradcam_retention_days * 24 * 60 * 60,
    )
    removed_temporary_inputs = _remove_older_than(
        inference_temp_dir,
        "bone-marrow-input-*.*",
        current_time - temp_retention_hours * 60 * 60,
    )
    return {
        "gradcam": removed_gradcams,
        "temporary_inputs": removed_temporary_inputs,
    }
