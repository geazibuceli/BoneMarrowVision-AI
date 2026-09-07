from __future__ import annotations

import logging
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)


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