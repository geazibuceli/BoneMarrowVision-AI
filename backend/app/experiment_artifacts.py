"""Read the explicitly supported artifacts of local training runs."""

import json
import math
import re
from pathlib import Path

from fastapi import HTTPException

from app.core.config import settings

ARTIFACTS = {
    "status": "status.json",
    "configuration": "config.json",
    "comparison": "comparison.json",
    "history": "classifier/best.history.json",
    "control_history": "control/best.history.json",
    "test_report": "test_report.json",
    "control_test_report": "control_test_report.json",
    "class_counts": "real_class_counts.json",
}


def runs_root() -> Path:
    return settings.resolve_path("data/experiments")


def run_directory(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", run_id):
        raise HTTPException(404, "Experiment run not found")
    root = runs_root()
    candidate = (root / run_id).resolve()
    if candidate.parent != root or not candidate.is_dir():
        raise HTTPException(404, "Experiment run not found")
    return candidate


def artifact_path(directory: Path, filename: str) -> Path:
    candidate = (directory / filename).resolve()
    if not candidate.is_relative_to(directory):
        raise HTTPException(404, "Experiment artifact not found")
    return candidate


def read_artifact(directory: Path, key: str):
    path = artifact_path(directory, ARTIFACTS[key])
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Artifact exceeds the size limit")
        value = json.loads(path.read_text(encoding="utf-8"))
        # Python accepts NaN/Infinity on input, but HTTP JSON responses cannot encode them.
        json.dumps(value, allow_nan=False)
        expected_type = list if key.endswith("history") else dict
        if not isinstance(value, expected_type):
            raise ValueError("Unexpected artifact structure")
        if key == "status" and not isinstance(value.get("stage"), str):
            raise ValueError("Missing run stage")
        if key.endswith("history") and any(
            not isinstance(row, dict) or any(
                not isinstance(number, (int, float)) or not math.isfinite(number)
                for number in row.values()
            ) or "epoch" not in row
            for row in value
        ):
            raise ValueError("Invalid epoch metrics")
        return value
    except (OSError, ValueError) as error:
        raise HTTPException(503, "Experiment artifact is unavailable or invalid") from error


def run_summary(directory: Path) -> dict:
    return {"id": directory.name, "status": read_artifact(directory, "status") or {"stage": "unknown"}}
