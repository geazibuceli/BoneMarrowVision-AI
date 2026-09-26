"""Read the explicitly supported artifacts of local training runs."""

import json
import math
import re
from pathlib import Path

import yaml
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
    architecture_run = run_id.startswith("models~")
    if architecture_run:
        run_id = run_id.removeprefix("models~")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", run_id):
        raise HTTPException(404, "Experiment run not found")
    root = settings.resolve_path(settings.model_dir) if architecture_run else runs_root()
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
    if directory.parent == settings.resolve_path(settings.model_dir):
        status = architecture_status(directory)
        return {"id": f"models~{directory.name}", "name": directory.name,
                "kind": "architecture", "status": status}
    return {"id": directory.name, "status": read_artifact(directory, "status") or {"stage": "unknown"}}


ARCHITECTURES = ("resnet50", "efficientnet_b0", "convnext_tiny")
METRICS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")


def _read_json(directory: Path, filename: str):
    path = artifact_path(directory, filename)
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Artifact exceeds the size limit")
        value = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(value, allow_nan=False)
        return value
    except (OSError, ValueError) as error:
        raise HTTPException(503, "Experiment artifact is unavailable or invalid") from error


def architecture_status(directory: Path) -> dict:
    value = _read_json(directory, "status.json")
    if not isinstance(value, dict) or value.get("status") not in {"running", "completed", "failed", "smoke_completed"}:
        raise HTTPException(503, "Experiment artifact is unavailable or invalid")
    # Runner commands and error text may contain private host paths. Expose only status fields.
    return {"stage": value["status"], **{
        key: value[key] for key in ("started_at", "finished_at") if isinstance(value.get(key), str)
    }}


def architecture_details(directory: Path, run_id: str) -> dict:
    status = architecture_status(directory)
    rows = []
    signatures = []
    for architecture in ARCHITECTURES:
        config_path = artifact_path(directory, f"{architecture}.yaml")
        if not config_path.is_file():
            continue
        try:
            if config_path.stat().st_size > 100_000:
                raise ValueError("Configuration exceeds size limit")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or config.get("architecture") != architecture:
                raise ValueError("Invalid architecture configuration")
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise HTTPException(503, "Experiment artifact is unavailable or invalid") from error
        configuration = {key: config[key] for key in (
            "architecture", "seed", "epochs", "image_size", "batch_size", "learning_rate",
            "weight_decay", "num_workers", "pretrained", "class_strategy", "deterministic",
        ) if isinstance(config.get(key), (str, int, float, bool))}
        history = _read_json(directory, f"{architecture}/best.history.json")
        if history is not None and (not isinstance(history, list) or any(
            not isinstance(row, dict) or "epoch" not in row or any(
                not isinstance(number, (int, float)) or not math.isfinite(number) for number in row.values()
            ) for row in history
        )):
            raise HTTPException(503, "Experiment artifact is unavailable or invalid")
        report = _read_json(directory, f"{architecture}/test_report.json")
        filtered_report = None
        if report is not None:
            try:
                if any(not isinstance(report[k], (int, float)) or not 0 <= report[k] <= 1 for k in METRICS):
                    raise ValueError("Invalid metric")
                names, matrix = report["class_names"], report["confusion_matrix"]
                if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
                    raise ValueError("Invalid classes")
                if not isinstance(matrix, list) or len(matrix) != len(names) or any(
                    not isinstance(row, list) or len(row) != len(names) or any(
                        not isinstance(n, int) or n < 0 for n in row
                    ) for row in matrix
                ):
                    raise ValueError("Invalid confusion matrix")
                digest = report["provenance"]["test_split"]["sha256"]
                if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                    raise ValueError("Invalid split digest")
                signatures.append((digest, names))
                filtered_report = {k: report[k] for k in (*METRICS, "class_names", "confusion_matrix")}
                filtered_report["test_split_sha256"] = digest
            except (KeyError, TypeError, ValueError) as error:
                raise HTTPException(503, "Experiment artifact is unavailable or invalid") from error
        rows.append({"architecture": architecture, "configuration": configuration,
                     "history": history, "test_report": filtered_report})
    return {"id": run_id, "kind": "architecture", "name": directory.name, "status": status,
            "architectures": rows, "matching_test_splits": len(signatures) > 1 and all(
                signature == signatures[0] for signature in signatures[1:]
            )}
