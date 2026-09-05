from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from bone_marrow_vision.data.dataset import sha256_file
from bone_marrow_vision.paths import portable_path

METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)


def write_report(
    metrics: dict[str, Any], output_path: str | Path, title: str = "Experiment Report"
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "This report contains research metrics only. It does not establish clinical validity.",
        "",
    ]
    for key in METRIC_KEYS:
        if key in metrics:
            if not math.isfinite(float(metrics[key])):
                raise ValueError(f"Metric {key!r} must be finite")
            lines.append(f"- **{key}**: {metrics[key]:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_confusion_matrix(
    metrics: dict[str, Any], class_names: list[str], output_path: str | Path
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set(xlabel="Predicted class", ylabel="True class")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted", *class_names])
        writer.writerows(
            [[class_names[index], *row] for index, row in enumerate(metrics["confusion_matrix"])]
        )


def write_artifact_manifest(
    output_path: str | Path,
    artifacts: dict[str, str | Path],
    *,
    inputs: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    manifest = {
        "format_version": 1,
        "artifacts": {
            name: {"path": portable_path(path), "sha256": sha256_file(path)}
            for name, path in sorted(artifacts.items())
        },
        "inputs": {
            name: {"path": portable_path(path), "sha256": sha256_file(path)}
            for name, path in sorted((inputs or {}).items())
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
