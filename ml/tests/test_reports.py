import json

import pytest

from bone_marrow_vision.data.dataset import sha256_file
from bone_marrow_vision.evaluation.reports import (
    plot_confusion_matrix,
    write_artifact_manifest,
    write_report,
)


def _metrics():
    return {
        "accuracy": 0.75,
        "balanced_accuracy": 0.75,
        "macro_precision": 0.8,
        "macro_recall": 0.75,
        "macro_f1": 0.74,
        "weighted_f1": 0.74,
        "confusion_matrix": [[2, 1], [0, 1]],
    }


def test_reports_and_artifact_manifest_include_verifiable_hashes(tmp_path):
    markdown = tmp_path / "report.md"
    image = tmp_path / "matrix.png"
    source = tmp_path / "metrics.json"
    source.write_text(json.dumps(_metrics()), encoding="utf-8")
    write_report(_metrics(), markdown)
    plot_confusion_matrix(_metrics(), ["A", "B"], image)

    manifest_path = tmp_path / "artifacts.json"
    manifest = write_artifact_manifest(
        manifest_path,
        {"report": markdown, "matrix": image, "matrix_csv": image.with_suffix(".csv")},
        inputs={"metrics": source},
    )
    assert manifest["artifacts"]["report"]["sha256"] == sha256_file(markdown)
    assert manifest["inputs"]["metrics"]["sha256"] == sha256_file(source)


def test_report_rejects_non_finite_metrics(tmp_path):
    metrics = _metrics()
    metrics["accuracy"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        write_report(metrics, tmp_path / "invalid.md")
