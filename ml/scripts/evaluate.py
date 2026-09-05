from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from bone_marrow_vision.data.dataset import (
    BoneMarrowDataset,
    sha256_file,
    verify_split_manifest,
)
from bone_marrow_vision.evaluation.reports import (
    plot_confusion_matrix,
    write_artifact_manifest,
    write_report,
)
from bone_marrow_vision.models import build_model
from bone_marrow_vision.paths import portable_path
from bone_marrow_vision.training import (
    load_checkpoint,
    make_generator,
    seed_everything,
    seed_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained bone marrow model on the held-out test split."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="reports/metrics/test_report.json")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--allow-unverified-split",
        action="store_true",
        help="Explicitly evaluate without a valid split manifest (not recommended).",
    )
    return parser.parse_args()


def compute_metrics(
    actual: list[int], predicted: list[int], class_names: list[str]
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(actual, predicted, labels=labels)
    precision, recall, macro_f1, _support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    class_names = checkpoint["class_names"]
    architecture = checkpoint["architecture"]
    image_size = int(checkpoint.get("image_size", 224))
    config = checkpoint.get("config", {})
    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=bool(config.get("deterministic", True)))

    model = build_model(architecture, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_dir = Path(args.data_dir)
    if not args.allow_unverified_split:
        verify_split_manifest(data_dir)
    test_path = data_dir / "test.json"
    test_samples = json.loads(test_path.read_text(encoding="utf-8"))
    unknown = sorted({sample["label"] for sample in test_samples} - set(class_names))
    if unknown:
        raise ValueError(f"Test split contains classes absent from checkpoint: {unknown}")
    test_dataset = BoneMarrowDataset(test_samples, class_names, image_size=image_size)
    num_workers = (
        int(args.num_workers) if args.num_workers is not None else int(config.get("num_workers", 0))
    )
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader = DataLoader(
        test_dataset,
        batch_size=int(checkpoint.get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_generator(seed + 2),
        persistent_workers=num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Evaluation device: {device} ({device_name})")
    model.to(device)
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(targets.tolist())

    metrics = compute_metrics(actual, predicted, class_names)
    manifest_path = data_dir / "manifest.json"
    metrics.update(
        {
            "format_version": 1,
            "class_names": class_names,
            "checkpoint_role": checkpoint.get("checkpoint_role", "legacy_unspecified"),
            "provenance": {
                "checkpoint": {
                    "path": portable_path(checkpoint_path),
                    "sha256": sha256_file(checkpoint_path),
                },
                "test_split": {
                    "path": portable_path(test_path),
                    "sha256": sha256_file(test_path),
                },
                "split_manifest": (
                    {
                        "path": portable_path(manifest_path),
                        "sha256": sha256_file(manifest_path),
                    }
                    if manifest_path.is_file()
                    else None
                ),
            },
        }
    )

    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    markdown_path = report_path.with_suffix(".md")
    image_path = report_path.with_suffix(".png")
    write_report(metrics, markdown_path, title="Bone Marrow Cell Evaluation")
    plot_confusion_matrix(metrics, class_names, image_path)
    artifact_manifest_path = report_path.with_name(f"{report_path.stem}.artifacts.json")
    write_artifact_manifest(
        artifact_manifest_path,
        {
            "metrics": report_path,
            "markdown_report": markdown_path,
            "confusion_matrix_image": image_path,
            "confusion_matrix_csv": image_path.with_suffix(".csv"),
        },
        inputs={"checkpoint": checkpoint_path, "test_split": test_path},
    )
    print(
        json.dumps({"metrics": metrics, "artifact_manifest": str(artifact_manifest_path)}, indent=2)
    )


if __name__ == "__main__":
    main()
