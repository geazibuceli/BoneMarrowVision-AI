from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from bone_marrow_vision.data.dataset import sha256_file
from bone_marrow_vision.evaluation.reports import write_artifact_manifest
from bone_marrow_vision.paths import portable_path
from bone_marrow_vision.training import load_checkpoint

parser = argparse.ArgumentParser(
    description="Create plots and tables for a trained model evaluation."
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--metrics", required=True, help="Evaluation JSON produced by evaluate.py.")
parser.add_argument("--data-dir", default="data/processed")
parser.add_argument("--output-dir", default="reports/analysis")
args = parser.parse_args()

checkpoint = load_checkpoint(args.checkpoint)
metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
class_names = checkpoint["class_names"]
confusion = (
    metrics["metrics"]["confusion_matrix"] if "metrics" in metrics else metrics["confusion_matrix"]
)
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)

history = checkpoint.get("history", [])
if history:
    epochs = [item["epoch"] for item in history]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, [item["train_loss"] for item in history], label="Training loss", linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "loss_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        epochs,
        [item["validation_macro_f1"] for item in history],
        label="Validation macro F1",
        color="#c2410c",
        linewidth=2,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Macro F1")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "validation_f1_curve.png", dpi=180)
    plt.close(figure)

    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

figure, axis = plt.subplots(figsize=(12, 10))
sns.heatmap(
    confusion,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axis,
)
axis.set_xlabel("Predicted class")
axis.set_ylabel("True class")
axis.set_title("Confusion matrix")
figure.tight_layout()
figure.savefig(output / "confusion_matrix.png", dpi=180)
plt.close(figure)

actual = []
predicted = []
for true_index, row in enumerate(confusion):
    for predicted_index, count in enumerate(row):
        actual.extend([true_index] * count)
        predicted.extend([predicted_index] * count)
report = classification_report(
    actual,
    predicted,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
class_metrics = {
    name: {
        key: float(value)
        for key, value in report[name].items()
        if key in {"precision", "recall", "f1-score", "support"}
    }
    for name in class_names
}
(output / "metrics_by_class.json").write_text(json.dumps(class_metrics, indent=2), encoding="utf-8")

confusions = [
    {
        "true_class": class_names[true_index],
        "predicted_class": class_names[predicted_index],
        "count": int(count),
    }
    for true_index, row in enumerate(confusion)
    for predicted_index, count in enumerate(row)
    if true_index != predicted_index and count
]
confusions.sort(key=lambda item: item["count"], reverse=True)
(output / "most_confused_pairs.json").write_text(json.dumps(confusions, indent=2), encoding="utf-8")

class_counts = Counter()
for split_name in ("train", "validation", "test"):
    split_path = Path(args.data_dir) / f"{split_name}.json"
    samples = json.loads(split_path.read_text(encoding="utf-8"))
    class_counts.update(sample["label"] for sample in samples)
(output / "class_distribution.json").write_text(
    json.dumps(dict(sorted(class_counts.items())), indent=2), encoding="utf-8"
)

figure, axis = plt.subplots(figsize=(12, 6))
ordered = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
axis.bar([item[0] for item in ordered], [item[1] for item in ordered], color="#155e75")
axis.set_xlabel("Class")
axis.set_ylabel("Images")
axis.set_title("Image distribution by class")
axis.tick_params(axis="x", rotation=60)
figure.tight_layout()
figure.savefig(output / "class_distribution.png", dpi=180)
plt.close(figure)

summary = {
    "checkpoint": portable_path(args.checkpoint),
    "checkpoint_sha256": sha256_file(args.checkpoint),
    "metrics_sha256": sha256_file(args.metrics),
    "metrics": metrics.get("metrics", metrics),
    "best_validation_macro_f1": max(
        (item["validation_macro_f1"] for item in history), default=None
    ),
    "best_validation_epoch": max(history, key=lambda item: item["validation_macro_f1"])["epoch"]
    if history
    else None,
    "most_confused_pairs": confusions[:10],
    "class_distribution": dict(sorted(class_counts.items())),
}
summary_path = output / "analysis_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
artifacts = {
    path.name: path for path in output.iterdir() if path.is_file() and path.name != "artifacts.json"
}
write_artifact_manifest(
    output / "artifacts.json",
    artifacts,
    inputs={"checkpoint": args.checkpoint, "metrics": args.metrics},
)
print(json.dumps(summary, indent=2))
