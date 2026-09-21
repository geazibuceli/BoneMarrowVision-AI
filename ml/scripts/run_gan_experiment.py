"""Run an isolated GAN augmentation experiment and a 50-epoch classifier training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from bone_marrow_vision.data.dataset import BoneMarrowDataset, sha256_file, verify_split_manifest
from bone_marrow_vision.models import build_model
from bone_marrow_vision.provenance import build_provenance
from bone_marrow_vision.training.conditional_gan import generate_samples, train_gan
from bone_marrow_vision.training.trainer import validation_metrics


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def inspect_image(row):
    if row.get("is_synthetic", False):
        raise ValueError("Source splits must contain only real images")
    path = Path(row["path"])
    digest = sha256_file(path)
    if row.get("sha256") and row["sha256"] != digest:
        raise ValueError(f"Source image hash mismatch: {path.name}")
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return {**row, "sha256": digest, "pixel_sha256": pixel_hash, "is_synthetic": False}


def write_splits(directory: Path, splits: dict, provenance: dict):
    directory.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name, rows in splits.items():
        path = directory / f"{name}.json"
        write_json(path, rows)
        entries[name] = {"file": path.name, "sha256": sha256_file(path),
                         "count": len(rows), "class_counts": dict(Counter(r["label"] for r in rows))}
    write_json(directory / "manifest.json", {
        "format_version": 1, "splits": entries, "duplicate_policy": "fail",
        "cross_split_duplicates": [], "provenance": provenance,
        "group_key": None, "independence": "image-level; patient metadata unavailable",
    })
    verify_split_manifest(directory)


def prepare_real_splits(source: Path, destination: Path, *, duplicate_policy="fail"):
    """Audit legacy images without replacing or silently reshuffling held-out splits."""
    splits = {}
    groups = defaultdict(list)
    for name in ("validation", "test", "train"):
        rows = json.loads((source / f"{name}.json").read_text(encoding="utf-8"))
        if not rows:
            raise ValueError(f"Empty source split: {name}")
        checked = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for index, row in enumerate(pool.map(inspect_image, rows), start=1):
                digest = row["pixel_sha256"]
                groups[digest].append({"split": name, "label": row["label"], "path": row["path"]})
                checked.append(row)
                if index % 10000 == 0:
                    print(f"Audited {name}: {index}/{len(rows)}", flush=True)
        splits[name] = checked
    conflicts = {digest: rows for digest, rows in groups.items()
                 if len({r["label"] for r in rows}) > 1}
    overlaps = {digest: rows for digest, rows in groups.items()
                if len({r["split"] for r in rows}) > 1}
    excluded = set(conflicts) | set(overlaps)
    report = {"policy": duplicate_policy, "conflicting_label_groups": conflicts,
              "cross_split_groups": overlaps,
              "excluded_by_split": {name: sum(r["pixel_sha256"] in excluded for r in rows)
                                    for name, rows in splits.items()}}
    write_json(destination.parent / "source_audit.json", report)
    if duplicate_policy == "fail":
        if conflicts:
            raise ValueError("Identical pixels have conflicting labels; review source_audit.json")
        if overlaps:
            raise ValueError("Identical pixels cross source splits; review source_audit.json")
    elif duplicate_policy == "quarantine":
        splits = {name: [r for r in rows if r["pixel_sha256"] not in excluded]
                  for name, rows in splits.items()}
        print(json.dumps({"quarantined": report["excluded_by_split"]}), flush=True)
    else:
        raise ValueError("Unknown source duplicate policy")
    if any(not rows for rows in splits.values()):
        raise ValueError("No usable images remain in a source split")
    known = {r["label"] for r in splits["train"]}
    if any(r["label"] not in known for name in ("validation", "test") for r in splits[name]):
        raise ValueError("Held-out class is absent from training")
    provenance = build_provenance({}, input_files={
        name: source / f"{name}.json" for name in splits
    })
    provenance["source_audit_sha256"] = sha256_file(destination.parent / "source_audit.json")
    write_splits(destination, splits, provenance)
    return splits


def export_history(checkpoint: Path, output: Path):
    history_path = checkpoint.with_suffix(".history.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    with (output / "classifier_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = [entry["epoch"] for entry in history]
    for axis, keys, title in zip(axes, [
        ["train_loss", "validation_loss"], ["train_accuracy", "validation_accuracy"],
        ["validation_macro_precision", "validation_macro_f1"],
    ], ["Loss (train weighted; validation unweighted)", "Accuracy", "Macro metrics"], strict=True):
        for key in keys:
            axis.plot(epochs, [entry[key] for entry in history], label=key)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "classifier_curves.png", dpi=160)
    plt.close(figure)


def evaluate_checkpoint(checkpoint_path: Path, split_path: Path):
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    names = checkpoint["class_names"]
    samples = json.loads(split_path.read_text(encoding="utf-8"))
    if set(names) != {r["label"] for r in samples}:
        raise ValueError("Comparison checkpoint has a different class vocabulary")
    model = build_model(checkpoint["architecture"], len(names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(BoneMarrowDataset(samples, names, checkpoint.get("image_size", 224)),
                        batch_size=32, num_workers=0)
    metrics = validation_metrics(model, loader, device)
    return {"checkpoint_sha256": sha256_file(checkpoint_path),
            "test_split_sha256": sha256_file(split_path),
            **{key.replace("validation_", "test_"): value for key, value in metrics.items()}}


def train_classifier(config_path: Path, splits: Path, checkpoint: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "ml/scripts/train.py", "--config", str(config_path),
               "--data-dir", str(splits), "--checkpoint", str(checkpoint)]
    with subprocess.Popen(command) as process:
        try:
            while process.poll() is None:
                if checkpoint.with_suffix(".history.json").is_file():
                    export_history(checkpoint, output)
                time.sleep(20)
        except BaseException:
            process.terminate()
            process.wait()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    export_history(checkpoint, output)


def verify_matched_holdouts(real_dir: Path, augmented_dir: Path):
    for name in ("validation", "test"):
        if sha256_file(real_dir / f"{name}.json") != sha256_file(augmented_dir / f"{name}.json"):
            raise RuntimeError("Held-out split changed")


def paired_comparison(augmented: dict, control: dict) -> dict:
    if augmented["test_split_sha256"] != control["test_split_sha256"]:
        raise ValueError("Cannot compare models evaluated on different test splits")
    keys = ("test_accuracy", "test_macro_precision", "test_macro_f1", "test_loss")
    return {
        "gan_augmented": augmented, "real_only_control": control,
        "comparison_type": "matched_real_only_control",
        "delta_gan_minus_control": {key: augmented[key] - control[key] for key in keys},
        "limitations": "Single-seed matched comparison; repeat with multiple seeds. "
                       "GAN labels are unreviewed. No patient-level split. "
                       "Equal epochs do not imply equal optimizer steps because training set sizes differ.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/conditional_gan_experiment.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", required=True, help="New directory under ignored local artifacts")
    parser.add_argument("--seed", type=int, help="Override GAN, generation, and both classifier seeds")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else config["seed"]
    config["seed"] = config["gan"]["seed"] = config["classifier"]["seed"] = seed
    if config["classifier"].get("early_stopping_patience", 0) <= config["classifier"]["epochs"]:
        raise ValueError("Matched comparison requires early stopping patience greater than epochs")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "status.json"
    started_at = datetime.now(UTC).isoformat()
    started_clock = time.monotonic()
    stages = []

    def status(stage, **fields):
        updated_at = datetime.now(UTC).isoformat()
        stages.append({"stage": stage, "at": updated_at})
        write_json(status_path, {"stage": stage, "started_at": started_at,
                                "updated_at": updated_at, "elapsed_seconds": time.monotonic() - started_clock,
                                "stages": stages, **fields})
        print(json.dumps({"stage": stage, **fields}), flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("This full experiment requires CUDA; use CPU only in unit tests")
        write_json(output / "config.json", config)
        write_json(output / "provenance.json", build_provenance(config, input_files={
            "config": args.config, "runner": __file__,
            "gan_code": "ml/src/bone_marrow_vision/training/conditional_gan.py",
            "trainer_code": "ml/src/bone_marrow_vision/training/trainer.py",
        }))
        status("auditing_real_splits")
        real_dir = output / "real_splits"
        splits = prepare_real_splits(
            Path(args.data_dir), real_dir, duplicate_policy=config.get("duplicate_policy", "fail")
        )
        counts = dict(Counter(row["label"] for row in splits["train"]))
        write_json(output / "real_class_counts.json", counts)
        status("training_gan", epochs=config["gan"]["epochs"])
        gan_checkpoint = train_gan(real_dir, output / "gan", config["gan"])
        status("generating_synthetic_images")
        synthetic = generate_samples(gan_checkpoint, real_dir, output / "synthetic", config["seed"])
        augmented = {**splits, "train": splits["train"] + synthetic}
        augmented_dir = output / "augmented_splits"
        write_splits(augmented_dir, augmented, {
            "real_manifest_sha256": sha256_file(real_dir / "manifest.json"),
            "synthetic_manifest_sha256": sha256_file(output / "synthetic" / "manifest.json"),
            "morphology_review": "unreviewed_experimental",
        })
        verify_matched_holdouts(real_dir, augmented_dir)
        classifier_config = output / "classifier.yaml"
        classifier_config.write_text(yaml.safe_dump(config["classifier"]), encoding="utf-8")
        checkpoint = output / "classifier" / "best.pt"
        control_checkpoint = output / "control" / "best.pt"
        status("training_real_only_control", epochs=config["classifier"]["epochs"], seed=seed)
        train_classifier(classifier_config, real_dir, control_checkpoint, output / "control")
        status("training_classifier", epochs=config["classifier"]["epochs"],
               synthetic_count=len(synthetic))
        train_classifier(classifier_config, augmented_dir, checkpoint, output)
        verify_matched_holdouts(real_dir, augmented_dir)
        status("evaluating_real_test")
        subprocess.run([
            sys.executable, "-u", "ml/scripts/evaluate.py", "--checkpoint", str(checkpoint),
            "--data-dir", str(augmented_dir), "--output", str(output / "test_report.json"),
        ], check=True)
        subprocess.run([
            sys.executable, "-u", "ml/scripts/evaluate.py", "--checkpoint", str(control_checkpoint),
            "--data-dir", str(real_dir), "--output", str(output / "control_test_report.json"),
        ], check=True)
        comparison = paired_comparison(
            evaluate_checkpoint(checkpoint, real_dir / "test.json"),
            evaluate_checkpoint(control_checkpoint, real_dir / "test.json"),
        )
        comparison["seed"] = seed
        baseline = config.get("baseline_checkpoint")
        if baseline:
            comparison["historical_baseline"] = evaluate_checkpoint(Path(baseline), real_dir / "test.json")
            comparison["historical_limitations"] = (
                "Historical baseline is not a matched run; training duration and conditions differ. "
                "It may have seen duplicate groups excluded from this experiment."
            )
        write_json(output / "comparison.json", comparison)
        status("completed", comparison="comparison.json", curves="classifier_curves.png")
    except BaseException as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
