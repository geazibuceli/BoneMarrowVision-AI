from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from bone_marrow_vision.data.dataset import BoneMarrowDataset, verify_split_manifest
from bone_marrow_vision.models import build_model
from bone_marrow_vision.provenance import build_provenance
from bone_marrow_vision.training import (
    final_checkpoint_path,
    make_generator,
    seed_everything,
    seed_worker,
)
from bone_marrow_vision.training.recovery import recovery_path
from bone_marrow_vision.training.trainer import make_sampler, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a bone marrow cell classifier.")
    parser.add_argument("--config", required=True, help="Path to the training config YAML file.")
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing split JSON files and manifest.json.",
    )
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", help="Resume from this run's .latest.pt checkpoint; epochs is the total target.")
    parser.add_argument(
        "--allow-unverified-splits",
        action="store_true",
        help="Explicitly train without a valid split manifest (not recommended).",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("Training config must contain a YAML mapping")
    return config


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    checkpoint_path = Path(args.checkpoint)
    if args.resume:
        if Path(args.resume).resolve() != recovery_path(checkpoint_path).resolve():
            raise ValueError("--resume must be the .latest.pt file associated with --checkpoint")
        if args.allow_unverified_splits:
            raise ValueError("Resume requires verified split manifests")
    elif any(path.exists() for path in (
        checkpoint_path, recovery_path(checkpoint_path), final_checkpoint_path(checkpoint_path),
    )):
        raise ValueError("Training output already exists; choose a new path or use --resume")
    if args.epochs is not None:
        config["epochs"] = args.epochs
    seed = int(config.get("seed", 42))
    seed_everything(seed, deterministic=bool(config.get("deterministic", True)))

    processed_dir = Path(args.data_dir)
    if not args.allow_unverified_splits:
        verify_split_manifest(processed_dir)
    train_path = processed_dir / "train.json"
    validation_path = processed_dir / "validation.json"
    train_samples = json.loads(train_path.read_text(encoding="utf-8"))
    valid_samples = json.loads(validation_path.read_text(encoding="utf-8"))
    class_names = sorted({sample["label"] for sample in train_samples})
    unknown_validation = sorted({sample["label"] for sample in valid_samples} - set(class_names))
    if unknown_validation:
        raise ValueError(f"Validation contains classes absent from training: {unknown_validation}")

    train_dataset = BoneMarrowDataset(
        train_samples,
        class_names,
        image_size=int(config["image_size"]),
        training=True,
    )
    valid_dataset = BoneMarrowDataset(
        valid_samples,
        class_names,
        image_size=int(config["image_size"]),
        training=False,
    )
    num_workers = int(config.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        # Restart workers at epoch boundaries so their seeds can be recovered.
        "persistent_workers": False,
    }
    train_sampler = (
        make_sampler(
            [train_dataset.class_to_index[sample["label"]] for sample in train_samples],
            generator=make_generator(seed),
        )
        if bool(config.get("balanced_sampling", False))
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=make_generator(seed),
        **loader_options,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        generator=make_generator(seed + 1),
        **loader_options,
    )

    input_files = {
        "config": config_path,
        "train_split": train_path,
        "validation_split": validation_path,
    }
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.is_file():
        input_files["split_manifest"] = manifest_path
    provenance = build_provenance(config, input_files=input_files)
    model = build_model(
        config["architecture"],
        len(class_names),
        pretrained=bool(config.get("pretrained", True)) and not args.resume,
    )
    checkpoint_path = Path(args.checkpoint)
    history = train_model(
        model,
        train_loader,
        valid_loader,
        config,
        checkpoint_path,
        class_names,
        provenance=provenance,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "best_checkpoint": str(checkpoint_path),
                "final_checkpoint": str(final_checkpoint_path(checkpoint_path)),
                "recovery_checkpoint": str(recovery_path(checkpoint_path)),
                "class_names": class_names,
                "final_history_entry": history[-1] if history else {},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
