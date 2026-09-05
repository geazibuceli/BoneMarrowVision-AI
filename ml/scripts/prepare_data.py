from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from bone_marrow_vision.data.dataset import discover_samples, make_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare auditable train/validation/test splits for the bone marrow dataset."
    )
    parser.add_argument("--config", required=True, help="Path to the training config YAML file.")
    parser.add_argument(
        "--data-dir", default="data/raw", help="Root directory containing class folders."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where split files and their manifest are written.",
    )
    parser.add_argument("--metadata", help="CSV or JSON metadata file matched by image path.")
    parser.add_argument(
        "--metadata-path-key",
        default="path",
        help="Metadata column containing the image path.",
    )
    parser.add_argument(
        "--group-key",
        help="Metadata field used to keep patients/slides/sources isolated between splits.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("fail", "report"),
        default=None,
        help="Fail on duplicate content (default), or record it in the manifest for review.",
    )
    parser.add_argument(
        "--allow-invalid-images",
        action="store_true",
        help="Explicitly skip images that cannot be fully decoded; strict failure is the default.",
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
    config = load_config(args.config)
    group_key = args.group_key or config.get("group_key")
    if group_key and not args.metadata:
        raise ValueError("--metadata is required when a group key is configured")

    samples = discover_samples(
        args.data_dir,
        metadata_path=args.metadata,
        metadata_path_key=args.metadata_path_key,
        strict=not args.allow_invalid_images,
    )
    class_counts = Counter(sample["label"] for sample in samples)
    print(
        json.dumps(
            {
                "data_directory": str(Path(args.data_dir)),
                "image_count": len(samples),
                "class_count": len(class_counts),
                "class_counts": dict(sorted(class_counts.items())),
                "metadata": args.metadata,
                "group_key": group_key,
            },
            indent=2,
        )
    )

    make_splits(
        samples,
        args.output_dir,
        seed=int(config.get("seed", 42)),
        validation_size=float(config.get("validation_size", 0.15)),
        test_size=float(config.get("test_size", 0.15)),
        group_key=group_key,
        metadata_path=args.metadata,
        duplicate_policy=args.duplicate_policy or config.get("duplicate_policy", "fail"),
    )
    print(f"Prepared dataset splits and manifest under {Path(args.output_dir)}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Dataset preparation failed: {error}") from None
