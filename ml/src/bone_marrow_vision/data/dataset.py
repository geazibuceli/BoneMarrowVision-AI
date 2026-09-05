from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

from bone_marrow_vision.paths import portable_dataset_path, portable_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SPLIT_NAMES = ("train", "validation", "test")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def infer_label(root: Path, image_path: Path) -> str:
    relative_parts = image_path.relative_to(root).parts
    if "BM_cytomorphology_data" in relative_parts:
        class_index = relative_parts.index("BM_cytomorphology_data") + 1
        if class_index < len(relative_parts) - 1:
            return relative_parts[class_index]
    return image_path.parent.name


def _read_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open(newline="", encoding="utf-8") as file:
        if metadata_path.suffix.lower() == ".csv":
            return list(csv.DictReader(file))
        payload = json.load(file)
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return payload
    if isinstance(payload, dict) and all(isinstance(row, dict) for row in payload.values()):
        return [{"path": key, **row} for key, row in payload.items()]
    raise ValueError("Metadata must be a CSV, a JSON list of objects, or a path-keyed JSON object")


def _path_keys(path: Path, root: Path) -> set[str]:
    keys = {path.as_posix(), str(path), path.name}
    try:
        relative = path.relative_to(root)
    except ValueError:
        pass
    else:
        keys.update({relative.as_posix(), str(relative)})
    return {key.replace("\\", "/").casefold() for key in keys if key}


def _merge_metadata(
    samples: list[dict[str, Any]],
    root: Path,
    metadata_path: Path,
    path_key: str,
) -> None:
    rows = _read_metadata(metadata_path)
    lookup: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows, start=1):
        raw_path = row.get(path_key)
        if not raw_path:
            raise ValueError(f"Metadata row {row_index} has no value for path key {path_key!r}")
        for key in _path_keys(Path(str(raw_path)), root):
            lookup[key].append(row)

    matched_images: defaultdict[int, list[str]] = defaultdict(list)
    for sample in samples:
        matches = {
            id(row): row
            for key in _path_keys(Path(sample["path"]), root)
            for row in lookup.get(key, [])
        }
        if len(matches) > 1:
            raise ValueError(f"Multiple metadata rows match image {sample['path']!r}")
        if not matches:
            continue
        row_id, row = next(iter(matches.items()))
        matched_images[row_id].append(str(sample["path"]))
        sample.update(
            {
                key: value
                for key, value in row.items()
                if key != path_key and value is not None and value != ""
            }
        )
    ambiguous = [paths for paths in matched_images.values() if len(paths) > 1]
    if ambiguous:
        raise ValueError(
            "A metadata path matched multiple images; use paths relative to the dataset root "
            f"instead of ambiguous basenames: {ambiguous[0][:3]}"
        )


def discover_samples(
    data_dir: str | Path,
    metadata_path: str | Path | None = None,
    *,
    metadata_path_key: str = "path",
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Decode and hash all supported images, optionally merging external metadata."""
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")

    samples: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, UnidentifiedImageError, ValueError):
            unreadable.append(path.relative_to(root).as_posix())
            continue
        samples.append(
            {
                "path": str(path),
                "label": infer_label(root, path),
                "sha256": sha256_file(path),
            }
        )

    if unreadable and strict:
        preview = ", ".join(unreadable[:5])
        suffix = " ..." if len(unreadable) > 5 else ""
        raise ValueError(f"Found {len(unreadable)} unreadable image(s): {preview}{suffix}")
    if not samples:
        raise ValueError(f"No readable supported images found under {root}")

    if metadata_path:
        metadata_file = Path(metadata_path)
        if not metadata_file.is_file():
            raise ValueError(f"Metadata file does not exist: {metadata_file}")
        _merge_metadata(samples, root, metadata_file, metadata_path_key)
    return samples


def _ensure_hashes(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for position, original in enumerate(samples):
        sample = dict(original)
        if not sample.get("path") or not sample.get("label"):
            raise ValueError(f"Sample {position} must contain non-empty 'path' and 'label' values")
        source_path = str(sample["path"])
        path = portable_dataset_path(source_path)
        if path in seen_paths:
            raise ValueError(f"Duplicate sample path: {path}")
        seen_paths.add(path)
        sample["path"] = path
        sample["sha256"] = sample.get("sha256") or sha256_file(source_path)
        prepared.append(sample)
    if not prepared:
        raise ValueError("At least one sample is required")
    return prepared


def _cross_split_duplicates(
    splits: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    locations: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for split_name, rows in splits.items():
        for sample in rows:
            locations[str(sample["sha256"])][split_name].append(str(sample["path"]))
    return [
        {"sha256": digest, "splits": dict(sorted(by_split.items()))}
        for digest, by_split in sorted(locations.items())
        if len(by_split) > 1
    ]


def _source_duplicates(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(sample["sha256"])].append(sample)
    return [
        {
            "sha256": digest,
            "paths": sorted(str(sample["path"]) for sample in rows),
            "labels": sorted({str(sample["label"]) for sample in rows}),
        }
        for digest, rows in sorted(groups.items())
        if len(rows) > 1
    ]


def _validate_group_isolation(
    splits: dict[str, list[dict[str, Any]]], group_key: str | None
) -> None:
    if not group_key:
        return
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for split_name, rows in splits.items():
        for sample in rows:
            group_splits[str(sample[group_key])].add(split_name)
    leaked = sorted(group for group, names in group_splits.items() if len(names) > 1)
    if leaked:
        raise RuntimeError(f"Group leakage detected for {group_key!r}: {leaked[:5]}")


def _split_indices(
    samples: list[dict[str, Any]],
    seed: int,
    validation_size: float,
    test_size: float,
    group_key: str | None,
) -> dict[str, list[int]]:
    labels = [sample["label"] for sample in samples]
    indices = list(range(len(samples)))
    try:
        if group_key:
            groups = [sample[group_key] for sample in samples]
            first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_valid, test = next(first.split(indices, labels, groups))
            second = GroupShuffleSplit(
                n_splits=1,
                test_size=validation_size / (1 - test_size),
                random_state=seed,
            )
            train, valid = next(
                second.split(
                    train_valid,
                    [labels[index] for index in train_valid],
                    [groups[index] for index in train_valid],
                )
            )
        else:
            first = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_valid, test = next(first.split(indices, labels))
            second = StratifiedShuffleSplit(
                n_splits=1,
                test_size=validation_size / (1 - test_size),
                random_state=seed,
            )
            train, valid = next(second.split(train_valid, [labels[index] for index in train_valid]))
    except ValueError as error:
        mode = "group-aware" if group_key else "stratified"
        raise ValueError(f"Unable to create a valid {mode} split: {error}") from error
    return {
        "train": sorted(int(train_valid[index]) for index in train),
        "validation": sorted(int(train_valid[index]) for index in valid),
        "test": sorted(int(index) for index in test),
    }


def make_splits(
    samples: list[dict[str, Any]],
    output_dir: str | Path,
    seed: int = 42,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    group_key: str | None = None,
    *,
    metadata_path: str | Path | None = None,
    duplicate_policy: str = "fail",
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic splits and a hash-verifiable split manifest."""
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must each be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")
    if duplicate_policy not in {"fail", "report"}:
        raise ValueError("duplicate_policy must be 'fail' or 'report'")

    prepared = _ensure_hashes(samples)
    if group_key:
        missing = [
            sample["path"]
            for sample in prepared
            if sample.get(group_key) is None or not str(sample[group_key]).strip()
        ]
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            raise ValueError(
                f"Group key {group_key!r} is missing for {len(missing)} sample(s): {preview}"
            )

    source_duplicates = _source_duplicates(prepared)
    if source_duplicates and duplicate_policy == "fail":
        conflicting = sum(len(entry["labels"]) > 1 for entry in source_duplicates)
        conflict_note = f", including {conflicting} with conflicting labels" if conflicting else ""
        raise ValueError(
            f"Duplicate image content found in {len(source_duplicates)} group(s){conflict_note}; "
            "remove or review duplicates, or use duplicate_policy='report' for an explicit audit"
        )

    selected = _split_indices(
        prepared,
        seed=seed,
        validation_size=validation_size,
        test_size=test_size,
        group_key=group_key,
    )
    result = {name: [prepared[index] for index in selected[name]] for name in SPLIT_NAMES}
    _validate_group_isolation(result, group_key)
    cross_split_duplicates = _cross_split_duplicates(result)
    if cross_split_duplicates and duplicate_policy == "fail":
        raise ValueError(
            f"Content-hash leakage found across splits for {len(cross_split_duplicates)} file(s); "
            "remove duplicates or use duplicate_policy='report' to emit an explicit audit"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, Any] = {}
    for name, rows in result.items():
        split_path = output / f"{name}.json"
        split_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        split_manifest[name] = {
            "file": split_path.name,
            "sha256": sha256_file(split_path),
            "sample_count": len(rows),
            "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
            "content_set_sha256": _sha256_json(sorted(row["sha256"] for row in rows)),
        }

    manifest = {
        "format_version": 1,
        "seed": seed,
        "validation_size": validation_size,
        "test_size": test_size,
        "group_key": group_key,
        "duplicate_policy": duplicate_policy,
        "metadata": (
            {"path": portable_path(metadata_path), "sha256": sha256_file(metadata_path)}
            if metadata_path
            else None
        ),
        "source_sample_count": len(prepared),
        "source_content_set_sha256": _sha256_json(
            sorted((row["sha256"], row["label"]) for row in prepared)
        ),
        "source_duplicates": source_duplicates,
        "cross_split_duplicates": cross_split_duplicates,
        "splits": split_manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def verify_split_manifest(data_dir: str | Path) -> dict[str, Any]:
    """Fail if a prepared split no longer matches its recorded SHA-256 digest."""
    directory = Path(data_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Split manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("splits"), dict):
        raise ValueError(f"Invalid split manifest: {manifest_path}")
    for split_name in SPLIT_NAMES:
        entry = manifest["splits"].get(split_name)
        if not isinstance(entry, dict) or not entry.get("file") or not entry.get("sha256"):
            raise ValueError(f"Manifest has no complete {split_name!r} entry")
        split_path = directory / entry["file"]
        if not split_path.is_file():
            raise ValueError(f"Manifest references a missing split file: {split_path}")
        actual = sha256_file(split_path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {split_path}: expected {entry['sha256']}, got {actual}"
            )
    if manifest.get("cross_split_duplicates") and manifest.get("duplicate_policy") == "fail":
        raise ValueError("Manifest records forbidden content duplicates across splits")
    return manifest


class BoneMarrowDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        class_names: list[str],
        image_size: int = 224,
        training: bool = False,
    ) -> None:
        self.samples = samples
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.transform = build_transforms(image_size, training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample["path"]) as opened:
            image = opened.convert("RGB")
        return self.transform(image), self.class_to_index[sample["label"]]


def build_transforms(image_size: int, training: bool = False):
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def seed_everything(seed: int) -> None:
    """Backward-compatible import for the full reproducibility helper."""
    from bone_marrow_vision.training.reproducibility import seed_everything as seed_all

    seed_all(seed)
