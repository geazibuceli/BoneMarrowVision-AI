"""Audit existing splits without changing their membership or overwriting a manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from bone_marrow_vision.data.dataset import SPLIT_NAMES, sha256_file


def inspect_image(path: str) -> str:
    with Image.open(path) as image:
        image.load()
    return sha256_file(path)


def audit_splits(directory: Path, workers: int = 4) -> dict:
    manifest = {
        "format_version": 1,
        "origin": "audit_of_existing_splits",
        "group_key": None,
        "group_isolation_verified": False,
        "duplicate_policy": "fail",
        "splits": {},
    }
    locations = defaultdict(list)
    labels_by_split = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in SPLIT_NAMES:
            path = directory / f"{name}.json"
            original_hash = sha256_file(path)
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not rows:
                raise ValueError(f"Empty split: {name}")
            if len({row["path"] for row in rows}) != len(rows):
                raise ValueError(f"Repeated paths within {name}")
            labels_by_split[name] = {row["label"] for row in rows}
            for row, digest in zip(
                rows, pool.map(inspect_image, [r["path"] for r in rows]), strict=True
            ):
                locations[digest].append(
                    {"split": name, "path": row["path"], "label": row["label"]}
                )
            if sha256_file(path) != original_hash:
                raise ValueError(f"Split changed during audit: {name}")
            manifest["splits"][name] = {
                "file": path.name,
                "sha256": original_hash,
                "sample_count": len(rows),
                "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
            }
            print(f"Audited {name}: {len(rows)} images", flush=True)
    if any(labels != labels_by_split["train"] for labels in labels_by_split.values()):
        raise ValueError("Class vocabulary differs between splits")
    manifest["source_duplicates"] = [
        {"sha256": digest, "occurrences": entries}
        for digest, entries in locations.items()
        if len(entries) > 1
    ]
    manifest["cross_split_duplicates"] = [
        entry
        for entry in manifest["source_duplicates"]
        if len({row["split"] for row in entry["occurrences"]}) > 1
    ]
    manifest["conflicting_labels"] = [
        entry
        for entry in manifest["source_duplicates"]
        if len({row["label"] for row in entry["occurrences"]}) > 1
    ]
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.data_dir / "manifest.json"
    if output.exists():
        raise SystemExit("Manifest already exists; refusing to overwrite it")
    manifest = audit_splits(args.data_dir, args.workers)
    blocked = bool(manifest["cross_split_duplicates"] or manifest["conflicting_labels"])
    destination = args.data_dir / "existing_split_audit.json" if blocked else output
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(
        json.dumps(
            {
                "output": str(destination),
                "duplicate_groups": len(manifest["source_duplicates"]),
                "cross_split_duplicate_groups": len(manifest["cross_split_duplicates"]),
                "conflicting_label_groups": len(manifest["conflicting_labels"]),
                "group_isolation_verified": False,
            },
            indent=2,
        )
    )
    if blocked:
        raise SystemExit("Audit failed: review duplicates before preparing a controlled comparison")


if __name__ == "__main__":
    main()
