from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from bone_marrow_vision.data.dataset import infer_label, sha256_file
from bone_marrow_vision.paths import portable_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
INCOMPLETE_DOWNLOAD_EXTENSIONS = {".partial", ".part", ".crdownload", ".aspera-ckpt"}


def inspect_dataset(data_dir: str | Path, report_path: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {portable_path(root)}")
    report_file = Path(report_path)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    image_files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    class_counts: Counter[str] = Counter()
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    unreadable: list[str] = []
    image_manifest: list[dict[str, Any]] = []

    for path in image_files:
        relative = path.relative_to(root).as_posix()
        class_name = infer_label(root, path) or "unknown"
        class_counts[class_name] += 1
        digest = sha256_file(path)
        entry: dict[str, Any] = {
            "path": relative,
            "label": class_name,
            "sha256": digest,
        }
        try:
            with Image.open(path) as image:
                image.load()
                entry.update(
                    {
                        "decoded": True,
                        "format": image.format,
                        "mode": image.mode,
                        "width": image.width,
                        "height": image.height,
                    }
                )
            hashes[digest].append(relative)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            unreadable.append(relative)
            entry.update({"decoded": False, "error": type(error).__name__})
        image_manifest.append(entry)

    duplicate_groups = [members for members in hashes.values() if len(members) > 1]
    manifest_path = report_file.with_name(f"{report_file.stem}.images.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in image_manifest),
        encoding="utf-8",
    )
    metadata_files = [
        path.relative_to(root).as_posix()
        for path in files
        if path.suffix.lower() in {".csv", ".json", ".jsonl", ".xlsx", ".xml"}
    ]
    incomplete_download_files = [
        path.relative_to(root).as_posix()
        for path in files
        if path.suffix.lower() in INCOMPLETE_DOWNLOAD_EXTENSIONS
    ]
    report: dict[str, Any] = {
        "data_directory": portable_path(root),
        "image_count": len(image_files),
        "readable_image_count": len(image_files) - len(unreadable),
        "class_counts": dict(sorted(class_counts.items())),
        "file_extensions": dict(sorted(Counter(path.suffix.lower() for path in files).items())),
        "metadata_files": metadata_files,
        "incomplete_download_files": incomplete_download_files,
        "unreadable_images": unreadable,
        "duplicate_groups": duplicate_groups,
        "duplicate_hashes": [
            {"sha256": digest, "paths": members}
            for digest, members in sorted(hashes.items())
            if len(members) > 1
        ],
        "image_manifest": {
            "path": portable_path(manifest_path),
            "sha256": sha256_file(manifest_path),
            "entry_count": len(image_manifest),
            "fully_decoded_count": len(image_manifest) - len(unreadable),
        },
        "possible_group_leakage": {
            "status": "requires_metadata_review" if metadata_files else "metadata_not_found",
            "signals": ["patient", "slide", "source", "case"],
            "assumption": "Folder names are treated as labels only until metadata is inspected.",
        },
        "assumptions": [
            "Image labels are inferred from the class directory under BM_cytomorphology_data, or the immediate parent for simpler layouts.",
            "Group-aware splitting requires patient, slide, or source identifiers in a metadata file.",
            "No clinical performance or diagnostic capability is inferred from this inspection.",
        ],
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
