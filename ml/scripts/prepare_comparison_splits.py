"""Copy audited splits and exclude all occurrences of conflicting-label content."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from audit_existing_splits import audit_splits
from bone_marrow_vision.data.dataset import SPLIT_NAMES, sha256_file


def prepare_comparison(source: Path, output: Path) -> None:
    audit_path = source / "existing_split_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["cross_split_duplicates"]:
        raise ValueError("Cross-split duplicates require a separate split-repair protocol")
    for name in SPLIT_NAMES:
        if sha256_file(source / f"{name}.json") != audit["splits"][name]["sha256"]:
            raise ValueError(f"Source split changed since audit: {name}")
    excluded = {
        (entry["split"], entry["path"])
        for group in audit["conflicting_labels"]
        for entry in group["occurrences"]
    }
    if not excluded:
        raise ValueError("No conflicting-label samples were recorded")
    output.mkdir(parents=True, exist_ok=False)
    removals = []
    for name in SPLIT_NAMES:
        source_path = source / f"{name}.json"
        rows = json.loads(source_path.read_text(encoding="utf-8"))
        kept = []
        for row in rows:
            if (name, row["path"]) in excluded:
                removals.append({"split": name, **row})
            else:
                kept.append(row)
        destination = output / source_path.name
        if len(kept) == len(rows):
            shutil.copyfile(source_path, destination)
        else:
            destination.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    (output / "exclusions.json").write_text(
        json.dumps(
            {
                "reason": "Identical file contents with conflicting labels; exclude every occurrence",
                "source_audit_sha256": sha256_file(audit_path),
                "excluded": removals,
                "historical_model_comparison": "Retrain both models; validation membership changed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = audit_splits(output)
    if manifest["cross_split_duplicates"] or manifest["conflicting_labels"]:
        raise ValueError("Corrected split audit failed; no manifest was written")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {output}; excluded {len(removals)} samples; original splits preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/resnet50_comparison"))
    args = parser.parse_args()
    prepare_comparison(args.source, args.output)
