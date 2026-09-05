from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from bone_marrow_vision.data.dataset import sha256_file
from bone_marrow_vision.paths import portable_path


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_commit(start: Path) -> str | None:
    environment_commit = os.environ.get("GITHUB_SHA")
    if environment_commit:
        return environment_commit
    for parent in (start, *start.parents):
        git_directory = parent / ".git"
        head = git_directory / "HEAD"
        if not head.is_file():
            continue
        value = head.read_text(encoding="utf-8").strip()
        if not value.startswith("ref: "):
            return value or None
        reference = value.removeprefix("ref: ")
        ref_file = git_directory / reference
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip() or None
        packed_refs = git_directory / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.endswith(f" {reference}"):
                    return line.split(" ", 1)[0]
    return None


def build_provenance(
    config: Mapping[str, Any],
    *,
    input_files: Mapping[str, str | Path] | None = None,
    working_directory: str | Path | None = None,
) -> dict[str, Any]:
    directory = Path(working_directory or Path.cwd())
    files = {
        name: {"path": portable_path(path, base=directory), "sha256": sha256_file(path)}
        for name, path in sorted((input_files or {}).items())
    }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(directory),
        "config_sha256": canonical_json_sha256(config),
        "input_files": files,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation": sys.implementation.name,
            "packages": {
                package: _package_version(package)
                for package in (
                    "torch",
                    "torchvision",
                    "numpy",
                    "scikit-learn",
                    "Pillow",
                )
            },
        },
    }
