from __future__ import annotations

import os
from pathlib import Path


def _absolute_without_resolving(path: str | Path) -> Path:
    """Return a normalized absolute path without following dataset symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def portable_path(path: str | Path, *, base: str | Path | None = None) -> str:
    """Render a path without exposing directories outside the working tree."""
    candidate = _absolute_without_resolving(path)
    anchor = _absolute_without_resolving(base or Path.cwd())
    try:
        relative = candidate.relative_to(anchor)
    except ValueError:
        return Path(path).name or "<external>"
    rendered = relative.as_posix()
    return rendered or "."


def portable_dataset_path(path: str | Path, *, base: str | Path | None = None) -> str:
    """Return a runnable workspace-relative path or reject a non-portable sample path."""
    candidate = _absolute_without_resolving(path)
    anchor = _absolute_without_resolving(base or Path.cwd())
    try:
        relative = candidate.relative_to(anchor)
    except ValueError as error:
        name = Path(path).name or "<external>"
        raise ValueError(
            "Dataset sample paths must be inside the working directory so split manifests "
            f"remain portable; {name!r} is outside it. Place or link the dataset under the "
            "project, or run the command from a common parent directory."
        ) from error
    return relative.as_posix()
