from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load tensor-only checkpoints without enabling arbitrary pickle globals."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    required = {"model_state", "architecture", "class_names"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {', '.join(missing)}")
    return checkpoint


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def final_checkpoint_path(best_path: str | Path) -> Path:
    path = Path(best_path)
    suffix = path.suffix or ".pt"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}.final{suffix}")


def atomic_torch_save(payload: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
