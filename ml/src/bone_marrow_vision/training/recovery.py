"""Epoch-boundary recovery with a self-contained best checkpoint."""

from pathlib import Path

import torch

from .checkpoints import load_checkpoint


def recovery_path(best_path: str | Path) -> Path:
    path = Path(best_path)
    return path.with_name(f"{path.stem}.latest{path.suffix or '.pt'}")


def loader_state(loader) -> dict:
    generators = {}
    for name, owner in (("loader", loader), ("sampler", loader.sampler)):
        generator = getattr(owner, "generator", None)
        if generator is not None:
            generators[name] = generator.get_state()
    return {
        "generators": generators,
        "num_workers": loader.num_workers,
        "persistent_workers": loader.persistent_workers,
        "batch_size": loader.batch_size,
        "drop_last": loader.drop_last,
        "sampler": type(loader.sampler).__name__,
        "dataset_size": len(loader.dataset),
    }


def restore_loader(loader, state: dict) -> None:
    current = loader_state(loader)
    if current["persistent_workers"] or state["persistent_workers"]:
        raise ValueError("Resume requires non-persistent data workers")
    if {k: v for k, v in current.items() if k != "generators"} != {
        k: v for k, v in state.items() if k != "generators"
    } or set(current["generators"]) != set(state["generators"]):
        raise ValueError("Resume data loader configuration changed")
    for name, owner in (("loader", loader), ("sampler", loader.sampler)):
        if name in state["generators"]:
            owner.generator.set_state(state["generators"][name])


def validated_recovery(path, config, class_names, provenance, device, *, finalize_completed=False) -> dict:
    checkpoint = load_checkpoint(path)
    if checkpoint.get("recovery_version") != 1:
        raise ValueError("Resume requires a new .latest.pt recovery checkpoint; legacy checkpoints are inference-only")
    ignored = {"epochs", "checkpoint_every"}
    if {k: v for k, v in config.items() if k not in ignored} != {
        k: v for k, v in checkpoint["config"].items() if k not in ignored
    }:
        raise ValueError("Resume training configuration changed")
    if checkpoint["class_names"] != class_names:
        raise ValueError("Resume class order changed")
    def inputs(value):
        return {k: v["sha256"] for k, v in value.get("input_files", {}).items() if k != "config"}
    if inputs(checkpoint["provenance"]) != inputs(provenance or {}):
        raise ValueError("Resume dataset or split manifest changed")
    state = checkpoint["recovery"]
    if state["device"] != device.type or state["cuda_devices"] != (
        torch.cuda.device_count() if device.type == "cuda" else 0
    ):
        raise ValueError("Resume device topology changed")
    epoch = checkpoint["epoch"]
    if not checkpoint["history"] or [r["epoch"] for r in checkpoint["history"]] != list(range(1, epoch + 1)):
        raise ValueError("Resume checkpoint history is inconsistent")
    if int(config["epochs"]) < epoch or (int(config["epochs"]) == epoch and not finalize_completed):
        raise ValueError("Target epochs must exceed the last completed epoch")
    if state["stale_epochs"] >= int(config.get("early_stopping_patience", 5)) and not finalize_completed:
        raise ValueError("This run already reached early stopping")
    return checkpoint
