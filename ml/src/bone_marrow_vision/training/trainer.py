from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from bone_marrow_vision.training.checkpoints import (
    atomic_torch_save,
    clone_state_dict,
    final_checkpoint_path,
)
from bone_marrow_vision.training.reproducibility import capture_rng_state


def build_loss(strategy: str, labels: list[int], num_classes: int) -> nn.Module:
    if strategy == "weighted_cross_entropy":
        counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float().clamp_min(1)
        return nn.CrossEntropyLoss(weight=counts.sum() / (num_classes * counts))
    if strategy == "focal_loss":
        return FocalLoss()
    if strategy != "cross_entropy":
        raise ValueError(f"Unsupported class strategy: {strategy}")
    return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.functional.cross_entropy(logits, targets, reduction="none")
        return ((1 - cross_entropy.neg().exp()) ** self.gamma * cross_entropy).mean()


def make_sampler(
    labels: list[int], *, generator: torch.Generator | None = None
) -> WeightedRandomSampler:
    counts = torch.bincount(torch.tensor(labels)).float().clamp_min(1)
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    return WeightedRandomSampler(
        weights,
        len(weights),
        replacement=True,
        generator=generator,
    )


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return value


def _checkpoint_payload(
    *,
    role: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    class_names: list[str],
    history: list[dict[str, Any]],
    epoch: int,
    best_epoch: int,
    best_f1: float,
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "checkpoint_role": role,
        "model_state": clone_state_dict(model.state_dict()),
        "architecture": config["architecture"],
        "class_names": list(class_names),
        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "low_confidence_threshold": config.get("low_confidence_threshold", 0.60),
        "config": _clone_to_cpu(config),
        "history": _clone_to_cpu(history),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "monitored_metric": "validation_macro_f1",
        "provenance": _clone_to_cpu(provenance or {}),
        "training_state": {
            "optimizer": _clone_to_cpu(optimizer.state_dict()),
            "scheduler": _clone_to_cpu(scheduler.state_dict()),
            "scaler": _clone_to_cpu(scaler.state_dict()),
            "rng": capture_rng_state(),
        },
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    class_names: list[str],
    *,
    provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Train and write distinct, self-describing best and final checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Training device: {device} ({device_name})")
    model.to(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5,
    )
    labels = [
        train_loader.dataset.class_to_index[sample["label"]]
        for sample in train_loader.dataset.samples
    ]
    criterion = build_loss(
        config.get("class_strategy", "cross_entropy"), labels, len(class_names)
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1, best_epoch, stale_epochs = -1.0, 0, 0
    history: list[dict[str, Any]] = []
    best_checkpoint: dict[str, Any] | None = None

    for epoch in range(int(config["epochs"])):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_seen = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite classifier loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(targets)
            train_correct += int((logits.argmax(1) == targets).sum().item())
            train_seen += len(targets)
        validation = validation_metrics(model, valid_loader, device)
        valid_f1 = validation["validation_macro_f1"]
        scheduler.step(valid_f1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss / train_seen,
                "train_accuracy": train_correct / train_seen,
                **validation,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if valid_f1 > best_f1:
            best_f1, best_epoch, stale_epochs = valid_f1, epoch + 1, 0
            best_checkpoint = _checkpoint_payload(
                role="best",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                class_names=class_names,
                history=history,
                epoch=epoch + 1,
                best_epoch=best_epoch,
                best_f1=best_f1,
                provenance=provenance,
            )
            atomic_torch_save(best_checkpoint, checkpoint_path)
        else:
            stale_epochs += 1
        interval = int(config.get("checkpoint_every", 0))
        if interval > 0 and (epoch + 1) % interval == 0:
            snapshot = _checkpoint_payload(
                role="epoch", model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, config=config, class_names=class_names, history=history,
                epoch=epoch + 1, best_epoch=best_epoch, best_f1=best_f1, provenance=provenance,
            )
            atomic_torch_save(
                snapshot, checkpoint_path.parent / f"epoch_{epoch + 1:03d}.pt"
            )
        history_path = checkpoint_path.with_suffix(".history.json")
        temporary = history_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
        temporary.replace(history_path)
        print(json.dumps(history[-1]), flush=True)
        if stale_epochs >= int(config.get("early_stopping_patience", 5)):
            break

    if not history or best_checkpoint is None:
        raise ValueError("Training must run for at least one epoch")

    final_epoch = history[-1]["epoch"]
    run_final_rng = capture_rng_state()
    final_checkpoint = _checkpoint_payload(
        role="final",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        class_names=class_names,
        history=history,
        epoch=final_epoch,
        best_epoch=best_epoch,
        best_f1=best_f1,
        provenance=provenance,
    )
    summary = {
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "stopped_early": final_epoch < int(config["epochs"]),
    }
    final_checkpoint["training_summary"] = summary
    final_checkpoint["run_final_rng_state"] = run_final_rng
    atomic_torch_save(final_checkpoint, final_checkpoint_path(checkpoint_path))

    best_checkpoint["history"] = _clone_to_cpu(history)
    best_checkpoint["best_epoch"] = best_epoch
    best_checkpoint["best_validation_macro_f1"] = best_f1
    best_checkpoint["training_summary"] = summary
    best_checkpoint["run_final_rng_state"] = run_final_rng
    atomic_torch_save(best_checkpoint, checkpoint_path)
    return history


def _validation_f1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    return validation_metrics(model, loader, device)["validation_macro_f1"]


def validation_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    total_loss = 0.0
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            total_loss += nn.functional.cross_entropy(
                logits, targets.to(device), reduction="sum"
            ).item()
            predicted.extend(logits.argmax(1).cpu().tolist())
            actual.extend(targets.tolist())
    if not actual:
        raise ValueError("Evaluation split must not be empty")
    labels = list(range(logits.shape[1]))
    return {
        "validation_loss": total_loss / len(actual),
        "validation_accuracy": float(accuracy_score(actual, predicted)),
        "validation_macro_precision": float(precision_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        )),
        "validation_macro_f1": float(f1_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        )),
    }
