from __future__ import annotations

import hashlib
import logging
import secrets
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from app.core.config import settings

LOGGER = logging.getLogger(__name__)
SUPPORTED_ARCHITECTURES = {"baseline_cnn", "resnet50", "efficientnet_b0", "convnext_tiny"}
_MODEL_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: torch.nn.Module
    architecture: str
    class_names: tuple[str, ...]
    image_size: int
    sha256: str


def resolve_project_path(path: str | Path) -> Path:
    return settings.resolve_path(path)


def _require_within(path: Path, root: Path, label: str) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside {resolved_root}") from error
    return resolved_path


def resolve_model_checkpoint(path: str | Path) -> Path:
    return _require_within(
        resolve_project_path(path),
        settings.resolve_path(settings.model_dir),
        "Model checkpoint",
    )


def checkpoint_sha256(path: str | Path) -> str:
    checkpoint_file = resolve_model_checkpoint(path)
    file_size = checkpoint_file.stat().st_size
    if file_size > settings.max_checkpoint_bytes:
        raise ValueError("Model checkpoint exceeds the configured size limit")

    digest = hashlib.sha256()
    with checkpoint_file.open("rb") as checkpoint_stream:
        for chunk in iter(lambda: checkpoint_stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _load_model_cached(
    checkpoint_path: str,
    modified_at_ns: int,
    file_size: int,
    expected_architecture: str,
    expected_sha256: str | None,
) -> LoadedCheckpoint:
    del modified_at_ns  # This value intentionally participates in the cache key.
    from bone_marrow_vision.models import build_model

    if file_size > settings.max_checkpoint_bytes:
        raise ValueError("Model checkpoint exceeds the configured size limit")

    payload = Path(checkpoint_path).read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and not secrets.compare_digest(actual_sha256, expected_sha256):
        raise ValueError("Model checkpoint does not match its registered SHA-256 digest")

    checkpoint = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Model checkpoint payload must be a mapping")

    required_keys = {"model_state", "architecture", "class_names"}
    missing_keys = sorted(required_keys - checkpoint.keys())
    if missing_keys:
        raise ValueError(f"Model checkpoint is missing required keys: {', '.join(missing_keys)}")

    architecture = checkpoint["architecture"]
    if not isinstance(architecture, str) or architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError("Model checkpoint uses an unsupported architecture")
    if architecture != expected_architecture:
        raise ValueError("Model checkpoint architecture does not match the registry")

    raw_class_names = checkpoint["class_names"]
    if not isinstance(raw_class_names, (list, tuple)) or not raw_class_names:
        raise ValueError("Model checkpoint class_names must be a non-empty sequence")
    class_names = tuple(raw_class_names)
    if not all(isinstance(class_name, str) and class_name for class_name in class_names):
        raise ValueError("Model checkpoint contains an invalid class name")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Model checkpoint contains duplicate class names")

    image_size = checkpoint.get("image_size", 224)
    if not isinstance(image_size, int) or not 32 <= image_size <= 2048:
        raise ValueError("Model checkpoint contains an invalid image size")
    model_state = checkpoint["model_state"]
    if not isinstance(model_state, Mapping):
        raise TypeError("Model checkpoint model_state must be a mapping")

    model = build_model(architecture, len(class_names), pretrained=False)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    LOGGER.info(
        "Loaded model checkpoint architecture=%s sha256=%s path=%s",
        architecture,
        actual_sha256,
        checkpoint_path,
    )
    return LoadedCheckpoint(model, architecture, class_names, image_size, actual_sha256)


def load_model(
    checkpoint_path: str | Path,
    expected_architecture: str,
    expected_sha256: str | None = None,
) -> LoadedCheckpoint:
    checkpoint_file = resolve_model_checkpoint(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")
    stat = checkpoint_file.stat()
    with _MODEL_LOAD_LOCK:
        return _load_model_cached(
            str(checkpoint_file),
            stat.st_mtime_ns,
            stat.st_size,
            expected_architecture,
            expected_sha256,
        )


def predict(
    image_path: Path,
    checkpoint_path: str,
    output_path: Path,
    threshold: float,
    *,
    expected_architecture: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    from bone_marrow_vision.data.dataset import build_transforms
    from bone_marrow_vision.explainability.gradcam import GradCAM, save_overlay
    from bone_marrow_vision.models import gradcam_target_layer

    if not 0 <= threshold <= 1:
        raise ValueError("Low-confidence threshold must be between zero and one")
    loaded = load_model(checkpoint_path, expected_architecture, expected_sha256)

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = build_transforms(loaded.image_size, training=False)(image)

    with _INFERENCE_LOCK:
        target_layer = gradcam_target_layer(loaded.model, loaded.architecture)
        explainer = GradCAM(loaded.model, target_layer)
        try:
            with torch.enable_grad():
                logits = loaded.model(tensor.unsqueeze(0))
                probabilities = logits.softmax(dim=1)[0]
                predicted_index = int(probabilities.argmax().item())
                heatmap = explainer.generate(tensor.unsqueeze(0), predicted_index)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_overlay(image_path, heatmap, output_path)
        finally:
            explainer.close()

    top_indices = probabilities.topk(min(3, len(loaded.class_names))).indices.tolist()
    confidence = probabilities[predicted_index].item()
    return {
        "predicted_class": loaded.class_names[predicted_index],
        "confidence": confidence,
        "top_predictions_json": [
            {"class": loaded.class_names[index], "probability": probabilities[index].item()} for index in top_indices
        ],
        "low_confidence": confidence < threshold,
    }
