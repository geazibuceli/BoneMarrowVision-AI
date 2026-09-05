from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from PIL import Image


class GradCAM:
    """Gradient-weighted activation map for convolutional model layers."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor | None, ...],
        output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if output[0] is None:
            raise RuntimeError("The target Grad-CAM layer did not produce gradients")
        self.gradients = output[0].detach()

    def generate(self, image: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        class_index = int(logits.argmax(1).item()) if class_index is None else class_index
        if not 0 <= class_index < logits.shape[1]:
            raise ValueError(f"class_index must be between 0 and {logits.shape[1] - 1}")
        logits[:, class_index].sum().backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).relu()
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def save_overlay(image_path: str | Path, heatmap: np.ndarray, output_path: str | Path) -> None:
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    image = np.asarray(source, dtype=np.float32) / 255
    heatmap_image = Image.fromarray((heatmap.clip(0, 1) * 255).astype(np.uint8)).resize(source.size)
    heatmap = np.asarray(heatmap_image, dtype=np.float32) / 255
    colors = matplotlib.colormaps["magma"](heatmap)[..., :3]
    blended = (0.55 * image + 0.45 * colors).clip(0, 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((blended * 255).astype(np.uint8)).save(output)
