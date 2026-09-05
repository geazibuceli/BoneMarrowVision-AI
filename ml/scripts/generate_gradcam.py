from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bone_marrow_vision.data.dataset import BoneMarrowDataset, sha256_file
from bone_marrow_vision.explainability.gradcam import GradCAM, save_overlay
from bone_marrow_vision.models import build_model, gradcam_target_layer
from bone_marrow_vision.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM overlay for one image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="reports/gradcam/overlay.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)
    output_path = Path(args.output)
    checkpoint = load_checkpoint(checkpoint_path)
    class_names = checkpoint["class_names"]
    architecture = checkpoint["architecture"]
    model = build_model(architecture, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    sample = [{"path": str(image_path), "label": class_names[0]}]
    tensor, _ = BoneMarrowDataset(
        sample,
        class_names,
        int(checkpoint.get("image_size", 224)),
    )[0]
    batch = tensor.unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(batch).softmax(dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
    with GradCAM(model, gradcam_target_layer(model, architecture)) as explainer:
        with torch.enable_grad():
            heatmap = explainer.generate(batch, predicted_index)
    save_overlay(image_path, heatmap, output_path)
    top_indices = probabilities.topk(min(3, len(class_names))).indices.tolist()
    confidence = float(probabilities[predicted_index])
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "image_sha256": sha256_file(image_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "predicted_class": class_names[predicted_index],
                "confidence": confidence,
                "top_predictions": [
                    {"class": class_names[index], "probability": float(probabilities[index])}
                    for index in top_indices
                ],
                "low_confidence": confidence
                < float(checkpoint.get("low_confidence_threshold", 0.60)),
                "disclaimer": (
                    "Grad-CAM is an interpretability aid, not proof of clinical correctness."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
