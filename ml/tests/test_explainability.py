import numpy as np
import torch
from PIL import Image

from bone_marrow_vision.explainability.gradcam import GradCAM, save_overlay
from bone_marrow_vision.models.architectures import BaselineCNN, gradcam_target_layer


def test_gradcam_and_overlay_support_original_image_size(tmp_path):
    model = BaselineCNN(num_classes=2).eval()
    target_layer = gradcam_target_layer(model, "baseline_cnn")
    assert isinstance(target_layer, torch.nn.Conv2d)
    explainer = GradCAM(model, target_layer)
    heatmap = explainer.generate(torch.rand(1, 3, 32, 32))
    assert heatmap.shape == (32, 32)
    assert np.isfinite(heatmap).all()

    source = tmp_path / "source.png"
    output = tmp_path / "overlay.png"
    Image.new("RGB", (47, 29), (120, 80, 40)).save(source)
    save_overlay(source, heatmap, output)
    assert Image.open(output).size == (47, 29)
    explainer.close()


def test_gradcam_rejects_invalid_class_index():
    model = BaselineCNN(num_classes=2).eval()
    with GradCAM(model, gradcam_target_layer(model, "baseline_cnn")) as explainer:
        try:
            explainer.generate(torch.rand(1, 3, 16, 16), class_index=2)
        except ValueError as error:
            assert "class_index" in str(error)
        else:
            raise AssertionError("Expected an invalid class index to fail")
