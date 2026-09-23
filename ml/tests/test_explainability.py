import numpy as np
import torch
from PIL import Image

from bone_marrow_vision.explainability.gradcam import GradCAM, save_overlay
from bone_marrow_vision.models.architectures import BaselineCNN, build_model, gradcam_target_layer


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


def test_convnext_classification_and_gradcam():
    model = build_model("convnext_tiny", num_classes=3, pretrained=False).eval()
    inputs = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert model(inputs).shape == (1, 3)
    with GradCAM(model, gradcam_target_layer(model, "convnext_tiny")) as explainer:
        heatmap = explainer.generate(inputs, class_index=2)
        assert heatmap.shape == (64, 64)
        assert np.isfinite(heatmap).all()
        assert explainer.activations.shape[-2:] == (2, 2)
        assert torch.isfinite(explainer.gradients).all()
        assert explainer.gradients.abs().sum() > 0
    assert model.classifier[2].weight.grad is not None
