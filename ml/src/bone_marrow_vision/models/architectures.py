from __future__ import annotations

import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs):
        return self.classifier(self.features(inputs).flatten(1))


def build_model(architecture: str, num_classes: int, pretrained: bool = True):
    if architecture == "baseline_cnn":
        return BaselineCNN(num_classes)
    if architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")


def gradcam_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Return a spatial convolutional layer suitable for Grad-CAM."""
    if architecture == "resnet50":
        return model.layer4[-1]
    if architecture == "efficientnet_b0":
        return model.features[-1]
    if architecture == "baseline_cnn":
        return next(layer for layer in reversed(model.features) if isinstance(layer, nn.Conv2d))
    raise ValueError(f"Unsupported architecture: {architecture}")
