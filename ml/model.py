"""EfficientNet-B0 binary classifier for DeepTrace."""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


NUM_CLASSES = 2


def build_model(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Build a fake/real EfficientNet-B0 classifier.

    Labels follow ml.data.dataset.LABEL_MAP: fake=0, real=1.
    When pretrained=True, ImageNet weights are downloaded if not cached.
    The feature extractor is frozen by default for initial classifier training.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, NUM_CLASSES)
    return model


def get_device() -> torch.device:
    """Choose CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
