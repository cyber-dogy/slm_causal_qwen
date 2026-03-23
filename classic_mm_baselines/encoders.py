from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    efficientnet_b0,
    resnet18,
    resnet34,
    resnet50,
    vit_b_16,
)


def build_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    if backbone_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Identity()
        return model, 512

    if backbone_name == "resnet34":
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Identity()
        return model, 512

    if backbone_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Identity()
        return model, 2048

    if backbone_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = nn.Identity()
        return model, 1280

    if backbone_name == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads = nn.Identity()
        return model, 768

    raise ValueError(f"Unsupported backbone_name: {backbone_name}")


class MultiViewImageEncoder(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, self.feature_dim = build_backbone(backbone_name=backbone_name, pretrained=pretrained)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, channels, height, width = images.shape
        flat = images.view(batch_size * num_views, channels, height, width)
        feats = self.backbone(flat)
        return feats.view(batch_size, num_views, self.feature_dim)
