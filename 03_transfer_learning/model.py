"""An ImageNet-pretrained backbone with a fresh classification head.

The whole project rests on one observation: the features an ImageNet backbone
learned - edges, textures, parts - are not specific to ImageNet's 1000 classes.
Throw away the final linear layer, keep everything below it, and you have a
general-purpose image descriptor.

What is left is a choice:

* **frozen** - treat the backbone as a fixed feature extractor and train only
  the new head. Cheap, needs little data, cannot adapt the features.
* **finetune** - keep training the backbone too, at a much smaller learning
  rate so the pretrained weights are nudged rather than destroyed.

``train.py --mode both`` runs the two and prints the comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# name -> (constructor, weights enum, attribute holding the classifier)
BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights, "fc"),
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights,
        "classifier",
    ),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights, "classifier"),
}

# Every torchvision ImageNet checkpoint expects these statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _strip_classifier(backbone: nn.Module, head_attr: str) -> int:
    """Replace the backbone's own classifier with identity, return the feature width."""
    head = getattr(backbone, head_attr)
    if isinstance(head, nn.Linear):
        features = head.in_features
        setattr(backbone, head_attr, nn.Identity())
        return features

    # mobilenet / efficientnet keep a small MLP; only the last Linear is
    # ImageNet-specific, so the layer before it stays part of the features.
    if isinstance(head, nn.Sequential):
        for index in range(len(head) - 1, -1, -1):
            if isinstance(head[index], nn.Linear):
                features = head[index].in_features
                head[index] = nn.Identity()
                return features
    raise TypeError(f"cannot strip classifier of type {type(head).__name__}")


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return ``(backbone, feature_dim)`` with the ImageNet head removed."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}, expected one of {sorted(BACKBONES)}")
    constructor, weights_enum, head_attr = BACKBONES[name]

    weights = weights_enum.DEFAULT if pretrained else None
    try:
        backbone = constructor(weights=weights)
    except Exception as error:  # noqa: BLE001 - the cause is almost always the network
        raise RuntimeError(
            f"could not build {name} with pretrained weights ({error}).\n"
            "torchvision downloads them from download.pytorch.org on first use; "
            "check the connection, or pass --no-pretrained to start from scratch "
            "(which defeats the point of this project, but proves the pipeline runs)."
        ) from error

    return backbone, _strip_classifier(backbone, head_attr)


class TransferModel(nn.Module):
    """Pretrained backbone + a new linear head sized for the target dataset."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 102,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, self.feature_dim = build_backbone(backbone_name, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone output only - what the frozen mode caches once and reuses."""
        return self.backbone(x)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def parameter_groups(self, head_lr: float, backbone_lr: float) -> list[dict]:
        """Discriminative learning rates: gentle on pretrained weights, brisk on the head."""
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)
    for name in BACKBONES:
        model = TransferModel(name, num_classes=102, pretrained=False)
        total = sum(p.numel() for p in model.parameters())
        head = sum(p.numel() for p in model.head.parameters())
        print(
            f"{name:20s} features {model.feature_dim:5d}  "
            f"logits {tuple(model(dummy).shape)}  "
            f"params {total:,} (head {head:,})"
        )
