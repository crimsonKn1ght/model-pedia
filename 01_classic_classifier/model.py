"""The two reference classifiers: a plain MLP and a LeNet-style CNN.

Both take a 1x28x28 tensor and return raw logits over ``num_classes``.
Keeping the interface identical is the whole point of this project: the same
training loop can drive either one, so the accuracy gap is attributable to the
architecture rather than to the surrounding code.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Flatten the image and push it through fully connected layers.

    Every pixel gets its own weight into the first hidden layer, so the model
    has no notion of which pixels are neighbours; spatial structure has to be
    relearned from scratch for every position.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int] = (1, 28, 28),
        hidden: tuple[int, ...] = (256, 128),
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        in_features = int(torch.tensor(in_shape).prod())

        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeNet5(nn.Module):
    """LeNet-5 in its modern form: ReLU and max-pooling instead of tanh/avg.

    The convolutions share weights across the image, so an edge detector
    learned in one corner is available everywhere. That weight sharing is why
    this network beats the MLP with roughly a quarter of the parameters.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # padding=2 turns a 28x28 input into the 32x32 LeNet expects.
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 6 x 14 x 14
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 x 5 x 5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


MODELS = {"mlp": MLP, "lenet": LeNet5}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    return MODELS[name](num_classes=num_classes)


if __name__ == "__main__":
    dummy = torch.randn(2, 1, 28, 28)
    for name in MODELS:
        model = build_model(name)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:6s} -> logits {tuple(model(dummy).shape)}  params {params:,}")
