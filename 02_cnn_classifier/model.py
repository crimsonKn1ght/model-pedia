"""ResNets written from scratch, plus a residual-free control network.

Two families live here:

* ``resnet20`` / ``resnet32`` - the CIFAR ResNets from the original paper.
  A 3x3 stem, then three stages of basic blocks at widths 16/32/64.
* ``resnet18`` - the ImageNet layout at widths 64/128/256/512, with the 7x7
  stride-2 stem and max-pool swapped for a 3x3 stride-1 stem. A 32x32 image
  cannot afford to lose 3/4 of its resolution before the first block.

``plain20`` is ``resnet20`` with the skip connections removed. It exists so the
comparison in ``compare.py`` isolates one thing: the identity shortcut.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with an identity shortcut around them.

    The shortcut lets gradients reach early layers unattenuated, which is why
    a 20-layer residual net trains where a 20-layer plain net stalls.
    """

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, residual: bool = True) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # When the block changes shape, the shortcut needs a 1x1 projection.
        self.shortcut = nn.Identity()
        if residual and (stride != 1 or in_ch != out_ch):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    """Generic small-image ResNet: a 3x3 stem, N stages, global pooling, linear head."""

    def __init__(
        self,
        blocks_per_stage: tuple[int, ...],
        widths: tuple[int, ...],
        num_classes: int = 10,
        in_channels: int = 3,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if len(blocks_per_stage) != len(widths):
            raise ValueError("blocks_per_stage and widths must have the same length")

        self.stem = nn.Sequential(
            conv3x3(in_channels, widths[0]),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        stages, in_ch = [], widths[0]
        for stage_idx, (n_blocks, width) in enumerate(zip(blocks_per_stage, widths)):
            for block_idx in range(n_blocks):
                # Halve the resolution at the start of every stage but the first.
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                stages.append(BasicBlock(in_ch, width, stride=stride, residual=residual))
                in_ch = width
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        return self.fc(torch.flatten(self.pool(x), 1))


def _cifar_resnet(n: int, residual: bool = True, **kwargs) -> ResNet:
    """CIFAR ResNet with 6n+2 layers."""
    return ResNet((n, n, n), (16, 32, 64), residual=residual, **kwargs)


MODELS = {
    "resnet20": lambda **kw: _cifar_resnet(3, **kw),
    "resnet32": lambda **kw: _cifar_resnet(5, **kw),
    "plain20": lambda **kw: _cifar_resnet(3, residual=False, **kw),
    "plain32": lambda **kw: _cifar_resnet(5, residual=False, **kw),
    "resnet18": lambda **kw: ResNet((2, 2, 2, 2), (64, 128, 256, 512), **kw),
}


def build_model(name: str, num_classes: int = 10, in_channels: int = 3) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    return MODELS[name](num_classes=num_classes, in_channels=in_channels)


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 32, 32)
    for name in MODELS:
        model = build_model(name)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:9s} -> logits {tuple(model(dummy).shape)}  params {params:,}")
