"""A U-Net for per-pixel classification, plus two controls.

    python model.py

Segmentation asks for a label at every pixel, which pulls in two directions. To
know *what* something is you need a large receptive field, and the cheap way to
get one is to downsample. To know *where* its edges are you need full resolution,
which downsampling destroys. Every architecture here is one answer to that
tension:

* ``unet`` - downsample for context, then upsample, and pass full-resolution
  features across skip connections so the decoder can put the edges back.
* ``unet_noskip`` - the same network with the skips cut. Same depth, same
  receptive field, about 10% fewer parameters (the decoder no longer takes
  concatenated input), and no route for fine detail. The gap between the two is
  what the skips are worth, and it shows up on the thin classes first.
* ``dilated`` - never downsample at all. Dilated convolutions grow the receptive
  field while keeping full resolution, so no edge detail is ever thrown away. The
  trade is the other way round: it runs every layer at full size, so it can only
  afford a narrow, shallow stack (about a seventh of the U-Net's parameters for
  roughly the same wall-clock), and its receptive field grows with depth alone.

The output is ``(N, num_classes, H, W)`` raw logits - no softmax, because
``CrossEntropyLoss`` applies its own.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x 2, with an optional dilation."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSegmenter(nn.Module):
    """Three-level U-Net. Any input size divisible by 4 works."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
        skips: bool = True,
    ) -> None:
        super().__init__()
        self.skips = skips
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.bottleneck = DoubleConv(c2, c3)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 * 2 if skips else c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 * 2 if skips else c1, c1)
        self.classifier = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)  # full resolution
        e2 = self.enc2(self.pool(e1))  # half
        b = self.bottleneck(self.pool(e2))  # quarter - the widest receptive field

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1) if self.skips else d2)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1) if self.skips else d1)
        return self.classifier(d1)


class DilatedNet(nn.Module):
    """Full-resolution control: dilation instead of downsampling.

    Dilations of 1, 2, 4, 8 give roughly the receptive field of the U-Net's three
    levels without ever losing a pixel. The catch is arithmetic - every block runs
    at full resolution, so this is the slowest model here by a wide margin.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        blocks = []
        channels = in_channels
        for dilation in dilations:
            blocks.append(DoubleConv(channels, base_channels, dilation=dilation))
            channels = base_channels
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.blocks(x))


MODELS = {
    "unet": {"kind": "unet", "skips": True},
    "unet_noskip": {"kind": "unet", "skips": False},
    "dilated": {"kind": "dilated"},
}


def build_model(
    name: str = "unet",
    in_channels: int = 3,
    num_classes: int = 3,
    base_channels: int = 32,
) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    spec = MODELS[name]
    if spec["kind"] == "unet":
        return UNetSegmenter(in_channels, num_classes, base_channels, skips=spec["skips"])
    return DilatedNet(in_channels, num_classes, base_channels)


if __name__ == "__main__":
    for size in (32, 64, 96):
        dummy = torch.rand(2, 3, size, size)
        for name in MODELS:
            model = build_model(name, 3, 4)
            params = sum(p.numel() for p in model.parameters())
            print(f"{size}x{size} {name:12s} out {tuple(model(dummy).shape)}  params {params:,}")
        print()

    # Receptive field sanity check: change one pixel, see how far the output moves.
    model = build_model("unet", 3, 4).eval()
    base = torch.zeros(1, 3, 64, 64)
    poked = base.clone()
    poked[0, :, 32, 32] = 1.0
    with torch.no_grad():
        difference = (model(poked) - model(base)).abs().sum(dim=1)[0]
    rows = difference.sum(dim=1).nonzero().flatten()
    print(f"one pixel at (32, 32) influences rows {rows.min().item()}-{rows.max().item()}")
