"""Pix2Pix: paired image-to-image translation with a U-Net and a PatchGAN.

Given pairs ``(source, target)`` -- an edge map and the photo it came from, a
facade layout and the photograph of that facade -- Pix2Pix learns the mapping
``source -> target``.  Two ideas carry the method:

**The U-Net generator.**  Translation preserves structure: an edge in the input
belongs at the same place in the output.  Skip connections wire each encoder
stage directly to the matching decoder stage, so spatial detail bypasses the
bottleneck instead of having to be reconstructed from it.  Drop the skips and
outputs immediately turn mushy.

**The PatchGAN discriminator.**  Rather than one verdict per image, the
discriminator outputs a grid of verdicts, each covering a local patch.  This
targets the adversarial loss at *texture*, which is what L1 cannot model, while
an explicit L1 term handles low-frequency correctness.  The combined objective::

    L = L_cGAN(G, D)  +  lambda_L1 * || target - G(source) ||_1

``lambda_L1 = 100`` in the paper: the adversarial term is there to sharpen, not
to lead.  Set ``--lambda-l1 0`` to watch the output collapse into plausible but
unfaithful texture.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Encoder stage: halve the resolution, double the channels."""

    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder stage: double the resolution, then concatenate the skip tensor."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            # Dropout at test time is Pix2Pix's only source of output diversity.
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.block(x), skip], dim=1)


class UNetGenerator(nn.Module):
    """Encoder-decoder with skip connections at every resolution."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        self.depth = depth
        widths = [min(base * 2**i, base * 8) for i in range(depth)]

        self.downs = nn.ModuleList()
        in_ch = in_channels
        for i, out_ch in enumerate(widths):
            self.downs.append(DownBlock(in_ch, out_ch, normalize=i > 0))
            in_ch = out_ch

        # Bottleneck at the lowest resolution.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True)
        )

        self.ups = nn.ModuleList()
        reversed_widths = widths[::-1]
        for i, skip_ch in enumerate(reversed_widths[1:], start=1):
            # After each UpBlock the skip tensor is concatenated, so the next
            # block sees `out_ch + skip_ch` channels.
            self.ups.append(UpBlock(in_ch, skip_ch, dropout=0.5 if i <= 2 else 0.0))
            in_ch = skip_ch * 2

        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_channels, 4, 2, 1), nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for down in self.downs:
            h = down(h)
            skips.append(h)
        h = self.bottleneck(h)

        # skips[-1] is the bottleneck's own resolution and is not re-used.
        for up, skip in zip(self.ups, skips[-2::-1]):
            h = up(h, skip)
        return self.final(h)


class PatchDiscriminator(nn.Module):
    """Classifies overlapping patches rather than whole images.

    The output is an ``N x N`` grid of logits; each element sees a limited
    receptive field ("patch") of the input pair.
    """

    def __init__(self, in_channels: int = 6, base: int = 64, layers: int = 3):
        super().__init__()
        blocks: List[nn.Module] = [
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = base
        for i in range(1, layers):
            out_ch = min(base * 2**i, base * 8)
            stride = 2 if i < layers - 1 else 1
            blocks += [
                nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        blocks.append(nn.Conv2d(in_ch, 1, 4, 1, 1))
        self.net = nn.Sequential(*blocks)
        self.apply(weights_init)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # The discriminator always sees the pair, never the output alone: that
        # is what makes it judge *correspondence* and not just realism.
        return self.net(torch.cat([source, target], dim=1))


def weights_init(module: nn.Module) -> None:
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
