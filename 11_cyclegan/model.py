"""CycleGAN: unpaired image-to-image translation.

Pix2Pix needs matched ``(source, target)`` pairs.  CycleGAN removes that
requirement: it learns to map between two *collections* of images that have no
correspondence at all -- a pile of horses and a pile of zebras.

Without pairs, the adversarial loss alone is hopelessly under-constrained.  Any
mapping that lands in domain B satisfies the discriminator, including one that
ignores the input entirely.  **Cycle consistency** supplies the missing
constraint: with a second generator running the other way, require

    F(G(a)) ~= a      and      G(F(b)) ~= b

so a translation must retain enough of the input to be reversible.  That single
idea is what makes the whole thing work.

Two smaller details from the paper are kept here because they matter in
practice:

* the **least-squares GAN loss** (squared error against 1/0 rather than binary
  cross entropy), which is markedly more stable;
* the **identity loss**, ``G(b) ~= b``, which stops the generator from shifting
  colours it has no reason to touch.
"""

from __future__ import annotations

import random
from typing import List

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Instance-normalised residual block, the body of the CycleGAN generator."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """Downsample twice, apply residual blocks, upsample back.

    Instance normalisation rather than batch normalisation: style transfer works
    per image, and batch statistics leak information between samples.
    """

    def __init__(self, channels: int = 3, base: int = 48, num_blocks: int = 4):
        super().__init__()
        layers: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, base, 7),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
        ]
        in_ch = base
        for _ in range(2):
            out_ch = in_ch * 2
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        layers += [ResidualBlock(in_ch) for _ in range(num_blocks)]

        for _ in range(2):
            out_ch = in_ch // 2
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        layers += [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, channels, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN, as in Pix2Pix, but judging a single image rather than a pair."""

    def __init__(self, channels: int = 3, base: int = 48, layers: int = 3):
        super().__init__()
        blocks: List[nn.Module] = [
            nn.Conv2d(channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = base
        for i in range(1, layers):
            out_ch = min(base * 2**i, base * 8)
            stride = 2 if i < layers - 1 else 1
            blocks += [
                nn.Conv2d(in_ch, out_ch, 4, stride, 1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        blocks.append(nn.Conv2d(in_ch, 1, 4, 1, 1))
        self.net = nn.Sequential(*blocks)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImagePool:
    """Buffer of previously generated images, from the CycleGAN paper.

    The discriminator is shown a mix of the generator's current output and its
    recent history.  This stops the pair from chasing each other around a cycle
    of one-step-behind responses, which is a common unpaired-GAN failure.
    """

    def __init__(self, size: int = 50):
        self.size = size
        self.images: List[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images
        out = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(image.detach().clone())
                out.append(image)
            elif random.random() > 0.5:
                idx = random.randrange(self.size)
                out.append(self.images[idx].clone())
                self.images[idx] = image.detach().clone()
            else:
                out.append(image)
        return torch.cat(out, dim=0)


def weights_init(module: nn.Module) -> None:
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0)
    elif "InstanceNorm" in name and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
