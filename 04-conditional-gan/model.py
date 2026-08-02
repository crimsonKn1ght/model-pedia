"""Conditional GANs: cGAN and ACGAN, two ways of steering what gets generated.

A DCGAN produces *some* digit; a conditional GAN produces *the digit you asked
for*.  Both variants here condition the generator by concatenating a learned
class embedding to the noise vector.  They differ in how the discriminator is
told about the label, and that difference is the whole lesson:

**cGAN** (Mirza & Osindero, 2014) feeds the label to the discriminator too, as
extra constant-valued input channels.  The discriminator answers one question:
"is this a real image *of this class*?"  A real image paired with the wrong
label is a fake.

**ACGAN** (Odena et al., 2017) hides the label from the discriminator's input
and instead gives it a second head that must *classify* the image.  The
discriminator answers two questions: "is this real?" and "which class is it?".
The auxiliary classification loss is applied to real and generated images alike,
which pushes the generator to make class-distinctive samples.

ACGAN's classifier head doubles as a free evaluation signal, and the same idea
underpins the classifier-accuracy metric in ``evaluate.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _stages(image_size: int, min_size: int = 4) -> int:
    n = 0
    size = image_size
    while size > min_size:
        if size % 2:
            raise ValueError(f"image_size={image_size} must be a power-of-two multiple of {min_size}")
        size //= 2
        n += 1
    return n


class ConditionalGenerator(nn.Module):
    """``(z, y)`` -> image.  Identical for cGAN and ACGAN."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        channels: int = 1,
        image_size: int = 32,
        base: int = 64,
        embed_dim: int = 32,
    ):
        super().__init__()
        stages = _stages(image_size)
        widths = [min(base * 2**i, base * 8) for i in range(stages)][::-1]
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        layers = [
            nn.ConvTranspose2d(latent_dim + embed_dim, widths[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        ]
        in_ch = widths[0]
        for out_ch in widths[1:]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        layers += [nn.ConvTranspose2d(in_ch, channels, 4, 2, 1), nn.Tanh()]

        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.apply(weights_init)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Conditioning by concatenation: the label becomes part of the input code.
        h = torch.cat([z, self.label_embedding(y)], dim=1)
        return self.net(h.view(h.shape[0], -1, 1, 1))

    @torch.no_grad()
    def sample(self, n: int, device: torch.device, labels: torch.Tensor = None) -> torch.Tensor:
        if labels is None:
            labels = torch.randint(0, self.num_classes, (n,), device=device)
        z = torch.randn(n, self.latent_dim, device=device)
        return self(z, labels)

    @torch.no_grad()
    def sample_class_grid(self, per_class: int, device: torch.device) -> torch.Tensor:
        """One row per class, ``per_class`` samples wide -- the money shot."""
        labels = torch.arange(self.num_classes, device=device).repeat_interleave(per_class)
        z = torch.randn(len(labels), self.latent_dim, device=device)
        return self(z, labels)


class ProjectionDiscriminator(nn.Module):
    """Shared convolutional trunk with a mode-dependent head.

    ``mode="cgan"``  -- the label enters as extra input channels.
    ``mode="acgan"`` -- the label is not an input; an auxiliary head predicts it.
    """

    def __init__(
        self,
        num_classes: int = 10,
        channels: int = 1,
        image_size: int = 32,
        base: int = 64,
        mode: str = "acgan",
    ):
        super().__init__()
        if mode not in {"cgan", "acgan"}:
            raise ValueError("mode must be 'cgan' or 'acgan'")
        self.mode = mode
        self.num_classes = num_classes
        self.image_size = image_size

        in_channels = channels + (num_classes if mode == "cgan" else 0)
        stages = _stages(image_size)
        widths = [min(base * 2**i, base * 8) for i in range(stages)]

        layers = [nn.Conv2d(in_channels, widths[0], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        in_ch = widths[0]
        for out_ch in widths[1:]:
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        self.trunk = nn.Sequential(*layers)

        self.adversarial_head = nn.Conv2d(in_ch, 1, 4, 1, 0)
        self.classifier_head = nn.Conv2d(in_ch, num_classes, 4, 1, 0) if mode == "acgan" else None
        self.apply(weights_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(real/fake logit, class logits or None)``."""
        if self.mode == "cgan":
            # One constant-valued plane per class, all zero except the true label.
            onehot = torch.zeros(x.shape[0], self.num_classes, device=x.device)
            onehot.scatter_(1, y.view(-1, 1), 1.0)
            planes = onehot.view(x.shape[0], self.num_classes, 1, 1).expand(
                -1, -1, x.shape[2], x.shape[3]
            )
            x = torch.cat([x, planes], dim=1)

        h = self.trunk(x)
        validity = self.adversarial_head(h).view(-1)
        class_logits = self.classifier_head(h).flatten(1) if self.classifier_head else None
        return validity, class_logits


def weights_init(module: nn.Module) -> None:
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
