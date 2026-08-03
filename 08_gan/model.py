"""DCGAN: a generator and a discriminator that train by competing.

    python model.py

Every model before this one had a loss you could evaluate on a single example. A GAN
does not. There is no likelihood, no reconstruction target, nothing to compare the
output against - only a second network whose job is to tell generated images from real
ones, and a generator whose job is to stop it. The training signal is entirely
adversarial, which is what makes GANs produce sharp images and what makes them
difficult.

The DCGAN paper is a list of architectural choices that made this stable enough to
work, and they are all here:

* **strided convolutions, no pooling** - the discriminator learns its own
  downsampling, the generator its own upsampling;
* **batch normalisation everywhere except** the discriminator's first layer and the
  generator's last;
* **LeakyReLU in the discriminator, ReLU in the generator**, because the
  discriminator's gradients have to survive being wrong;
* **tanh output**, so the generator's range is bounded. The rest of this repository
  keeps images in ``[0, 1]``, so ``Generator.forward`` rescales on the way out and
  ``Discriminator.forward`` rescales on the way in - the architecture is the paper's,
  the interface is the repository's.

The loss is the **non-saturating** form: the generator maximises ``log D(G(z))``
rather than minimising ``log(1 - D(G(z)))``. Both have the same optimum and only one
has usable gradients when the discriminator is winning, which it is at the start of
every run.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES = ("bce", "hinge")


def _levels_for(image_size: int) -> int:
    """Upsampling stages between a 4x4 seed and the image."""
    if image_size % 4 or not float(math.log2(image_size)).is_integer():
        raise ValueError(f"image size {image_size} must be a power of two, at least 8")
    return int(math.log2(image_size)) - 2


class Generator(nn.Module):
    """``z`` -> a 4x4 seed -> transposed convolutions up to the image."""

    def __init__(
        self,
        latent_dim: int = 64,
        out_channels: int = 1,
        image_size: int = 32,
        width: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        levels = _levels_for(image_size)
        channels = width * 2 ** (levels - 1)

        layers = [
            # A 1x1 "image" of latent values projected to a 4x4 feature map.
            nn.ConvTranspose2d(latent_dim, channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(levels - 1):
            layers += [
                nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ]
            channels //= 2
        # No batch norm before the output: it would fight the tanh.
        layers += [nn.ConvTranspose2d(channels, out_channels, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(dcgan_init)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Returns images in ``[0, 1]``."""
        if latent.dim() == 2:
            latent = latent.view(latent.size(0), -1, 1, 1)
        return (self.net(latent) + 1.0) / 2.0

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return self(torch.randn(n, self.latent_dim, device=device))


class Discriminator(nn.Module):
    """Strided convolutions down to a single logit."""

    def __init__(self, in_channels: int = 1, image_size: int = 32, width: int = 64) -> None:
        super().__init__()
        levels = _levels_for(image_size)
        channels = width

        # No batch norm on the first layer: it would normalise away the very
        # statistics that distinguish a real batch from a generated one.
        layers = [
            nn.Conv2d(in_channels, channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(levels - 1):
            layers += [
                nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels *= 2
        layers += [nn.Conv2d(channels, 1, 4, 1, 0)]
        self.net = nn.Sequential(*layers)
        self.apply(dcgan_init)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Takes images in ``[0, 1]``, returns one logit per image."""
        return self.net(images * 2.0 - 1.0).flatten(1).squeeze(1)


def dcgan_init(module: nn.Module) -> None:
    """The paper's initialisation: normal(0, 0.02) on convolutions and batch norms."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    loss: str = "bce",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Push real logits up and fake logits down.

    ``label_smoothing`` targets ``1 - eps`` for real images instead of 1. A
    discriminator that is *certain* stops producing gradient, so refusing to let it be
    certain is one of the cheapest stabilisers available.
    """
    if loss == "hinge":
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    real_target = torch.full_like(real_logits, 1.0 - label_smoothing)
    fake_target = torch.zeros_like(fake_logits)
    return (
        F.binary_cross_entropy_with_logits(real_logits, real_target)
        + F.binary_cross_entropy_with_logits(fake_logits, fake_target)
    )


def generator_loss(fake_logits: torch.Tensor, loss: str = "bce") -> torch.Tensor:
    """Non-saturating generator objective."""
    if loss == "hinge":
        return -fake_logits.mean()
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def build_models(
    latent_dim: int = 64,
    in_channels: int = 1,
    image_size: int = 32,
    width: int = 64,
) -> tuple[Generator, Discriminator]:
    return (
        Generator(latent_dim, in_channels, image_size, width),
        Discriminator(in_channels, image_size, width),
    )


if __name__ == "__main__":
    for channels, size in ((1, 32), (3, 32), (3, 64)):
        generator, discriminator = build_models(64, channels, size)
        images = generator.sample(4)
        logits = discriminator(images)
        print(
            f"C={channels} {size}x{size}: samples {tuple(images.shape)} in "
            f"[{images.min():.2f}, {images.max():.2f}]  logits {tuple(logits.shape)}  "
            f"G {sum(p.numel() for p in generator.parameters()):,} "
            f"D {sum(p.numel() for p in discriminator.parameters()):,}"
        )

    generator, discriminator = build_models(64, 1, 32)
    real = torch.rand(8, 1, 32, 32)
    fake = generator(torch.randn(8, 64))
    for loss in LOSSES:
        d_loss = discriminator_loss(discriminator(real), discriminator(fake.detach()), loss)
        g_loss = generator_loss(discriminator(fake), loss)
        print(f"{loss:6s}: D {d_loss.item():.4f}  G {g_loss.item():.4f}")

    # At initialisation the discriminator should be near chance, and the BCE loss
    # near 2 log 2 = 1.386. Far from it means the initialisation is wrong.
    with torch.no_grad():
        baseline = discriminator_loss(discriminator(real), discriminator(fake), "bce")
    print(f"\nD loss at init {baseline.item():.4f} (chance is {2 * math.log(2):.4f})")
