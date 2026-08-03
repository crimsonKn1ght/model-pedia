"""Conditional GANs: cGAN and ACGAN, on the same DCGAN backbone.

    python model.py

An unconditional GAN (project 08) samples *something* from the data distribution. A
conditional one samples something of a **requested class**, which is both more useful
and much easier to evaluate: you can ask an independent classifier whether it got the
class you asked for.

The generator's side is the same in both variants - embed the label, concatenate it to
the latent vector, carry on. The interesting difference is what the discriminator does
with the label, and there are two answers:

**cGAN** gives the label to the discriminator too, as extra input channels holding a
broadcast class embedding. The discriminator is then judging *pairs*: is this a real
image **of this class**? A generated horse handed to it with the label "car" is fake
even if it is a perfect horse. Direct, and it makes the discriminator's job harder.

**ACGAN** keeps the discriminator's real/fake head unconditional and bolts on a second
head that classifies the image. The generator is rewarded when that classifier agrees
with the label it was given. Cheaper, and it has a known failure mode: the generator
can win the auxiliary term by producing over-typical, low-diversity examples of each
class, so it trades recall for class accuracy. ``compare.py`` measures that trade.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("cgan", "acgan")


def _levels_for(image_size: int) -> int:
    if image_size % 4 or not float(math.log2(image_size)).is_integer():
        raise ValueError(f"image size {image_size} must be a power of two, at least 8")
    return int(math.log2(image_size)) - 2


def dcgan_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class ConditionalGenerator(nn.Module):
    """DCGAN generator whose input is ``[z ; embed(y)]``."""

    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 10,
        out_channels: int = 1,
        image_size: int = 32,
        width: int = 64,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.label_embed = nn.Embedding(num_classes, embed_dim)

        levels = _levels_for(image_size)
        channels = width * 2 ** (levels - 1)
        layers = [
            nn.ConvTranspose2d(latent_dim + embed_dim, channels, 4, 1, 0, bias=False),
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
        layers += [nn.ConvTranspose2d(channels, out_channels, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(dcgan_init)

    def forward(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        conditioned = torch.cat([latent, self.label_embed(labels)], dim=1)
        return (self.net(conditioned.view(conditioned.size(0), -1, 1, 1)) + 1.0) / 2.0

    @torch.no_grad()
    def sample(self, n: int, device=None, labels: torch.Tensor | None = None):
        """Sample ``n`` images; ``labels`` defaults to a balanced sweep of the classes."""
        device = device or next(self.parameters()).device
        if labels is None:
            labels = torch.arange(n, device=device) % self.num_classes
        latent = torch.randn(n, self.latent_dim, device=device)
        return self(latent, labels), labels


class ConditionalDiscriminator(nn.Module):
    """Shared trunk; ``mode`` decides how the label enters.

    ``cgan`` appends a broadcast class embedding to the input channels, so the trunk
    sees the label. ``acgan`` leaves the trunk unconditional and adds a classification
    head next to the real/fake head.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        image_size: int = 32,
        width: int = 64,
        mode: str = "acgan",
        embed_channels: int = 8,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        self.mode = mode
        self.num_classes = num_classes
        self.image_size = image_size

        if mode == "cgan":
            self.label_map = nn.Embedding(num_classes, embed_channels)
            trunk_in = in_channels + embed_channels
        else:
            self.label_map = None
            trunk_in = in_channels

        levels = _levels_for(image_size)
        channels = width
        layers = [nn.Conv2d(trunk_in, channels, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(levels - 1):
            layers += [
                nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels *= 2
        self.trunk = nn.Sequential(*layers)
        self.adversarial = nn.Conv2d(channels, 1, 4, 1, 0)
        self.classifier = nn.Conv2d(channels, num_classes, 4, 1, 0) if mode == "acgan" else None
        self.apply(dcgan_init)

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        """Returns ``(adversarial logit, class logits or None)``."""
        x = images * 2.0 - 1.0
        if self.mode == "cgan":
            embedded = self.label_map(labels)  # (N, C)
            maps = embedded.view(embedded.size(0), -1, 1, 1).expand(-1, -1, *x.shape[-2:])
            x = torch.cat([x, maps], dim=1)
        features = self.trunk(x)
        adversarial = self.adversarial(features).flatten(1).squeeze(1)
        classes = (
            self.classifier(features).flatten(1) if self.classifier is not None else None
        )
        return adversarial, classes


def discriminator_loss(
    real_out, fake_out, real_labels, fake_labels, aux_weight: float = 1.0
) -> dict:
    real_logit, real_classes = real_out
    fake_logit, fake_classes = fake_out
    adversarial = F.binary_cross_entropy_with_logits(
        real_logit, torch.ones_like(real_logit)
    ) + F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))

    auxiliary = torch.zeros((), device=real_logit.device)
    if real_classes is not None:
        # The classifier head is trained on real images and on generated ones. Using
        # both is the ACGAN recipe; using only real images also works and is steadier.
        auxiliary = F.cross_entropy(real_classes, real_labels) + F.cross_entropy(
            fake_classes, fake_labels
        )
    return {"loss": adversarial + aux_weight * auxiliary,
            "adversarial": adversarial, "auxiliary": auxiliary}


def generator_loss(fake_out, fake_labels, aux_weight: float = 1.0) -> dict:
    fake_logit, fake_classes = fake_out
    adversarial = F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
    auxiliary = torch.zeros((), device=fake_logit.device)
    if fake_classes is not None:
        auxiliary = F.cross_entropy(fake_classes, fake_labels)
    return {"loss": adversarial + aux_weight * auxiliary,
            "adversarial": adversarial, "auxiliary": auxiliary}


def build_models(
    latent_dim: int = 64,
    num_classes: int = 10,
    in_channels: int = 1,
    image_size: int = 32,
    width: int = 64,
    mode: str = "acgan",
):
    return (
        ConditionalGenerator(latent_dim, num_classes, in_channels, image_size, width),
        ConditionalDiscriminator(num_classes, in_channels, image_size, width, mode),
    )


if __name__ == "__main__":
    for mode in MODES:
        generator, discriminator = build_models(64, 10, 1, 32, mode=mode)
        images, labels = generator.sample(8)
        outputs = discriminator(images, labels)
        d_loss = discriminator_loss(
            discriminator(torch.rand(8, 1, 32, 32), labels), outputs, labels, labels
        )
        g_loss = generator_loss(outputs, labels)
        print(
            f"{mode:6s}: samples {tuple(images.shape)}  adv logit {tuple(outputs[0].shape)}  "
            f"class logits {tuple(outputs[1].shape) if outputs[1] is not None else None}  "
            f"D {d_loss['loss'].item():.3f} G {g_loss['loss'].item():.3f}  "
            f"G params {sum(p.numel() for p in generator.parameters()):,}"
        )

    generator, _ = build_models(64, 10, 1, 32)
    images, labels = generator.sample(20)
    print(f"\nbalanced sampling gives labels {labels[:12].tolist()} ...")

    # Same latent, different label: a conditional generator must change the image.
    latent = torch.randn(1, 64).repeat(2, 1)
    pair = generator(latent, torch.tensor([0, 7]))
    print("same latent, different label -> different image:",
          bool((pair[0] - pair[1]).abs().mean() > 1e-4))
