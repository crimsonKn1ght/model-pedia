"""CycleGAN: translation learned from two unpaired collections.

    python model.py            # shapes, parameter counts and a cycle round-trip

Pix2Pix (project 10) is given matched pairs and can ask "is this output close to the
right answer". Remove the pairs and that question has no answer, which breaks the method
rather than merely weakening it: the adversarial loss alone is satisfied by *any* mapping
whose outputs land in domain B, including one that ignores its input completely and emits
the same plausible image every time. Nothing in the objective forbids it.

**Cycle consistency** supplies the missing constraint. Train two generators at once, one
each way, and require that going there and back returns where you started:

    F(G(a)) ~= a        and        G(F(b)) ~= b

A mapping that discards its input cannot be inverted, so it cannot satisfy this. That one
idea is the whole method, and ``compare.py --study cycle`` is what shows it: at
``lambda_cycle=0`` the adversarial losses keep improving while the translations stop having
anything to do with the inputs.

Three details from the paper are kept because they change the outcome rather than the
score by a percent:

**Least-squares adversarial loss.** Squared error against 1 and 0 instead of binary cross
entropy. BCE saturates once the discriminator is confident and stops producing gradient
exactly when the generator most needs it; the squared error keeps pushing.

**Instance normalisation, not batch.** Translation is a per-image job, and batch statistics
leak one image's content into another's normalisation.

**An image buffer for the discriminators.** They are shown a mix of the generator's current
output and its recent history. Without it the two networks chase each other's most recent
move and oscillate.

The generator output is ``(tanh + 1) / 2``, so images stay in ``[0, 1]`` like everywhere
else in this repository.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn


def weights_init(module: nn.Module) -> None:
    """DCGAN-style initialisation, which CycleGAN inherited."""
    name = module.__class__.__name__
    if "Conv" in name and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm" in name and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class ResidualBlock(nn.Module):
    """Instance-normalised residual block; reflection padding avoids edge artefacts."""

    def __init__(self, channels: int) -> None:
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
    """Downsample twice, several residual blocks, upsample back.

    No skip connections, unlike the Pix2Pix U-Net in project 10. A U-Net's skips carry the
    input's spatial detail straight to the output, which is what you want when the output
    must align with the input pixel for pixel. Here the two domains may differ in shape,
    and the bottleneck is doing useful work by forcing the content through a representation
    that both domains share.
    """

    def __init__(self, channels: int = 3, base: int = 32, num_blocks: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, base, 7),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
        ]

        in_channels = base
        for _ in range(2):
            out_channels = in_channels * 2
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        layers += [ResidualBlock(in_channels) for _ in range(num_blocks)]

        for _ in range(2):
            out_channels = in_channels // 2
            layers += [
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1,
                                   output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        layers += [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, channels, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Back to [0, 1], the convention every project here uses.
        return (self.net(x) + 1.0) / 2.0


class PatchDiscriminator(nn.Module):
    """PatchGAN, as in Pix2Pix, but judging one image rather than a pair.

    It scores overlapping patches instead of the whole image, so "does this look like
    domain B" is asked about local texture. That is the right question for translation:
    global layout is the input's business, and the discriminator should not be able to
    reject an output for having its content in an unusual place.
    """

    def __init__(self, channels: int = 3, base: int = 32, layers: int = 3) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            nn.Conv2d(channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels = base
        for index in range(1, layers):
            out_channels = min(base * 2**index, base * 8)
            stride = 2 if index < layers - 1 else 1
            blocks += [
                nn.Conv2d(in_channels, out_channels, 4, stride, 1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_channels = out_channels
        blocks.append(nn.Conv2d(in_channels, 1, 4, 1, 1))
        self.net = nn.Sequential(*blocks)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImagePool:
    """A buffer of previously generated images, from the CycleGAN paper.

    Half the time the discriminator sees a stored image from the generator's past instead
    of its current output. That breaks the one-step-behind oscillation the pair otherwise
    falls into, and it costs one tensor per slot.
    """

    def __init__(self, size: int = 50) -> None:
        self.size = size
        self.images: list[torch.Tensor] = []

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
                index = random.randrange(self.size)
                out.append(self.images[index].clone())
                self.images[index] = image.detach().clone()
            else:
                out.append(image)
        return torch.cat(out, dim=0)


def least_squares_loss(prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    """LSGAN: squared error against 1 for real, 0 for fake."""
    target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
    return ((prediction - target) ** 2).mean()


def build_models(
    channels: int = 3, base: int = 32, num_blocks: int = 4, patch_layers: int = 3
):
    """The four networks: two generators and two discriminators."""
    return (
        ResnetGenerator(channels, base, num_blocks),   # G: A -> B
        ResnetGenerator(channels, base, num_blocks),   # F: B -> A
        PatchDiscriminator(channels, base, patch_layers),  # D_A, judges domain A
        PatchDiscriminator(channels, base, patch_layers),  # D_B, judges domain B
    )


def generator_losses(
    g_ab: nn.Module,
    g_ba: nn.Module,
    d_a: nn.Module,
    d_b: nn.Module,
    real_a: torch.Tensor,
    real_b: torch.Tensor,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 0.5,
):
    """The generator objective, returned in parts so each can be logged.

    Returns ``(total, parts, fake_a, fake_b)``. The fakes come back because the
    discriminator step needs them and recomputing would double the cost.
    """
    fake_b = g_ab(real_a)
    fake_a = g_ba(real_b)

    adversarial = (
        least_squares_loss(d_b(fake_b), True) + least_squares_loss(d_a(fake_a), True)
    )

    # Cycle consistency, the constraint that makes the problem well posed.
    cycle = (
        (g_ba(fake_b) - real_a).abs().mean() + (g_ab(fake_a) - real_b).abs().mean()
    )

    # Identity: a generator handed an image already in its target domain should leave it
    # alone. Without it, generators drift the palette for no reason.
    identity = torch.zeros((), device=real_a.device)
    if lambda_identity > 0:
        identity = (
            (g_ab(real_b) - real_b).abs().mean() + (g_ba(real_a) - real_a).abs().mean()
        )

    total = adversarial + lambda_cycle * cycle + lambda_cycle * lambda_identity * identity
    parts = {
        "adversarial": float(adversarial.detach()),
        "cycle": float(cycle.detach()),
        "identity": float(identity.detach()),
        "generator": float(total.detach()),
    }
    return total, parts, fake_a.detach(), fake_b.detach()


def discriminator_loss(
    discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
    """Halved, as in the paper, so the discriminators learn at half the generators' rate."""
    return 0.5 * (
        least_squares_loss(discriminator(real), True)
        + least_squares_loss(discriminator(fake), False)
    )


def main() -> None:
    from utils import count_parameters

    torch.manual_seed(0)
    g_ab, g_ba, d_a, d_b = build_models(channels=3, base=32, num_blocks=4)
    images_a = torch.rand(2, 3, 64, 64)
    images_b = torch.rand(2, 3, 64, 64)

    print(f"generator G (A->B) : {count_parameters(g_ab):,} parameters")
    print(f"generator F (B->A) : {count_parameters(g_ba):,} parameters")
    print(f"discriminator D_B  : {count_parameters(d_b):,} parameters")
    print(f"total trainable    : "
          f"{sum(count_parameters(m) for m in (g_ab, g_ba, d_a, d_b)):,}")

    translated = g_ab(images_a)
    cycled = g_ba(translated)
    print(f"\nA {tuple(images_a.shape)} -> B {tuple(translated.shape)} "
          f"-> A {tuple(cycled.shape)}")
    print(f"output range       : [{translated.min():.3f}, {translated.max():.3f}]")
    print(f"patch verdict      : {tuple(d_b(translated).shape)} (one score per patch)")

    total, parts, _, _ = generator_losses(g_ab, g_ba, d_a, d_b, images_a, images_b)
    print("\nuntrained generator losses")
    for name, value in parts.items():
        print(f"  {name:12s} {value:.4f}")
    total.backward()
    grad = next(p.grad for p in g_ab.parameters() if p.grad is not None)
    print(f"\ngradient reaches the generator: {bool(grad.abs().sum() > 0)}")


if __name__ == "__main__":
    main()
