"""Pix2Pix: a U-Net generator and a PatchGAN discriminator.

    python model.py

Paired image-to-image translation has an obvious loss - L1 against the target - and using
only that loss produces blurry output. The reason is worth understanding, because it is the
same reason a VAE blurs: when several outputs are plausible, the L1-optimal answer is a
compromise between them, and a compromise between two sharp images is a blurry one.

Pix2Pix adds a discriminator, so the generator is scored not only on being close to the
target but on being *the kind of image that could be a target*. The combination is the
whole method, and ``compare.py --study l1_weight`` is the ablation:

* L1 only - correct on average, blurry
* GAN only - sharp, and free to invent content that does not match the input
* both - what the paper ships

Two architectural choices carry the rest:

**A U-Net generator, not an encoder-decoder.** For translation the input and output are
aligned pixel by pixel, so most of the information the decoder needs is already sitting at
the matching encoder resolution. The skip connections carry it across, and without them the
output is soft in exactly the way project 05's no-skip control was.

**A PatchGAN discriminator.** Rather than one verdict per image, it outputs a grid of
verdicts, each covering a patch. That makes the discriminator a texture critic: it has
enough receptive field to judge local realism and not enough to police global layout, which
is the generator's job to get from the input. It is also far smaller than a whole-image
discriminator. ``--patch-layers`` sets the receptive field, and the whole-image variant is
available for comparison.

The discriminator sees the input **and** the output, concatenated. Without that it could
only judge whether an image looks real, not whether it matches what was asked for.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GAN_LOSSES = ("bce", "hinge")


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: bool = True) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False) -> None:
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            # Pix2Pix uses dropout at test time as its only source of stochasticity.
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """Symmetric U-Net; ``skips=False`` builds the control that shows what they buy."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 64,
        levels: int = 4,
        skips: bool = True,
    ) -> None:
        super().__init__()
        self.skips = skips
        self.levels = levels

        widths = [min(width * 2**level, width * 8) for level in range(levels)]
        self.downs = nn.ModuleList()
        channels = in_channels
        for index, out in enumerate(widths):
            self.downs.append(DownBlock(channels, out, norm=index > 0))
            channels = out

        self.ups = nn.ModuleList()
        for index in range(levels - 1, 0, -1):
            skip = widths[index - 1] if skips else 0
            self.ups.append(UpBlock(channels, widths[index - 1], dropout=index >= levels - 1))
            channels = widths[index - 1] + skip
        self.final = nn.Sequential(
            nn.ConvTranspose2d(channels, out_channels, 4, 2, 1), nn.Tanh()
        )
        self.apply(init_weights)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        x = source * 2.0 - 1.0
        features = []
        for down in self.downs:
            x = down(x)
            features.append(x)

        x = features[-1]
        for index, up in enumerate(self.ups):
            x = up(x)
            if self.skips:
                x = torch.cat([x, features[-(index + 2)]], dim=1)
        return (self.final(x) + 1.0) / 2.0


class PatchDiscriminator(nn.Module):
    """Verdicts on overlapping patches of the concatenated (input, output) pair."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 64,
        layers: int = 3,
        whole_image: bool = False,
    ) -> None:
        super().__init__()
        self.whole_image = whole_image
        channels = in_channels + out_channels

        blocks = [nn.Conv2d(channels, width, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        current = width
        for index in range(1, layers):
            out = min(width * 2**index, width * 8)
            blocks += [
                nn.Conv2d(current, out, 4, 2 if index < layers - 1 else 1, 1, bias=False),
                nn.BatchNorm2d(out),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            current = out
        blocks.append(nn.Conv2d(current, 1, 4, 1, 1))
        self.net = nn.Sequential(*blocks)
        self.apply(init_weights)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([source * 2 - 1, target * 2 - 1], dim=1)
        verdicts = self.net(pair)
        # A whole-image discriminator is the same trunk with its grid pooled to one value.
        return verdicts.mean(dim=(2, 3)) if self.whole_image else verdicts


def discriminator_loss(real_logits, fake_logits, loss: str = "bce") -> torch.Tensor:
    if loss == "hinge":
        return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    return F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)
    ) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))


def generator_loss(
    fake_logits: torch.Tensor,
    generated: torch.Tensor,
    target: torch.Tensor,
    loss: str = "bce",
    l1_weight: float = 100.0,
) -> dict:
    """Adversarial term plus a weighted L1 term. ``l1_weight=0`` is GAN only."""
    adversarial = (
        -fake_logits.mean()
        if loss == "hinge"
        else F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    )
    l1 = F.l1_loss(generated, target)
    return {"loss": adversarial + l1_weight * l1, "adversarial": adversarial, "l1": l1}


def build_models(
    in_channels: int = 3,
    out_channels: int = 3,
    width: int = 64,
    levels: int = 4,
    skips: bool = True,
    patch_layers: int = 3,
    whole_image: bool = False,
):
    return (
        UNetGenerator(in_channels, out_channels, width, levels, skips),
        PatchDiscriminator(in_channels, out_channels, width, patch_layers, whole_image),
    )


if __name__ == "__main__":
    for size, levels in ((64, 4), (128, 5)):
        generator, discriminator = build_models(3, 3, 64, levels)
        source = torch.rand(2, 3, size, size)
        generated = generator(source)
        verdicts = discriminator(source, generated)
        print(
            f"{size}x{size} levels {levels}: output {tuple(generated.shape)} in "
            f"[{generated.min():.2f}, {generated.max():.2f}]  verdicts {tuple(verdicts.shape)}  "
            f"G {sum(p.numel() for p in generator.parameters()):,} "
            f"D {sum(p.numel() for p in discriminator.parameters()):,}"
        )

    for layers in (1, 3, 5):
        _, discriminator = build_models(3, 3, 64, 4, patch_layers=layers)
        grid = discriminator(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
        print(f"patch layers {layers}: {tuple(grid.shape)[-2:]} verdicts per image, "
              f"{sum(p.numel() for p in discriminator.parameters()):,} parameters")

    _, whole = build_models(3, 3, 64, 4, whole_image=True)
    print(f"whole image      : {tuple(whole(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)).shape)}")

    generator, _ = build_models(3, 3, 64, 4, skips=False)
    print(f"\nno-skip generator: {sum(p.numel() for p in generator.parameters()):,} parameters, "
          f"output {tuple(generator(torch.rand(2, 3, 64, 64)).shape)}")
