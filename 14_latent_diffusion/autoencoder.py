"""The first stage: a small autoencoder with a *spatial* latent.

    python autoencoder.py

Project 06's VAE compresses an image to a single vector, which is the wrong shape for
diffusion - a U-Net needs something with height and width to convolve over. So the first
stage here keeps the latent spatial: a 32x32 image becomes a 4x8x8 tensor. That is 256
values instead of 1024 for greyscale (4x fewer) or instead of 3072 for colour (12x fewer),
and it is still a grid.

That factor is the entire argument for latent diffusion. Diffusion cost scales with the
number of values being denoised, and the expensive part of an image is high-frequency
detail that carries almost no semantic information. Let a cheap autoencoder handle the
detail once, and run the expensive iterative model on the small tensor that is left.

The latent is lightly KL-regularised, as in Stable Diffusion's first stage. The point of
that term is not to build a prior worth sampling from - the diffusion model does that job -
but to stop the latent's scale from drifting, because the diffusion process assumes its
input has roughly unit variance. ``latent_scale`` handles the rest: it is measured from
the data after stage one and stored, so stage two always sees a standardised tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAutoencoder(nn.Module):
    """Strided-convolution encoder/decoder around a ``(latent_channels, h, w)`` latent."""

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 32,
        base_channels: int = 32,
        latent_channels: int = 4,
        levels: int = 2,
        kl_weight: float = 1e-6,
    ) -> None:
        super().__init__()
        if image_size % 2**levels:
            raise ValueError(f"image size {image_size} must divide by {2 ** levels}")
        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.levels = levels
        self.kl_weight = kl_weight
        self.grid = image_size // 2**levels

        encoder, channels = [], in_channels
        for level in range(levels):
            out_channels = base_channels * 2**level
            encoder += [
                nn.Conv2d(channels, out_channels, 4, 2, 1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(inplace=True),
            ]
            channels = out_channels
        # Two heads: the latent mean and its log-variance.
        encoder.append(nn.Conv2d(channels, latent_channels * 2, 1))
        self.encoder = nn.Sequential(*encoder)

        decoder = [
            nn.Conv2d(latent_channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        ]
        for level in reversed(range(levels)):
            out_channels = base_channels * 2 ** max(level - 1, 0)
            decoder += [
                nn.ConvTranspose2d(channels, out_channels, 4, 2, 1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(inplace=True),
            ]
            channels = out_channels
        decoder += [nn.Conv2d(channels, in_channels, 3, 1, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)

    @property
    def compression(self) -> float:
        """Values in the image divided by values in the latent."""
        return (self.in_channels * self.image_size**2) / (self.latent_channels * self.grid**2)

    def encode(self, images: torch.Tensor):
        mean, logvar = self.encoder(images).chunk(2, dim=1)
        return mean, logvar.clamp(-8.0, 8.0)

    def sample_latent(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mean
        return mean + (0.5 * logvar).exp() * torch.randn_like(mean)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, images: torch.Tensor) -> dict:
        mean, logvar = self.encode(images)
        latent = self.sample_latent(mean, logvar)
        return {"reconstruction": self.decode(latent), "mean": mean,
                "logvar": logvar, "latent": latent}

    def loss(self, images: torch.Tensor, outputs: dict) -> dict:
        reconstruction = F.mse_loss(outputs["reconstruction"], images)
        kl = 0.5 * (
            outputs["mean"].pow(2) + outputs["logvar"].exp() - 1.0 - outputs["logvar"]
        ).flatten(1).sum(dim=1).mean()
        return {
            "loss": reconstruction + self.kl_weight * kl,
            "reconstruction": reconstruction,
            "kl": kl,
        }


def build_autoencoder(
    in_channels: int = 1,
    image_size: int = 32,
    base_channels: int = 32,
    latent_channels: int = 4,
    levels: int = 2,
    kl_weight: float = 1e-6,
) -> SpatialAutoencoder:
    return SpatialAutoencoder(
        in_channels, image_size, base_channels, latent_channels, levels, kl_weight
    )


def load_autoencoder(path: str, device):
    """Load a trained first stage, frozen and in eval mode, with its config."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_autoencoder(
        ckpt["in_channels"], ckpt["image_size"], ckpt["base_channels"],
        ckpt["latent_channels"], ckpt["levels"], ckpt["kl_weight"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, ckpt


if __name__ == "__main__":
    for channels, size, levels in ((1, 32, 2), (3, 32, 2), (3, 64, 3)):
        model = build_autoencoder(channels, size, levels=levels)
        model.train()
        images = torch.rand(4, channels, size, size)
        outputs = model(images)
        losses = model.loss(images, outputs)
        print(
            f"C={channels} {size}x{size} levels {levels}: latent "
            f"{tuple(outputs['latent'].shape[1:])}  {model.compression:.0f}x fewer values  "
            f"recon {losses['reconstruction'].item():.4f}  "
            f"params {sum(p.numel() for p in model.parameters()):,}"
        )

    model = build_autoencoder(1, 32)
    model.eval()
    latent, _ = model.encode(torch.rand(2, 1, 32, 32))
    print(f"\nlatent grid {model.grid}x{model.grid} with {model.latent_channels} channels")
    print(f"decode round trip: {tuple(model.decode(latent).shape)}")
