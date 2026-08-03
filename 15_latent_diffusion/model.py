"""Latent diffusion: run the diffusion process in a compressed space.

Pixel-space diffusion spends most of its capacity on detail that a cheap
autoencoder could have reproduced anyway, and every one of its hundreds of
sampling steps pays the full resolution cost.  Latent diffusion (Rombach et al.,
2022 -- the basis of Stable Diffusion) splits the problem in two:

1. **Stage 1, perceptual compression.**  An autoencoder maps a ``C x H x W``
   image to a much smaller ``d x H/f x W/f`` latent and back.  It is trained
   once, on reconstruction alone, and then frozen.
2. **Stage 2, semantic generation.**  A DDPM is trained *in that latent space*.
   With a downsampling factor of ``f = 4`` the diffusion U-Net operates on
   1/16th of the spatial positions, so both training and sampling get much
   cheaper -- which is the entire claim.

Generation is: sample a latent with the diffusion model, then decode it once.
The expensive iterative part happens at low resolution; the single expensive
upsampling happens once.

The autoencoder here is a light KL-regularised VAE.  The regulariser is weak on
purpose: too strong and the latent loses detail, too weak and it becomes a
high-variance space that diffusion struggles to model.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(math.gcd(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(math.gcd(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AutoencoderKL(nn.Module):
    """Stage-1 autoencoder: image <-> latent, with a light KL regulariser.

    ``downsample_factor`` is the ``f`` of the paper.  At ``f = 4`` a 32x32 image
    becomes a ``latent_channels x 8 x 8`` tensor.
    """

    def __init__(
        self,
        channels: int = 1,
        base: int = 32,
        latent_channels: int = 4,
        downsample_factor: int = 4,
    ):
        super().__init__()
        stages = int(math.log2(downsample_factor))
        if 2**stages != downsample_factor:
            raise ValueError("downsample_factor must be a power of two")
        self.downsample_factor = downsample_factor
        self.latent_channels = latent_channels

        encoder = [nn.Conv2d(channels, base, 3, padding=1)]
        in_ch = base
        for i in range(stages):
            out_ch = min(base * 2 ** (i + 1), base * 4)
            encoder += [ResidualBlock(in_ch, out_ch), nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)]
            in_ch = out_ch
        encoder += [ResidualBlock(in_ch, in_ch), nn.GroupNorm(math.gcd(8, in_ch), in_ch), nn.SiLU()]
        # Two heads: the mean and log-variance of the latent posterior.
        encoder += [nn.Conv2d(in_ch, latent_channels * 2, 1)]
        self.encoder = nn.Sequential(*encoder)

        decoder = [nn.Conv2d(latent_channels, in_ch, 3, padding=1), ResidualBlock(in_ch, in_ch)]
        for i in reversed(range(stages)):
            out_ch = min(base * 2**i, base * 4)
            decoder += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                ResidualBlock(out_ch, out_ch),
            ]
            in_ch = out_ch
        decoder += [
            nn.GroupNorm(math.gcd(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, channels, 3, padding=1),
            nn.Tanh(),
        ]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar

    def loss(self, x: torch.Tensor, kl_weight: float = 1e-6):
        """Reconstruction plus a deliberately tiny KL term.

        The KL is there only to stop the latent scale from drifting away; a
        VAE-strength weight would blur exactly the detail stage 2 relies on.
        """
        recon, mu, logvar = self(x)
        rec_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec_loss + kl_weight * kl, rec_loss, kl

    @torch.no_grad()
    def latent_shape(self, image_size: int) -> Tuple[int, int, int]:
        size = image_size // self.downsample_factor
        return (self.latent_channels, size, size)


class LatentScaler:
    """Rescales latents to roughly unit variance before diffusion.

    Diffusion assumes its data has a sane scale -- the noise schedule is defined
    against a unit-variance signal. Stage 1 makes no such promise, so a single
    scalar is estimated from the training set and applied in both directions.
    Stable Diffusion does exactly this with its 0.18215 constant.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    @classmethod
    def fit(cls, autoencoder: AutoencoderKL, loader, device, max_batches: int = 20):
        values = []
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= max_batches:
                    break
                mu, _ = autoencoder.encode(x.to(device))
                values.append(mu.flatten())
        std = torch.cat(values).std().item()
        return cls(scale=1.0 / max(std, 1e-6))

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.scale

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.scale
