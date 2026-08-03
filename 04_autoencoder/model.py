"""A convolutional autoencoder with an explicit, resizable bottleneck.

The encoder halves the resolution three times (32 -> 16 -> 8 -> 4) while
widening the channels, then a single linear layer squeezes everything into a
``latent_dim``-vector. The decoder mirrors that exactly.

The bottleneck is the whole model. A 32x32 grayscale image is 1024 numbers; a
32-dimensional code is a 32x compression, and the only way to survive it is to
discard whatever is predictable and keep whatever is not. That is what makes an
autoencoder a representation learner rather than an expensive identity
function - remove the bottleneck and it learns exactly that identity.

Images are expected in ``[0, 1]``; the decoder ends in a sigmoid to match.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SPATIAL = 32  # every dataset is resized to 32x32
BOTTLENECK_SPATIAL = SPATIAL // 8  # after three stride-2 convolutions


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 32,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.flat_dim = c3 * BOTTLENECK_SPATIAL * BOTTLENECK_SPATIAL

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.to_latent = nn.Linear(self.flat_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, self.flat_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, in_channels, 4, stride=2, padding=1),  # 32x32
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(torch.flatten(self.encoder_conv(x), 1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = h.view(-1, self.base_channels * 4, BOTTLENECK_SPATIAL, BOTTLENECK_SPATIAL)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def compression_ratio(self) -> float:
        """How many input numbers each latent number stands in for."""
        return (self.in_channels * SPATIAL * SPATIAL) / self.latent_dim


def build_model(in_channels: int = 1, latent_dim: int = 32, base_channels: int = 32):
    return ConvAutoencoder(in_channels, latent_dim, base_channels)


if __name__ == "__main__":
    for channels in (1, 3):
        for latent in (8, 32, 128):
            model = build_model(channels, latent)
            dummy = torch.rand(2, channels, SPATIAL, SPATIAL)
            code = model.encode(dummy)
            out = model(dummy)
            params = sum(p.numel() for p in model.parameters())
            print(
                f"C={channels} latent={latent:3d}  code {tuple(code.shape)}  "
                f"out {tuple(out.shape)}  ratio {model.compression_ratio():5.1f}x  "
                f"params {params:,}"
            )
