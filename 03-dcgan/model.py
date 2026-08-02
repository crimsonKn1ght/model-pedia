"""DCGAN: the convolutional GAN architecture that made adversarial training work.

A generator ``G`` turns a noise vector ``z ~ N(0, I)`` into an image; a
discriminator ``D`` tries to tell real images from generated ones.  They play a
minimax game::

    min_G max_D  E_x[ log D(x) ]  +  E_z[ log(1 - D(G(z))) ]

In practice the generator is trained with the **non-saturating** variant --
maximise ``log D(G(z))`` instead of minimising ``log(1 - D(G(z)))`` -- because
the original form has vanishing gradients exactly when the generator is bad,
which is when it most needs them.

The DCGAN recipe, which this file follows, is a short list of architectural
rules that turned a notoriously unstable idea into something reproducible:

* strided convolutions instead of pooling (discriminator) and transposed
  convolutions for upsampling (generator);
* batch normalisation everywhere except the generator output and the
  discriminator input;
* ReLU in the generator, LeakyReLU(0.2) in the discriminator;
* ``tanh`` on the generator output, so images live in ``[-1, 1]``;
* Adam with ``lr = 2e-4`` and ``beta1 = 0.5``.
"""

from __future__ import annotations

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


class Generator(nn.Module):
    """``z`` of shape ``(B, latent_dim)`` -> image in ``[-1, 1]``."""

    def __init__(
        self,
        latent_dim: int = 100,
        channels: int = 1,
        image_size: int = 32,
        base: int = 64,
    ):
        super().__init__()
        # `stages` stride-2 steps take a 4x4 map up to `image_size`.
        stages = _stages(image_size)
        widths = [min(base * 2**i, base * 8) for i in range(stages)][::-1]
        start = widths[0]

        layers = [
            # Project the noise vector to a 4x4 feature map.
            nn.ConvTranspose2d(latent_dim, start, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(start),
            nn.ReLU(inplace=True),
        ]
        in_ch = start
        for out_ch in widths[1:]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        # No batch norm on the output layer: it would fight the tanh.
        layers += [nn.ConvTranspose2d(in_ch, channels, 4, stride=2, padding=1), nn.Tanh()]

        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.shape[0], -1, 1, 1)
        return self.net(z)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        return self(torch.randn(n, self.latent_dim, device=device))


class Discriminator(nn.Module):
    """Image -> a single real/fake logit."""

    def __init__(self, channels: int = 1, image_size: int = 32, base: int = 64):
        super().__init__()
        # `stages` stride-2 steps take `image_size` down to 4x4, then a valid
        # 4x4 convolution collapses that to a single logit.
        stages = _stages(image_size)
        widths = [min(base * 2**i, base * 8) for i in range(stages)]

        # No batch norm on the input layer, per the DCGAN recipe.
        layers = [
            nn.Conv2d(channels, widths[0], 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = widths[0]
        for out_ch in widths[1:]:
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        layers += [nn.Conv2d(in_ch, 1, 4, stride=1, padding=0)]

        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def weights_init(module: nn.Module) -> None:
    """DCGAN initialisation: N(0, 0.02) for conv weights, N(1, 0.02) for batch norm."""
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
