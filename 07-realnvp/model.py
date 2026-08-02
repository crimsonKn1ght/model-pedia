"""RealNVP: a normalizing flow with exact likelihoods.

VAEs optimise a *bound* on the likelihood; GANs never touch it.  A normalizing
flow is built entirely from invertible transforms, so the change-of-variables
formula gives the exact log-density::

    log p(x) = log p(z) + log | det (dz / dx) |     with  z = f(x)

The catch is that a general Jacobian determinant costs O(D^3).  RealNVP's
**affine coupling layer** sidesteps it: split the input, leave one half
untouched, and use it to predict a scale and shift for the other half::

    z_a = x_a
    z_b = x_b * exp(s(x_a)) + t(x_a)

The Jacobian is triangular, so its determinant is just ``sum(s(x_a))`` -- cheap
to compute, and the layer is trivially invertible no matter how complicated
``s`` and ``t`` are.  Alternating which half stays fixed lets every dimension be
transformed.

Two masking patterns are alternated, as in the paper:

* **checkerboard** -- captures local spatial structure;
* **channel-wise**, applied after a squeeze that trades resolution for channels
  -- captures longer-range structure.

The headline metric is **bits per dimension**, the negative log-likelihood in
base 2 per subpixel. It is a real, comparable number: lower is strictly better,
and unlike FID it needs no reference network.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Coupling network
# --------------------------------------------------------------------------- #
class CouplingNet(nn.Module):
    """Small residual CNN that predicts the scale and shift of a coupling layer."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )
        # Start as the identity transform: zero scale and shift.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffineCoupling(nn.Module):
    """One masked affine coupling layer.

    ``mask`` is 1 where values pass through unchanged and 0 where they are
    transformed. The masked half is what conditions the transform, so the
    layer stays invertible by construction.
    """

    def __init__(self, channels: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = CouplingNet(channels, hidden, channels * 2)
        # Learned per-channel cap on |s|, which keeps early training stable.
        self.scale_cap = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x * self.mask)
        shift, scale = h.chunk(2, dim=1)
        # tanh bounds the log-scale; the learned cap lets it grow if needed.
        scale = torch.tanh(scale) * torch.exp(self.scale_cap.clamp(-5.0, 5.0))
        return shift * (1 - self.mask), scale * (1 - self.mask)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self._params(x)
        z = x * torch.exp(scale) + shift
        return z, scale.flatten(1).sum(dim=1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        shift, scale = self._params(z)  # depends only on the unchanged half
        return (z - shift) * torch.exp(-scale)


def checkerboard_mask(channels: int, height: int, width: int, invert: bool) -> torch.Tensor:
    grid = torch.arange(height).view(-1, 1) + torch.arange(width).view(1, -1)
    mask = (grid % 2).float()
    if invert:
        mask = 1 - mask
    return mask.view(1, 1, height, width).repeat(1, channels, 1, 1)


def channel_mask(channels: int, height: int, width: int, invert: bool) -> torch.Tensor:
    mask = torch.zeros(1, channels, 1, 1)
    mask[:, : channels // 2] = 1.0
    if invert:
        mask = 1 - mask
    return mask.repeat(1, 1, height, width)


def squeeze(x: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W)`` -> ``(B, 4C, H/2, W/2)``; trades resolution for channels."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * 4, h // 2, w // 2)


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(b, c // 4, h * 2, w * 2)


# --------------------------------------------------------------------------- #
# The flow
# --------------------------------------------------------------------------- #
class RealNVP(nn.Module):
    """Multi-scale RealNVP over images.

    Structure per scale: checkerboard couplings at full resolution, a squeeze,
    then channel-wise couplings at half resolution.
    """

    def __init__(
        self,
        channels: int = 1,
        image_size: int = 32,
        hidden: int = 64,
        num_scales: int = 2,
        couplings_per_scale: int = 3,
        alpha: float = 0.05,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.alpha = alpha
        self.num_scales = num_scales
        self.couplings_per_scale = couplings_per_scale

        self.scales = nn.ModuleList()
        c, s = channels, image_size
        for _ in range(num_scales):
            layers: List[nn.Module] = []
            for i in range(couplings_per_scale):
                layers.append(AffineCoupling(c, hidden, checkerboard_mask(c, s, s, i % 2 == 1)))
            c, s = c * 4, s // 2
            for i in range(couplings_per_scale):
                layers.append(AffineCoupling(c, hidden, channel_mask(c, s, s, i % 2 == 1)))
            self.scales.append(nn.ModuleList(layers))
        self.final_channels = c
        self.final_size = s

    # -- data preprocessing -------------------------------------------------- #
    def preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantise and logit-transform. ``x`` arrives in ``[0, 1]``.

        Uniform dequantisation turns the discrete pixel values into a continuous
        density, which is what a flow can actually model.  The logit transform
        then moves the data off the bounded interval, where an unconstrained
        Gaussian latent would otherwise be a poor fit.
        """
        x = (x * 255.0 + torch.rand_like(x)) / 256.0
        s = self.alpha + (1 - 2 * self.alpha) * x
        z = torch.log(s) - torch.log1p(-s)
        logdet = (
            math.log(1 - 2 * self.alpha) - torch.log(s) - torch.log1p(-s)
        ).flatten(1).sum(dim=1)
        return z, logdet

    def postprocess(self, z: torch.Tensor) -> torch.Tensor:
        """Undo the logit transform, returning images in ``[0, 1]``."""
        x = torch.sigmoid(z)
        return ((x - self.alpha) / (1 - 2 * self.alpha)).clamp(0.0, 1.0)

    # -- forward / inverse ---------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Image -> latent, returning ``(z, total log|det|)``."""
        h, logdet = self.preprocess(x)
        for layers in self.scales:
            for i, layer in enumerate(layers):
                if i == self.couplings_per_scale:
                    h = squeeze(h)
                h, ld = layer(h)
                logdet = logdet + ld
        return h, logdet

    @torch.no_grad()
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Latent -> image, running every coupling layer backwards.

        The coupling networks contain batch normalisation, whose behaviour
        differs between train and eval mode.  Inversion is only exact when both
        directions see the same statistics, so eval mode is forced here and the
        previous mode restored afterwards.
        """
        was_training = self.training
        self.eval()
        try:
            h = z
            for layers in reversed(self.scales):
                for i, layer in enumerate(reversed(list(layers))):
                    h = layer.inverse(h)
                    if i == self.couplings_per_scale - 1:
                        h = unsqueeze(h)
            return self.postprocess(h)
        finally:
            self.train(was_training)

    # -- likelihood ------------------------------------------------------------ #
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact log-density of the dequantised data, in nats per image."""
        z, logdet = self(x)
        prior = -0.5 * (z**2 + math.log(2 * math.pi)).flatten(1).sum(dim=1)
        return prior + logdet

    def bits_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood in bits per subpixel.

        The ``+ 8`` accounts for the 1/256 rescaling applied during
        dequantisation, which converts the density over ``[0, 1]`` into one over
        the original 8-bit pixel grid.
        """
        dims = x[0].numel()
        return -self.log_prob(x) / (dims * math.log(2.0)) + 8.0

    @torch.no_grad()
    def sample(self, n: int, device: torch.device, temperature: float = 1.0) -> torch.Tensor:
        """Draw from the prior and push it through the inverse flow.

        ``temperature < 1`` shrinks the prior, which trades diversity for
        typicality -- the same trick Glow uses for its cleanest samples.
        """
        z = torch.randn(
            n, self.final_channels, self.final_size, self.final_size, device=device
        ) * temperature
        return self.inverse(z)
