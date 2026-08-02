"""Convolutional VAE / beta-VAE.

The encoder maps an image to the parameters of a diagonal Gaussian
``q(z|x) = N(mu, diag(sigma^2))``.  A sample is drawn with the reparameterisation
trick ``z = mu + sigma * eps`` so gradients flow through the sampling step, and
the decoder turns ``z`` back into an image.

Training maximises the evidence lower bound::

    ELBO = E_q(z|x)[ log p(x|z) ]  -  beta * KL( q(z|x) || N(0, I) )

``beta = 1`` is the plain VAE.  ``beta > 1`` is the beta-VAE: the KL term is
weighted more heavily, which pushes the posterior towards the isotropic prior
and encourages individual latent dimensions to capture independent factors of
variation, at the cost of blurrier reconstructions.  Running the same script
with ``--beta 1`` and ``--beta 4`` and comparing the traversal figures is the
whole point of the project.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_downsamples(image_size: int, min_size: int = 4) -> int:
    """How many stride-2 stages fit between ``image_size`` and ``min_size``."""
    n = 0
    size = image_size
    while size > min_size and size % 2 == 0:
        size //= 2
        n += 1
    if n == 0:
        raise ValueError(f"image_size={image_size} is too small or not divisible by 2")
    return n


class Encoder(nn.Module):
    """Image -> (mu, logvar) of the approximate posterior."""

    def __init__(self, channels: int, image_size: int, latent_dim: int, base: int = 32):
        super().__init__()
        stages = _num_downsamples(image_size)
        layers = []
        in_ch = channels
        out_ch = base
        for _ in range(stages):
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            ]
            in_ch = out_ch
            out_ch = min(out_ch * 2, base * 8)
        self.body = nn.Sequential(*layers)
        self.spatial = image_size // (2**stages)
        self.flat_dim = in_ch * self.spatial * self.spatial
        self.to_mu = nn.Linear(self.flat_dim, latent_dim)
        self.to_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x).flatten(1)
        # Clamping keeps sigma in a sane range early in training.
        return self.to_mu(h), self.to_logvar(h).clamp(-8.0, 8.0)


class Decoder(nn.Module):
    """Latent vector -> image."""

    def __init__(self, channels: int, image_size: int, latent_dim: int, base: int = 32):
        super().__init__()
        stages = _num_downsamples(image_size)
        self.spatial = image_size // (2**stages)
        widths = [min(base * 2**i, base * 8) for i in range(stages)][::-1]
        self.start_ch = widths[0]
        self.fc = nn.Linear(latent_dim, self.start_ch * self.spatial * self.spatial)

        layers = []
        in_ch = self.start_ch
        for out_ch in widths[1:]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            ]
            in_ch = out_ch
        layers += [nn.ConvTranspose2d(in_ch, channels, 4, stride=2, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.start_ch, self.spatial, self.spatial)
        return self.body(h)  # raw logits / unbounded means


class VAE(nn.Module):
    """Convolutional VAE with a Bernoulli or Gaussian observation model.

    ``likelihood="bernoulli"`` treats pixel intensities in ``[0, 1]`` as
    Bernoulli probabilities and uses binary cross entropy, which is the standard
    choice for MNIST.  ``likelihood="gaussian"`` uses a fixed-variance Gaussian
    (i.e. mean squared error) and suits natural images such as CelebA.
    """

    def __init__(
        self,
        channels: int = 1,
        image_size: int = 32,
        latent_dim: int = 16,
        base: int = 32,
        likelihood: str = "bernoulli",
    ):
        super().__init__()
        if likelihood not in {"bernoulli", "gaussian"}:
            raise ValueError("likelihood must be 'bernoulli' or 'gaussian'")
        self.encoder = Encoder(channels, image_size, latent_dim, base)
        self.decoder = Decoder(channels, image_size, latent_dim, base)
        self.latent_dim = latent_dim
        self.likelihood = likelihood
        self.channels = channels
        self.image_size = image_size

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    # -- outputs in image space ------------------------------------------- #
    def decode_to_image(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return torch.sigmoid(out) if self.likelihood == "bernoulli" else out.clamp(0.0, 1.0)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Draw ``n`` images from the prior ``p(z) = N(0, I)``."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode_to_image(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return self.decode_to_image(mu)  # posterior mean, not a random sample


def vae_loss(
    recon_logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    likelihood: str = "bernoulli",
):
    """Negative ELBO, averaged over the batch and summed over pixels/dimensions.

    Summing over pixels (rather than averaging) keeps the reconstruction term
    and the KL term on the same scale, which is what makes ``beta`` interpretable.
    """
    batch = x.shape[0]
    if likelihood == "bernoulli":
        recon = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum") / batch
    else:
        recon = F.mse_loss(recon_logits, x, reduction="sum") / batch

    # KL( N(mu, sigma^2) || N(0, I) ) in closed form.
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()) / batch
    return recon + beta * kl, recon, kl


def bits_per_dim(negative_elbo: float, channels: int, image_size: int) -> float:
    """Convert a nats-per-image bound into bits per dimension."""
    dims = channels * image_size * image_size
    return negative_elbo / (dims * math.log(2.0))
