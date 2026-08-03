"""A convolutional VAE, and the beta knob that turns it into a beta-VAE.

    python model.py

An autoencoder (project 04) learns a *point* in latent space per image. A VAE learns
a *distribution*: the encoder outputs a mean and a log-variance, a sample is drawn
from it, and the decoder has to reconstruct from that sample. Two things follow, and
they are the whole reason to prefer it:

**The latent space becomes usable.** Because the encoder must place a whole Gaussian
where the image lives, nearby codes have to decode to plausible images - there is no
room for the empty pockets a plain autoencoder leaves between its training points.
So interpolations stay on the data manifold, and sampling from the prior gives
something rather than noise.

**There is a loss you can write down.** The objective is a lower bound on the log
likelihood, the ELBO:

    log p(x) >= E[log p(x|z)] - KL(q(z|x) || p(z))

The first term is reconstruction, the second is a pull towards the unit Gaussian
prior. That is the entire model.

The two terms fight, and ``beta`` sets the exchange rate. At ``beta=1`` you have a
VAE. Raise it and the KL term wins: the latent gets tidier and more disentangled,
reconstructions get blurrier, and eventually dimensions stop carrying information at
all - **posterior collapse**, which ``active_units`` measures directly.

The reparameterisation trick is the one line that makes any of it trainable:
sampling ``z ~ N(mu, sigma^2)`` has no gradient, so instead sample
``eps ~ N(0, 1)`` and compute ``z = mu + sigma * eps``, which does.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LIKELIHOODS = ("bernoulli", "gaussian")


class ConvVAE(nn.Module):
    """Encoder and decoder of strided convolutions around a Gaussian latent."""

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 32,
        latent_dim: int = 16,
        base_channels: int = 32,
        beta: float = 1.0,
        likelihood: str = "bernoulli",
        levels: int = 3,
    ) -> None:
        super().__init__()
        if likelihood not in LIKELIHOODS:
            raise ValueError(f"unknown likelihood {likelihood!r}, expected one of {LIKELIHOODS}")
        if image_size % 2**levels:
            raise ValueError(f"image size {image_size} must be divisible by {2 ** levels}")

        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.likelihood = likelihood
        self.grid = image_size // 2**levels

        channels = [in_channels] + [base_channels * 2**level for level in range(levels)]
        encoder = []
        for level in range(levels):
            encoder += [
                nn.Conv2d(channels[level], channels[level + 1], 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels[level + 1]),
                nn.ReLU(inplace=True),
            ]
        self.encoder = nn.Sequential(*encoder)
        self.flat_dim = channels[-1] * self.grid**2

        # One linear layer produces both mu and log-variance.
        self.to_latent = nn.Linear(self.flat_dim, 2 * latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.flat_dim)

        decoder = []
        for level in range(levels, 0, -1):
            out_channels = channels[level - 1] if level > 1 else base_channels
            decoder += [
                nn.ConvTranspose2d(channels[level], out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.decoder = nn.Sequential(*decoder)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        self.decoder_channels = channels[-1]

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(images).flatten(1)
        mu, logvar = self.to_latent(hidden).chunk(2, dim=1)
        # Clamping stops a diverging variance from producing inf in the KL term.
        return mu, logvar.clamp(-8.0, 8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu  # the posterior mean is the deterministic code
        standard_deviation = (0.5 * logvar).exp()
        return mu + standard_deviation * torch.randn_like(standard_deviation)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Returns logits for ``bernoulli``, or the mean for ``gaussian``."""
        hidden = self.from_latent(latent)
        hidden = hidden.view(-1, self.decoder_channels, self.grid, self.grid)
        return self.out_conv(self.decoder(hidden))

    def reconstruct_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Turn decoder output into an image in ``[0, 1]``."""
        return logits.sigmoid() if self.likelihood == "bernoulli" else logits.clamp(0, 1)

    def forward(self, images: torch.Tensor) -> dict:
        mu, logvar = self.encode(images)
        latent = self.reparameterize(mu, logvar)
        logits = self.decode(latent)
        return {"logits": logits, "mu": mu, "logvar": logvar, "latent": latent}

    def loss(self, images: torch.Tensor, outputs: dict) -> dict:
        """Negative ELBO, in nats per image, plus its two halves separately."""
        logits, mu, logvar = outputs["logits"], outputs["mu"], outputs["logvar"]

        if self.likelihood == "bernoulli":
            # Bernoulli log-likelihood on continuous [0, 1] targets. Not a proper
            # density (the data is not binary) but it is the standard VAE choice and
            # it trains far better than a fixed-variance Gaussian on greyscale.
            reconstruction = F.binary_cross_entropy_with_logits(
                logits, images, reduction="none"
            ).flatten(1).sum(dim=1)
        else:
            # Gaussian with unit variance: squared error plus the normalising constant,
            # so the number is comparable across likelihoods rather than off by a factor.
            squared = (logits.clamp(0, 1) - images).pow(2).flatten(1).sum(dim=1)
            constant = 0.5 * math.log(2 * math.pi) * images[0].numel()
            reconstruction = 0.5 * squared + constant

        # KL between N(mu, sigma^2) and N(0, 1), per latent dimension.
        kl_per_dimension = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl = kl_per_dimension.sum(dim=1)

        return {
            "loss": (reconstruction + self.beta * kl).mean(),
            "reconstruction": reconstruction.mean(),
            "kl": kl.mean(),
            "neg_elbo": (reconstruction + kl).mean(),  # beta-free, so arms compare
            "kl_per_dimension": kl_per_dimension.mean(dim=0).detach(),
        }

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """Draw from the prior and decode. This is the test a plain AE fails."""
        device = device or next(self.parameters()).device
        latent = torch.randn(n, self.latent_dim, device=device)
        return self.reconstruct_from_logits(self.decode(latent))

    @torch.no_grad()
    def interpolate(self, first: torch.Tensor, second: torch.Tensor, steps: int = 8):
        """Decode a straight line between two images' posterior means."""
        mu_a, _ = self.encode(first.unsqueeze(0))
        mu_b, _ = self.encode(second.unsqueeze(0))
        weights = torch.linspace(0, 1, steps, device=mu_a.device).view(-1, 1)
        latents = (1 - weights) * mu_a + weights * mu_b
        return self.reconstruct_from_logits(self.decode(latents))

    @torch.no_grad()
    def traverse(self, base: torch.Tensor, dimension: int, span: float = 3.0, steps: int = 8):
        """Vary one latent dimension, hold the rest. The beta-VAE picture."""
        mu, _ = self.encode(base.unsqueeze(0))
        latents = mu.repeat(steps, 1)
        latents[:, dimension] = torch.linspace(-span, span, steps, device=mu.device)
        return self.reconstruct_from_logits(self.decode(latents))


def active_units(kl_per_dimension: torch.Tensor, threshold: float = 0.01) -> int:
    """Latent dimensions carrying more than ``threshold`` nats of information.

    A dimension whose KL is ~0 has posterior equal to prior: the encoder ignores it
    and the decoder cannot read it. Counting them is the cheapest honest measure of
    posterior collapse, and it is what makes the beta study legible - the latent does
    not degrade smoothly, dimensions switch off one at a time.
    """
    return int((kl_per_dimension > threshold).sum())


def build_model(
    in_channels: int = 1,
    image_size: int = 32,
    latent_dim: int = 16,
    base_channels: int = 32,
    beta: float = 1.0,
    likelihood: str = "bernoulli",
) -> ConvVAE:
    return ConvVAE(
        in_channels=in_channels,
        image_size=image_size,
        latent_dim=latent_dim,
        base_channels=base_channels,
        beta=beta,
        likelihood=likelihood,
    )


if __name__ == "__main__":
    for channels, size in ((1, 32), (3, 32), (3, 64)):
        images = torch.rand(4, channels, size, size)
        for likelihood in LIKELIHOODS:
            model = build_model(channels, size, latent_dim=16, likelihood=likelihood)
            model.train()
            outputs = model(images)
            losses = model.loss(images, outputs)
            print(
                f"C={channels} {size}x{size} {likelihood:9s} "
                f"-ELBO {losses['neg_elbo'].item():9.2f} nats  "
                f"recon {losses['reconstruction'].item():9.2f}  kl {losses['kl'].item():7.3f}  "
                f"params {sum(p.numel() for p in model.parameters()):,}"
            )

    model = build_model(1, 32, latent_dim=16)
    model.eval()
    print("\nprior samples :", tuple(model.sample(6).shape))
    print("interpolation :", tuple(model.interpolate(torch.rand(1, 32, 32),
                                                     torch.rand(1, 32, 32)).shape))
    print("traversal     :", tuple(model.traverse(torch.rand(1, 32, 32), 0).shape))

    # In eval mode the code is the posterior mean, so encoding is deterministic.
    first = model(torch.rand(2, 1, 32, 32))["latent"]
    second = model(torch.rand(2, 1, 32, 32))["latent"]
    print("eval mode is deterministic:", not torch.allclose(first, second))
