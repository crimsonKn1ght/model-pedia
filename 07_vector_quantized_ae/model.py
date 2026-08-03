"""A VQ-VAE, and the autoregressive prior that makes it a generative model.

    python model.py

A VAE (project 06) has a continuous latent with a Gaussian prior you can sample from
directly. A VQ-VAE replaces that with a **grid of discrete codes**: the encoder
produces a feature map, each position is snapped to its nearest entry in a learned
codebook, and the decoder works from the snapped version. A 32x32 image becomes an
8x8 grid of integers.

Three problems come with that, and the three pieces below are their answers.

**1. argmin has no gradient.** The straight-through estimator copies the gradient
from the quantised tensor straight back to the continuous one:
``z_q = z_e + (z_q - z_e).detach()``. Forward pass sees the code, backward pass
pretends quantisation was the identity. It is a biased estimator and it works.

**2. The codebook has to be trained too.** Either by a loss term pulling entries
towards the encoder outputs assigned to them, or - the more stable option, and the
default here - by an exponential moving average of those assignments, which is
k-means in disguise and needs no gradient at all. The commitment term stays in both
cases, pulling the *encoder* towards its chosen code so the two do not drift apart.

**3. There is no prior to sample from.** The codes are integers with no distribution
attached, so a trained VQ-VAE can compress and reconstruct but cannot generate. The
fix is a second model: an autoregressive prior over the code grid, trained after the
fact on the codes the encoder produces. ``CodePrior`` here is a small causal
Transformer over the 64 flattened positions; the original paper used a PixelCNN.

That two-stage structure is not an implementation detail - it is why VQ-VAE became
the backbone of so much later work. Stage one learns a short discrete description of
an image; stage two is then an ordinary sequence-modelling problem.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Nearest-neighbour lookup into a learned codebook, with straight-through gradients."""

    def __init__(
        self,
        num_codes: int = 128,
        code_dim: int = 64,
        commitment: float = 0.25,
        ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment = commitment
        self.ema = ema
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        if ema:
            # The codebook is not updated by the optimiser in EMA mode, so its
            # statistics live in buffers and travel with the checkpoint.
            self.embedding.weight.requires_grad_(False)
            self.register_buffer("cluster_size", torch.zeros(num_codes))
            self.register_buffer("embed_average", self.embedding.weight.data.clone())

    def forward(self, features: torch.Tensor) -> dict:
        """``features`` is ``(B, D, H, W)``. Returns the quantised tensor and indices."""
        batch, dim, height, width = features.shape
        flat = features.permute(0, 2, 3, 1).reshape(-1, dim)

        # ||z - e||^2 expanded, so no (N, K, D) tensor is ever materialised.
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantised = self.embedding(indices).view(batch, height, width, dim).permute(0, 3, 1, 2)

        if self.ema and self.training:
            self._ema_update(flat, indices)

        codebook_loss = (
            torch.zeros((), device=features.device)
            if self.ema
            else F.mse_loss(quantised, features.detach())
        )
        commitment_loss = F.mse_loss(features, quantised.detach())

        # Straight-through: identical value, gradient routed around the argmin.
        quantised = features + (quantised - features).detach()
        return {
            "quantised": quantised,
            "indices": indices.view(batch, height, width),
            "codebook_loss": codebook_loss,
            "commitment_loss": self.commitment * commitment_loss,
        }

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        one_hot = F.one_hot(indices, self.num_codes).to(flat.dtype)
        self.cluster_size.mul_(self.decay).add_(one_hot.sum(dim=0), alpha=1 - self.decay)
        self.embed_average.mul_(self.decay).add_(one_hot.t() @ flat, alpha=1 - self.decay)

        # Laplace smoothing keeps an unused code from dividing by zero.
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (total + self.num_codes * self.eps) * total
        self.embedding.weight.data.copy_(self.embed_average / smoothed.unsqueeze(1))

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """``(B, H, W)`` integers -> ``(B, D, H, W)`` code vectors."""
        return self.embedding(indices).permute(0, 3, 1, 2)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 32,
        base_channels: int = 64,
        code_dim: int = 64,
        num_codes: int = 128,
        commitment: float = 0.25,
        ema: bool = True,
        levels: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.levels = levels
        self.grid = image_size // 2**levels
        self.num_codes = num_codes

        encoder = []
        channels = in_channels
        for _ in range(levels):
            encoder += [
                nn.Conv2d(channels, base_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            ]
            channels = base_channels
        encoder += [nn.Conv2d(channels, code_dim, 3, 1, 1)]
        self.encoder = nn.Sequential(*encoder)

        self.quantizer = VectorQuantizer(num_codes, code_dim, commitment, ema)

        decoder = [
            nn.Conv2d(code_dim, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(levels):
            decoder += [
                nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            ]
        decoder += [nn.Conv2d(base_channels, in_channels, 3, 1, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)

    @property
    def compression(self) -> float:
        """Pixels in, integers out - the ratio this representation actually buys."""
        pixels = self.in_channels * self.image_size**2
        return pixels / self.grid**2

    def forward(self, images: torch.Tensor) -> dict:
        quantised = self.quantizer(self.encoder(images))
        reconstruction = self.decoder(quantised["quantised"])
        return {**quantised, "reconstruction": reconstruction}

    def loss(self, images: torch.Tensor, outputs: dict) -> dict:
        reconstruction = F.mse_loss(outputs["reconstruction"], images)
        total = reconstruction + outputs["codebook_loss"] + outputs["commitment_loss"]
        return {
            "loss": total,
            "reconstruction": reconstruction,
            "codebook": outputs["codebook_loss"],
            "commitment": outputs["commitment_loss"],
        }

    @torch.no_grad()
    def encode_indices(self, images: torch.Tensor) -> torch.Tensor:
        return self.quantizer(self.encoder(images))["indices"]

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantizer.lookup(indices))


class PriorBlock(nn.Module):
    """Pre-norm causal transformer block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class CodePrior(nn.Module):
    """Autoregressive model over the flattened code grid, in raster order.

    Trained on the codes a *frozen* VQ-VAE produces. Its cross entropy is a real
    likelihood over the discrete latent, so unlike the VAE's ELBO there is no bound
    involved - but it is a likelihood over codes, not over pixels, so it says nothing
    about bits per dimension of the image.
    """

    def __init__(
        self,
        num_codes: int,
        sequence_length: int,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.sequence_length = sequence_length
        self.start_token = num_codes  # one extra embedding row for "sequence start"

        self.embedding = nn.Embedding(num_codes + 1, dim)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, dim))
        self.blocks = nn.ModuleList([PriorBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_codes)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        x = self.embedding(tokens) + self.position[:, :length]
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))

    def loss(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes`` is ``(B, L)``. Predict each position from the ones before it."""
        start = torch.full((codes.size(0), 1), self.start_token, device=codes.device)
        inputs = torch.cat([start, codes[:, :-1]], dim=1)
        logits = self(inputs)
        return F.cross_entropy(logits.reshape(-1, self.num_codes), codes.reshape(-1))

    @torch.no_grad()
    def sample(self, n: int, device: torch.device, temperature: float = 1.0) -> torch.Tensor:
        """Ancestral sampling, one position at a time. Returns ``(n, L)`` codes."""
        self.eval()
        tokens = torch.full((n, 1), self.start_token, device=device)
        for _ in range(self.sequence_length):
            logits = self(tokens)[:, -1] / max(temperature, 1e-6)
            nxt = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens[:, 1:]


def build_model(
    in_channels: int = 1,
    image_size: int = 32,
    base_channels: int = 64,
    code_dim: int = 64,
    num_codes: int = 128,
    commitment: float = 0.25,
    ema: bool = True,
) -> VQVAE:
    return VQVAE(
        in_channels=in_channels,
        image_size=image_size,
        base_channels=base_channels,
        code_dim=code_dim,
        num_codes=num_codes,
        commitment=commitment,
        ema=ema,
    )


def build_prior(
    num_codes: int, sequence_length: int, dim: int = 128, depth: int = 4, heads: int = 4
) -> CodePrior:
    return CodePrior(num_codes, sequence_length, dim, depth, heads)


if __name__ == "__main__":
    for channels, size in ((1, 32), (3, 32), (3, 64)):
        images = torch.rand(4, channels, size, size)
        for ema in (True, False):
            model = build_model(channels, size, num_codes=128, ema=ema)
            model.train()
            outputs = model(images)
            losses = model.loss(images, outputs)
            print(
                f"C={channels} {size}x{size} {'ema      ' if ema else 'loss-based'} "
                f"grid {model.grid}x{model.grid}  recon {losses['reconstruction'].item():.4f}  "
                f"commit {losses['commitment'].item():.4f}  "
                f"compression {model.compression:.0f}x  "
                f"params {sum(p.numel() for p in model.parameters()):,}"
            )

    model = build_model(1, 32, num_codes=128)
    model.eval()
    indices = model.encode_indices(torch.rand(4, 1, 32, 32))
    print(f"\ncodes        : {tuple(indices.shape)} integers in [0, 128)")
    print(f"round trip   : {tuple(model.decode_indices(indices).shape)}")

    prior = build_prior(128, model.grid**2)
    codes = indices.flatten(1)
    print(f"prior loss   : {prior.loss(codes).item():.4f} nats/code "
          f"({sum(p.numel() for p in prior.parameters()):,} parameters)")
    sampled = prior.sample(2, torch.device("cpu"))
    print(f"prior samples: {tuple(sampled.shape)} -> "
          f"{tuple(model.decode_indices(sampled.view(-1, model.grid, model.grid)).shape)}")

    # Straight-through must pass gradient to the encoder despite the argmin.
    model.train()
    images = torch.rand(2, 1, 32, 32)
    model.loss(images, model(images))["loss"].backward()
    grad = model.encoder[0].weight.grad
    print("encoder receives gradient through the quantiser:", bool(grad.abs().sum() > 0))
