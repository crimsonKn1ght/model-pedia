"""VQ-VAE: an autoencoder with a *discrete* latent space, plus a PixelCNN prior.

A plain VAE has a continuous latent code.  A VQ-VAE instead keeps a codebook of
``K`` embedding vectors and snaps every spatial position of the encoder output
to its nearest codebook entry.  The latent representation of an image is
therefore a small grid of integers -- for a 32x32 input downsampled by 4 that is
an 8x8 grid of indices drawn from ``K`` symbols.

Two consequences drive the whole design:

1. ``argmin`` has no gradient.  The **straight-through estimator** copies the
   gradient from the decoder input straight back to the encoder output, and two
   auxiliary losses (codebook loss and commitment loss) keep the two in step.
2. Because the latent is discrete, *sampling* needs a prior over index grids.
   That is what the PixelCNN is for: it models ``p(indices)`` autoregressively,
   and decoding a sampled grid produces a new image.  A VQ-VAE on its own is a
   compressor; VQ-VAE + prior is a generative model.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Vector quantiser
# --------------------------------------------------------------------------- #
class VectorQuantizer(nn.Module):
    """Nearest-neighbour lookup into a learned codebook.

    ``decay > 0`` selects the EMA variant from the original paper: the codebook
    is updated with an exponential moving average of the encoder outputs
    assigned to it rather than by gradient descent, which trains noticeably more
    stably.  ``decay = 0`` falls back to the plain codebook loss.
    """

    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        restart_threshold: float = 1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        # Codes whose EMA cluster size falls below this are considered dead and
        # get reseeded from real encoder outputs. Set to 0 to disable.
        self.restart_threshold = restart_threshold

        embed = torch.randn(num_embeddings, embedding_dim) * 0.1
        if decay > 0:
            # EMA statistics are buffers, not parameters: no gradient touches them.
            self.register_buffer("embedding", embed)
            self.register_buffer("cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("embed_avg", embed.clone())
        else:
            self.embedding = nn.Parameter(embed)

    def _codebook(self) -> torch.Tensor:
        return self.embedding

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise ``z`` of shape ``(B, D, H, W)``.

        Returns ``(quantized, loss, indices, perplexity)``.
        """
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)  # (BHW, D)
        codebook = self._codebook()

        # ||z - e||^2 expanded so no (BHW, K, D) tensor is ever materialised.
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        one_hot = F.one_hot(indices, self.num_embeddings).type(flat.dtype)
        quantized = (one_hot @ codebook).view(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        if self.decay > 0 and self.training:
            with torch.no_grad():
                counts = one_hot.sum(dim=0)
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                embed_sum = one_hot.t() @ flat
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                # Laplace smoothing stops unused codes from collapsing to zero.
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )
                self.embedding.copy_(self.embed_avg / cluster_size.unsqueeze(1))

                # Dead-code restarts. Codes that stop winning any inputs never
                # receive an EMA update again, so they stay dead forever and the
                # effective codebook shrinks -- the classic VQ-VAE collapse.
                # Reseeding them from real encoder outputs puts them back
                # somewhere the data actually is.
                if self.restart_threshold > 0:
                    dead = self.cluster_size < self.restart_threshold
                    num_dead = int(dead.sum())
                    if num_dead > 0:
                        pick = torch.randint(flat.shape[0], (num_dead,), device=flat.device)
                        seeds = flat[pick]
                        self.embedding[dead] = seeds
                        self.embed_avg[dead] = seeds
                        self.cluster_size[dead] = 1.0
            loss = self.commitment_cost * F.mse_loss(z, quantized.detach())
        else:
            codebook_loss = F.mse_loss(quantized, z.detach())
            commitment = F.mse_loss(z, quantized.detach())
            loss = codebook_loss + self.commitment_cost * commitment

        # Straight-through: forward uses `quantized`, backward pretends it was `z`.
        quantized = z + (quantized - z).detach()

        # Perplexity measures how many codes are effectively in use.
        probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
        return quantized, loss, indices.view(b, h, w), perplexity

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Index grid ``(B, H, W)`` -> embedding map ``(B, D, H, W)``."""
        flat = F.embedding(indices.reshape(-1), self._codebook())
        b, h, w = indices.shape
        return flat.view(b, h, w, self.embedding_dim).permute(0, 3, 1, 2).contiguous()


# --------------------------------------------------------------------------- #
# Encoder / decoder
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    """Downsamples by 4 and projects to the codebook dimension."""

    def __init__(self, channels: int, hidden: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, hidden, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Decoder(nn.Module):
    """Mirror of the encoder: upsamples by 4 back to image resolution."""

    def __init__(self, channels: int, hidden: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, 3, padding=1),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden // 2, channels, 4, stride=2, padding=1),
            nn.Tanh(),  # data lives in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.body(z)


class VQVAE(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        hidden: int = 128,
        embedding_dim: int = 64,
        num_embeddings: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        restart_threshold: float = 1.0,
    ):
        super().__init__()
        self.encoder = Encoder(channels, hidden, embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost, decay, restart_threshold=restart_threshold
        )
        self.decoder = Decoder(channels, hidden, embedding_dim)
        self.channels = channels
        self.num_embeddings = num_embeddings

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quantized, vq_loss, indices, perplexity = self.quantizer(z)
        return self.decoder(quantized), vq_loss, indices, perplexity

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        _, _, indices, _ = self.quantizer(self.encoder(x))
        return indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantizer.lookup(indices))


# --------------------------------------------------------------------------- #
# PixelCNN prior over the index grid
# --------------------------------------------------------------------------- #
class MaskedConv2d(nn.Conv2d):
    """Convolution that may only look at pixels already generated.

    Mask 'A' also blanks the centre pixel and is used for the first layer, so
    the prediction for a position never depends on that position's own value.
    Mask 'B' keeps the centre and is used everywhere after.
    """

    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")
        _, _, kh, kw = self.weight.shape
        mask = torch.ones(kh, kw)
        mask[kh // 2, kw // 2 + (mask_type == "B") :] = 0
        mask[kh // 2 + 1 :] = 0
        self.register_buffer("mask", mask.view(1, 1, kh, kw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data.mul_(self.mask)
        return super().forward(x)


class PixelCNNPrior(nn.Module):
    """Autoregressive model of the discrete latent grid.

    Trained with plain cross entropy to predict each index from the indices
    above and to the left of it.  Sampling is a raster-scan loop, which is slow
    but only over an 8x8 grid.
    """

    def __init__(self, num_embeddings: int = 256, hidden: int = 96, layers: int = 6):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed = nn.Embedding(num_embeddings, hidden)
        blocks = [MaskedConv2d("A", hidden, hidden, 7, padding=3), nn.ReLU(inplace=True)]
        for _ in range(layers - 1):
            blocks += [
                MaskedConv2d("B", hidden, hidden, 3, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ]
        blocks += [nn.Conv2d(hidden, num_embeddings, 1)]
        self.body = nn.Sequential(*blocks)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """``(B, H, W)`` indices -> ``(B, K, H, W)`` logits."""
        h = self.embed(indices).permute(0, 3, 1, 2).contiguous()
        return self.body(h)

    @torch.no_grad()
    def sample(self, n: int, height: int, width: int, device: torch.device,
               temperature: float = 1.0) -> torch.Tensor:
        """Generate index grids one position at a time, in raster order."""
        self.eval()
        grid = torch.zeros(n, height, width, dtype=torch.long, device=device)
        for i in range(height):
            for j in range(width):
                logits = self(grid)[:, :, i, j] / max(temperature, 1e-6)
                grid[:, i, j] = torch.multinomial(F.softmax(logits, dim=1), 1).squeeze(1)
        return grid
