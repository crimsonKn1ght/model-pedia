"""DDPM: denoising diffusion probabilistic models, with a small U-Net.

Diffusion splits generation into two processes.

**Forward (fixed, no learning).**  Gaussian noise is added to an image over
``T`` steps until nothing but noise is left.  The useful algebraic fact is that
any step can be reached in closed form::

    x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * eps

so training never has to simulate the chain.  Pick a random ``t``, jump straight
there, and ask the network to recover the noise.

**Reverse (learned).**  A U-Net ``eps_theta(x_t, t)`` predicts the noise that was
added.  Sampling starts from pure noise and walks back down the chain, removing
a little predicted noise at each step.

The training objective is almost embarrassingly simple -- mean squared error
between the true and predicted noise::

    L = E_{x_0, t, eps} || eps  -  eps_theta(x_t, t) ||^2

That is the whole loss.  No adversarial game, no posterior approximation, no
likelihood bound to wrestle with, which is a large part of why diffusion took
over.  The cost is paid at sampling time: ``T`` sequential network evaluations,
which is exactly the problem project 09 (DDIM) addresses.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Timestep conditioning
# --------------------------------------------------------------------------- #
class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style positional encoding, applied to the timestep.

    The network is shared across all ``T`` noise levels, so it must be told
    which one it is looking at.  A smooth encoding lets it interpolate between
    neighbouring timesteps rather than memorising each independently.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResidualBlock(nn.Module):
    """Convolutional block with the timestep embedding injected as a bias."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        groups_in = math.gcd(groups, in_ch)
        groups_out = math.gcd(groups, out_ch)
        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over spatial positions.

    Used only at the lowest resolution, where the token count is small enough
    for the quadratic cost to be irrelevant.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(math.gcd(groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(dim=1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.out(out)


class UNet(nn.Module):
    """Compact U-Net noise predictor.

    Skip connections matter here for the same reason they do in Pix2Pix: the
    output has the same spatial layout as the input, so detail should travel
    sideways rather than be squeezed through the bottleneck.
    """

    def __init__(
        self,
        channels: int = 1,
        base: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 2),
        time_dim: int = 128,
        attention_at: int = 2,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # Optional class conditioning, used by the latent-diffusion project.
        self.label_embedding = nn.Embedding(num_classes, time_dim) if num_classes else None

        widths = [base * m for m in channel_multipliers]
        levels = len(widths)
        self.stem = nn.Conv2d(channels, widths[0], 3, padding=1)

        # Encoder: one residual block per level, then a stride-2 downsample
        # everywhere except the last level. Exactly one skip is stored per
        # level, so the decoder pops exactly one per level and the resolutions
        # always line up.
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = widths[0]
        for level, width in enumerate(widths):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(in_ch, width, time_dim),
                        SelfAttention(width) if level >= attention_at else nn.Identity(),
                    ]
                )
            )
            in_ch = width
            last = level == levels - 1
            self.downsamples.append(
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1) if not last else nn.Identity()
            )

        self.mid1 = ResidualBlock(in_ch, in_ch, time_dim)
        self.mid_attn = SelfAttention(in_ch)
        self.mid2 = ResidualBlock(in_ch, in_ch, time_dim)

        # Decoder: mirror the encoder. At level `i` the popped skip has
        # `widths[i]` channels and matches the current resolution.
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(levels)):
            width = widths[level]
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(in_ch + width, width, time_dim),
                        SelfAttention(width) if level >= attention_at else nn.Identity(),
                    ]
                )
            )
            in_ch = width
            first = level == 0
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_ch, in_ch, 3, padding=1),
                )
                if not first
                else nn.Identity()
            )

        self.out_norm = nn.GroupNorm(math.gcd(8, in_ch), in_ch)
        self.out_conv = nn.Conv2d(in_ch, channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        if self.label_embedding is not None and y is not None:
            t_emb = t_emb + self.label_embedding(y)

        h = self.stem(x)
        skips = []
        for (block, attn), down in zip(self.down_blocks, self.downsamples):
            h = attn(block(h, t_emb))
            skips.append(h)  # one skip per level, before downsampling
            h = down(h)

        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)

        for (block, attn), up in zip(self.up_blocks, self.upsamples):
            h = attn(block(torch.cat([h, skips.pop()], dim=1), t_emb))
            if not isinstance(up, nn.Identity):
                h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# --------------------------------------------------------------------------- #
# Noise schedule and the diffusion process
# --------------------------------------------------------------------------- #
def make_beta_schedule(kind: str, timesteps: int) -> torch.Tensor:
    """Variance schedule for the forward process.

    ``linear`` is the original DDPM schedule.  ``cosine`` (Nichol & Dhariwal)
    destroys information more gradually, which helps noticeably at low
    resolutions where the linear schedule turns the image into noise too early.
    """
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    if kind == "cosine":
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(1e-8, 0.999).float()
    raise ValueError(f"unknown schedule {kind!r}")


class GaussianDiffusion(nn.Module):
    """Holds the schedule and implements training loss and ancestral sampling."""

    def __init__(self, model: nn.Module, timesteps: int = 400, schedule: str = "cosine"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = make_beta_schedule(schedule, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        # Everything below is precomputed constants, not learned parameters.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        # Variance of the true posterior q(x_{t-1} | x_t, x_0).
        self.register_buffer(
            "posterior_variance", betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )

    @staticmethod
    def _gather(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, t)
        return out.view(-1, *([1] * (len(shape) - 1)))

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Jump straight to step ``t`` of the forward process."""
        noise = torch.randn_like(x0) if noise is None else noise
        mean = self._gather(self.sqrt_alpha_bars, t, x0.shape) * x0
        std = self._gather(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return mean + std * noise, noise

    def loss(self, x0: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The simple objective: predict the noise that was added."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        predicted = self.model(x_t, t, y) if y is not None else self.model(x_t, t)
        return F.mse_loss(predicted, noise)

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Invert the forward equation to recover an estimate of the clean image."""
        sqrt_ab = self._gather(self.sqrt_alpha_bars, t, x_t.shape)
        sqrt_1mab = self._gather(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        return ((x_t - sqrt_1mab * noise) / sqrt_ab).clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        shape: Tuple[int, int, int],
        device: torch.device,
        y: Optional[torch.Tensor] = None,
        record_every: int = 0,
    ):
        """Ancestral sampling: walk the full chain from noise back to an image.

        With ``record_every > 0`` the intermediate states are returned too,
        which is what the denoising-trajectory figure is built from.
        """
        self.model.eval()
        x = torch.randn(n, *shape, device=device)
        trajectory = []

        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            noise_pred = self.model(x, t, y) if y is not None else self.model(x, t)
            x0_pred = self.predict_x0(x, t, noise_pred)

            # Posterior mean of q(x_{t-1} | x_t, x_0), with x_0 replaced by its estimate.
            beta = self._gather(self.betas, t, x.shape)
            alpha = self._gather(self.alphas, t, x.shape)
            alpha_bar = self._gather(self.alpha_bars, t, x.shape)
            alpha_bar_prev = self._gather(self.alpha_bars_prev, t, x.shape)
            mean = (
                beta * alpha_bar_prev.sqrt() / (1 - alpha_bar) * x0_pred
                + (1 - alpha_bar_prev) * alpha.sqrt() / (1 - alpha_bar) * x
            )
            if step > 0:
                variance = self._gather(self.posterior_variance, t, x.shape)
                x = mean + variance.clamp(min=1e-20).sqrt() * torch.randn_like(x)
            else:
                x = mean  # the last step is deterministic

            if record_every and (step % record_every == 0 or step == 0):
                trajectory.append(x.clone())

        return (x, trajectory) if record_every else x
