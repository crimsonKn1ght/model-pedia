"""DDPM: a time-conditioned U-Net, and the diffusion process it is trained inside.

    python model.py

A diffusion model turns generation into a long sequence of small, easy problems.

**Forward process** (no learning): repeatedly add a little Gaussian noise to an image
until nothing is left but noise. Because each step is Gaussian, the composition of
``t`` steps is Gaussian too, and there is a closed form - ``q_sample`` below - that
jumps straight to any timestep. That is what makes training cheap: a training step
picks a random ``t``, jumps there in one shot, and asks the network to undo it.

**Reverse process** (all the learning): a single network, told which timestep it is
looking at, predicts the noise that was added. Sampling then starts from pure noise and
walks back down, subtracting a little predicted noise at a time.

Two design choices are worth naming.

*Predicting the noise, not the image.* Algebraically equivalent - either determines
the other - but the noise target has the same scale at every timestep, while the clean
image is nearly free at ``t=1`` and impossible at ``t=T``. ``--prediction x0`` switches
to the other parameterisation so you can watch it be worse.

*The schedule matters.* A linear beta schedule destroys most of the signal in the first
third of the trajectory, so much of the network's capacity is spent on timesteps where
there is nothing left to learn from. The cosine schedule spends the budget more evenly
and is the default here.

The network is a small U-Net: skip connections carry high-resolution detail (the same
argument as project 05), the timestep enters every residual block through a sinusoidal
embedding, and one self-attention layer sits at the lowest resolution where it is cheap.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEDULES = ("cosine", "linear")
PREDICTIONS = ("noise", "x0")


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of an integer timestep, as in the transformer."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResidualBlock(nn.Module):
    """Two convolutions with the timestep embedding added in between."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # One scalar per channel, broadcast over space: the network's only knowledge
        # of how noisy its input is.
        h = h + self.time(F.silu(time)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head spatial attention, used only at the lowest resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(batch, 3, channels, height * width).unbind(1)
        attention = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(channels), dim=-1)
        out = (v @ attention.transpose(1, 2)).reshape(batch, channels, height, width)
        return x + self.out(out)


class UNet(nn.Module):
    """Three-resolution U-Net conditioned on the timestep."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 2),
        time_dim: int = 128,
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.base_channels = base_channels

        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        widths = [base_channels * multiplier for multiplier in channel_multipliers]

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = base_channels
        for index, width in enumerate(widths):
            self.down_blocks.append(ResidualBlock(channels, width, time_dim))
            channels = width
            self.downsamples.append(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1)
                if index < len(widths) - 1
                else nn.Identity()
            )

        self.mid_block1 = ResidualBlock(channels, channels, time_dim)
        self.mid_attention = SelfAttention(channels) if attention else nn.Identity()
        self.mid_block2 = ResidualBlock(channels, channels, time_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index, width in reversed(list(enumerate(widths))):
            self.upsamples.append(
                nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
                if index < len(widths) - 1
                else nn.Identity()
            )
            self.up_blocks.append(ResidualBlock(channels + width, width, time_dim))
            channels = width

        self.out_norm = nn.GroupNorm(8, channels)
        self.out_conv = nn.Conv2d(channels, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time = self.time_mlp(timestep_embedding(timesteps, self.base_channels))
        h = self.stem(x)

        skips = []
        for block, downsample in zip(self.down_blocks, self.downsamples):
            h = block(h, time)
            skips.append(h)
            h = downsample(h)

        h = self.mid_block2(self.mid_attention(self.mid_block1(h, time)), time)

        for upsample, block in zip(self.upsamples, self.up_blocks):
            h = upsample(h)
            h = block(torch.cat([h, skips.pop()], dim=1), time)

        return self.out_conv(F.silu(self.out_norm(h)))


def make_betas(schedule: str, steps: int) -> torch.Tensor:
    """Noise schedule. Cosine spends the budget more evenly than linear."""
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, steps)
    if schedule == "cosine":
        # Nichol & Dhariwal: define alpha_bar directly, then read betas off it.
        t = torch.linspace(0, 1, steps + 1)
        alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(1e-8, 0.999)
    raise ValueError(f"unknown schedule {schedule!r}, expected one of {SCHEDULES}")


class GaussianDiffusion(nn.Module):
    """The forward and reverse processes, with the network as an argument.

    Images live in ``[0, 1]`` everywhere else in this repository; diffusion needs them
    centred, so ``scale``/``unscale`` map to ``[-1, 1]`` at the boundary.
    """

    def __init__(
        self,
        steps: int = 200,
        schedule: str = "cosine",
        prediction: str = "noise",
    ) -> None:
        super().__init__()
        if prediction not in PREDICTIONS:
            raise ValueError(f"unknown prediction {prediction!r}, expected {PREDICTIONS}")
        self.steps = steps
        self.schedule = schedule
        self.prediction = prediction

        betas = make_betas(schedule, steps)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer(
            "alphas_cumprod_prev", torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        )

    @staticmethod
    def scale(images: torch.Tensor) -> torch.Tensor:
        return images * 2.0 - 1.0

    @staticmethod
    def unscale(images: torch.Tensor) -> torch.Tensor:
        return ((images + 1.0) / 2.0).clamp(0, 1)

    def _gather(self, values: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        return values.gather(0, t).view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Jump straight to timestep ``t`` in one step - the closed form."""
        return (
            self._gather(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def loss(self, model: UNet, images: torch.Tensor) -> torch.Tensor:
        """Pick a random timestep per image, corrupt, and predict."""
        x0 = self.scale(images)
        t = torch.randint(0, self.steps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        noisy = self.q_sample(x0, t, noise)
        prediction = model(noisy, t)
        target = noise if self.prediction == "noise" else x0
        return F.mse_loss(prediction, target)

    def predicted_noise(self, model: UNet, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Whatever the network predicts, expressed as noise."""
        output = model(x_t, t)
        if self.prediction == "noise":
            return output
        # Invert x_t = sqrt(a) x0 + sqrt(1-a) eps for eps.
        sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alpha * output) / sqrt_one_minus.clamp_min(1e-8)

    @torch.no_grad()
    def sample(
        self,
        model: UNet,
        shape: tuple[int, ...],
        device: torch.device,
        record: int = 0,
    ):
        """Ancestral sampling from pure noise. ``record`` keeps that many intermediates."""
        model.eval()
        x_t = torch.randn(shape, device=device)
        keep_at = set(
            torch.linspace(self.steps - 1, 0, record).long().tolist() if record else []
        )
        trajectory = []

        for step in reversed(range(self.steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            noise_prediction = self.predicted_noise(model, x_t, t)

            alpha_cumprod = self._gather(self.alphas_cumprod, t, x_t.shape)
            alpha_cumprod_prev = self._gather(self.alphas_cumprod_prev, t, x_t.shape)
            beta = self._gather(self.betas, t, x_t.shape)

            x0 = ((x_t - (1 - alpha_cumprod).sqrt() * noise_prediction)
                  / alpha_cumprod.sqrt().clamp_min(1e-8)).clamp(-1, 1)
            # Posterior mean of q(x_{t-1} | x_t, x0), the DDPM update.
            mean = (
                alpha_cumprod_prev.sqrt() * beta / (1 - alpha_cumprod).clamp_min(1e-8) * x0
                + ((1 - beta).sqrt() * (1 - alpha_cumprod_prev)
                   / (1 - alpha_cumprod).clamp_min(1e-8)) * x_t
            )
            if step > 0:
                variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod).clamp_min(1e-8)
                x_t = mean + variance.sqrt() * torch.randn_like(x_t)
            else:
                x_t = mean

            if step in keep_at:
                trajectory.append(self.unscale(x_t).cpu())

        images = self.unscale(x_t)
        return (images, trajectory) if record else images


def build_model(
    in_channels: int = 1,
    base_channels: int = 32,
    attention: bool = True,
) -> UNet:
    return UNet(in_channels=in_channels, base_channels=base_channels, attention=attention)


def build_diffusion(
    steps: int = 200, schedule: str = "cosine", prediction: str = "noise"
) -> GaussianDiffusion:
    return GaussianDiffusion(steps=steps, schedule=schedule, prediction=prediction)


if __name__ == "__main__":
    for channels, size in ((1, 32), (3, 32)):
        model = build_model(channels)
        diffusion = build_diffusion(steps=50)
        images = torch.rand(4, channels, size, size)
        loss = diffusion.loss(model, images)
        print(
            f"C={channels} {size}x{size}: loss {loss.item():.4f}  "
            f"U-Net {sum(p.numel() for p in model.parameters()):,} parameters"
        )

    for schedule in SCHEDULES:
        diffusion = build_diffusion(steps=200, schedule=schedule)
        alpha_bar = diffusion.alphas_cumprod
        # Signal-to-noise ratio at a few points along the trajectory.
        points = [0, 49, 99, 149, 199]
        snr = [f"{(alpha_bar[p] / (1 - alpha_bar[p])).sqrt():.2f}" for p in points]
        print(f"{schedule:7s} sqrt-SNR at t={points}: {snr}")

    model = build_model(1)
    diffusion = build_diffusion(steps=20)
    samples, trajectory = diffusion.sample(model, (2, 1, 32, 32), torch.device("cpu"), record=4)
    print(f"\nsamples {tuple(samples.shape)} in [{samples.min():.2f}, {samples.max():.2f}], "
          f"{len(trajectory)} recorded intermediates")

    # The closed-form forward process must match iterating one step at a time.
    diffusion = build_diffusion(steps=100, schedule="linear")
    torch.manual_seed(0)
    x0 = torch.rand(1, 1, 8, 8) * 2 - 1
    t = torch.tensor([40])
    closed = diffusion.q_sample(x0, t, torch.zeros_like(x0))
    expected = diffusion.sqrt_alphas_cumprod[40] * x0
    print("q_sample matches the closed form:", bool(torch.allclose(closed, expected, atol=1e-6)))
