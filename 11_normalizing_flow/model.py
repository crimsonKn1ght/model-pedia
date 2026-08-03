"""RealNVP: an invertible network, and therefore an exact likelihood.

    python model.py

Every generative model so far has been unable to tell you the likelihood of an image. A
VAE reports a *bound* (project 06). A GAN has no likelihood at all (project 08). A
diffusion model has one in principle and it is expensive to evaluate. A normalizing flow
reports the exact number, and the reason is a constraint on the architecture: **every
layer must be invertible, with a tractable Jacobian determinant.**

The change of variables formula is the whole idea. If ``z = f(x)`` is invertible then

    log p(x) = log p_z(f(x)) + log |det df/dx|

Choose ``p_z`` to be a unit Gaussian, and computing ``log p(x)`` is just running the
network forwards and adding a correction term. Sampling is the same network backwards.

The trick that makes the determinant tractable is the **affine coupling layer**. Split
the input in two. Pass the first half through unchanged, and use it to predict a scale
and a shift for the second half:

    y_a = x_a
    y_b = x_b * exp(s(x_a)) + t(x_a)

Inverting is trivial - subtract and divide, using ``x_a = y_a`` which you already have -
and the Jacobian is triangular, so its determinant is just ``sum(s(x_a))``. The functions
``s`` and ``t`` can be arbitrarily complicated convolutional networks, because they are
never inverted.

Two supporting pieces:

* **masks alternate**, checkerboard then channel-wise, so that every dimension gets
  transformed and every dimension gets to condition. A coupling layer that always passed
  the same half through would leave those dimensions Gaussian.
* **squeeze** trades resolution for channels (space-to-depth), which is what lets the
  channel-wise masks see spatial neighbours.

``preprocess`` is not a detail either - see the note on dequantisation in its docstring.
Skipping it produces bits-per-dimension numbers that look wonderful and mean nothing,
which ``compare.py --study dequantisation`` demonstrates.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MASKS = ("checkerboard", "channel")


class CouplingNet(nn.Module):
    """The ``s`` and ``t`` predictor. Never inverted, so it can be anything."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 3, padding=1),
        )
        # Start as the identity: zero output means scale 1, shift 0, so an untrained
        # flow is exactly the standard Gaussian and training starts from a sane place.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.net(x).chunk(2, dim=1)
        # Bounded log-scale. Unbounded scales are the usual reason a flow diverges.
        return torch.tanh(scale) * 2.0, shift


class AffineCoupling(nn.Module):
    """Transform the masked-out half, conditioned on the half left alone."""

    def __init__(self, channels: int, hidden: int, mask_type: str, parity: int) -> None:
        super().__init__()
        if mask_type not in MASKS:
            raise ValueError(f"unknown mask {mask_type!r}, expected one of {MASKS}")
        self.mask_type = mask_type
        self.parity = parity
        self.net = CouplingNet(channels, hidden)
        self.register_buffer("mask", self._build_mask(channels), persistent=False)

    def _build_mask(self, channels: int) -> torch.Tensor:
        if self.mask_type == "channel":
            mask = torch.zeros(1, channels, 1, 1)
            half = channels // 2
            mask[:, :half] = 1.0
            if self.parity:
                mask = 1.0 - mask
            return mask
        # Checkerboard is built at 2x2 and tiled to whatever the input size is.
        mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).view(1, 1, 2, 2)
        if self.parity:
            mask = 1.0 - mask
        return mask

    def _expanded_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_type == "channel":
            return self.mask
        height, width = x.shape[-2:]
        return self.mask.repeat(1, 1, height // 2 + 1, width // 2 + 1)[..., :height, :width]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self._expanded_mask(x)
        frozen = x * mask
        scale, shift = self.net(frozen)
        scale, shift = scale * (1 - mask), shift * (1 - mask)
        y = frozen + (1 - mask) * (x * scale.exp() + shift)
        return y, scale.flatten(1).sum(dim=1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        mask = self._expanded_mask(y)
        frozen = y * mask
        scale, shift = self.net(frozen)
        scale, shift = scale * (1 - mask), shift * (1 - mask)
        return frozen + (1 - mask) * ((y - shift) * (-scale).exp())


def squeeze(x: torch.Tensor) -> torch.Tensor:
    """``(N, C, H, W)`` -> ``(N, 4C, H/2, W/2)``: space traded for channels."""
    batch, channels, height, width = x.shape
    x = x.view(batch, channels, height // 2, 2, width // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(batch, channels * 4, height // 2, width // 2)


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    x = x.view(batch, channels // 4, 2, 2, height, width)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(batch, channels // 4, height * 2, width * 2)


class RealNVP(nn.Module):
    """Checkerboard couplings, squeeze, channel couplings, squeeze, channel couplings."""

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 32,
        hidden: int = 64,
        couplings_per_stage: int = 3,
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.alpha = alpha
        self.dimension = in_channels * image_size * image_size

        self.stages = nn.ModuleList()
        self.stage_kinds: list[str] = []

        channels = in_channels
        for index in range(couplings_per_stage):
            self.stages.append(AffineCoupling(channels, hidden, "checkerboard", index % 2))
            self.stage_kinds.append("coupling")

        for _ in range(2):
            self.stages.append(nn.Identity())
            self.stage_kinds.append("squeeze")
            channels *= 4
            for index in range(couplings_per_stage):
                self.stages.append(AffineCoupling(channels, hidden, "channel", index % 2))
                self.stage_kinds.append("coupling")

        self.latent_channels = channels

    def preprocess(self, images: torch.Tensor, dequantise: bool = True):
        """Discrete pixels -> unbounded reals, returning the log-determinant.

        Two steps, and both are necessary for the likelihood to mean anything.

        **Dequantisation.** The data is discrete - 256 grey levels - and a continuous
        density can put unbounded mass on a finite set of points, so a flow fitted
        directly to quantised pixels reports arbitrarily good bits-per-dimension while
        modelling nothing. Adding uniform noise within each quantisation bin turns the
        problem back into a continuous one, and the resulting likelihood is a proper
        lower bound on the discrete one.

        **Logit transform.** Pixels live in ``[0, 1]`` and a Gaussian does not, so the
        flow would have to spend capacity squashing its output into a box. Mapping to
        logit space removes that job. The ``alpha`` inset keeps the logarithm finite at
        pure black and pure white.
        """
        if dequantise and self.training:
            images = (images * 255.0 + torch.rand_like(images)) / 256.0
        scaled = self.alpha + (1 - 2 * self.alpha) * images
        y = torch.log(scaled) - torch.log1p(-scaled)
        # d/dx log(a + b x) - log(1 - a - b x) with b = 1 - 2 alpha.
        log_det = (
            math.log(1 - 2 * self.alpha)
            - torch.log(scaled)
            - torch.log1p(-scaled)
        ).flatten(1).sum(dim=1)
        return y, log_det

    def postprocess(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of the logit transform, back to ``[0, 1]``."""
        return ((y.sigmoid() - self.alpha) / (1 - 2 * self.alpha)).clamp(0, 1)

    def forward(self, images: torch.Tensor, dequantise: bool = True):
        """Returns ``(latent, total log-determinant)``."""
        x, log_det = self.preprocess(images, dequantise)
        for module, kind in zip(self.stages, self.stage_kinds):
            if kind == "squeeze":
                x = squeeze(x)
            else:
                x, coupling_log_det = module(x)
                log_det = log_det + coupling_log_det
        return x, log_det

    @torch.no_grad()
    def inverse(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent
        for module, kind in reversed(list(zip(self.stages, self.stage_kinds))):
            if kind == "squeeze":
                x = unsqueeze(x)
            else:
                x = module.inverse(x)
        return self.postprocess(x)

    def log_prob(self, images: torch.Tensor, dequantise: bool = True) -> torch.Tensor:
        """Exact ``log p(x)`` per image, in nats, for the continuous density."""
        latent, log_det = self(images, dequantise)
        prior = -0.5 * (latent.flatten(1).pow(2) + math.log(2 * math.pi)).sum(dim=1)
        return prior + log_det

    def bits_per_dimension(self, images: torch.Tensor, dequantise: bool = True) -> torch.Tensor:
        """The standard reporting unit.

        The density is over ``[0, 1]`` while the data is 256 levels, so converting to
        bits per dimension of the *discrete* data adds ``log2(256) = 8``. This is the
        same convention every flow and autoregressive paper uses, which is what makes
        the number comparable outside this repository - unlike the FID values elsewhere.
        """
        log_prob = self.log_prob(images, dequantise)
        return -log_prob / (self.dimension * math.log(2)) + 8.0

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | None = None, temperature: float = 1.0):
        device = device or next(self.parameters()).device
        grid = self.image_size // 4
        latent = torch.randn(n, self.latent_channels, grid, grid, device=device) * temperature
        return self.inverse(latent)


def build_model(
    in_channels: int = 1,
    image_size: int = 32,
    hidden: int = 64,
    couplings_per_stage: int = 3,
) -> RealNVP:
    return RealNVP(
        in_channels=in_channels,
        image_size=image_size,
        hidden=hidden,
        couplings_per_stage=couplings_per_stage,
    )


if __name__ == "__main__":
    for channels, size in ((1, 32), (3, 32)):
        model = build_model(channels, size)
        model.eval()
        images = torch.rand(4, channels, size, size)
        latent, log_det = model(images, dequantise=False)
        bpd = model.bits_per_dimension(images, dequantise=False)
        print(
            f"C={channels} {size}x{size}: latent {tuple(latent.shape)}  "
            f"bits/dim {bpd.mean().item():7.3f}  "
            f"params {sum(p.numel() for p in model.parameters()):,}"
        )

    model = build_model(1, 32)
    model.eval()
    images = torch.rand(2, 1, 32, 32)

    # The defining property: the network must invert exactly.
    latent, _ = model(images, dequantise=False)
    round_trip = model.inverse(latent)
    error = (round_trip - images).abs().max().item()
    print(f"\ninvertibility: max round-trip error {error:.2e}")

    # squeeze/unsqueeze must also be exact inverses.
    x = torch.randn(2, 1, 8, 8)
    print(f"squeeze round-trip exact: {bool(torch.equal(unsqueeze(squeeze(x)), x))}")

    print(f"samples: {tuple(model.sample(4).shape)}")
    # An untrained flow is the identity, so its samples are the prior pushed through
    # the logit inverse - and its bits/dim should be close to the Gaussian's.
    print(f"untrained bits/dim: {model.bits_per_dimension(images, False).mean().item():.3f}")
