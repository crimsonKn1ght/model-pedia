"""DDIM: the same trained model, a shorter walk back from noise.

    python sampler.py

A DDPM samples by walking down every one of its ``T`` timesteps, one network evaluation
each. That is where all the cost is - 200 forward passes per image against a GAN's one -
and DDIM removes most of it without retraining anything.

Two ideas do the work.

**1. The update can be written in terms of the predicted clean image.** Given ``x_t``
and the network's noise prediction, you can estimate ``x_0`` directly, then jump to
*any* earlier timestep by re-noising that estimate to the right level:

    x_prev  =  sqrt(a_prev) * x0_hat  +  sqrt(1 - a_prev - sigma^2) * eps_hat  +  sigma * z

Nothing requires ``prev`` to be ``t - 1``. Pick a subsequence of 20 timesteps out of 200
and the same trained network works, at a tenth of the cost.

**2. The noise can be removed entirely.** ``sigma`` above is a free parameter. DDPM
corresponds to one particular choice; ``eta = 0`` sets it to zero and makes sampling
**deterministic** - one latent maps to exactly one image, for a given step count. That
is what makes latent interpolation meaningful for a diffusion model, and what
``--eta 0`` gives you here.

The trade is real and this project measures it: fewer steps costs image quality, and the
curve is not linear. Most of the quality survives a 10x reduction; below that it falls
away quickly.
"""

from __future__ import annotations

import torch

import _paths  # noqa: F401  - puts project 12 on the import path
from model import GaussianDiffusion, build_diffusion, build_model


class DDIMSampler:
    """Deterministic (or partly stochastic) sampling over a subsequence of timesteps."""

    def __init__(self, diffusion: GaussianDiffusion, eta: float = 0.0) -> None:
        self.diffusion = diffusion
        self.eta = eta

    def timesteps(self, steps: int) -> list[int]:
        """``steps`` timesteps spread evenly over the trained trajectory, descending."""
        total = self.diffusion.steps
        steps = max(1, min(steps, total))
        chosen = torch.linspace(total - 1, 0, steps).round().long().tolist()
        # Keep them strictly decreasing after rounding.
        unique = []
        for value in chosen:
            if not unique or value < unique[-1]:
                unique.append(int(value))
        return unique

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: tuple[int, ...],
        device: torch.device,
        steps: int = 20,
        record: int = 0,
        latent: torch.Tensor | None = None,
    ):
        """Sample with ``steps`` network evaluations per image."""
        model.eval()
        diffusion = self.diffusion
        schedule = self.timesteps(steps)

        x_t = torch.randn(shape, device=device) if latent is None else latent.to(device)
        keep_every = max(len(schedule) // record, 1) if record else 0
        trajectory = []

        for index, step in enumerate(schedule):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            noise_prediction = diffusion.predicted_noise(model, x_t, t)

            alpha = diffusion.alphas_cumprod[step]
            previous_step = schedule[index + 1] if index + 1 < len(schedule) else None
            alpha_prev = (
                diffusion.alphas_cumprod[previous_step]
                if previous_step is not None
                else torch.ones_like(alpha)
            )

            x0 = ((x_t - (1 - alpha).sqrt() * noise_prediction) / alpha.sqrt().clamp_min(1e-8))
            x0 = x0.clamp(-1, 1)

            # eta = 0 removes the noise term entirely and the sampler becomes a
            # deterministic map from the initial latent to an image.
            sigma = (
                self.eta
                * ((1 - alpha_prev) / (1 - alpha).clamp_min(1e-8)).sqrt()
                * (1 - alpha / alpha_prev.clamp_min(1e-8)).clamp_min(0).sqrt()
            )
            direction = (1 - alpha_prev - sigma**2).clamp_min(0).sqrt() * noise_prediction
            x_t = alpha_prev.sqrt() * x0 + direction
            if self.eta > 0 and previous_step is not None:
                x_t = x_t + sigma * torch.randn_like(x_t)

            if keep_every and (index % keep_every == 0 or previous_step is None):
                trajectory.append(diffusion.unscale(x_t).cpu())

        images = diffusion.unscale(x_t)
        return (images, trajectory) if record else images


def load_checkpoint(checkpoint: str, device: torch.device):
    """Return ``(model, diffusion, config)`` from a project 12 checkpoint."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model = build_model(ckpt["in_channels"], ckpt["base_channels"], ckpt["attention"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    diffusion = build_diffusion(ckpt["steps"], ckpt["schedule"], ckpt["prediction"]).to(device)
    return model, diffusion, ckpt


if __name__ == "__main__":
    device = torch.device("cpu")
    model = build_model(1, base_channels=16, attention=False)
    diffusion = build_diffusion(steps=100, schedule="cosine")

    for steps in (5, 10, 25, 100):
        sampler = DDIMSampler(diffusion, eta=0.0)
        chosen = sampler.timesteps(steps)
        images = sampler.sample(model, (2, 1, 32, 32), device, steps=steps)
        print(f"steps {steps:4d}: {len(chosen)} network evaluations, "
              f"t from {chosen[0]} down to {chosen[-1]}, images {tuple(images.shape)}")

    # eta = 0 must be deterministic: the same latent gives the same image twice.
    sampler = DDIMSampler(diffusion, eta=0.0)
    latent = torch.randn(2, 1, 32, 32)
    first = sampler.sample(model, (2, 1, 32, 32), device, steps=10, latent=latent)
    second = sampler.sample(model, (2, 1, 32, 32), device, steps=10, latent=latent)
    print(f"\neta=0 is deterministic: {bool(torch.allclose(first, second, atol=1e-6))}")

    stochastic = DDIMSampler(diffusion, eta=1.0)
    a = stochastic.sample(model, (2, 1, 32, 32), device, steps=10, latent=latent)
    b = stochastic.sample(model, (2, 1, 32, 32), device, steps=10, latent=latent)
    print(f"eta=1 is not:           {not bool(torch.allclose(a, b, atol=1e-6))}")
