"""DDIM: the same trained model, sampled in a fraction of the steps.

A DDPM needs one network evaluation per timestep -- 300, 1000, sometimes more --
which makes sampling slow enough to be the method's main practical drawback.

DDIM (Song et al., 2020) starts from an observation about the DDPM objective:
the training loss only ever depends on the marginals ``q(x_t | x_0)``, never on
the particular Markov chain that connects them.  So the *sampling* chain can be
replaced by a different one with the same marginals -- including a
**non-Markovian** and **deterministic** one, and one defined over an arbitrary
subsequence of timesteps.

The update rule::

    x_{t-1} = sqrt(a_{t-1}) * x0_hat
            + sqrt(1 - a_{t-1} - sigma^2) * eps_hat
            + sigma * z

    sigma_t = eta * sqrt((1 - a_{t-1}) / (1 - a_t)) * sqrt(1 - a_t / a_{t-1})

Two consequences:

* ``eta = 0`` removes the noise term entirely.  Sampling becomes deterministic,
  so a latent code maps to exactly one image and latents can be interpolated
  the way GAN latents are.
* Because the marginals are respected at any spacing, 20-50 steps often produce
  what the DDPM needed hundreds for.

**No retraining is involved.**  This project loads the checkpoint from project
08 and only changes how it is sampled, which is the entire point.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class DDIMSampler(nn.Module):
    """Deterministic (or partially stochastic) sampler over a timestep subsequence."""

    def __init__(self, diffusion: nn.Module):
        super().__init__()
        self.diffusion = diffusion
        self.timesteps = diffusion.timesteps

    def timestep_subsequence(self, num_steps: int, method: str = "uniform") -> List[int]:
        """Pick which of the original timesteps to actually visit.

        The subsequence must span the *whole* range, ending at ``timesteps - 1``.
        Sampling starts from pure noise, so the first timestep the model is told
        about has to be the one whose marginal actually is pure noise. A
        subsequence that stops short (say at 240 of 300) hands the network a
        fully-noised image while claiming a much lower noise level, and the
        resulting mismatch corrupts the entire trajectory -- worsening sharply
        as the step count drops, which is precisely where DDIM is supposed to
        shine.
        """
        num_steps = max(2, min(num_steps, self.timesteps))
        last = self.timesteps - 1
        if method == "uniform":
            steps = [round(i * last / (num_steps - 1)) for i in range(num_steps)]
        elif method == "quadratic":
            # Spends more steps at the low-noise end, where detail is decided.
            steps = [
                int((i / max(num_steps - 1, 1)) ** 2 * (self.timesteps - 1))
                for i in range(num_steps)
            ]
        else:
            raise ValueError(f"unknown method {method!r}")
        return sorted(set(steps))

    @torch.no_grad()
    def sample(
        self,
        n: int,
        shape: Tuple[int, int, int],
        device: torch.device,
        num_steps: int = 50,
        eta: float = 0.0,
        method: str = "uniform",
        noise: Optional[torch.Tensor] = None,
        record: bool = False,
    ):
        """Generate ``n`` images using ``num_steps`` network evaluations.

        ``eta = 0`` is fully deterministic DDIM; ``eta = 1`` reproduces DDPM's
        ancestral sampling on the chosen subsequence.
        """
        diffusion = self.diffusion
        diffusion.model.eval()

        subsequence = self.timestep_subsequence(num_steps, method)
        x = torch.randn(n, *shape, device=device) if noise is None else noise.to(device)
        trajectory = []

        for i in reversed(range(len(subsequence))):
            step = subsequence[i]
            prev_step = subsequence[i - 1] if i > 0 else -1

            t = torch.full((n,), step, device=device, dtype=torch.long)
            eps = diffusion.model(x, t)
            x0 = diffusion.predict_x0(x, t, eps)

            alpha_bar = diffusion.alpha_bars[step]
            alpha_bar_prev = (
                diffusion.alpha_bars[prev_step]
                if prev_step >= 0
                else torch.ones((), device=device)
            )

            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            )
            # The direction term deliberately re-uses the predicted noise: this
            # is what keeps the marginals correct while skipping timesteps.
            direction = torch.sqrt((1 - alpha_bar_prev - sigma**2).clamp(min=0.0)) * eps
            x = alpha_bar_prev.sqrt() * x0 + direction
            if eta > 0 and prev_step >= 0:
                x = x + sigma * torch.randn_like(x)

            if record:
                trajectory.append(x.clone())

        return (x, trajectory) if record else x

    @torch.no_grad()
    def interpolate(
        self,
        shape: Tuple[int, int, int],
        device: torch.device,
        rows: int = 4,
        steps: int = 8,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Spherically interpolate between latent codes.

        Only meaningful because ``eta = 0`` makes sampling deterministic: with
        DDPM's stochastic chain, the same latent gives a different image every
        time.  Slerp rather than lerp keeps the interpolated codes on the
        typical-radius shell of the Gaussian prior.
        """
        z_a = torch.randn(rows, *shape, device=device)
        z_b = torch.randn(rows, *shape, device=device)
        outputs = []
        for i in range(steps):
            alpha = i / max(steps - 1, 1)
            a_flat = z_a.flatten(1)
            b_flat = z_b.flatten(1)
            cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=1).clamp(-1, 1)
            omega = torch.acos(cos).view(-1, 1)
            sin = torch.sin(omega).clamp(min=1e-6)
            z = (
                torch.sin((1 - alpha) * omega) / sin * a_flat
                + torch.sin(alpha * omega) / sin * b_flat
            ).view_as(z_a)
            outputs.append(self.sample(rows, shape, device, num_steps=num_steps, noise=z))
        # Reorder so each row is one interpolation path.
        stacked = torch.stack(outputs, dim=1)
        return stacked.reshape(rows * steps, *shape)
