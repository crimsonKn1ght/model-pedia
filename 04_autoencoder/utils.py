"""Helpers for the reconstruction projects: seeding, PSNR, SSIM, plotting.

PSNR and SSIM are implemented here rather than pulled from scikit-image, both
to keep the dependency list short and because the two metrics are worth
reading once: PSNR is log-scaled mean squared error, SSIM compares local means,
variances and covariance, which is why it notices blur that PSNR forgives.

Every function expects images in ``[0, 1]`` with shape ``(N, C, H, W)``.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB, one value per image.

    A perfect reconstruction is infinite, so the MSE is floored to keep the
    logarithm finite on the (rare) exact match.
    """
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    mse = mse.clamp_min(1e-12)
    return 10.0 * torch.log10(data_range**2 / mse)


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Structural similarity, one value per image (1.0 is identical).

    Gaussian-weighted local statistics, following Wang et al. 2004 with the
    usual K1=0.01, K2=0.03.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

    channels = pred.size(1)
    window_size = min(window_size, pred.size(-1), pred.size(-2))
    if window_size % 2 == 0:  # the window has to have a centre pixel
        window_size -= 1
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    pad = window_size // 2

    def filt(x: torch.Tensor) -> torch.Tensor:
        # Reflect-pad, like the reference implementation; the border is dropped
        # from the average below because its statistics are unreliable anyway.
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), window, groups=channels)

    mu_pred, mu_target = filt(pred), filt(target)
    mu_pred_sq, mu_target_sq = mu_pred**2, mu_target**2
    mu_cross = mu_pred * mu_target

    sigma_pred = filt(pred * pred) - mu_pred_sq
    sigma_target = filt(target * target) - mu_target_sq
    sigma_cross = filt(pred * target) - mu_cross

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2)
    ssim_map = numerator / denominator
    if pad > 0:
        ssim_map = ssim_map[..., pad:-pad, pad:-pad]
    return ssim_map.flatten(1).mean(dim=1)


def plot_curves(history: dict, path: str | Path, keys: tuple[str, ...] = ("loss", "psnr")) -> None:
    """Plot every ``train_<key>`` / ``val_<key>`` pair present in ``history``."""
    present = [key for key in keys if f"train_{key}" in history or f"val_{key}" in history]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), squeeze=False)
    for ax, key in zip(axes[0], present):
        for split in ("train", "val"):
            values = history.get(f"{split}_{key}")
            if values:
                ax.plot(range(1, len(values) + 1), values, label=split)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def show_image(ax, image: torch.Tensor) -> None:
    """Draw a ``(C, H, W)`` tensor in ``[0, 1]``, grayscale or RGB."""
    array = image.detach().cpu().clamp(0, 1).numpy()
    if array.shape[0] == 1:
        ax.imshow(array[0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(np.transpose(array, (1, 2, 0)))
    ax.axis("off")


def plot_image_rows(
    rows: list[tuple[str, torch.Tensor]],
    path: str | Path,
    title: str | None = None,
    column_labels: list[str] | None = None,
) -> None:
    """Save a grid where each row is a labelled batch of images.

    Used for original-vs-reconstruction strips and for noisy/denoised triptychs.
    """
    n_rows = len(rows)
    n_cols = min(len(images) for _, images in rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.4 * n_cols, 1.5 * n_rows + 0.4), squeeze=False
    )
    for row_idx, (label, images) in enumerate(rows):
        for col_idx in range(n_cols):
            show_image(axes[row_idx][col_idx], images[col_idx])
            if col_idx == 0:
                axes[row_idx][col_idx].set_ylabel(label)
                axes[row_idx][col_idx].axis("on")
                axes[row_idx][col_idx].set_xticks([])
                axes[row_idx][col_idx].set_yticks([])
        if column_labels and row_idx == 0:
            for col_idx in range(min(n_cols, len(column_labels))):
                axes[row_idx][col_idx].set_title(column_labels[col_idx], fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bits_per_image(shape: tuple[int, ...]) -> int:
    """Number of scalars in an image, for the compression-ratio report."""
    return int(math.prod(shape))
