"""Figures: sample grids, training curves, latent scatter plots, sweeps.

Everything writes a PNG to disk; nothing opens a window, so the scripts run
head-less on a server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.utils import make_grid, save_image  # noqa: E402

from common.data import denormalize  # noqa: E402
from common.utils import ensure_dir  # noqa: E402


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    normalize: str = "tanh",
    padding: int = 2,
) -> Path:
    """Write a grid of images, converting from the project's value range to [0, 1]."""
    path = Path(path)
    ensure_dir(path.parent)
    grid = denormalize(images.detach().cpu().float(), normalize)
    save_image(grid, path, nrow=nrow, padding=padding)
    return path


def save_comparison_grid(
    rows: Dict[str, torch.Tensor],
    path: str | Path,
    normalize: str = "tanh",
    max_columns: int = 8,
) -> Path:
    """Stack several labelled rows of images (e.g. input / output / target)."""
    path = Path(path)
    ensure_dir(path.parent)
    n = min(max_columns, min(v.shape[0] for v in rows.values()))
    fig, axes = plt.subplots(len(rows), n, figsize=(1.4 * n, 1.5 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (label, batch) in enumerate(rows.items()):
        imgs = denormalize(batch[:n].detach().cpu().float(), normalize)
        for c in range(n):
            ax = axes[r, c]
            img = imgs[c]
            ax.imshow(img.permute(1, 2, 0).squeeze(), cmap="gray" if img.shape[0] == 1 else None)
            ax.axis("off")
        axes[r, 0].set_ylabel(label)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_curves(
    history: Dict[str, Sequence[float]],
    path: str | Path,
    title: str = "training curves",
    xlabel: str = "epoch",
) -> Path:
    """One panel per tracked scalar."""
    path = Path(path)
    ensure_dir(path.parent)
    keys = [k for k, v in history.items() if len(v) > 0]
    if not keys:
        return path
    cols = min(3, len(keys))
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i // cols][i % cols]
        values = history[key]
        ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3)
        ax.set_title(key)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    for j in range(len(keys), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_latent_scatter(
    codes: torch.Tensor,
    labels: Optional[torch.Tensor],
    path: str | Path,
    title: str = "latent space",
    class_names: Sequence[str] = (),
) -> Path:
    """2-D scatter of latent codes; >2 dimensions are reduced with PCA."""
    path = Path(path)
    ensure_dir(path.parent)
    z = codes.detach().cpu().float().numpy()
    if z.shape[1] > 2:
        centred = z - z.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        z = centred @ vt[:2].T
        title = f"{title} (PCA to 2-D)"
    fig, ax = plt.subplots(figsize=(6, 5.4))
    if labels is None:
        ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.5)
    else:
        y = labels.detach().cpu().numpy()
        scatter = ax.scatter(z[:, 0], z[:, 1], c=y, s=4, alpha=0.6, cmap="tab10")
        handles, _ = scatter.legend_elements()
        names = list(class_names) if class_names else [str(i) for i in sorted(set(y.tolist()))]
        ax.legend(handles, names[: len(handles)], fontsize=7, loc="best", markerscale=1.5)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_line(
    x: Sequence[float],
    y: Sequence[float],
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series_label: Optional[str] = None,
    extra: Optional[Dict[str, Sequence[float]]] = None,
) -> Path:
    """Single line chart, optionally with extra named series sharing the x axis."""
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o", label=series_label)
    for name, values in (extra or {}).items():
        ax.plot(x, values, marker="s", label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if series_label or extra:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_bars(
    labels: Sequence[str],
    values: Sequence[float],
    path: str | Path,
    title: str,
    ylabel: str,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), 4))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_animation_strip(
    frames: Sequence[torch.Tensor],
    path: str | Path,
    normalize: str = "tanh",
    max_columns: int = 10,
) -> Path:
    """Lay a sequence of batches out as a strip: one column per time step.

    Used for the diffusion denoising trajectory, where each frame is the model's
    view of the image at one noise level.
    """
    path = Path(path)
    ensure_dir(path.parent)
    step = max(1, len(frames) // max_columns)
    picked = list(frames)[::step][:max_columns]
    stacked = torch.cat([f[:1] for f in picked], dim=0)
    grid = make_grid(denormalize(stacked.detach().cpu().float(), normalize), nrow=len(picked))
    save_image(grid, path)
    return path
