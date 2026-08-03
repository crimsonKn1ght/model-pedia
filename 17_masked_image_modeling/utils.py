"""Seeding, metrics and plotting for the masked-autoencoder project.

Reconstruction quality is reported as PSNR over the *masked* patches only. The
visible patches are copied through unchanged, so including them would inflate
every number and hide the thing being measured - at 75% masking a quarter of the
image is free.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


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


def count_parameters(model: nn.Module) -> int:
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


def masked_psnr(
    prediction_patches: torch.Tensor,
    target_patches: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """PSNR in dB over masked patches only.

    ``prediction_patches`` and ``target_patches`` are ``(N, L, patch_dim)`` in
    ``[0, 1]``; ``mask`` is ``(N, L)`` with 1 where the patch was hidden.
    """
    squared_error = (prediction_patches - target_patches).pow(2).mean(dim=-1)
    mse = (squared_error * mask).sum() / mask.sum().clamp_min(1.0)
    mse = mse.clamp_min(1e-12)
    return float(10.0 * torch.log10(torch.tensor(data_range**2) / mse))


@torch.no_grad()
def extract_cls_features(encoder: nn.Module, loader, device: torch.device):
    """Class-token features from a frozen encoder, ``(features, labels)``.

    Extracting once and fitting on the cached result makes the linear probe cost
    a single pass over the data instead of one per probe epoch.
    """
    encoder.eval()
    features, labels = [], []
    for images, targets in loader:
        tokens, _, _ = encoder(images.to(device))
        features.append(tokens[:, 0].cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def fit_linear_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
    """Fit one linear layer on cached features; returns train/test accuracy.

    Features are standardised first so a single learning rate works for encoders
    whose activations differ in scale - otherwise the probe partly measures
    feature magnitude rather than feature quality.
    """
    device = device or torch.device("cpu")
    generator = torch.Generator().manual_seed(seed)

    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = ((train_features - mean) / std).to(device)
    test_x = ((test_features - mean) / std).to(device)
    train_y, test_y = train_labels.to(device), test_labels.to(device)

    head = nn.Linear(train_x.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        head.train()
        order = torch.randperm(train_x.size(0), generator=generator)
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            criterion(head(train_x[batch]), train_y[batch]).backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    with torch.no_grad():
        train_acc = (head(train_x).argmax(dim=1) == train_y).float().mean().item()
        test_acc = (head(test_x).argmax(dim=1) == test_y).float().mean().item()
    return {"train_accuracy": train_acc, "test_accuracy": test_acc}


@torch.no_grad()
def classification_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, targets in loader:
        logits = model(images.to(device))
        correct += (logits.argmax(dim=1).cpu() == targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1)


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
                ax.plot(range(1, len(values) + 1), values, label=split, marker="o", markersize=3)
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
) -> None:
    """Grid with one labelled row per batch of images."""
    n_rows = len(rows)
    n_cols = min(len(images) for _, images in rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.4 * n_cols, 1.5 * n_rows + 0.5), squeeze=False
    )
    for row_idx, (label, images) in enumerate(rows):
        for col_idx in range(n_cols):
            show_image(axes[row_idx][col_idx], images[col_idx])
            if col_idx == 0:
                axes[row_idx][col_idx].set_ylabel(label, fontsize=8)
                axes[row_idx][col_idx].axis("on")
                axes[row_idx][col_idx].set_xticks([])
                axes[row_idx][col_idx].set_yticks([])

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ratio_sweep(rows: list[dict], train_ratio: float, path: str | Path) -> None:
    """Reconstruction quality against mask ratio, with the trained ratio marked."""
    ratios = [row["mask_ratio"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(ratios, [row["masked_psnr_db"] for row in rows], marker="o")
    axes[0].set_ylabel("masked-patch PSNR (dB)")
    axes[1].plot(ratios, [row["loss"] for row in rows], marker="o", color="tab:orange")
    axes[1].set_ylabel("training objective on masked patches")

    for ax in axes:
        ax.axvline(train_ratio, color="tab:green", ls="--", alpha=0.7, label="pretraining ratio")
        ax.set_xlabel("mask ratio at test time")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("how much can be hidden before reconstruction fails")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bars(
    labels: list[str],
    series: dict[str, list[float]],
    path: str | Path,
    ylabel: str = "accuracy",
    title: str | None = None,
) -> None:
    """Grouped bar chart, used by ``evaluate.py`` and ``compare.py``."""
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)

    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 3, 4))
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)

    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
