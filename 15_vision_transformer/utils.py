"""Metrics and plotting for the ViT/ResNet comparison.

Two things here are specific to this project.

``measure_throughput`` times a forward-backward step, because a comparison at matched
*parameter count* is only half a control. Attention costs O(tokens^2) and convolution
costs O(pixels x channels), so two models of equal size can differ by a factor in what
they cost to run - and reporting accuracy per parameter while quietly spending twice the
compute is one of the easier ways to publish a misleading comparison.

``plot_attention_maps`` overlays the attention rollout on the input. It is the thing a
ViT gives you that a CNN does not, and it is worth looking at even when the ViT is losing
on accuracy.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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


def measure_throughput(
    model: nn.Module, shape: tuple[int, ...], device: torch.device, steps: int = 5
) -> dict:
    """Seconds per training step and per inference image, measured not estimated."""
    model.train()
    images = torch.randn(*shape, device=device)
    targets = torch.randint(0, 2, (shape[0],), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    criterion = nn.CrossEntropyLoss()

    for _ in range(2):  # warm up
        optimizer.zero_grad(set_to_none=True)
        criterion(model(images), targets).backward()
        optimizer.step()

    started = time.time()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        criterion(model(images), targets).backward()
        optimizer.step()
    train_step = (time.time() - started) / steps

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(images)
        started = time.time()
        for _ in range(steps):
            model(images)
        inference = (time.time() - started) / steps

    return {
        "train_seconds_per_step": train_step,
        "train_images_per_second": shape[0] / max(train_step, 1e-9),
        "inference_images_per_second": shape[0] / max(inference, 1e-9),
    }


@torch.no_grad()
def accuracy_and_confusion(model, loader, device, num_classes: int):
    """Top-1 accuracy plus the confusion matrix, in one pass."""
    model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    correct = total = 0
    for images, targets in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        for target, prediction in zip(targets, predictions):
            matrix[int(target), int(prediction)] += 1
        correct += (predictions == targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1), matrix


def per_class_accuracy(matrix: torch.Tensor) -> list[float]:
    totals = matrix.sum(dim=1).clamp_min(1)
    return (matrix.diag().float() / totals.float()).tolist()


def show_image(ax, image: torch.Tensor) -> None:
    array = image.detach().cpu().clamp(0, 1).numpy()
    if array.shape[0] == 1:
        ax.imshow(array[0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(np.transpose(array, (1, 2, 0)))
    ax.axis("off")


def plot_image_grid(
    images: torch.Tensor, path: str | Path, columns: int = 8, title: str | None = None
) -> None:
    n = images.size(0)
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.3 * columns, 1.35 * rows + 0.4),
                             squeeze=False)
    for index in range(rows * columns):
        ax = axes[index // columns][index % columns]
        if index < n:
            show_image(ax, images[index])
        else:
            ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention_maps(
    images: torch.Tensor,
    rollout: torch.Tensor,
    labels: list[str],
    path: str | Path,
    title: str | None = None,
) -> None:
    """Input on top, attention rollout in the middle, the two overlaid at the bottom."""
    n = min(images.size(0), rollout.size(0), 8)
    upsampled = F.interpolate(
        rollout[:n].unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False
    )

    fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 5.0), squeeze=False)
    for column in range(n):
        show_image(axes[0][column], images[column])
        axes[0][column].set_title(labels[column] if column < len(labels) else "", fontsize=7)
        axes[1][column].imshow(upsampled[column, 0].cpu().numpy(), cmap="inferno")
        axes[1][column].axis("off")
        show_image(axes[2][column], images[column])
        axes[2][column].imshow(upsampled[column, 0].cpu().numpy(), cmap="inferno", alpha=0.55)
    for row, name in enumerate(("input", "attention", "overlay")):
        axes[row][0].set_ylabel(name, fontsize=8)
        axes[row][0].axis("on")
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: torch.Tensor, class_names: list[str], path: str | Path, title: str | None = None
) -> None:
    normalised = matrix.float() / matrix.sum(dim=1, keepdim=True).clamp_min(1)
    size = len(class_names)
    fig, ax = plt.subplots(figsize=(0.5 * size + 3, 0.45 * size + 2.5))
    image = ax.imshow(normalised.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(size), class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(size), class_names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    if size <= 12:
        for i in range(size):
            for j in range(size):
                value = normalised[i, j].item()
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if value > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title(title or "confusion matrix (row-normalised)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(history: dict, path: str | Path, keys: tuple[str, ...] = ("loss", "accuracy")):
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


def plot_bars(
    labels: list[str],
    series: dict[str, list[float]],
    path: str | Path,
    ylabel: str = "score",
    title: str | None = None,
) -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 3, 4))
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(bars, fmt="%.3g", fontsize=7, padding=1)
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
