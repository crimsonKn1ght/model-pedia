"""Helpers for self-supervised learning: seeding, feature extraction, the two
label-free probes, and plotting.

The interesting part of this file is that it contains the *entire* evaluation
protocol for self-supervised learning. Pretraining has no accuracy to report -
its loss is measured against its own augmentations, and a lower loss does not
have to mean better features. So the representation is judged from the outside,
by asking how much a *frozen* encoder already knows:

* ``knn_accuracy`` - classify a test image by the labels of its nearest
  neighbours in feature space. No training at all, so nothing can leak.
* ``linear_probe`` - fit one linear layer on frozen features. If a linear
  boundary separates the classes, the encoder has done the hard part.

Both are cheap because the features are extracted once and cached.
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


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen encoder over a loader, returning ``(features, labels)``.

    Features come from the encoder itself, never from the projection head. The
    head exists only to shape the pretraining loss and is thrown away - the
    layer *below* it is consistently the better representation, which is one of
    the more surprising practical findings in this literature.
    """
    encoder.eval()
    features, labels = [], []
    for index, (images, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        features.append(encoder(images.to(device)).flatten(1).cpu())
        labels.append(targets)
    if not features:
        raise ValueError("the loader produced no batches")
    return torch.cat(features), torch.cat(labels)


def knn_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    k: int = 20,
    temperature: float = 0.07,
) -> float:
    """Weighted cosine k-NN classification on frozen features.

    Nothing is fitted: the training features are the classifier. Votes are
    weighted by ``exp(similarity / temperature)`` so a close neighbour counts
    for more than a distant one, following the protocol used to monitor
    contrastive pretraining runs.
    """
    train_norm = F.normalize(train_features, dim=1)
    test_norm = F.normalize(test_features, dim=1)
    k = min(k, train_norm.size(0))

    correct = 0
    for start in range(0, test_norm.size(0), 256):
        chunk = test_norm[start : start + 256]
        similarity = chunk @ train_norm.t()
        sim_k, idx_k = similarity.topk(k, dim=1)
        neighbour_labels = train_labels[idx_k]

        weights = (sim_k / temperature).exp()
        scores = torch.zeros(chunk.size(0), num_classes)
        scores.scatter_add_(1, neighbour_labels, weights)
        correct += (scores.argmax(dim=1) == test_labels[start : start + 256]).sum().item()

    return correct / max(test_labels.numel(), 1)


def linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
    """Fit a single linear layer on cached frozen features.

    Features are standardised first, which is what makes a fixed learning rate
    work across encoders of very different scale - without it the probe result
    partly measures feature magnitude rather than feature quality.
    """
    device = device or torch.device("cpu")
    generator = torch.Generator().manual_seed(seed)

    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = ((train_features - mean) / std).to(device)
    test_x = ((test_features - mean) / std).to(device)
    train_y = train_labels.to(device)
    test_y = test_labels.to(device)

    head = nn.Linear(train_x.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        head.train()
        order = torch.randperm(train_x.size(0), generator=generator)
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    with torch.no_grad():
        train_acc = (head(train_x).argmax(dim=1) == train_y).float().mean().item()
        logits = head(test_x)
        test_acc = (logits.argmax(dim=1) == test_y).float().mean().item()
        test_loss = criterion(logits, test_y).item()
    return {"train_accuracy": train_acc, "test_accuracy": test_acc, "test_loss": test_loss}


def subsample_per_class(
    labels: torch.Tensor, fraction: float, num_classes: int, seed: int = 0
) -> torch.Tensor:
    """Indices of a class-balanced subset, for the label-efficiency study."""
    generator = torch.Generator().manual_seed(seed)
    keep = []
    for cls in range(num_classes):
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        if cls_idx.numel() == 0:
            continue
        n = max(1, int(round(cls_idx.numel() * fraction)))
        perm = torch.randperm(cls_idx.numel(), generator=generator)[:n]
        keep.append(cls_idx[perm])
    return torch.cat(keep) if keep else torch.arange(0)


def plot_curves(history: dict, path: str | Path, keys: tuple[str, ...] = ("loss", "knn")) -> None:
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


def pca_2d(features: torch.Tensor) -> torch.Tensor:
    """Project features onto their first two principal components."""
    centred = features - features.mean(dim=0, keepdim=True)
    # torch.pca_lowrank is enough here and keeps the dependency list short.
    _, _, components = torch.pca_lowrank(centred, q=min(8, *centred.shape))
    return centred @ components[:, :2]


def plot_feature_scatter(
    panels: list[tuple[str, torch.Tensor, torch.Tensor]],
    path: str | Path,
    class_names: list[str] | None = None,
    max_points: int = 2000,
    title: str | None = None,
) -> None:
    """PCA scatter of frozen features, one panel per encoder.

    Two dimensions cannot show what a 128-dimensional space is doing, so treat
    this as a sanity check rather than evidence: pretrained features come out
    visibly clustered by class, random ones come out as a single blob.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6), squeeze=False)
    for ax, (label, features, labels) in zip(axes[0], panels):
        n = min(max_points, features.size(0))
        points = pca_2d(features[:n]).numpy()
        targets = labels[:n].numpy()
        for cls in np.unique(targets):
            mask = targets == cls
            name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
            ax.scatter(points[mask, 0], points[mask, 1], s=4, alpha=0.5, label=name)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][-1].legend(fontsize=6, markerscale=2, loc="best")

    if title:
        fig.suptitle(title)
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
    """Grouped bar chart, used by the comparison scripts."""
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
    """Save a grid where each row is a labelled batch of images."""
    n_rows = len(rows)
    n_cols = min(len(images) for _, images in rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.4 * n_cols, 1.5 * n_rows + 0.4), squeeze=False
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
