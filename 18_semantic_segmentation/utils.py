"""Metrics and plotting for segmentation.

Everything is computed from one confusion matrix, accumulated over the split.
That is worth doing rather than averaging per-image scores: a class that is absent
from an image has an undefined IoU, and averaging over images silently decides
what to do about it. Accumulating counts first and dividing once at the end gives
the standard dataset-level IoU with no such ambiguity.

Three numbers, and they disagree in useful ways:

* **pixel accuracy** - the optimistic one. On Oxford Pets, predicting "background"
  everywhere already scores well over half.
* **IoU / mean IoU** - intersection over union per class, then the unweighted mean.
  A thin class counts as much as a huge one, which is why mIoU is the number
  segmentation papers report.
* **Dice** - the same information with the intersection weighted double. Standard
  in medical imaging, and less brutal than IoU on small structures.
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

# Colour-blind-safe enough for four or five classes, and stable across figures.
PALETTE = torch.tensor(
    [
        [0.10, 0.10, 0.12],  # background
        [0.90, 0.35, 0.20],
        [0.20, 0.55, 0.90],
        [0.95, 0.80, 0.25],
        [0.35, 0.75, 0.45],
        [0.75, 0.40, 0.85],
    ]
)


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


class ConfusionMatrix:
    """Accumulates ``(target, prediction)`` pixel counts. Rows are ground truth."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.flatten().cpu()
        target = target.flatten().cpu()
        valid = (target >= 0) & (target < self.num_classes)
        index = self.num_classes * target[valid] + prediction[valid]
        self.matrix += torch.bincount(index, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    @property
    def _counts(self):
        true_positive = self.matrix.diag().float()
        predicted = self.matrix.sum(dim=0).float()
        actual = self.matrix.sum(dim=1).float()
        return true_positive, predicted - true_positive, actual - true_positive

    def iou(self) -> torch.Tensor:
        tp, fp, fn = self._counts
        return tp / (tp + fp + fn).clamp_min(1e-9)

    def dice(self) -> torch.Tensor:
        tp, fp, fn = self._counts
        return 2 * tp / (2 * tp + fp + fn).clamp_min(1e-9)

    def mean_iou(self) -> float:
        """Mean over classes that actually appear in the ground truth."""
        present = self.matrix.sum(dim=1) > 0
        if not present.any():
            return 0.0
        return float(self.iou()[present].mean())

    def mean_dice(self) -> float:
        present = self.matrix.sum(dim=1) > 0
        if not present.any():
            return 0.0
        return float(self.dice()[present].mean())

    def pixel_accuracy(self) -> float:
        total = self.matrix.sum()
        return float(self.matrix.diag().sum() / total.clamp_min(1))

    def class_pixel_share(self) -> torch.Tensor:
        actual = self.matrix.sum(dim=1).float()
        return actual / actual.sum().clamp_min(1)

    def summary(self, class_names: list[str] | None = None) -> dict:
        names = class_names or [str(i) for i in range(self.num_classes)]
        return {
            "mean_iou": self.mean_iou(),
            "mean_dice": self.mean_dice(),
            "pixel_accuracy": self.pixel_accuracy(),
            "per_class": [
                {
                    "class": name,
                    "iou": float(iou),
                    "dice": float(dice),
                    "pixel_share": float(share),
                }
                for name, iou, dice, share in zip(
                    names, self.iou(), self.dice(), self.class_pixel_share()
                )
            ],
        }


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice over classes, as a loss.

    Cross entropy is a per-pixel loss, so a class covering 2% of the pixels
    contributes 2% of the gradient and the model learns to ignore it. Dice is
    computed per class over the whole image and normalised by that class's own
    area, which makes a thin class count as much as a large one. Using
    ``ce + dice`` is the usual compromise: cross entropy for stable gradients
    early, Dice to stop the rare classes being written off.
    """
    num_classes = logits.size(1)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    cardinality = probabilities.sum(dims) + one_hot.sum(dims)
    dice = (2 * intersection + eps) / (cardinality + eps)
    return 1 - dice.mean()


def build_criterion(name: str, dice_weight: float = 1.0):
    """``ce``, ``dice`` or ``ce+dice``."""
    cross_entropy = nn.CrossEntropyLoss()
    if name == "ce":
        return cross_entropy
    if name == "dice":
        return lambda logits, targets: dice_loss(logits, targets)
    if name == "ce+dice":
        return lambda logits, targets: cross_entropy(logits, targets) + dice_weight * dice_loss(
            logits, targets
        )
    raise ValueError(f"unknown loss {name!r}, expected ce, dice or ce+dice")


def colorize(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``(H, W)`` label map -> ``(3, H, W)`` image in ``[0, 1]``."""
    palette = PALETTE[: max(num_classes, 1)]
    if num_classes > PALETTE.size(0):  # wrap round for datasets with many classes
        repeats = (num_classes + PALETTE.size(0) - 1) // PALETTE.size(0)
        palette = PALETTE.repeat(repeats, 1)[:num_classes]
    return palette.to(mask.device)[mask.clamp(0, num_classes - 1)].permute(2, 0, 1)


def overlay(image: torch.Tensor, mask: torch.Tensor, num_classes: int, alpha: float = 0.5):
    """Blend a label map over the image, leaving background untouched."""
    coloured = colorize(mask, num_classes).to(image.device)
    weight = torch.where(mask > 0, alpha, 0.0).unsqueeze(0).to(image.device)
    return (image * (1 - weight) + coloured * weight).clamp(0, 1)


def _draw(ax, image: torch.Tensor) -> None:
    array = image.detach().cpu().clamp(0, 1).numpy()
    if array.shape[0] == 1:
        ax.imshow(array[0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(np.transpose(array, (1, 2, 0)))
    ax.axis("off")


def plot_segmentation_rows(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor | None,
    num_classes: int,
    path: str | Path,
    title: str | None = None,
) -> None:
    """Image / ground truth / prediction / prediction overlay, one column per sample."""
    rows = [("image", [img for img in images]),
            ("truth", [colorize(mask, num_classes) for mask in targets])]
    if predictions is not None:
        rows.append(("predicted", [colorize(mask, num_classes) for mask in predictions]))
        rows.append(
            (
                "overlay",
                [overlay(img, mask, num_classes) for img, mask in zip(images, predictions)],
            )
        )
    else:
        rows.append(
            ("overlay", [overlay(img, mask, num_classes) for img, mask in zip(images, targets)])
        )

    n_cols = min(len(entry) for _, entry in rows)
    fig, axes = plt.subplots(
        len(rows), n_cols, figsize=(1.5 * n_cols, 1.6 * len(rows) + 0.5), squeeze=False
    )
    for row_idx, (label, entries) in enumerate(rows):
        for col_idx in range(n_cols):
            _draw(axes[row_idx][col_idx], entries[col_idx])
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


def plot_confusion_matrix(
    matrix: torch.Tensor, class_names: list[str], path: str | Path, title: str | None = None
) -> None:
    """Row-normalised confusion matrix - where the pixels of each class end up."""
    normalised = matrix.float() / matrix.sum(dim=1, keepdim=True).clamp_min(1)
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 3, 1.1 * len(class_names) + 2.5))
    image = ax.imshow(normalised.numpy(), cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = normalised[i, j].item()
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                color="white" if value > 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title(title or "confusion matrix (row-normalised)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(history: dict, path: str | Path, keys: tuple[str, ...] = ("loss", "miou")) -> None:
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
