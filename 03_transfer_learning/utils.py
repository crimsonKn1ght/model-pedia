"""Small helpers shared by the scripts in this project.

Everything here is deliberately dependency-light: numpy plus matplotlib.
The classification metrics are implemented by hand so that the project stays
readable and does not pull in scikit-learn just for a confusion matrix.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the scripts only ever write figures to disk

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve ``auto`` to cuda/mps/cpu, or honour an explicit request."""
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
    """Running mean of a scalar, weighted by batch size."""

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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are the true class, columns the predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)
    return cm


def per_class_metrics(cm: np.ndarray) -> dict:
    """Precision / recall / F1 / support derived from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual.astype(np.int64),
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * actual).sum() / max(actual.sum(), 1)),
    }


def format_per_class_table(
    metrics: dict, class_names: list[str], max_rows: int | None = None
) -> str:
    """Render the per-class metrics as a plain-text table.

    With more classes than ``max_rows`` only the strongest and weakest classes
    by F1 are listed, which is what you actually want to read on CIFAR-100.
    """
    order = list(range(len(class_names)))
    elided_at = None
    if max_rows is not None and len(order) > max_rows:
        half = max_rows // 2
        by_f1 = sorted(order, key=lambda i: metrics["f1"][i], reverse=True)
        order = by_f1[:half] + by_f1[-half:]
        elided_at = half

    width = max([len("overall")] + [len(class_names[i]) for i in order])
    lines = [f"{'class'.ljust(width)}  {'prec':>6} {'recall':>6} {'f1':>6} {'n':>6}"]
    lines.append("-" * len(lines[0]))
    for position, i in enumerate(order):
        if position == elided_at:
            lines.append(f"{'...'.ljust(width)}  {'':>6} {'':>6} {'':>6} {'':>6}")
        lines.append(
            f"{class_names[i].ljust(width)}  "
            f"{metrics['precision'][i]:6.3f} {metrics['recall'][i]:6.3f} "
            f"{metrics['f1'][i]:6.3f} {metrics['support'][i]:6d}"
        )
    lines.append("-" * len(lines[0]))
    lines.append(
        f"{'overall'.ljust(width)}  {'':>6} {'':>6} {metrics['macro_f1']:6.3f} "
        f"{int(metrics['support'].sum()):6d}"
    )
    return "\n".join(lines)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], path: str | Path, normalize: bool = True
) -> None:
    """Save a confusion-matrix heatmap with the cell values written in."""
    data = cm.astype(np.float64)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)

    # With many classes the tick labels and per-cell numbers stop being readable.
    detailed = len(class_names) <= 20
    side = min(1.0 + 0.6 * len(class_names), 12.0)

    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() > 0 else 1)
    fig.colorbar(im, ax=ax, fraction=0.046)

    if detailed:
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("confusion matrix" + (" (row-normalised)" if normalize else ""))

    if detailed:
        threshold = data.max() / 2 if data.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = f"{data[i, j]:.2f}" if normalize else f"{cm[i, j]:d}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if data[i, j] > threshold else "black",
                )

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: dict, path: str | Path) -> None:
    """Save loss and accuracy curves side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
