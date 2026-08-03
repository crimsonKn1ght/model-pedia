"""Evaluate a trained CNN checkpoint on the held-out test split.

    python evaluate.py --checkpoint outputs/cifar10_resnet20/best.pt

Reports top-1 and top-5 accuracy, per-class accuracy, the most-confused class
pairs, and writes the confusion matrix and a grid of predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import dataset_info, get_dataloaders, get_datasets
from model import build_model
from utils import (
    confusion_matrix,
    format_per_class_table,
    get_device,
    per_class_metrics,
    plot_confusion_matrix,
    save_json,
    set_seed,
)


@torch.no_grad()
def collect_predictions(model, loader, device, topk: int = 5):
    """Return true labels, top-1 predictions, top-k hit rate and mean loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    y_true, y_pred, total_loss, topk_hits, seen = [], [], 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item()

        k = min(topk, logits.size(1))
        top = logits.topk(k, dim=1).indices
        topk_hits += (top == targets.view(-1, 1)).any(dim=1).sum().item()

        y_true.append(targets.cpu().numpy())
        y_pred.append(top[:, 0].cpu().numpy())
        seen += images.size(0)

    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        topk_hits / max(seen, 1),
        total_loss / max(seen, 1),
    )


def most_confused(cm: np.ndarray, class_names: list[str], top: int = 5) -> list[dict]:
    """The class pairs the model mixes up most often, as a fraction of the true class."""
    counts = cm.astype(np.float64).copy()
    np.fill_diagonal(counts, 0)
    row_sums = cm.sum(axis=1, keepdims=True)
    rates = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    pairs = []
    for true_idx, pred_idx in zip(*np.unravel_index(np.argsort(rates, axis=None)[::-1], rates.shape)):
        if rates[true_idx, pred_idx] <= 0:
            break
        pairs.append(
            {
                "true": class_names[true_idx],
                "predicted": class_names[pred_idx],
                "count": int(cm[true_idx, pred_idx]),
                "rate": float(rates[true_idx, pred_idx]),
            }
        )
        if len(pairs) >= top:
            break
    return pairs


def denormalize(images: torch.Tensor, dataset: str) -> torch.Tensor:
    info = dataset_info(dataset)
    mean = torch.tensor(info["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(info["std"]).view(1, -1, 1, 1)
    return (images.cpu() * std + mean).clamp(0, 1)


@torch.no_grad()
def plot_predictions(model, loader, device, dataset, class_names, path, n: int = 16) -> None:
    """Grid of test images titled with the prediction; wrong ones are marked in red."""
    model.eval()
    images, targets = next(iter(loader))
    images, targets = images[:n], targets[:n]
    preds = model(images.to(device)).argmax(dim=1).cpu()

    shown = denormalize(images, dataset)
    cols = 4
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
    for ax, image, target, pred in zip(np.atleast_1d(axes).ravel(), shown, targets, preds):
        ax.imshow(image.permute(1, 2, 0).numpy())
        correct = target.item() == pred.item()
        ax.set_title(
            f"{class_names[pred]}" if correct else f"{class_names[pred]}\n(true {class_names[target]})",
            fontsize=8,
            color="green" if correct else "red",
        )
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(shown) :]:
        ax.axis("off")
    fig.suptitle("test predictions (red = wrong)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evaluation(
    checkpoint: str,
    data_root: str = "data",
    batch_size: int = 256,
    num_workers: int = 2,
    device: str = "auto",
    seed: int = 0,
    synthetic: bool = False,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    dev = get_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=True)
    dataset, arch, class_names = ckpt["dataset"], ckpt["arch"], ckpt["class_names"]

    model = build_model(arch, num_classes=len(class_names)).to(dev)
    model.load_state_dict(ckpt["model_state"])

    train_ds, val_ds, test_ds, _ = get_datasets(
        dataset, root=data_root, seed=seed, synthetic=synthetic
    )
    _, _, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=num_workers
    )

    y_true, y_pred, top5, test_loss = collect_predictions(model, test_loader, dev)
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    metrics = per_class_metrics(cm)
    confusions = most_confused(cm, class_names)

    out_dir = Path(checkpoint).parent
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    plot_predictions(model, test_loader, dev, dataset, class_names, out_dir / "predictions.png")

    result = {
        "arch": arch,
        "dataset": dataset,
        "checkpoint": str(checkpoint),
        "test_loss": test_loss,
        "top1_accuracy": metrics["accuracy"],
        "top5_accuracy": top5,
        "macro_f1": metrics["macro_f1"],
        "per_class_accuracy": {
            name: float(metrics["recall"][i]) for i, name in enumerate(class_names)
        },
        "most_confused": confusions,
        "confusion_matrix": cm.tolist(),
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"model          : {arch} on {dataset}")
        print(f"test loss      : {test_loss:.4f}")
        print(f"top-1 accuracy : {metrics['accuracy']:.4f}")
        print(f"top-5 accuracy : {top5:.4f}")
        print(f"macro F1       : {metrics['macro_f1']:.4f}\n")
        print(format_per_class_table(metrics, class_names, max_rows=20))
        print("\nmost confused pairs")
        for pair in confusions:
            print(
                f"  {pair['true']} -> {pair['predicted']}: "
                f"{pair['count']} images ({pair['rate']:.1%} of the class)"
            )
        print(f"\nconfusion matrix -> {out_dir / 'confusion_matrix.png'}")
        print(f"predictions      -> {out_dir / 'predictions.png'}")
        print(f"metrics          -> {out_dir / 'metrics.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CNN classifier checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/cifar10_resnet20/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on random tensors")
    args = parser.parse_args()

    run_evaluation(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        device=args.device,
        seed=args.seed,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
