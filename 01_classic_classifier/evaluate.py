"""Evaluate a trained checkpoint on the held-out test split.

    python evaluate.py --checkpoint outputs/mnist_lenet/best.pt

Prints accuracy, macro-F1 and a per-class table, and writes the confusion
matrix plus a grid of the model's mistakes next to the checkpoint.
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
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(true_labels, predicted_labels, mean_loss)`` over a loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    y_true, y_pred, total_loss, seen = [], [], 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item()
        seen += images.size(0)
        y_true.append(targets.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), total_loss / max(seen, 1)


def denormalize(images: torch.Tensor, dataset: str) -> torch.Tensor:
    info = dataset_info(dataset)
    mean = torch.tensor(info["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(info["std"]).view(1, -1, 1, 1)
    return (images.cpu() * std + mean).clamp(0, 1)


@torch.no_grad()
def plot_mistakes(model, loader, device, dataset, class_names, path, max_items: int = 16) -> int:
    """Save a grid of misclassified test images: true label vs prediction."""
    model.eval()
    images_out, titles = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu()
        wrong = (preds != targets).nonzero(as_tuple=True)[0]
        for idx in wrong.tolist():
            images_out.append(images[idx])
            titles.append(f"true {class_names[targets[idx]]}\npred {class_names[preds[idx]]}")
            if len(images_out) >= max_items:
                break
        if len(images_out) >= max_items:
            break

    if not images_out:
        print("no misclassified test images found - nothing to plot")
        return 0

    shown = denormalize(torch.stack(images_out), dataset)
    cols = 4
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.6 * rows))
    for ax, image, title in zip(np.atleast_1d(axes).ravel(), shown, titles):
        ax.imshow(image.squeeze(0).numpy(), cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(shown) :]:
        ax.axis("off")
    fig.suptitle("test images the model gets wrong")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return len(shown)


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
    dataset, model_name = ckpt["dataset"], ckpt["model_name"]
    class_names = ckpt["class_names"]

    model = build_model(model_name, num_classes=len(class_names)).to(dev)
    model.load_state_dict(ckpt["model_state"])

    train_ds, val_ds, test_ds, _ = get_datasets(
        dataset, root=data_root, seed=seed, synthetic=synthetic
    )
    _, _, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=num_workers
    )

    y_true, y_pred, test_loss = collect_predictions(model, test_loader, dev)
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    metrics = per_class_metrics(cm)

    out_dir = Path(checkpoint).parent
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    n_wrong = plot_mistakes(model, test_loader, dev, dataset, class_names, out_dir / "mistakes.png")

    result = {
        "model": model_name,
        "dataset": dataset,
        "checkpoint": str(checkpoint),
        "test_loss": test_loss,
        "test_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": {
            name: {
                "precision": float(metrics["precision"][i]),
                "recall": float(metrics["recall"][i]),
                "f1": float(metrics["f1"][i]),
                "support": int(metrics["support"][i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"model         : {model_name} on {dataset}")
        print(f"test loss     : {test_loss:.4f}")
        print(f"test accuracy : {metrics['accuracy']:.4f}")
        print(f"macro F1      : {metrics['macro_f1']:.4f}\n")
        print(format_per_class_table(metrics, class_names))
        print(f"\nconfusion matrix -> {out_dir / 'confusion_matrix.png'}")
        if n_wrong:
            print(f"mistakes         -> {out_dir / 'mistakes.png'}")
        print(f"metrics          -> {out_dir / 'metrics.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a classifier checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/mnist_lenet/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="evaluate on random tensors, to check the pipeline without downloading",
    )
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
