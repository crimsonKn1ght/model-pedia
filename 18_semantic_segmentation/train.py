"""Train a segmentation network.

    python train.py --dataset shapes --epochs 8
    python train.py --dataset oxford-pets --epochs 10 --image-size 96

Checkpoint selection is on **validation mean IoU**, not on loss and not on pixel
accuracy. That choice matters more here than in a classification project: both
datasets are dominated by background, so pixel accuracy rewards a model for
getting the easy majority right, and a network that quietly gives up on the thin
classes can improve its accuracy while getting worse at the job.

Every epoch prints mIoU next to the ``predict background everywhere`` baseline,
computed from the validation masks themselves. Until the model beats that number
it has not learned to segment anything - it has learned which class is biggest.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import DATASETS, get_splits, make_loader
from model import MODELS, build_model
from utils import (
    AverageMeter,
    ConfusionMatrix,
    build_criterion,
    count_parameters,
    get_device,
    plot_curves,
    plot_segmentation_rows,
    save_json,
    set_seed,
)

LOSSES = ("ce", "dice", "ce+dice")


@torch.no_grad()
def evaluate_split(model, loader, criterion, device, num_classes: int) -> dict:
    """Loss and confusion-matrix metrics over a whole split."""
    model.eval()
    loss_meter = AverageMeter()
    confusion = ConfusionMatrix(num_classes)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss_meter.update(criterion(logits, masks).item(), images.size(0))
        confusion.update(logits.argmax(dim=1), masks)

    return {
        "loss": loss_meter.avg,
        "miou": confusion.mean_iou(),
        "dice": confusion.mean_dice(),
        "pixel_accuracy": confusion.pixel_accuracy(),
        "confusion": confusion,
    }


def majority_class_baseline(confusion: ConfusionMatrix) -> dict:
    """Metrics for always predicting the most common class in the split.

    Built from the ground-truth column sums, so it costs nothing and is exactly
    the number a real constant predictor would score.
    """
    actual = confusion.matrix.sum(dim=1).float()
    total = actual.sum().clamp_min(1)
    majority = int(actual.argmax())
    # Only the majority class has any overlap; its IoU is its own share, all
    # others are zero, and the mean is taken over the classes that appear.
    present = (actual > 0).sum().clamp_min(1)
    return {
        "class": majority,
        "pixel_accuracy": float(actual[majority] / total),
        "miou": float((actual[majority] / total) / present),
    }


@torch.no_grad()
def save_predictions(model, loader, device, num_classes: int, path, n: int = 8) -> None:
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images[:n].to(device), masks[:n].to(device)
    predictions = model(images).argmax(dim=1)
    plot_segmentation_rows(
        images.cpu(),
        masks.cpu(),
        predictions.cpu(),
        num_classes,
        path,
        title="validation predictions",
    )


def run_training(
    model_name: str = "unet",
    dataset: str = "shapes",
    image_size: int | None = None,
    base_channels: int = 32,
    loss: str = "ce",
    dice_weight: float = 1.0,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    val_split: float = 0.1,
    augment: bool = True,
    train_subset: int | None = 4000,
    num_workers: int = 2,
    seed: int = 0,
    device: str = "auto",
    data_root: str = "data",
    out_dir: str | None = None,
    synthetic: bool = False,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    dev = get_device(device)
    out_path = Path(out_dir or f"outputs/{dataset}_{model_name}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_base, val_base, _, info = get_splits(
        dataset,
        root=data_root,
        val_split=val_split,
        train_subset=train_subset,
        seed=seed,
        synthetic=synthetic,
    )
    size = image_size or info["size"]
    num_classes = info["classes"]

    train_loader = make_loader(
        train_base, info, size, batch_size=batch_size, shuffle=True,
        augment=augment, num_workers=num_workers,
    )
    val_loader = make_loader(
        val_base, info, size, batch_size=batch_size, num_workers=num_workers
    )

    model = build_model(model_name, info["channels"], num_classes, base_channels).to(dev)
    criterion = build_criterion(loss, dice_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    config = {
        "model": model_name,
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": num_classes,
        "class_names": info["class_names"],
        "image_size": size,
        "base_channels": base_channels,
        "loss": loss,
    }

    if verbose:
        print(f"device      : {dev}")
        print(f"model       : {model_name} ({count_parameters(model):,} parameters)")
        print(f"data        : {len(train_base)} train / {len(val_base)} val "
              f"at {info['channels']}x{size}x{size}")
        print(f"classes     : {num_classes}  {info['class_names']}")
        print(f"loss        : {loss}\n")

    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_dice": []}
    best_miou, best_epoch = -float("inf"), 0
    ckpt_path = out_path / "best.pt"
    baseline = None
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for images, masks in batches:
            images, masks = images.to(dev), masks.to(dev)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = criterion(model(images), masks)
            batch_loss.backward()
            optimizer.step()
            loss_meter.update(batch_loss.item(), images.size(0))
            batches.set_postfix(loss=f"{loss_meter.avg:.4f}")
        scheduler.step()

        val = evaluate_split(model, val_loader, criterion, dev, num_classes)
        baseline = baseline or majority_class_baseline(val["confusion"])
        history["train_loss"].append(loss_meter.avg)
        history["val_loss"].append(val["loss"])
        history["val_miou"].append(val["miou"])
        history["val_dice"].append(val["dice"])

        if val["miou"] >= best_miou:
            best_miou, best_epoch = val["miou"], epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_miou": val["miou"],
                 **config},
                ckpt_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  train loss {loss_meter.avg:.4f}  "
                f"val mIoU {val['miou']:.4f}  Dice {val['dice']:.4f}  "
                f"pixel acc {val['pixel_accuracy']:.4f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss", "miou", "dice"))
        save_predictions(model, val_loader, dev, num_classes, out_path / "predictions_val.png")

    summary = {
        **config,
        "parameters": count_parameters(model),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "majority_class_baseline": baseline,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose and epochs:
        print(f"\nbest val mIoU {best_miou:.4f} at epoch {best_epoch} ({elapsed:.1f}s)")
        print(
            f"predict-majority-class baseline: mIoU {baseline['miou']:.4f}, "
            f"pixel accuracy {baseline['pixel_accuracy']:.4f}"
        )
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a segmentation network.")
    parser.add_argument("--model", default="unet", choices=sorted(MODELS))
    parser.add_argument("--dataset", default="shapes", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--loss", default="ce", choices=LOSSES)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--train-subset", type=int, default=4000, help="cap the training images (0 = all)"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch on generated 32x32 data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        model_name=args.model,
        dataset=args.dataset,
        image_size=args.image_size,
        base_channels=args.base_channels,
        loss=args.loss,
        dice_weight=args.dice_weight,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=8 if args.smoke_test else args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        augment=not args.no_augment,
        train_subset=args.train_subset or None,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out_dir,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
