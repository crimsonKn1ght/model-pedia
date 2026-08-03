"""Train a ViT or the ResNet control.

    python train.py --arch vit --dataset cifar10 --epochs 15
    python train.py --arch resnet --dataset cifar10 --epochs 15

Both architectures go through this same script with the same schedule, the same
augmentation and the same seed, because the comparison is the point of the project and a
comparison where the two arms had different recipes would not be one.

The wall-clock numbers printed at the end matter as much as the accuracy. Matching
parameter counts is only half a control: attention costs O(tokens^2) and convolution
costs O(pixels x channels), so two models of equal size can differ by a factor in what
they cost to run. ``measure_throughput`` times a real forward-backward step so the
comparison can be made per-second as well as per-parameter.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import AUGMENTATIONS, DATASETS, get_splits, make_loader
from model import ARCHITECTURES, build_model
from utils import (
    AverageMeter,
    accuracy_and_confusion,
    count_parameters,
    get_device,
    measure_throughput,
    plot_curves,
    save_json,
    set_seed,
)


def run_training(
    architecture: str = "vit",
    dataset: str = "cifar10",
    image_size: int | None = None,
    patch_size: int = 4,
    dim: int = 192,
    depth: int = 6,
    heads: int = 3,
    dropout: float = 0.1,
    width: int = 64,
    augment: str = "basic",
    label_smoothing: float = 0.1,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    val_split: float = 0.05,
    train_subset: int | None = 20000,
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
    out_path = Path(out_dir or f"outputs/{dataset}_{architecture}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        dataset, root=data_root, image_size=image_size, val_split=val_split,
        train_subset=train_subset, augment=augment, seed=seed, synthetic=synthetic,
    )
    train_loader = make_loader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_set, 256, num_workers=num_workers)

    model = build_model(
        architecture, info["classes"], info["size"], info["channels"],
        patch_size, dim, depth, heads, dropout, width,
    ).to(dev)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    throughput = measure_throughput(
        model, (batch_size, info["channels"], info["size"], info["size"]), dev,
        steps=2 if synthetic else 5,
    )
    config = {
        "architecture": architecture,
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": info["classes"],
        "class_names": info["class_names"],
        "image_size": info["size"],
        "patch_size": patch_size,
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "width": width,
        "augment": augment,
    }

    if verbose:
        print(f"device     : {dev}")
        print(f"model      : {architecture} with {count_parameters(model):,} parameters")
        if architecture == "vit":
            print(f"tokens     : {(info['size'] // patch_size) ** 2} patches of "
                  f"{patch_size}x{patch_size} plus a class token")
        print(f"throughput : {throughput['train_images_per_second']:.0f} train images/s, "
              f"{throughput['inference_images_per_second']:.0f} inference images/s")
        print(f"data       : {len(train_set)} train / {len(val_set)} val, "
              f"augment={augment}\n")

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_accuracy, best_epoch = -1.0, 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter, correct, total = AverageMeter(), 0, 0
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False,
                       disable=not verbose)
        for images, targets in batches:
            images, targets = images.to(dev), targets.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_meter.update(loss.item(), images.size(0))
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
            batches.set_postfix(loss=f"{loss_meter.avg:.4f}")
        scheduler.step()

        model.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(dev), targets.to(dev)
                val_meter.update(criterion(model(images), targets).item(), images.size(0))
        val_accuracy, _ = accuracy_and_confusion(model, val_loader, dev, info["classes"])

        history["train_loss"].append(loss_meter.avg)
        history["train_accuracy"].append(correct / max(total, 1))
        history["val_loss"].append(val_meter.avg)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy >= best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch,
                 "val_accuracy": val_accuracy, **config},
                ckpt_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  train loss {loss_meter.avg:.4f} "
                f"acc {history['train_accuracy'][-1]:.4f}  |  "
                f"val loss {val_meter.avg:.4f} acc {val_accuracy:.4f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss", "accuracy"))

    summary = {
        **config,
        "parameters": count_parameters(model),
        "throughput": throughput,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose and epochs:
        print(f"\nbest val accuracy {best_accuracy:.4f} at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a ViT or a ResNet control.")
    parser.add_argument("--arch", default="vit", choices=ARCHITECTURES)
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=4, help="ViT only")
    parser.add_argument("--dim", type=int, default=192, help="ViT width")
    parser.add_argument("--depth", type=int, default=6, help="ViT blocks")
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=64, help="ResNet width")
    parser.add_argument("--augment", default="basic", choices=AUGMENTATIONS)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--train-subset", type=int, default=20000, help="cap images (0 = all)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch on generated data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        architecture=args.arch,
        dataset=args.dataset,
        image_size=args.image_size,
        patch_size=2 if args.smoke_test else args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        width=args.width,
        augment=args.augment,
        label_smoothing=args.label_smoothing,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=32 if args.smoke_test else args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
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
