"""Train a ResNet on CIFAR-10 / CIFAR-100 / SVHN.

    python train.py --arch resnet20 --dataset cifar10 --epochs 12

The defaults are tuned for a CPU: a 15k-image subset of the training split
finishes in a few minutes and already reaches a clearly-better-than-chance
model. For the real numbers use ``--train-subset 0 --epochs 30``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import DATASETS, get_dataloaders, get_datasets
from model import MODELS, build_model
from utils import AverageMeter, count_parameters, get_device, plot_history, save_json, set_seed


@torch.no_grad()
def evaluate_split(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_meter, correct, seen = AverageMeter(), 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_meter.update(criterion(logits, targets).item(), images.size(0))
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += images.size(0)
    return loss_meter.avg, correct / max(seen, 1)


def run_training(
    arch: str = "resnet20",
    dataset: str = "cifar10",
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.0,
    val_split: float = 0.1,
    augment: bool = True,
    train_subset: int | None = 15000,
    num_workers: int = 2,
    seed: int = 0,
    device: str = "auto",
    data_root: str = "data",
    out_dir: str | None = None,
    synthetic: bool = False,
    verbose: bool = True,
) -> dict:
    """Train one architecture and return a summary dict."""
    set_seed(seed)
    dev = get_device(device)
    out_path = Path(out_dir or f"outputs/{dataset}_{arch}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = get_datasets(
        dataset,
        root=data_root,
        val_split=val_split,
        augment=augment,
        train_subset=train_subset,
        seed=seed,
        synthetic=synthetic,
    )
    train_loader, val_loader, _ = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=num_workers
    )

    model = build_model(arch, num_classes=len(class_names)).to(dev)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
    )
    # One-cycle-ish: warm up for the first epoch, then cosine down to ~0.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=max(epochs, 1),
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.25,
    )

    if verbose:
        print(f"device      : {dev}")
        print(f"model       : {arch} ({count_parameters(model):,} trainable parameters)")
        print(f"train / val : {len(train_ds)} / {len(val_ds)}  augment={augment}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc, best_epoch = 0.0, 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter, correct, seen = AverageMeter(), 0, 0
        batches = tqdm(
            train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose
        )
        for images, targets in batches:
            images, targets = images.to(dev), targets.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), images.size(0))
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += images.size(0)
            batches.set_postfix(loss=f"{loss_meter.avg:.4f}")

        train_loss, train_acc = loss_meter.avg, correct / max(seen, 1)
        val_loss, val_acc = evaluate_split(model, val_loader, criterion, dev)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(scheduler.get_last_lr()[0])

        if val_acc >= best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": arch,
                    "dataset": dataset,
                    "class_names": list(class_names),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  "
                f"train loss {train_loss:.4f} acc {train_acc:.4f}  |  "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    summary = {
        "arch": arch,
        "dataset": dataset,
        "parameters": count_parameters(model),
        "epochs": epochs,
        "train_images": len(train_ds),
        "augment": augment,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")
    if epochs > 0:
        plot_history(history, out_path / "curves.png")

    if verbose:
        print(f"\nbest val accuracy {best_val_acc:.4f} at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CNN classifier on small images.")
    parser.add_argument("--arch", default="resnet20", choices=sorted(MODELS))
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1, help="peak LR for the one-cycle schedule")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true", help="disable crop/flip augmentation")
    parser.add_argument(
        "--train-subset",
        type=int,
        default=15000,
        help="cap the training images (0 = use the full split)",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 epoch on random tensors, to check the pipeline without downloading",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        arch=args.arch,
        dataset=args.dataset,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
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
