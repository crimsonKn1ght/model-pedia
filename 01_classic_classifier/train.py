"""Train the MLP or LeNet classifier.

    python train.py --model lenet --dataset fashion-mnist --epochs 5

Writes the best checkpoint, the epoch-by-epoch history and the training curves
into ``outputs/<dataset>_<model>/``.
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
    """Mean loss and accuracy over a loader."""
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
    model_name: str = "lenet",
    dataset: str = "mnist",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_split: float = 0.1,
    augment: bool = False,
    train_subset: int | None = None,
    num_workers: int = 2,
    seed: int = 0,
    device: str = "auto",
    data_root: str = "data",
    out_dir: str | None = None,
    synthetic: bool = False,
    verbose: bool = True,
) -> dict:
    """Train one model and return a summary dict. Also used by ``compare.py``."""
    set_seed(seed)
    dev = get_device(device)
    out_path = Path(out_dir or f"outputs/{dataset}_{model_name}")
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

    model = build_model(model_name, num_classes=len(class_names)).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    if verbose:
        print(f"device      : {dev}")
        print(f"model       : {model_name} ({count_parameters(model):,} trainable parameters)")
        print(f"train / val : {len(train_ds)} / {len(val_ds)}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
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

            loss_meter.update(loss.item(), images.size(0))
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += images.size(0)
            batches.set_postfix(loss=f"{loss_meter.avg:.4f}")

        scheduler.step()
        train_loss, train_acc = loss_meter.avg, correct / max(seen, 1)
        val_loss, val_acc = evaluate_split(model, val_loader, criterion, dev)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
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
        "model": model_name,
        "dataset": dataset,
        "parameters": count_parameters(model),
        "epochs": epochs,
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
    parser = argparse.ArgumentParser(description="Train an MLP or LeNet classifier.")
    parser.add_argument("--model", default="lenet", choices=sorted(MODELS))
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true", help="mild affine jitter on train images")
    parser.add_argument("--train-subset", type=int, default=None, help="cap the training images")
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
        model_name=args.model,
        dataset=args.dataset,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        augment=args.augment,
        train_subset=args.train_subset,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out_dir,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
