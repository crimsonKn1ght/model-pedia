"""Fine-tune (or linear-probe) a pretrained backbone on a small dataset.

    python train.py --mode both --dataset flowers102 --epochs 10

``--mode frozen`` caches the backbone's features once and then trains only the
linear head, which takes seconds. ``--mode finetune`` keeps training the
backbone at a smaller learning rate. ``--mode both`` runs the two and prints
the comparison the project exists to show.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import DATASETS, get_dataloaders, get_datasets
from model import BACKBONES, TransferModel
from utils import AverageMeter, count_parameters, get_device, plot_history, save_json, set_seed


@torch.no_grad()
def cache_features(model: TransferModel, loader, device, desc: str, verbose: bool = True):
    """Run the frozen backbone over a split once and keep the vectors in memory.

    A linear probe only ever sees these vectors, so recomputing them every
    epoch would be pure waste - this is what makes frozen mode near-instant.
    """
    model.eval()
    features, labels = [], []
    for images, targets in tqdm(loader, desc=desc, leave=False, disable=not verbose):
        features.append(model.extract_features(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def evaluate_head(head: nn.Module, features: torch.Tensor, labels: torch.Tensor, criterion, device):
    head.eval()
    logits = head(features.to(device))
    loss = criterion(logits, labels.to(device)).item()
    acc = (logits.argmax(dim=1).cpu() == labels).float().mean().item()
    return loss, acc


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


def _train_frozen(model, loaders, epochs, head_lr, weight_decay, device, verbose):
    """Linear probe on cached features."""
    train_loader, val_loader = loaders
    model.set_backbone_trainable(False)

    train_x, train_y = cache_features(model, train_loader, device, "caching train", verbose)
    val_x, val_y = cache_features(model, val_loader, device, "caching val", verbose)
    if verbose:
        print(f"cached features: train {tuple(train_x.shape)}  val {tuple(val_x.shape)}")

    head = model.head.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=head_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    feature_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = 0.0, 0, None

    for epoch in range(1, epochs + 1):
        head.train()
        loss_meter, correct, seen = AverageMeter(), 0, 0
        for batch_x, batch_y in feature_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), batch_x.size(0))
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            seen += batch_x.size(0)
        scheduler.step()

        train_loss, train_acc = loss_meter.avg, correct / max(seen, 1)
        val_loss, val_acc = evaluate_head(head, val_x, val_y, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  train loss {train_loss:.4f} acc {train_acc:.4f}  |  "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )
    return history, best_val_acc, best_epoch, best_state


def _train_finetune(
    model, loaders, epochs, head_lr, backbone_lr, weight_decay, label_smoothing, device, verbose
):
    """Update the whole network, backbone included, at two learning rates."""
    train_loader, val_loader = loaders
    model.set_backbone_trainable(True)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(head_lr=head_lr, backbone_lr=backbone_lr),
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[backbone_lr, head_lr],
        epochs=max(epochs, 1),
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.3,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = 0.0, 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter, correct, seen = AverageMeter(), 0, 0
        batches = tqdm(
            train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose
        )
        for images, targets in batches:
            images, targets = images.to(device), targets.to(device)
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
        val_loss, val_acc = evaluate_split(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  train loss {train_loss:.4f} acc {train_acc:.4f}  |  "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )
    return history, best_val_acc, best_epoch, best_state


def run_training(
    mode: str = "finetune",
    backbone: str = "resnet18",
    dataset: str = "flowers102",
    epochs: int = 10,
    batch_size: int = 32,
    image_size: int = 224,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    augment: bool = True,
    dropout: float = 0.2,
    num_workers: int = 2,
    seed: int = 0,
    device: str = "auto",
    data_root: str = "data",
    out_dir: str | None = None,
    pretrained: bool = True,
    synthetic: bool = False,
    verbose: bool = True,
) -> dict:
    """Train in one mode and return a summary dict."""
    if mode not in {"frozen", "finetune"}:
        raise ValueError("mode must be 'frozen' or 'finetune'")

    set_seed(seed)
    dev = get_device(device)
    out_path = Path(out_dir or f"outputs/{dataset}_{backbone}_{mode}")
    out_path.mkdir(parents=True, exist_ok=True)

    # Cached features cannot follow random augmentation, so a linear probe
    # always sees the deterministic centre crop.
    effective_augment = augment and mode == "finetune"

    train_ds, val_ds, test_ds, class_names = get_datasets(
        dataset,
        root=data_root,
        image_size=image_size,
        augment=effective_augment,
        seed=seed,
        synthetic=synthetic,
    )
    train_loader, val_loader, _ = get_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=(mode == "finetune"),
    )

    model = TransferModel(backbone, len(class_names), pretrained=pretrained, dropout=dropout).to(dev)

    if verbose:
        print(f"device      : {dev}")
        print(f"backbone    : {backbone} (pretrained={pretrained}, {model.feature_dim}-d features)")
        print(f"mode        : {mode}  augment={effective_augment}")
        print(f"train / val : {len(train_ds)} / {len(val_ds)}  classes={len(class_names)}")
        trainable = (
            count_parameters(model.head)
            if mode == "frozen"
            else count_parameters(model.head) + sum(p.numel() for p in model.backbone.parameters())
        )
        print(f"updating    : {trainable:,} parameters")

    started = time.time()
    if mode == "frozen":
        history, best_val_acc, best_epoch, best_state = _train_frozen(
            model, (train_loader, val_loader), epochs, head_lr, weight_decay, dev, verbose
        )
    else:
        history, best_val_acc, best_epoch, best_state = _train_finetune(
            model,
            (train_loader, val_loader),
            epochs,
            head_lr,
            backbone_lr,
            weight_decay,
            label_smoothing,
            dev,
            verbose,
        )
    elapsed = time.time() - started

    ckpt_path = out_path / "best.pt"
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "backbone": backbone,
            "dataset": dataset,
            "mode": mode,
            "image_size": image_size,
            "dropout": dropout,
            "class_names": list(class_names),
            "epoch": best_epoch,
            "val_acc": best_val_acc,
        },
        ckpt_path,
    )

    summary = {
        "mode": mode,
        "backbone": backbone,
        "dataset": dataset,
        "epochs": epochs,
        "image_size": image_size,
        "trainable_parameters": (
            count_parameters(model.head)
            if mode == "frozen"
            else count_parameters(model.head) + sum(p.numel() for p in model.backbone.parameters())
        ),
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
    parser = argparse.ArgumentParser(description="Transfer-learn a pretrained backbone.")
    parser.add_argument("--mode", default="both", choices=["frozen", "finetune", "both"])
    parser.add_argument("--backbone", default="resnet18", choices=sorted(BACKBONES))
    parser.add_argument("--dataset", default="flowers102", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="finetune mode only")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="random init instead of ImageNet weights - the control arm",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="2 epochs on random tensors with a random-init backbone",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    modes = ["frozen", "finetune"] if args.mode == "both" else [args.mode]

    summaries = []
    for mode in modes:
        print(f"\n=== {mode} ===")
        # Keep the two arms apart when one --out-dir covers both.
        out_dir = args.out_dir
        if out_dir and len(modes) > 1:
            out_dir = str(Path(out_dir) / mode)
        summaries.append(
            run_training(
                mode=mode,
                backbone=args.backbone,
                dataset=args.dataset,
                epochs=2 if args.smoke_test else args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
                head_lr=args.head_lr,
                backbone_lr=args.backbone_lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                augment=not args.no_augment,
                dropout=args.dropout,
                num_workers=0 if args.smoke_test else args.num_workers,
                seed=args.seed,
                device=args.device,
                data_root=args.data_root,
                out_dir=out_dir,
                # The smoke test must not depend on a weight download.
                pretrained=not (args.no_pretrained or args.smoke_test),
                synthetic=args.smoke_test,
            )
        )

    if len(summaries) > 1:
        print(f"\n=== {args.backbone} on {args.dataset}, {args.epochs} epochs ===")
        header = f"{'mode':10s} {'updated params':>15s} {'train s':>9s} {'best val acc':>13s}"
        print(header)
        print("-" * len(header))
        for summary in summaries:
            print(
                f"{summary['mode']:10s} {summary['trainable_parameters']:15,d} "
                f"{summary['train_seconds']:9.1f} {summary['best_val_acc']:13.4f}"
            )
        print("\nrun evaluate.py on each checkpoint for test-set numbers")


if __name__ == "__main__":
    main()
