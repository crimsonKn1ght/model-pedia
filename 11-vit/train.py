"""Train a ViT-Tiny (or the ResNet-18 baseline) for image classification.

    python train.py                              # ViT on CIFAR-10
    python train.py --model resnet18             # the CNN baseline
    python train.py --dataset flowers102 --image-size 64

Train both and compare -- that is what this project is for.  ``evaluate.py``
does the comparison properly, including a parameters-and-FLOPs matching so the
numbers mean something.

Note the augmentation and mixup defaults.  A ViT trained on CIFAR-10 without
them overfits badly and lands well below the ResNet; with them the gap narrows
considerably. Turn them off with ``--mixup 0 --no-augment`` to see it happen.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from model import build_model, mixup

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--model", default="vit", choices=["vit", "resnet18"])
    parser.add_argument("--dataset", default="cifar10", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=192, help="ViT embedding width")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--resnet-base", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--mixup", type=float, default=0.2, help="mixup alpha; 0 disables")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true")
    return parser


def evaluate_accuracy(model, loader, device, args) -> tuple:
    model.eval()
    top1, top5, loss_m = (utils.AverageMeter() for _ in range(3))
    with torch.no_grad():
        for x, y in utils.maybe_limit(loader, args):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_m.update(F.cross_entropy(logits, y).item(), x.shape[0])
            k = min(5, logits.shape[1])
            _, pred = logits.topk(k, dim=1)
            correct = pred.eq(y.view(-1, 1))
            top1.update(correct[:, 0].float().mean().item(), x.shape[0])
            top5.update(correct.any(dim=1).float().mean().item(), x.shape[0])
    return top1.avg, top5.avg, loss_m.avg


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    out_dir = utils.resolve_out_dir(args, str(HERE / "outputs" / args.model))

    meta = data_mod.info(args.dataset)
    train_loader = data_mod.get_dataloader(
        args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        train=True,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        num_workers=args.num_workers,
        subset=args.train_subset,
    )
    test_loader = data_mod.get_dataloader(
        args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    kwargs = (
        dict(patch_size=args.patch_size, dim=args.dim, depth=args.depth, heads=args.heads)
        if args.model == "vit"
        else dict(base=args.resnet_base)
    )
    model = build_model(args.model, args.image_size, meta.channels, meta.num_classes, **kwargs).to(device)
    print(utils.describe_model(args.model, model))
    print(f"dataset={args.dataset} classes={meta.num_classes} device={device}")

    # AdamW with cosine decay and a short warmup: the standard ViT recipe.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = 1 if args.quick else args.epochs
    steps_per_epoch = utils.quick_len(train_loader, args)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    history = utils.History()
    best = 0.0

    with utils.Timer() as timer:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_m, acc_m = utils.AverageMeter(), utils.AverageMeter()
            for x, y in utils.progress(
                utils.maybe_limit(train_loader, args), desc=f"epoch {epoch}/{epochs}",
                total=steps_per_epoch,
            ):
                x, y = x.to(device), y.to(device)
                mixed, y_a, y_b, lam = mixup(x, y, args.mixup)
                logits = model(mixed)
                loss = lam * F.cross_entropy(
                    logits, y_a, label_smoothing=args.label_smoothing
                ) + (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=args.label_smoothing)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                loss_m.update(loss.item(), x.shape[0])
                acc_m.update((logits.argmax(1) == y).float().mean().item(), x.shape[0])

            top1, top5, test_loss = evaluate_accuracy(model, test_loader, device, args)
            best = max(best, top1)
            history.add(
                train_loss=loss_m.avg,
                train_accuracy=acc_m.avg,
                test_loss=test_loss,
                test_accuracy=top1,
            )
            print(
                f"epoch {epoch:>3}  train loss {loss_m.avg:.4f}  train acc {acc_m.avg:.3f}  "
                f"test acc {top1:.4f}  test top-5 {top5:.4f}"
            )

    print(f"\ntrained {epochs} epochs in {timer.elapsed / 60:.1f} min; best test accuracy {best:.4f}")

    utils.save_checkpoint(
        out_dir / f"{args.model}.pt",
        model=model,
        config={
            "model": args.model,
            "dataset": args.dataset,
            "image_size": args.image_size,
            "channels": meta.channels,
            "num_classes": meta.num_classes,
            "kwargs": kwargs,
            "epochs": epochs,
            "best_test_accuracy": best,
            "train_minutes": timer.elapsed / 60,
        },
    )
    history.save(out_dir / "history.json")
    viz.plot_curves(history.records, out_dir / "training-curves.png", title=f"{args.model} training")
    print(f"saved checkpoint and figures to {out_dir}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
