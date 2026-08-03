"""Pretrain an encoder without labels.

    python train.py --method simclr
    python train.py --method byol

The training loss here is not a quality measure. SimCLR's loss falls when the
model gets better at telling batch-mates apart, which is correlated with useful
features but is not the same thing, and BYOL's loss can be driven to zero by a
representation that has collapsed to a constant. So every epoch also runs a
**k-NN probe** on the validation split: features are extracted from a small
memory bank of training images, validation images are classified by their nearest
neighbours, and *that* is what selects the checkpoint.

The probe costs a fraction of an epoch, needs no fitting, and is the honest
signal - watching it rise while the loss falls is the point of the curves figure.
``--epochs 0`` skips pretraining entirely and checkpoints the random encoder,
which is the control every number in this project is compared against.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import Subset
from tqdm import tqdm

from data import (
    AUGMENTATIONS,
    DATASETS,
    default_image_size,
    denormalize,
    get_splits,
    make_eval_loader,
    make_ssl_loader,
)
from model import METHODS, build_model
from utils import (
    AverageMeter,
    count_parameters,
    extract_features,
    get_device,
    knn_accuracy,
    plot_curves,
    plot_image_rows,
    save_json,
    set_seed,
)


def knn_monitor(
    encoder,
    bank_loader,
    val_loader,
    device: torch.device,
    num_classes: int,
    k: int = 20,
) -> float:
    """Cheap label-aware read-out of a label-free encoder."""
    bank_features, bank_labels = extract_features(encoder, bank_loader, device)
    val_features, val_labels = extract_features(encoder, val_loader, device)
    return knn_accuracy(
        bank_features, bank_labels, val_features, val_labels, num_classes, k=k
    )


def save_checkpoint(path: Path, model, config: dict, epoch: int, val_knn: float) -> None:
    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "epoch": epoch,
            "val_knn": val_knn,
            **config,
        },
        path,
    )


def run_training(
    method: str = "simclr",
    dataset: str = "cifar10",
    image_size: int | None = None,
    width: int = 32,
    blocks_per_stage: int = 2,
    proj_dim: int = 64,
    temperature: float = 0.5,
    momentum: float = 0.99,
    augment: str = "full",
    epochs: int = 6,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.05,
    train_subset: int | None = 10000,
    knn_bank: int = 2000,
    knn_k: int = 20,
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
    out_path = Path(out_dir or f"outputs/{dataset}_{method}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_base, val_base, _, info = get_splits(
        dataset,
        root=data_root,
        val_split=val_split,
        train_subset=train_subset,
        seed=seed,
        synthetic=synthetic,
    )
    size = image_size or (min(info["size"], 32) if synthetic else default_image_size(dataset))

    ssl_loader = make_ssl_loader(
        train_base, info, size, batch_size=batch_size, strength=augment, num_workers=num_workers
    )
    bank_base = Subset(train_base, range(min(knn_bank, len(train_base))))
    bank_loader = make_eval_loader(bank_base, info, size, num_workers=num_workers)
    val_loader = make_eval_loader(val_base, info, size, num_workers=num_workers)

    model = build_model(
        method,
        in_channels=info["channels"],
        width=width,
        blocks_per_stage=blocks_per_stage,
        proj_dim=proj_dim,
        temperature=temperature,
        momentum=momentum,
    ).to(dev)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    config = {
        "method": method,
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": info["classes"],
        "image_size": size,
        "width": width,
        "blocks_per_stage": blocks_per_stage,
        "augment": augment,
    }

    if verbose:
        print(f"device       : {dev}")
        print(f"method       : {method} - {METHODS[method]}")
        print(f"encoder      : {count_parameters(model.encoder):,} parameters, "
              f"{model.encoder.feature_dim}-d features")
        print(f"data         : {len(train_base)} unlabelled train / {len(val_base)} val "
              f"at {info['channels']}x{size}x{size}, augment={augment}")
        print(f"monitor      : {knn_k}-NN on a {len(bank_base)}-image bank\n")

    history = {"train_loss": [], "val_knn": []}
    ckpt_path = out_path / "best.pt"
    # Checkpoint the untrained encoder first, so --epochs 0 yields the control.
    baseline_knn = knn_monitor(model.encoder, bank_loader, val_loader, dev, info["classes"], knn_k)
    save_checkpoint(ckpt_path, model, config, 0, baseline_knn)
    best_knn, best_epoch = baseline_knn, 0
    if verbose:
        print(f"epoch  0/{epochs}  (untrained)          val {knn_k}-NN {baseline_knn:.4f}")

    total_steps = max(epochs * max(len(ssl_loader), 1), 1)
    step = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        batches = tqdm(ssl_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for view1, view2, _ in batches:  # the label is loaded and ignored
            view1, view2 = view1.to(dev), view2.to(dev)

            optimizer.zero_grad(set_to_none=True)
            loss = model(view1, view2)
            loss.backward()
            optimizer.step()
            step += 1
            model.update_target(step / total_steps)

            loss_meter.update(loss.item(), view1.size(0))
            batches.set_postfix(loss=f"{loss_meter.avg:.4f}")
        scheduler.step()

        val_knn = knn_monitor(model.encoder, bank_loader, val_loader, dev, info["classes"], knn_k)
        history["train_loss"].append(loss_meter.avg)
        history["val_knn"].append(val_knn)

        if val_knn >= best_knn:
            best_knn, best_epoch = val_knn, epoch
            save_checkpoint(ckpt_path, model, config, epoch, val_knn)

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  train loss {loss_meter.avg:.4f}  "
                f"val {knn_k}-NN {val_knn:.4f}  ({val_knn - baseline_knn:+.4f} vs untrained)"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss", "knn"))
    view1, view2, _ = next(iter(ssl_loader))
    plot_image_rows(
        [("view 1", denormalize(view1[:8], info)), ("view 2", denormalize(view2[:8], info))],
        out_path / "views.png",
        title=f"the pairs the model is asked to match (augment={augment})",
    )

    summary = {
        **config,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "encoder_parameters": count_parameters(model.encoder),
        "untrained_val_knn": baseline_knn,
        "best_val_knn": best_knn,
        "best_epoch": best_epoch,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose:
        print(
            f"\nbest val {knn_k}-NN {best_knn:.4f} at epoch {best_epoch} "
            f"(untrained encoder: {baseline_knn:.4f}) in {elapsed:.1f}s"
        )
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain an encoder without labels.")
    parser.add_argument("--method", default="simclr", choices=sorted(METHODS))
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--width", type=int, default=32, help="encoder base width")
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5, help="SimCLR only")
    parser.add_argument("--momentum", type=float, default=0.99, help="BYOL target EMA")
    parser.add_argument(
        "--augment",
        default="full",
        choices=AUGMENTATIONS,
        help="strength of the two-view augmentation; 'none' makes the task trivial",
    )
    parser.add_argument("--epochs", type=int, default=6, help="0 checkpoints the random encoder")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument(
        "--train-subset", type=int, default=10000, help="cap the unlabelled images (0 = all)"
    )
    parser.add_argument("--knn-bank", type=int, default=2000)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch on random tensors")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        method=args.method,
        dataset=args.dataset,
        image_size=args.image_size,
        width=args.width,
        blocks_per_stage=args.blocks_per_stage,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        momentum=args.momentum,
        augment=args.augment,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=32 if args.smoke_test else args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        train_subset=args.train_subset or None,
        knn_bank=args.knn_bank,
        knn_k=args.knn_k,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out_dir,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
