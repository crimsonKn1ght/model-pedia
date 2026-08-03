"""Pretrain a masked autoencoder.

    python train.py --dataset cifar10 --epochs 8
    python train.py --mask-ratio 0.5            # hide half instead of three quarters

No labels are used. The model sees a quarter of the patches and is scored on the
pixels of the three quarters it did not see, which is the whole objective.

Validation uses a **fixed mask seed**, so the epoch-to-epoch comparison is over
the same hidden patches every time. Without that, half of the movement in the
validation curve is just a different random mask.

Two numbers are tracked. The training objective (mean squared error on masked
patches, per-patch normalised by default) is what the optimiser sees; masked-patch
PSNR is the same thing in comparable units, and is the one to quote. Neither says
anything about whether the *features* are good - that is what ``evaluate.py`` is
for, and the gap between "reconstructs nicely" and "classifies well" is one of
the more useful things this project shows.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import DATASETS, get_splits, make_loader, pretrain_transform, eval_transform
from model import build_mae, patchify
from utils import (
    AverageMeter,
    count_parameters,
    get_device,
    masked_psnr,
    plot_curves,
    plot_image_rows,
    save_json,
    set_seed,
)

VAL_MASK_SEED = 1234


def _fork_rng(device: torch.device):
    """Reproducible masks inside a ``with`` block, without disturbing training."""
    devices = [device] if device.type == "cuda" else []
    return torch.random.fork_rng(devices=devices)


@torch.no_grad()
def evaluate_split(model, loader, device: torch.device, mask_ratio: float) -> dict:
    """Masked-patch loss and PSNR at one mask ratio, with a fixed mask."""
    model.eval()
    loss_meter, psnr_meter = AverageMeter(), AverageMeter()

    with _fork_rng(device):
        torch.manual_seed(VAL_MASK_SEED)
        for images, _ in loader:
            images = images.to(device)
            loss, prediction, mask = model(images, mask_ratio)
            _, reconstruction = model.reconstruct(images, prediction, mask)

            n = images.size(0)
            loss_meter.update(loss.item(), n)
            psnr_meter.update(
                masked_psnr(
                    patchify(reconstruction, model.encoder.patch_size),
                    patchify(images, model.encoder.patch_size),
                    mask,
                ),
                n,
            )
    return {"loss": loss_meter.avg, "psnr": psnr_meter.avg}


@torch.no_grad()
def save_reconstructions(model, loader, device, mask_ratio: float, path, n: int = 8) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)
    with _fork_rng(device):
        torch.manual_seed(VAL_MASK_SEED)
        _, prediction, mask = model(images, mask_ratio)
    masked, reconstruction = model.reconstruct(images, prediction, mask)
    plot_image_rows(
        [
            ("original", images.cpu()),
            (f"masked {mask_ratio:g}", masked.cpu()),
            ("reconstruction", reconstruction.cpu()),
        ],
        path,
        title="the encoder only ever saw the patches left in the middle row",
    )


def run_training(
    dataset: str = "cifar10",
    image_size: int | None = None,
    patch_size: int | None = None,
    dim: int = 128,
    depth: int = 4,
    heads: int = 4,
    decoder_dim: int = 64,
    decoder_depth: int = 2,
    mask_ratio: float = 0.75,
    norm_pixel_loss: bool = True,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1.5e-3,
    weight_decay: float = 0.05,
    val_split: float = 0.05,
    train_subset: int | None = 10000,
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
    out_path = Path(out_dir or f"outputs/{dataset}_mae")
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
    patch = patch_size or info["patch"]

    train_loader = make_loader(
        train_base,
        pretrain_transform(info, size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        val_base, eval_transform(info, size), batch_size=batch_size, num_workers=num_workers
    )

    model = build_mae(
        image_size=size,
        patch_size=patch,
        in_channels=info["channels"],
        dim=dim,
        depth=depth,
        heads=heads,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        norm_pixel_loss=norm_pixel_loss,
    ).to(dev)
    # AdamW with a large weight decay is the standard ViT recipe; the model is
    # small enough that the decay matters more than the schedule.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    encoder_params = count_parameters(model.encoder)
    total_params = count_parameters(model)
    visible = max(1, int(round(model.encoder.num_patches * (1 - mask_ratio))))

    config = {
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": info["classes"],
        "image_size": size,
        "patch_size": patch,
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "decoder_dim": decoder_dim,
        "decoder_depth": decoder_depth,
        "mask_ratio": mask_ratio,
        "norm_pixel_loss": norm_pixel_loss,
    }

    if verbose:
        print(f"device     : {dev}")
        print(f"encoder    : {encoder_params:,} parameters, dim {dim} x depth {depth}")
        print(f"decoder    : {total_params - encoder_params:,} parameters (discarded afterwards)")
        print(
            f"patches    : {model.encoder.num_patches} of {patch}x{patch}; "
            f"mask {mask_ratio:g} leaves {visible} for the encoder"
        )
        print(f"data       : {len(train_base)} train / {len(val_base)} val "
              f"at {info['channels']}x{size}x{size}\n")

    history = {"train_loss": [], "val_loss": [], "val_psnr": []}
    best_psnr, best_epoch = -float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for images, _ in batches:  # labels loaded and ignored
            images = images.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(images, mask_ratio)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.size(0))
            batches.set_postfix(loss=f"{loss_meter.avg:.4f}")
        scheduler.step()

        val = evaluate_split(model, val_loader, dev, mask_ratio)
        history["train_loss"].append(loss_meter.avg)
        history["val_loss"].append(val["loss"])
        history["val_psnr"].append(val["psnr"])

        if val["psnr"] >= best_psnr:
            best_psnr, best_epoch = val["psnr"], epoch
            torch.save(
                {
                    "encoder_state": model.encoder.state_dict(),
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_psnr": val["psnr"],
                    **config,
                },
                ckpt_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  train loss {loss_meter.avg:.4f}  "
                f"val loss {val['loss']:.4f}  val masked PSNR {val['psnr']:.2f} dB"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss", "psnr"))
        save_reconstructions(
            model, val_loader, dev, mask_ratio, out_path / "reconstruction_val.png"
        )
    else:
        # --epochs 0 checkpoints the untrained model, the control for evaluate.py.
        torch.save(
            {
                "encoder_state": model.encoder.state_dict(),
                "model_state": model.state_dict(),
                "epoch": 0,
                "val_psnr": float("nan"),
                **config,
            },
            ckpt_path,
        )

    summary = {
        **config,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "encoder_parameters": encoder_params,
        "decoder_parameters": total_params - encoder_params,
        "visible_patches": visible,
        "best_epoch": best_epoch,
        "best_val_psnr": best_psnr if epochs else float("nan"),
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose and epochs:
        print(f"\nbest val masked PSNR {best_psnr:.2f} dB at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain a masked autoencoder.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--dim", type=int, default=128, help="encoder width")
    parser.add_argument("--depth", type=int, default=4, help="encoder blocks")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--decoder-dim", type=int, default=64)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument(
        "--raw-pixel-loss",
        action="store_true",
        help="predict raw pixels instead of per-patch normalised ones",
    )
    parser.add_argument("--epochs", type=int, default=8, help="0 checkpoints the untrained model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--train-subset", type=int, default=10000, help="cap the images (0 = all)")
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
        dataset=args.dataset,
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        mask_ratio=args.mask_ratio,
        norm_pixel_loss=not args.raw_pixel_loss,
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
