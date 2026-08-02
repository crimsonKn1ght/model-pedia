"""Train the convolutional autoencoder.

    python train.py --dataset fashion-mnist --latent-dim 32 --epochs 10

There are no labels in the loss: the target is the input. Validation PSNR is
tracked alongside the loss because a reconstruction MSE of "0.008" means
nothing on its own, while "21 dB" is comparable across datasets.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import DATASETS, get_dataloaders, get_datasets
from model import build_model
from utils import (
    AverageMeter,
    count_parameters,
    get_device,
    plot_curves,
    plot_image_rows,
    psnr,
    save_json,
    set_seed,
    ssim,
)

LOSSES = {
    "mse": nn.MSELoss(),
    # BCE treats each pixel as a Bernoulli probability; it sharpens the
    # near-binary images of MNIST and Fashion-MNIST noticeably.
    "bce": nn.BCELoss(),
}


@torch.no_grad()
def evaluate_split(model, loader, criterion, device) -> dict:
    model.eval()
    loss_meter, psnr_meter, ssim_meter = AverageMeter(), AverageMeter(), AverageMeter()
    for images, _ in loader:
        images = images.to(device)
        recon = model(images)
        n = images.size(0)
        loss_meter.update(criterion(recon, images).item(), n)
        psnr_meter.update(psnr(recon, images).mean().item(), n)
        ssim_meter.update(ssim(recon, images).mean().item(), n)
    return {"loss": loss_meter.avg, "psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


@torch.no_grad()
def save_reconstruction_strip(model, loader, device, path, n: int = 8) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)
    recon = model(images)
    plot_image_rows(
        [("original", images.cpu()), ("reconstruction", recon.cpu())],
        path,
        title="validation reconstructions",
    )


def run_training(
    dataset: str = "fashion-mnist",
    latent_dim: int = 32,
    base_channels: int = 32,
    loss: str = "mse",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    val_split: float = 0.1,
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
    out_path = Path(out_dir or f"outputs/{dataset}_latent{latent_dim}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, info = get_datasets(
        dataset,
        root=data_root,
        val_split=val_split,
        train_subset=train_subset,
        seed=seed,
        synthetic=synthetic,
    )
    train_loader, val_loader, _ = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=num_workers
    )

    model = build_model(info["channels"], latent_dim, base_channels).to(dev)
    criterion = LOSSES[loss]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    if verbose:
        print(f"device      : {dev}")
        print(f"model       : latent {latent_dim} ({count_parameters(model):,} parameters)")
        print(
            f"compression : {model.compression_ratio():.1f}x "
            f"({info['channels'] * 32 * 32} numbers -> {latent_dim})"
        )
        print(f"train / val : {len(train_ds)} / {len(val_ds)}  loss={loss}")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
    }
    best_psnr, best_epoch = -float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        batches = tqdm(
            train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose
        )
        for images, _ in batches:
            images = images.to(dev)
            optimizer.zero_grad(set_to_none=True)
            recon = model(images)
            batch_loss = criterion(recon, images)
            batch_loss.backward()
            optimizer.step()

            loss_meter.update(batch_loss.item(), images.size(0))
            batches.set_postfix(loss=f"{loss_meter.avg:.5f}")
        scheduler.step()

        val = evaluate_split(model, val_loader, criterion, dev)
        history["train_loss"].append(loss_meter.avg)
        history["val_loss"].append(val["loss"])
        history["val_psnr"].append(val["psnr"])
        history["val_ssim"].append(val["ssim"])

        if val["psnr"] >= best_psnr:
            best_psnr, best_epoch = val["psnr"], epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "dataset": dataset,
                    "latent_dim": latent_dim,
                    "base_channels": base_channels,
                    "in_channels": info["channels"],
                    "loss": loss,
                    "epoch": epoch,
                    "val_psnr": val["psnr"],
                },
                ckpt_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  train loss {loss_meter.avg:.5f}  |  "
                f"val loss {val['loss']:.5f}  PSNR {val['psnr']:.2f} dB  "
                f"SSIM {val['ssim']:.4f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss", "psnr", "ssim"))
        save_reconstruction_strip(model, val_loader, dev, out_path / "reconstructions_val.png")

    summary = {
        "dataset": dataset,
        "latent_dim": latent_dim,
        "compression_ratio": model.compression_ratio(),
        "parameters": count_parameters(model),
        "loss": loss,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_psnr": best_psnr,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose:
        print(f"\nbest val PSNR {best_psnr:.2f} dB at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a convolutional autoencoder.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--loss", default="mse", choices=sorted(LOSSES))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--train-subset", type=int, default=20000, help="cap the training images (0 = all)"
    )
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
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        loss=args.loss,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
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
