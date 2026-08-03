"""Train the denoising U-Net.

    python train.py --dataset cifar10 --epochs 6

By default the model is trained *blind*: every batch gets a noise level drawn
uniformly from ``--sigma-range``, so at test time it is not told how noisy the
input is. That is the only realistic setting, and it costs surprisingly little
compared to training one model per noise level.

Validation always uses a fixed noise seed, otherwise the epoch-to-epoch
comparison would be measuring the random number generator.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import DATASETS, add_gaussian_noise, get_dataloaders, get_datasets
from model import MODELS, build_model
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

VAL_NOISE_SEED = 1234


@torch.no_grad()
def evaluate_split(model, loader, criterion, device, sigma: float) -> dict:
    """Denoising quality at one noise level, plus the do-nothing baseline."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(VAL_NOISE_SEED)
    loss_meter = AverageMeter()
    out_psnr, out_ssim = AverageMeter(), AverageMeter()
    noisy_psnr, noisy_ssim = AverageMeter(), AverageMeter()

    for images, _ in loader:
        images = images.to(device)
        noisy, _ = add_gaussian_noise(images, sigma, generator=generator)
        denoised = model(noisy)
        n = images.size(0)

        loss_meter.update(criterion(denoised, images).item(), n)
        out_psnr.update(psnr(denoised, images).mean().item(), n)
        out_ssim.update(ssim(denoised, images).mean().item(), n)
        noisy_psnr.update(psnr(noisy, images).mean().item(), n)
        noisy_ssim.update(ssim(noisy, images).mean().item(), n)

    return {
        "loss": loss_meter.avg,
        "psnr": out_psnr.avg,
        "ssim": out_ssim.avg,
        "noisy_psnr": noisy_psnr.avg,
        "noisy_ssim": noisy_ssim.avg,
    }


@torch.no_grad()
def save_triptych(model, loader, device, sigma, path, n: int = 8) -> None:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(VAL_NOISE_SEED)
    images, _ = next(iter(loader))
    images = images[:n].to(device)
    noisy, _ = add_gaussian_noise(images, sigma, generator=generator)
    denoised = model(noisy)
    plot_image_rows(
        [("clean", images.cpu()), (f"noisy s={sigma}", noisy.cpu()), ("denoised", denoised.cpu())],
        path,
        title="validation denoising",
    )


def run_training(
    model_name: str = "unet",
    dataset: str = "cifar10",
    sigma_range: tuple[float, float] = (0.05, 0.25),
    val_sigma: float = 0.15,
    base_channels: int = 32,
    predict_residual: bool = False,
    loss: str = "l1",
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    val_split: float = 0.05,
    augment: bool = True,
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
    out_path = Path(out_dir or f"outputs/{dataset}_{model_name}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, info = get_datasets(
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

    model = build_model(model_name, info["channels"], base_channels, predict_residual).to(dev)
    # L1 is the usual default for restoration: it is less willing than L2 to
    # hedge with a blur when it is unsure.
    criterion = nn.L1Loss() if loss == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    if verbose:
        print(f"device      : {dev}")
        print(f"model       : {model_name} ({count_parameters(model):,} parameters)")
        print(f"noise       : sigma ~ U({sigma_range[0]}, {sigma_range[1]}) during training")
        print(f"train / val : {len(train_ds)} / {len(val_ds)}  loss={loss}")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "noisy_psnr": [],
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
            noisy, _ = add_gaussian_noise(images, sigma_range)

            optimizer.zero_grad(set_to_none=True)
            denoised = model(noisy)
            batch_loss = criterion(denoised, images)
            batch_loss.backward()
            optimizer.step()

            loss_meter.update(batch_loss.item(), images.size(0))
            batches.set_postfix(loss=f"{loss_meter.avg:.5f}")
        scheduler.step()

        val = evaluate_split(model, val_loader, criterion, dev, val_sigma)
        history["train_loss"].append(loss_meter.avg)
        history["val_loss"].append(val["loss"])
        history["val_psnr"].append(val["psnr"])
        history["val_ssim"].append(val["ssim"])
        history["noisy_psnr"].append(val["noisy_psnr"])

        if val["psnr"] >= best_psnr:
            best_psnr, best_epoch = val["psnr"], epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "dataset": dataset,
                    "in_channels": info["channels"],
                    "base_channels": base_channels,
                    "predict_residual": predict_residual,
                    "sigma_range": list(sigma_range),
                    "epoch": epoch,
                    "val_psnr": val["psnr"],
                },
                ckpt_path,
            )

        if verbose:
            gain = val["psnr"] - val["noisy_psnr"]
            print(
                f"epoch {epoch:2d}/{epochs}  train loss {loss_meter.avg:.5f}  |  "
                f"val PSNR {val['psnr']:.2f} dB (noisy {val['noisy_psnr']:.2f}, "
                f"{gain:+.2f})  SSIM {val['ssim']:.4f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss", "psnr", "ssim"))
        save_triptych(model, val_loader, dev, val_sigma, out_path / "denoising_val.png")

    summary = {
        "model": model_name,
        "dataset": dataset,
        "sigma_range": list(sigma_range),
        "val_sigma": val_sigma,
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
    parser = argparse.ArgumentParser(description="Train a denoising U-Net.")
    parser.add_argument("--model", default="unet", choices=sorted(MODELS))
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument(
        "--sigma-range",
        type=float,
        nargs=2,
        default=[0.05, 0.25],
        metavar=("LOW", "HIGH"),
        help="noise levels sampled during training, in [0, 1] pixel units",
    )
    parser.add_argument("--val-sigma", type=float, default=0.15)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--predict-residual",
        action="store_true",
        help="learn the noise and subtract it, instead of predicting the clean image",
    )
    parser.add_argument("--loss", default="l1", choices=["l1", "l2"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--train-subset", type=int, default=10000, help="cap the training images (0 = all)"
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
        model_name=args.model,
        dataset=args.dataset,
        sigma_range=tuple(args.sigma_range),
        val_sigma=args.val_sigma,
        base_channels=args.base_channels,
        predict_residual=args.predict_residual,
        loss=args.loss,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
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
