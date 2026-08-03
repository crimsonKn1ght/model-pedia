"""Stage one: train the first-stage autoencoder that defines the latent space.

    python train_autoencoder.py --dataset fashion-mnist --epochs 10

This stage never sees a diffusion process. Its only job is to compress images into a
small spatial tensor that can be decoded back faithfully, because everything the second
stage can possibly achieve is bounded by how good this reconstruction is. That bound is
worth taking seriously: ``evaluate.py`` reports it as the "reconstruction ceiling", the
FID you would get by encoding and decoding real images with no diffusion at all.

At the end of training the latent's standard deviation is measured over the training set
and stored in the checkpoint. Stage two divides by it, because a diffusion process assumes
its input is roughly unit-variance and a latent whose scale drifted would quietly break
the noise schedule.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

import _paths  # noqa: F401
from autoencoder import build_autoencoder
from data import DATASETS, get_splits, make_loader
from utils import (
    AverageMeter,
    count_parameters,
    get_device,
    plot_curves,
    plot_image_rows,
    psnr,
    save_json,
    set_seed,
)


@torch.no_grad()
def evaluate_split(model, loader, device) -> dict:
    model.eval()
    meters = {key: AverageMeter() for key in ("loss", "reconstruction", "psnr")}
    for images, _ in loader:
        images = images.to(device)
        outputs = model(images)
        losses = model.loss(images, outputs)
        n = images.size(0)
        meters["loss"].update(losses["loss"].item(), n)
        meters["reconstruction"].update(losses["reconstruction"].item(), n)
        meters["psnr"].update(psnr(outputs["reconstruction"], images).mean().item(), n)
    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def measure_latent_scale(model, loader, device, batches: int = 20) -> float:
    """Standard deviation of the latent over the data - stage two divides by this."""
    model.eval()
    values = []
    for index, (images, _) in enumerate(loader):
        if index >= batches:
            break
        mean, _ = model.encode(images.to(device))
        values.append(mean.flatten().cpu())
    return float(torch.cat(values).std())


def run_training(
    dataset: str = "fashion-mnist",
    image_size: int | None = None,
    base_channels: int = 32,
    latent_channels: int = 4,
    levels: int = 2,
    kl_weight: float = 1e-6,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 2e-3,
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
    out_path = Path(out_dir or f"outputs/{dataset}_z{latent_channels}x{levels}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        dataset, root=data_root, image_size=image_size, train_subset=train_subset,
        seed=seed, synthetic=synthetic,
    )
    train_loader = make_loader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_set, 256, num_workers=num_workers)

    model = build_autoencoder(
        info["channels"], info["size"], base_channels, latent_channels, levels, kl_weight
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    config = {
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": info["classes"],
        "image_size": info["size"],
        "base_channels": base_channels,
        "latent_channels": latent_channels,
        "levels": levels,
        "kl_weight": kl_weight,
        "grid": model.grid,
    }
    if verbose:
        print(f"device      : {dev}")
        print(f"autoencoder : {count_parameters(model):,} parameters")
        print(f"latent      : {latent_channels}x{model.grid}x{model.grid}, "
              f"{model.compression:.0f}x fewer values than the image")
        print(f"data        : {len(train_set)} train / {len(val_set)} val\n")

    history = {"train_loss": [], "val_loss": [], "val_psnr": []}
    best_psnr, best_epoch = -float("inf"), 0
    ckpt_path = out_path / "autoencoder.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False,
                       disable=not verbose)
        for images, _ in batches:
            images = images.to(dev)
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(images, model(images))
            losses["loss"].backward()
            optimizer.step()
            meter.update(losses["loss"].item(), images.size(0))
            batches.set_postfix(loss=f"{meter.avg:.5f}")
        scheduler.step()

        val = evaluate_split(model, val_loader, dev)
        history["train_loss"].append(meter.avg)
        history["val_loss"].append(val["loss"])
        history["val_psnr"].append(val["psnr"])

        if val["psnr"] >= best_psnr:
            best_psnr, best_epoch = val["psnr"], epoch
            latent_scale = measure_latent_scale(model, train_loader, dev)
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_psnr": val["psnr"], "latent_scale": latent_scale, **config},
                       ckpt_path)

        if verbose:
            print(f"epoch {epoch:2d}/{epochs}  train {meter.avg:.5f}  "
                  f"val PSNR {val['psnr']:6.2f} dB"
                  f"{'  <- best' if epoch == best_epoch else ''}")

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "autoencoder_curves.png", keys=("loss", "psnr"))
        model.eval()
        images, _ = next(iter(val_loader))
        images = images[:8].to(dev)
        with torch.no_grad():
            reconstruction = model(images)["reconstruction"]
        plot_image_rows(
            [("input", images.cpu()), ("through the latent", reconstruction.cpu())],
            out_path / "autoencoder_reconstructions.png",
            title="stage one sets the ceiling: diffusion cannot beat this",
        )

    stored = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    summary = {
        **config,
        "parameters": count_parameters(model),
        "compression": model.compression,
        "latent_scale": stored["latent_scale"],
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_psnr": best_psnr,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "autoencoder_history.json")

    if verbose and epochs:
        print(f"\nbest val PSNR {best_psnr:.2f} dB at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"latent scale {stored['latent_scale']:.4f} (stage two divides by this)")
        print(f"autoencoder -> {ckpt_path}")
        print(f"next        : python train.py --autoencoder {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the first-stage autoencoder.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--levels", type=int, default=2, help="downsampling factor is 2^levels")
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--train-subset", type=int, default=20000)
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
        dataset=args.dataset,
        image_size=args.image_size,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        levels=1 if args.smoke_test else args.levels,
        kl_weight=args.kl_weight,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=32 if args.smoke_test else args.batch_size,
        lr=args.lr,
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
