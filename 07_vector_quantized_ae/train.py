"""Stage one: train the VQ-VAE (encoder, codebook, decoder).

    python train.py --dataset fashion-mnist --epochs 10
    python train.py --num-codes 512 --no-ema

There is no prior here and no sampling - this stage only learns to turn an image into
a grid of integers and back. Generation is stage two (``train_prior.py``).

The number to watch is **perplexity**, printed every epoch. It is ``exp`` of the
entropy of the code histogram, so it equals the codebook size when every entry is
used equally and 1 when the model has collapsed onto a single code. A 512-entry
codebook running at perplexity 12 is a 12-entry codebook that took 512 entries' worth
of memory, and no reconstruction metric will tell you that has happened.

Checkpoints are selected on validation PSNR rather than on the loss, because the loss
includes the commitment and codebook terms, which are about keeping the two halves of
the model in agreement rather than about reconstruction quality.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import DATASETS, get_splits, make_loader
from model import build_model
from utils import (
    AverageMeter,
    codebook_usage,
    count_parameters,
    get_device,
    plot_bars,
    plot_curves,
    plot_image_rows,
    psnr,
    save_json,
    set_seed,
    ssim,
)


@torch.no_grad()
def evaluate_split(model, loader, device) -> dict:
    """Reconstruction quality plus codebook statistics over a whole split."""
    model.eval()
    meters = {key: AverageMeter() for key in ("loss", "reconstruction", "psnr", "ssim")}
    all_indices = []

    for images, _ in loader:
        images = images.to(device)
        outputs = model(images)
        losses = model.loss(images, outputs)
        n = images.size(0)
        meters["loss"].update(losses["loss"].item(), n)
        meters["reconstruction"].update(losses["reconstruction"].item(), n)
        meters["psnr"].update(psnr(outputs["reconstruction"], images).mean().item(), n)
        meters["ssim"].update(ssim(outputs["reconstruction"], images).mean().item(), n)
        all_indices.append(outputs["indices"].flatten().cpu())

    usage = codebook_usage(torch.cat(all_indices), model.num_codes)
    return {**{key: meter.avg for key, meter in meters.items()}, "usage": usage}


@torch.no_grad()
def save_figures(model, loader, device, out_path: Path, usage: dict, n: int = 8) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)
    outputs = model(images)

    # The code map, drawn as an image: each pixel is one integer, scaled to [0, 1].
    codes = outputs["indices"].float().unsqueeze(1) / max(model.num_codes - 1, 1)
    codes = torch.nn.functional.interpolate(codes, size=images.shape[-2:], mode="nearest")
    plot_image_rows(
        [
            ("input", images.cpu()),
            (f"codes {model.grid}x{model.grid}", codes.expand(-1, images.size(1), -1, -1).cpu()),
            ("reconstruction", outputs["reconstruction"].cpu()),
        ],
        out_path / "reconstructions_val.png",
        title=f"one integer per code position ({model.compression:.0f}x fewer numbers)",
    )
    plot_bars(
        [str(i) if model.num_codes <= 32 else "" for i in range(model.num_codes)],
        {"times used": usage["histogram"]},
        out_path / "codebook_usage.png",
        ylabel="count",
        title=f"codebook histogram: {usage['codes_used']}/{usage['num_codes']} used, "
              f"perplexity {usage['perplexity']:.1f}",
    )


def run_training(
    dataset: str = "fashion-mnist",
    image_size: int | None = None,
    base_channels: int = 64,
    code_dim: int = 64,
    num_codes: int = 128,
    commitment: float = 0.25,
    ema: bool = True,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 2e-3,
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
    out_path = Path(out_dir or f"outputs/{dataset}_k{num_codes}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        dataset, root=data_root, image_size=image_size, val_split=val_split,
        train_subset=train_subset, seed=seed, synthetic=synthetic,
    )
    train_loader = make_loader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_set, batch_size, num_workers=num_workers)

    model = build_model(
        info["channels"], info["size"], base_channels, code_dim, num_codes, commitment, ema
    ).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    config = {
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": info["classes"],
        "image_size": info["size"],
        "base_channels": base_channels,
        "code_dim": code_dim,
        "num_codes": num_codes,
        "commitment": commitment,
        "ema": ema,
        "grid": model.grid,
    }

    if verbose:
        print(f"device     : {dev}")
        print(f"model      : {count_parameters(model):,} trainable parameters "
              f"({'EMA' if ema else 'loss-based'} codebook)")
        print(f"codes      : {num_codes} entries of {code_dim}d on a "
              f"{model.grid}x{model.grid} grid, {model.compression:.0f}x compression")
        print(f"data       : {len(train_set)} train / {len(val_set)} val at "
              f"{info['channels']}x{info['size']}x{info['size']}\n")

    history = {
        "train_loss": [], "train_reconstruction": [],
        "val_loss": [], "val_psnr": [], "val_ssim": [], "val_perplexity": [],
    }
    best_psnr, best_epoch = -float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        meters = {key: AverageMeter() for key in ("loss", "reconstruction", "commitment")}
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for images, _ in batches:
            images = images.to(dev)
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(images, model(images))
            losses["loss"].backward()
            optimizer.step()
            for key, meter in meters.items():
                meter.update(losses[key].item(), images.size(0))
            batches.set_postfix(loss=f"{meters['loss'].avg:.4f}")
        scheduler.step()

        val = evaluate_split(model, val_loader, dev)
        history["train_loss"].append(meters["loss"].avg)
        history["train_reconstruction"].append(meters["reconstruction"].avg)
        history["val_loss"].append(val["loss"])
        history["val_psnr"].append(val["psnr"])
        history["val_ssim"].append(val["ssim"])
        history["val_perplexity"].append(val["usage"]["perplexity"])

        if val["psnr"] >= best_psnr:
            best_psnr, best_epoch = val["psnr"], epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_psnr": val["psnr"],
                 **config},
                ckpt_path,
            )

        if verbose:
            usage = val["usage"]
            print(
                f"epoch {epoch:2d}/{epochs}  train loss {meters['loss'].avg:.4f}  |  "
                f"val PSNR {val['psnr']:6.2f} dB  SSIM {val['ssim']:.4f}  "
                f"codes {usage['codes_used']:3d}/{usage['num_codes']} "
                f"perplexity {usage['perplexity']:6.1f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss", "psnr", "perplexity"))
        save_figures(model, val_loader, dev, out_path, evaluate_split(model, val_loader, dev)["usage"])

    summary = {
        **config,
        "parameters": count_parameters(model),
        "compression": model.compression,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_psnr": best_psnr,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose and epochs:
        print(f"\nbest val PSNR {best_psnr:.2f} dB at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python train_prior.py --checkpoint {ckpt_path}   (to generate)")
        print(f"       then: python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a VQ-VAE.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--code-dim", type=int, default=64)
    parser.add_argument("--num-codes", type=int, default=128, help="codebook size")
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument(
        "--no-ema", action="store_true",
        help="train the codebook with a loss term instead of an exponential moving average",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
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
        dataset=args.dataset,
        image_size=args.image_size,
        base_channels=args.base_channels,
        code_dim=args.code_dim,
        num_codes=args.num_codes,
        commitment=args.commitment,
        ema=not args.no_ema,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=32 if args.smoke_test else args.batch_size,
        lr=args.lr,
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
