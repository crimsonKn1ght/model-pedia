"""Train Pix2Pix.

    python train.py --dataset shapes --epochs 20
    python train.py --l1-weight 0        # GAN only: sharp, and free to invent

Two losses fight, and the balance is the method. The L1 term keeps the output aligned with
the target; the adversarial term keeps it sharp. Neither alone works - see
``compare.py --study l1_weight``.

Because the task is paired, this project has something the unconditional GANs do not: a
**ground-truth target**, so quality can be measured directly with L1, PSNR and SSIM rather
than only through a feature network. Those are what select the checkpoint. FID is still
reported, because pixel metrics reward the blur that the adversarial term exists to remove,
and a run that improves L1 while getting worse to look at is a real outcome worth seeing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import DATASETS, get_splits, make_loader
from model import GAN_LOSSES, build_models, discriminator_loss, generator_loss
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


@torch.no_grad()
def evaluate_split(generator, loader, device) -> dict:
    generator.eval()
    meters = {key: AverageMeter() for key in ("l1", "psnr", "ssim")}
    for source, target in loader:
        source, target = source.to(device), target.to(device)
        generated = generator(source)
        n = source.size(0)
        meters["l1"].update((generated - target).abs().mean().item(), n)
        meters["psnr"].update(psnr(generated, target).mean().item(), n)
        meters["ssim"].update(ssim(generated, target).mean().item(), n)
    return {key: meter.avg for key, meter in meters.items()}


def run_training(
    dataset: str = "shapes",
    image_size: int | None = None,
    width: int = 64,
    levels: int = 4,
    skips: bool = True,
    patch_layers: int = 3,
    whole_image: bool = False,
    gan_loss: str = "bce",
    l1_weight: float = 100.0,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 2e-4,
    beta1: float = 0.5,
    train_size: int = 3000,
    val_size: int = 200,
    test_size: int = 500,
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
    out_path = Path(out_dir or f"outputs/{dataset}_l1{l1_weight:g}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        dataset, root=data_root, image_size=image_size, train_size=train_size,
        val_size=val_size, test_size=test_size, seed=seed, synthetic=synthetic,
    )
    size = info["size"]
    train_loader = make_loader(train_set, size, batch_size, shuffle=True, augment=True,
                               num_workers=num_workers)
    val_loader = make_loader(val_set, size, batch_size, num_workers=num_workers)

    generator, discriminator = build_models(
        info["channels"], info["channels"], width, levels, skips, patch_layers, whole_image
    )
    generator, discriminator = generator.to(dev), discriminator.to(dev)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    config = {
        "dataset": dataset,
        "in_channels": info["channels"],
        "image_size": size,
        "width": width,
        "levels": levels,
        "skips": skips,
        "patch_layers": patch_layers,
        "whole_image": whole_image,
        "gan_loss": gan_loss,
        "l1_weight": l1_weight,
    }
    if verbose:
        print(f"device        : {dev}")
        print(f"generator     : {count_parameters(generator):,} parameters"
              f"{'' if skips else ' (no skip connections)'}")
        print(f"discriminator : {count_parameters(discriminator):,} parameters, "
              f"{'whole image' if whole_image else f'{patch_layers}-layer PatchGAN'}")
        print(f"losses        : {gan_loss} adversarial + {l1_weight:g} x L1")
        print(f"data          : {len(train_set)} train pairs at "
              f"{info['channels']}x{size}x{size}\n")

    history = {"train_d_loss": [], "train_g_loss": [], "train_l1": [],
               "val_l1": [], "val_psnr": [], "val_ssim": []}
    best_ssim, best_epoch = -float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        meters = {key: AverageMeter() for key in ("d_loss", "g_loss", "l1", "adversarial")}
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False,
                       disable=not verbose)
        for source, target in batches:
            source, target = source.to(dev), target.to(dev)
            n = source.size(0)

            with torch.no_grad():
                generated = generator(source)
            d_loss = discriminator_loss(
                discriminator(source, target), discriminator(source, generated), gan_loss
            )
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            generated = generator(source)
            g_losses = generator_loss(
                discriminator(source, generated), generated, target, gan_loss, l1_weight
            )
            opt_g.zero_grad(set_to_none=True)
            g_losses["loss"].backward()
            opt_g.step()

            meters["d_loss"].update(d_loss.item(), n)
            meters["g_loss"].update(g_losses["loss"].item(), n)
            meters["l1"].update(g_losses["l1"].item(), n)
            meters["adversarial"].update(g_losses["adversarial"].item(), n)
            batches.set_postfix(d=f"{meters['d_loss'].avg:.3f}", l1=f"{meters['l1'].avg:.4f}")

        val = evaluate_split(generator, val_loader, dev)
        history["train_d_loss"].append(meters["d_loss"].avg)
        history["train_g_loss"].append(meters["g_loss"].avg)
        history["train_l1"].append(meters["l1"].avg)
        history["val_l1"].append(val["l1"])
        history["val_psnr"].append(val["psnr"])
        history["val_ssim"].append(val["ssim"])

        if val["ssim"] >= best_ssim:
            best_ssim, best_epoch = val["ssim"], epoch
            torch.save({"generator_state": generator.state_dict(),
                        "discriminator_state": discriminator.state_dict(),
                        "epoch": epoch, "val_ssim": val["ssim"], **config}, ckpt_path)

        if verbose:
            print(f"epoch {epoch:2d}/{epochs}  D {meters['d_loss'].avg:.3f}  "
                  f"adv {meters['adversarial'].avg:.3f}  L1 {meters['l1'].avg:.4f}  |  "
                  f"val L1 {val['l1']:.4f}  PSNR {val['psnr']:5.2f}  SSIM {val['ssim']:.4f}"
                  f"{'  <- best' if epoch == best_epoch else ''}")

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("l1", "psnr", "ssim"))
        generator.eval()
        source, target = next(iter(val_loader))
        source, target = source[:8].to(dev), target[:8].to(dev)
        with torch.no_grad():
            generated = generator(source)
        plot_image_rows(
            [("input", source.cpu()), ("generated", generated.cpu()), ("target", target.cpu())],
            out_path / "translations_val.png",
            title=f"validation translations (L1 weight {l1_weight:g})",
        )

    summary = {
        **config,
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_ssim": best_ssim,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose and epochs:
        print(f"\nbest val SSIM {best_ssim:.4f} at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Pix2Pix.")
    parser.add_argument("--dataset", default="shapes", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--no-skips", action="store_true", help="the control for the U-Net skips")
    parser.add_argument("--patch-layers", type=int, default=3, help="PatchGAN receptive field")
    parser.add_argument("--whole-image", action="store_true", help="one verdict per image")
    parser.add_argument("--gan-loss", default="bce", choices=GAN_LOSSES)
    parser.add_argument("--l1-weight", type=float, default=100.0, help="0 is GAN only")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch on generated pairs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        dataset=args.dataset,
        image_size=args.image_size,
        width=16 if args.smoke_test else args.width,
        levels=2 if args.smoke_test else args.levels,
        skips=not args.no_skips,
        patch_layers=args.patch_layers,
        whole_image=args.whole_image,
        gan_loss=args.gan_loss,
        l1_weight=args.l1_weight,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=8 if args.smoke_test else args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out_dir,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
