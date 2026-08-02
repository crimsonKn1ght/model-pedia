"""Evaluate a denoiser across a range of noise levels.

    python evaluate.py --checkpoint outputs/cifar10_unet/best.pt

Every number is reported next to the do-nothing baseline - the PSNR/SSIM of the
noisy input itself. That baseline is the whole point: a denoiser that returns
its input scores respectably, so "31 dB" means nothing until you know the input
was 24 dB.

The sweep deliberately runs past the training range so you can see where the
model stops generalising.

    python evaluate.py --checkpoint ... --test-dir path/to/BSD68
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data import ImageDirectory, add_gaussian_noise, get_dataloaders, get_datasets
from model import build_model
from utils import AverageMeter, get_device, plot_image_rows, psnr, save_json, set_seed, ssim

EVAL_NOISE_SEED = 7


@torch.no_grad()
def evaluate_at_sigma(model, loader, device, sigma: float) -> dict:
    """Denoised and noisy-input quality at one noise level."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(EVAL_NOISE_SEED)
    out_psnr, out_ssim = AverageMeter(), AverageMeter()
    in_psnr, in_ssim = AverageMeter(), AverageMeter()

    for images, _ in loader:
        images = images.to(device)
        noisy, _ = add_gaussian_noise(images, sigma, generator=generator)
        denoised = model(noisy)
        n = images.size(0)

        out_psnr.update(psnr(denoised, images).mean().item(), n)
        out_ssim.update(ssim(denoised, images).mean().item(), n)
        in_psnr.update(psnr(noisy, images).mean().item(), n)
        in_ssim.update(ssim(noisy, images).mean().item(), n)

    return {
        "sigma": sigma,
        "noisy_psnr_db": in_psnr.avg,
        "noisy_ssim": in_ssim.avg,
        "denoised_psnr_db": out_psnr.avg,
        "denoised_ssim": out_ssim.avg,
        "psnr_gain_db": out_psnr.avg - in_psnr.avg,
    }


def plot_sigma_sweep(rows: list[dict], sigma_range: list[float], path) -> None:
    sigmas = [row["sigma"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(sigmas, [row["noisy_psnr_db"] for row in rows], marker="s", label="noisy input")
    axes[0].plot(sigmas, [row["denoised_psnr_db"] for row in rows], marker="o", label="denoised")
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].plot(sigmas, [row["noisy_ssim"] for row in rows], marker="s", label="noisy input")
    axes[1].plot(sigmas, [row["denoised_ssim"] for row in rows], marker="o", label="denoised")
    axes[1].set_ylabel("SSIM")

    for ax in axes:
        ax.axvspan(
            sigma_range[0],
            sigma_range[1],
            color="tab:green",
            alpha=0.10,
            label="trained noise range",
        )
        ax.set_xlabel("noise sigma")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("denoiser vs the do-nothing baseline")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_examples(model, loader, device, sigmas: list[float], path, n: int = 6) -> None:
    """Clean / noisy / denoised, at a few noise levels, on the same images."""
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)

    rows = [("clean", images.cpu())]
    for sigma in sigmas:
        generator = torch.Generator(device=device).manual_seed(EVAL_NOISE_SEED)
        noisy, _ = add_gaussian_noise(images, sigma, generator=generator)
        rows.append((f"noisy {sigma:g}", noisy.cpu()))
        rows.append((f"denoised {sigma:g}", model(noisy).cpu()))

    plot_image_rows(rows, path, title="denoising at several noise levels")


def _test_loader(ckpt, args, synthetic: bool):
    """Either the dataset's own test split, or a folder of benchmark images."""
    if args.test_dir:
        dataset = ImageDirectory(args.test_dir, channels=ckpt["in_channels"])
        # Benchmark images have different sizes, so they go through one at a time.
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers), len(
            dataset
        )

    train_ds, val_ds, test_ds, _ = get_datasets(
        ckpt["dataset"], root=args.data_root, seed=args.seed, synthetic=synthetic
    )
    _, _, loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )
    return loader, len(test_ds)


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)

    model = build_model(
        ckpt["model_name"],
        ckpt["in_channels"],
        ckpt["base_channels"],
        ckpt["predict_residual"],
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])

    loader, n_images = _test_loader(ckpt, args, args.smoke_test)
    rows = [evaluate_at_sigma(model, loader, dev, sigma) for sigma in args.sigmas]

    out_dir = Path(args.checkpoint).parent
    suffix = f"_{Path(args.test_dir).name}" if args.test_dir else ""
    plot_sigma_sweep(rows, ckpt["sigma_range"], out_dir / f"sigma_sweep{suffix}.png")
    example_sigmas = [s for s in args.sigmas if s in (0.1, 0.25, 0.5)] or args.sigmas[:3]
    plot_examples(model, loader, dev, example_sigmas, out_dir / f"examples{suffix}.png")

    result = {
        "model": ckpt["model_name"],
        "dataset": args.test_dir or ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "trained_sigma_range": ckpt["sigma_range"],
        "test_images": n_images,
        "by_sigma": rows,
    }
    save_json(result, out_dir / f"metrics{suffix}.json")

    if verbose:
        low, high = ckpt["sigma_range"]
        print(f"model      : {ckpt['model_name']} trained on sigma in [{low}, {high}]")
        print(f"test set   : {result['dataset']}  ({n_images} images)\n")
        header = (
            f"{'sigma':>6s} {'noisy PSNR':>11s} {'denoised':>10s} {'gain':>7s} "
            f"{'noisy SSIM':>11s} {'denoised':>10s}"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            marker = " " if low <= row["sigma"] <= high else "*"
            print(
                f"{row['sigma']:6.2f}{marker}{row['noisy_psnr_db']:11.2f} "
                f"{row['denoised_psnr_db']:10.2f} {row['psnr_gain_db']:+7.2f} "
                f"{row['noisy_ssim']:11.4f} {row['denoised_ssim']:10.4f}"
            )
        print("\n* outside the noise range the model was trained on")
        print(f"\nsweep    -> {out_dir / f'sigma_sweep{suffix}.png'}")
        print(f"examples -> {out_dir / f'examples{suffix}.png'}")
        print(f"metrics  -> {out_dir / f'metrics{suffix}.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a denoiser across noise levels.")
    parser.add_argument("--checkpoint", default="outputs/cifar10_unet/best.pt")
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5],
        help="noise levels to evaluate; values outside the trained range are marked",
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="folder of images to use instead of the dataset test split (e.g. BSD68)",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on random tensors")
    args = parser.parse_args()

    if args.smoke_test:
        args.num_workers = 0
    run_evaluation(args)


if __name__ == "__main__":
    main()
