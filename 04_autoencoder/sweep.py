"""Sweep the bottleneck size and plot quality against compression.

    python sweep.py --dataset fashion-mnist --latent-dims 2 8 16 32 64 128

Trains one autoencoder per latent size and draws PSNR/SSIM against the
compression ratio. The curve has the shape you would hope for: steep at the
left, where every extra dimension buys real detail, and flat at the right,
where the images simply do not contain that much information.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from data import DATASETS, get_dataloaders, get_datasets
from evaluate import run_evaluation
from model import build_model
from train import run_training
from utils import get_device, plot_image_rows, save_json, set_seed


@torch.no_grad()
def compare_reconstructions(checkpoints: list[tuple[int, str]], args, path) -> None:
    """One figure: the same test images decoded through every bottleneck size."""
    set_seed(args.seed)
    dev = get_device(args.device)
    train_ds, val_ds, test_ds, _ = get_datasets(
        args.dataset, root=args.data_root, seed=args.seed, synthetic=args.smoke_test
    )
    _, _, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.columns, num_workers=0
    )
    images, _ = next(iter(test_loader))
    images = images[: args.columns].to(dev)

    rows = [("original", images.cpu())]
    for latent_dim, checkpoint in checkpoints:
        ckpt = torch.load(checkpoint, map_location=dev, weights_only=True)
        model = build_model(ckpt["in_channels"], ckpt["latent_dim"], ckpt["base_channels"]).to(dev)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        rows.append((f"latent {latent_dim}", model(images).cpu()))

    plot_image_rows(rows, path, title="the same images through different bottlenecks")


def plot_sweep(rows: list[dict], path) -> None:
    latent_dims = [row["latent_dim"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(latent_dims, [row["test_psnr_db"] for row in rows], marker="o")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("latent dimensions")
    axes[0].set_ylabel("test PSNR (dB)")
    axes[0].set_title("reconstruction quality vs bottleneck")
    axes[0].grid(alpha=0.3)

    axes[1].plot(latent_dims, [row["test_ssim"] for row in rows], marker="o", label="SSIM")
    axes[1].plot(
        latent_dims,
        [row["latent_1nn_accuracy"] for row in rows],
        marker="s",
        label="latent 1-NN accuracy",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("latent dimensions")
    axes[1].set_title("structure kept in the code")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the autoencoder bottleneck size.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[2, 8, 16, 32, 64, 128])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--columns", type=int, default=8, help="images in the comparison figure")
    parser.add_argument("--smoke-test", action="store_true", help="run on random tensors")
    args = parser.parse_args()

    workers = 0 if args.smoke_test else args.num_workers
    out_root = Path(args.out_dir or f"outputs/sweep_{args.dataset}")
    rows, checkpoints = [], []

    for latent_dim in args.latent_dims:
        print(f"\n=== latent {latent_dim} ===")
        summary = run_training(
            dataset=args.dataset,
            latent_dim=latent_dim,
            loss=args.loss,
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_subset=args.train_subset or None,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=str(out_root / f"latent{latent_dim}"),
            synthetic=args.smoke_test,
        )
        result = run_evaluation(
            checkpoint=summary["checkpoint"],
            data_root=args.data_root,
            num_workers=workers,
            device=args.device,
            seed=args.seed,
            synthetic=args.smoke_test,
            verbose=False,
        )
        rows.append(
            {
                "latent_dim": latent_dim,
                "compression_ratio": result["compression_ratio"],
                "train_seconds": summary["train_seconds"],
                "test_psnr_db": result["test_psnr_db"],
                "test_ssim": result["test_ssim"],
                "latent_1nn_accuracy": result["latent_1nn_accuracy"],
                "pixel_1nn_accuracy": result["pixel_1nn_accuracy"],
            }
        )
        checkpoints.append((latent_dim, summary["checkpoint"]))

    plot_sweep(rows, out_root / "sweep.png")
    compare_reconstructions(checkpoints, args, out_root / "bottleneck_comparison.png")
    save_json({"dataset": args.dataset, "epochs": args.epochs, "results": rows}, out_root / "sweep.json")

    print(f"\n=== {args.dataset}, {args.epochs} epochs each ===")
    header = (
        f"{'latent':>7s} {'ratio':>8s} {'train s':>8s} {'PSNR dB':>9s} "
        f"{'SSIM':>7s} {'1-NN':>7s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['latent_dim']:7d} {row['compression_ratio']:7.1f}x {row['train_seconds']:8.1f} "
            f"{row['test_psnr_db']:9.2f} {row['test_ssim']:7.4f} {row['latent_1nn_accuracy']:7.4f}"
        )
    print(f"\n1-NN on raw pixels for reference: {rows[0]['pixel_1nn_accuracy']:.4f}")
    print(f"figures -> {out_root / 'sweep.png'}, {out_root / 'bottleneck_comparison.png'}")


if __name__ == "__main__":
    main()
