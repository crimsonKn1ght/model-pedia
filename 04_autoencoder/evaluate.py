"""Evaluate a trained autoencoder on the test split.

    python evaluate.py --checkpoint outputs/fashion-mnist_latent32/best.pt

Reports reconstruction MSE / PSNR / SSIM, and then asks the more interesting
question: is the code actually *useful*? A 1-nearest-neighbour classifier run
on the latent vectors is compared against the same classifier run on raw
pixels. The latent is 32x smaller; if it classifies about as well, the encoder
kept the structure and threw away the redundancy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import dataset_info, get_dataloaders, get_datasets
from model import build_model
from utils import (
    AverageMeter,
    get_device,
    plot_image_rows,
    psnr,
    save_json,
    set_seed,
    ssim,
)


@torch.no_grad()
def reconstruction_metrics(model, loader, device) -> dict:
    model.eval()
    mse_meter, psnr_meter, ssim_meter = AverageMeter(), AverageMeter(), AverageMeter()
    mse = nn.MSELoss()
    for images, _ in loader:
        images = images.to(device)
        recon = model(images)
        n = images.size(0)
        mse_meter.update(mse(recon, images).item(), n)
        psnr_meter.update(psnr(recon, images).mean().item(), n)
        ssim_meter.update(ssim(recon, images).mean().item(), n)
    return {"mse": mse_meter.avg, "psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


@torch.no_grad()
def encode_split(model, loader, device, limit: int | None = None):
    """Return ``(latents, flattened_pixels, labels)`` for up to ``limit`` images."""
    model.eval()
    latents, pixels, labels = [], [], []
    seen = 0
    for images, targets in loader:
        latents.append(model.encode(images.to(device)).cpu())
        pixels.append(images.flatten(1))
        labels.append(targets)
        seen += images.size(0)
        if limit is not None and seen >= limit:
            break
    latents, pixels, labels = torch.cat(latents), torch.cat(pixels), torch.cat(labels)
    if limit is not None:
        latents, pixels, labels = latents[:limit], pixels[:limit], labels[:limit]
    return latents, pixels, labels


def knn_accuracy(
    reference: torch.Tensor,
    reference_labels: torch.Tensor,
    query: torch.Tensor,
    query_labels: torch.Tensor,
    chunk: int = 512,
) -> float:
    """1-nearest-neighbour accuracy, computed in chunks to bound memory."""
    correct = 0
    for start in range(0, query.size(0), chunk):
        block = query[start : start + chunk]
        nearest = torch.cdist(block, reference).argmin(dim=1)
        correct += (reference_labels[nearest] == query_labels[start : start + chunk]).sum().item()
    return correct / max(query.size(0), 1)


def plot_latent_pca(latents: torch.Tensor, labels: torch.Tensor, class_names, path) -> float:
    """Scatter the first two principal components of the latent space.

    Returns the fraction of variance those two components explain, which is the
    honest caveat to attach to any such picture.
    """
    centered = latents - latents.mean(dim=0, keepdim=True)
    _, singular, components = torch.pca_lowrank(centered, q=min(8, latents.size(1)))
    projected = centered @ components[:, :2]

    variances = singular**2
    explained = float(variances[:2].sum() / variances.sum().clamp_min(1e-12))

    fig, ax = plt.subplots(figsize=(7, 6))
    for class_idx in sorted(set(labels.tolist())):
        mask = labels == class_idx
        name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
        ax.scatter(
            projected[mask, 0], projected[mask, 1], s=6, alpha=0.6, label=name
        )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"latent space, first 2 PCs ({explained:.0%} of variance)")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return explained


@torch.no_grad()
def plot_interpolation(model, loader, device, path, pairs: int = 4, steps: int = 8) -> None:
    """Decode a straight line between the codes of two test images.

    A plain autoencoder is not a generative model - nothing forces the space
    between two codes to decode to anything sensible - so the midpoints are
    usually a soft blend rather than a plausible new image. Seeing that is the
    point; it is exactly the gap a VAE is designed to close.
    """
    model.eval()
    images, _ = next(iter(loader))
    if images.size(0) < 2 * pairs:
        pairs = max(images.size(0) // 2, 1)

    left = images[:pairs].to(device)
    right = images[pairs : 2 * pairs].to(device)
    z_left, z_right = model.encode(left), model.encode(right)

    rows = []
    weights = torch.linspace(0, 1, steps, device=device)
    for index in range(pairs):
        codes = torch.stack([(1 - w) * z_left[index] + w * z_right[index] for w in weights])
        rows.append((f"pair {index + 1}", model.decode(codes).cpu()))

    plot_image_rows(
        rows,
        path,
        title="decoding a straight line between two codes",
        column_labels=[f"{w:.2f}" for w in weights.tolist()],
    )


@torch.no_grad()
def plot_reconstructions(model, loader, device, path, n: int = 10) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)
    recon = model(images)
    error = (images - recon).abs()
    plot_image_rows(
        [
            ("original", images.cpu()),
            ("reconstruction", recon.cpu()),
            ("|error|", (error / error.amax().clamp_min(1e-8)).cpu()),
        ],
        path,
        title="test reconstructions (error row is rescaled to [0, 1])",
    )


def run_evaluation(
    checkpoint: str,
    data_root: str = "data",
    batch_size: int = 256,
    num_workers: int = 2,
    device: str = "auto",
    seed: int = 0,
    knn_reference: int = 5000,
    knn_query: int = 2000,
    synthetic: bool = False,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    dev = get_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=True)
    dataset = ckpt["dataset"]

    model = build_model(ckpt["in_channels"], ckpt["latent_dim"], ckpt["base_channels"]).to(dev)
    model.load_state_dict(ckpt["model_state"])

    train_ds, val_ds, test_ds, info = get_datasets(
        dataset, root=data_root, seed=seed, synthetic=synthetic
    )
    train_loader, _, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=num_workers
    )

    metrics = reconstruction_metrics(model, test_loader, dev)

    train_latents, train_pixels, train_labels = encode_split(model, train_loader, dev, knn_reference)
    test_latents, test_pixels, test_labels = encode_split(model, test_loader, dev, knn_query)
    latent_knn = knn_accuracy(train_latents, train_labels, test_latents, test_labels)
    pixel_knn = knn_accuracy(train_pixels, train_labels, test_pixels, test_labels)

    out_dir = Path(checkpoint).parent
    plot_reconstructions(model, test_loader, dev, out_dir / "reconstructions.png")
    plot_interpolation(model, test_loader, dev, out_dir / "interpolation.png")
    explained = plot_latent_pca(
        test_latents, test_labels, info["classes"], out_dir / "latent_pca.png"
    )

    result = {
        "dataset": dataset,
        "latent_dim": ckpt["latent_dim"],
        "compression_ratio": model.compression_ratio(),
        "checkpoint": str(checkpoint),
        "test_mse": metrics["mse"],
        "test_psnr_db": metrics["psnr"],
        "test_ssim": metrics["ssim"],
        "latent_1nn_accuracy": latent_knn,
        "pixel_1nn_accuracy": pixel_knn,
        "knn_reference_images": int(train_latents.size(0)),
        "knn_query_images": int(test_latents.size(0)),
        "latent_pca_explained_variance": explained,
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        numbers = ckpt["in_channels"] * 32 * 32
        print(f"dataset       : {dataset}")
        print(
            f"bottleneck    : {numbers} numbers -> {ckpt['latent_dim']} "
            f"({model.compression_ratio():.1f}x compression)"
        )
        print(f"test MSE      : {metrics['mse']:.6f}")
        print(f"test PSNR     : {metrics['psnr']:.2f} dB")
        print(f"test SSIM     : {metrics['ssim']:.4f}")
        print(
            f"\n1-NN accuracy on {test_latents.size(0)} test images "
            f"against {train_latents.size(0)} train images:"
        )
        print(f"  latent codes ({ckpt['latent_dim']} numbers) : {latent_knn:.4f}")
        print(f"  raw pixels   ({numbers} numbers) : {pixel_knn:.4f}")
        print(f"\nreconstructions -> {out_dir / 'reconstructions.png'}")
        print(f"interpolation   -> {out_dir / 'interpolation.png'}")
        print(f"latent PCA      -> {out_dir / 'latent_pca.png'}")
        print(f"metrics         -> {out_dir / 'metrics.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an autoencoder checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/fashion-mnist_latent32/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knn-reference", type=int, default=5000)
    parser.add_argument("--knn-query", type=int, default=2000)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on random tensors")
    args = parser.parse_args()

    run_evaluation(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        device=args.device,
        seed=args.seed,
        knn_reference=args.knn_reference,
        knn_query=args.knn_query,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
