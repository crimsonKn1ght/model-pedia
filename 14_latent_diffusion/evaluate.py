"""Evaluate latent diffusion, and price it against pixel-space diffusion.

    python evaluate.py --checkpoint outputs/fashion-mnist_z4x2/best.pt
    python evaluate.py --checkpoint ... --pixel-checkpoint ../12_diffusion/outputs/fashion-mnist_cosine_t200/best.pt

Three numbers matter here and only one of them is FID.

**The reconstruction ceiling.** Real test images are encoded and decoded with no diffusion
at all, and scored. Nothing the second stage does can beat that number, because every
sample it produces goes through the same decoder. If the ceiling is poor, the model is
capped by stage one and no amount of diffusion training will help - which is the failure
mode that makes latent diffusion look mysteriously bad when the first stage was
under-trained.

**Sample quality**, as everywhere else: FID, KID, precision and recall.

**Cost**, against the pixel-space model from project 12 on the same dataset. The U-Net
here denoises a 4x8x8 tensor instead of a 1x32x32 image, so each of its ``T`` steps is
cheaper; pass ``--pixel-checkpoint`` and the table reports both models' milliseconds per
image and peak memory side by side. That comparison is the entire reason latent diffusion
exists, and it is the one thing a FID table alone will not show you.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

import torch

import _paths  # noqa: F401
from data import feature_net_cache, get_splits, make_loader
from model import build_diffusion, build_model
from autoencoder import load_autoencoder
from utils import (
    features_from_loader,
    features_from_sampler,
    generative_metrics,
    get_device,
    load_or_train_feature_net,
    plot_bars,
    plot_image_grid,
    plot_image_rows,
    psnr,
    save_json,
    set_seed,
)


def _timed_samples(feature_net, sample_fn, total, batch, device):
    started = time.time()
    features = features_from_sampler(feature_net, sample_fn, total, batch, device)
    return features, time.time() - started


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    first_stage, ae_config = load_autoencoder(ckpt["autoencoder"], dev)
    latent_scale = ckpt["latent_scale"]

    train_set, _, test_set, info = get_splits(
        ckpt["dataset"], root=args.data_root, image_size=ckpt["image_size"],
        seed=args.seed, synthetic=args.smoke_test,
    )
    train_loader = make_loader(train_set, args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, num_workers=args.num_workers)

    model = build_model(ckpt["latent_channels"], ckpt["base_channels"], ckpt["attention"]).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    diffusion = build_diffusion(ckpt["steps"], ckpt["schedule"], ckpt["prediction"]).to(dev)
    grid = ckpt["latent_grid"]
    latent_shape = (ckpt["latent_channels"], grid, grid)

    @torch.no_grad()
    def sample_images(n: int) -> torch.Tensor:
        latents = diffusion.sample(model, (n, *latent_shape), dev)
        return first_stage.decode((latents * 2.0 - 1.0) * latent_scale)

    feature_net, feature_kind = load_or_train_feature_net(
        None if args.smoke_test
        else feature_net_cache(args.data_root, ckpt["dataset"], ckpt["image_size"]),
        train_loader, ckpt["in_channels"], info["classes"], dev,
        epochs=1 if args.smoke_test else 2, verbose=verbose,
    )
    real_features = features_from_loader(feature_net, test_loader, dev, limit=args.fid_samples)

    # The ceiling: real images through the autoencoder and nothing else.
    ceiling_features, ceiling_psnr = [], []
    with torch.no_grad():
        seen = 0
        for images, _ in test_loader:
            images = images.to(dev)
            mean, _ = first_stage.encode(images)
            decoded = first_stage.decode(mean)
            ceiling_features.append(feature_net.features(decoded).cpu())
            ceiling_psnr.append(psnr(decoded, images).cpu())
            seen += images.size(0)
            if seen >= real_features.size(0):
                break
    ceiling_features = torch.cat(ceiling_features)[: real_features.size(0)]
    ceiling = generative_metrics(
        real_features, ceiling_features, kid_subset_size=min(500, real_features.size(0))
    )

    tracemalloc.start()
    fake_features, latent_seconds = _timed_samples(
        feature_net, sample_images, real_features.size(0), args.sample_batch, dev
    )
    latent_peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    metrics = generative_metrics(
        real_features, fake_features, kid_subset_size=min(500, real_features.size(0))
    )

    result = {
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "latent": f"{ckpt['latent_channels']}x{grid}x{grid}",
        "compression": first_stage.compression,
        "steps": ckpt["steps"],
        "feature_net": feature_kind,
        "unet_parameters": sum(p.numel() for p in model.parameters()),
        "reconstruction_ceiling": {
            "fid": ceiling["fid"],
            "psnr_db": float(torch.cat(ceiling_psnr).mean()),
        },
        "samples": metrics,
        "cost": {
            "latent": {
                "seconds_per_image": latent_seconds / max(real_features.size(0), 1),
                "peak_python_mb": latent_peak,
                "values_denoised": int(ckpt["latent_channels"] * grid * grid),
            }
        },
    }

    if args.pixel_checkpoint:
        pixel_ckpt = torch.load(args.pixel_checkpoint, map_location=dev, weights_only=True)
        pixel_model = build_model(
            pixel_ckpt["in_channels"], pixel_ckpt["base_channels"], pixel_ckpt["attention"]
        ).to(dev)
        pixel_model.load_state_dict(pixel_ckpt["model_state"])
        pixel_model.eval()
        pixel_diffusion = build_diffusion(
            pixel_ckpt["steps"], pixel_ckpt["schedule"], pixel_ckpt["prediction"]
        ).to(dev)
        pixel_shape = (pixel_ckpt["in_channels"], pixel_ckpt["image_size"],
                       pixel_ckpt["image_size"])

        tracemalloc.start()
        pixel_features, pixel_seconds = _timed_samples(
            feature_net,
            lambda n: pixel_diffusion.sample(pixel_model, (n, *pixel_shape), dev),
            real_features.size(0), args.sample_batch, dev,
        )
        pixel_peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        pixel_metrics = generative_metrics(
            real_features, pixel_features, kid_subset_size=min(500, real_features.size(0))
        )
        result["pixel_space"] = {
            "checkpoint": str(args.pixel_checkpoint),
            "steps": pixel_ckpt["steps"],
            "unet_parameters": sum(p.numel() for p in pixel_model.parameters()),
            "samples": pixel_metrics,
            "seconds_per_image": pixel_seconds / max(real_features.size(0), 1),
            "peak_python_mb": pixel_peak,
            "values_denoised": int(pixel_ckpt["in_channels"] * pixel_ckpt["image_size"] ** 2),
        }
        result["speedup"] = pixel_seconds / max(latent_seconds, 1e-9)

    out_dir = Path(args.checkpoint).parent
    plot_image_grid(sample_images(64).cpu(), out_dir / "samples.png", columns=8,
                    title=f"latent diffusion, {ckpt['steps']} steps, FID {metrics['fid']:.1f}")
    images, _ = next(iter(test_loader))
    images = images[:8].to(dev)
    with torch.no_grad():
        mean, _ = first_stage.encode(images)
        decoded = first_stage.decode(mean)
    plot_image_rows(
        [("real", images.cpu()), ("through the latent", decoded.cpu()),
         ("sampled", sample_images(8).cpu())],
        out_dir / "ceiling.png",
        title="row 2 is the ceiling row 3 cannot beat",
    )
    series = {"FID": [metrics["fid"], ceiling["fid"]]}
    labels = ["latent diffusion", "autoencoder ceiling"]
    if "pixel_space" in result:
        labels.append("pixel diffusion")
        series["FID"].append(result["pixel_space"]["samples"]["fid"])
    plot_bars(labels, series, out_dir / "comparison.png", ylabel="FID",
              title="quality against the ceiling and the pixel-space model")

    save_json(result, out_dir / "metrics.json")

    if verbose:
        cost = result["cost"]["latent"]
        print(f"\nmodel    : latent {result['latent']} ({result['compression']:.0f}x fewer "
              f"values), {ckpt['steps']} steps, epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {real_features.size(0)} images\n")
        print(f"FID                   {metrics['fid']:9.3f}")
        print(f"KID                   {metrics['kid']:+9.5f} +- {metrics['kid_std']:.5f}")
        print(f"precision / recall    {metrics['precision']:9.3f} / {metrics['recall']:.3f}")
        print(f"autoencoder ceiling   {ceiling['fid']:9.3f} FID at "
              f"{result['reconstruction_ceiling']['psnr_db']:.2f} dB PSNR")
        print("nothing the diffusion model does can beat the ceiling row.\n")
        print(f"{'':22s}{'values':>9s} {'ms/image':>10s} {'FID':>9s}")
        print(f"{'latent diffusion':22s}{cost['values_denoised']:9d} "
              f"{cost['seconds_per_image'] * 1000:10.0f} {metrics['fid']:9.3f}")
        if "pixel_space" in result:
            pixel = result["pixel_space"]
            print(f"{'pixel diffusion':22s}{pixel['values_denoised']:9d} "
                  f"{pixel['seconds_per_image'] * 1000:10.0f} "
                  f"{pixel['samples']['fid']:9.3f}")
            print(f"\nsampling speedup: {result['speedup']:.2f}x")
        else:
            print("\npass --pixel-checkpoint to add the pixel-space comparison row.")
        print(f"\nfigures -> {out_dir}/samples.png, ceiling.png, comparison.png")
        print(f"metrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate latent diffusion.")
    parser.add_argument("--checkpoint", default="outputs/fashion-mnist_z4x2/best.pt")
    parser.add_argument("--pixel-checkpoint", default=None,
                        help="a project 12 checkpoint on the same dataset, for the cost table")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--fid-samples", type=int, default=500)
    parser.add_argument("--sample-batch", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on generated data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_workers = 0
        args.fid_samples = 64
        args.sample_batch = 32
    run_evaluation(args)


if __name__ == "__main__":
    main()
