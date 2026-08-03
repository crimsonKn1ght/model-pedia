"""Evaluate a flow: exact bits per dimension, samples, and the inverse it depends on.

    python evaluate.py --checkpoint outputs/fashion-mnist_h64/best.pt

**Bits per dimension is the headline and it is exact.** No bound, no feature network, no
proxy - the change of variables formula gives ``log p(x)`` and the standard ``+8``
conversion turns it into bits of the discrete data. That makes it the one number in this
half of the repository that is comparable with published results, and the reason to put
this project next to the VAE: project 06 reports an ELBO *bound* in the same units, and
the gap between a bound and the real thing is exactly what you cannot see from inside a
VAE.

The invertibility check is not ceremony. Every likelihood here is only correct if the
network really is a bijection, so the round-trip error is measured and reported. If it
were large, the bits/dim number would be meaningless.

Sample quality is also scored with FID and KID for comparison with projects 08 and 12 on
the same dataset - and the usual finding holds: flows report excellent likelihoods and
produce visibly worse samples than a GAN or a diffusion model. Likelihood and perceptual
quality are different objectives, and this is the cleanest place in the repository to see
them disagree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import DATASETS, feature_net_cache, get_splits, make_loader
from model import build_model
from train import evaluate_split
from utils import (
    features_from_loader,
    features_from_sampler,
    generative_metrics,
    get_device,
    load_or_train_feature_net,
    plot_image_grid,
    plot_image_rows,
    save_json,
    set_seed,
)


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)

    train_set, _, test_set, info = get_splits(
        ckpt["dataset"], root=args.data_root, image_size=ckpt["image_size"],
        seed=args.seed, synthetic=args.smoke_test,
    )
    train_loader = make_loader(train_set, args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, num_workers=args.num_workers)

    model = build_model(
        ckpt["in_channels"], ckpt["image_size"], ckpt["hidden"], ckpt["couplings_per_stage"]
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_bpd = evaluate_split(model, test_loader, dev)

    # Invertibility: the likelihood is only valid if the network is a bijection.
    images, _ = next(iter(test_loader))
    images = images[:16].to(dev)
    with torch.no_grad():
        latent, _ = model(images, dequantise=False)
        round_trip = model.inverse(latent)
    inversion_error = float((round_trip - images).abs().max())

    feature_net, feature_kind = load_or_train_feature_net(
        None if args.smoke_test
        else feature_net_cache(args.data_root, ckpt["dataset"], ckpt["image_size"]),
        train_loader, ckpt["in_channels"], info["classes"], dev,
        epochs=1 if args.smoke_test else 2, verbose=verbose,
    )
    real_features = features_from_loader(feature_net, test_loader, dev, limit=args.fid_samples)
    fake_features = features_from_sampler(
        feature_net, lambda n: model.sample(n, dev, args.temperature),
        real_features.size(0), args.batch_size, dev,
    )
    metrics = generative_metrics(
        real_features, fake_features, kid_subset_size=min(500, real_features.size(0))
    )

    out_dir = Path(args.checkpoint).parent
    plot_image_grid(model.sample(64, dev, args.temperature).cpu(), out_dir / "samples.png",
                    columns=8, title=f"{ckpt['dataset']} samples, temperature {args.temperature:g}")
    plot_image_rows(
        [("input", images[:8].cpu()), ("round trip", round_trip[:8].cpu())],
        out_dir / "invertibility.png",
        title=f"forwards then backwards: max error {inversion_error:.1e}",
    )
    # Lowering the temperature narrows the prior, which is the standard trick for
    # trading sample diversity for sample quality in a flow.
    rows = [(f"T={t:g}", model.sample(8, dev, t).cpu()) for t in (0.5, 0.7, 1.0)]
    plot_image_rows(rows, out_dir / "temperature.png",
                    title="a narrower prior gives cleaner, less varied samples")

    result = {
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "test_images": len(test_set),
        "test_bits_per_dimension": test_bpd,
        "uniform_baseline_bpd": 8.0,
        "bits_saved_per_pixel": 8.0 - test_bpd,
        "exact": True,
        "max_inversion_error": inversion_error,
        "feature_net": feature_kind,
        "temperature": args.temperature,
        "samples": metrics,
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"\nmodel    : {ckpt['hidden']}-wide couplings, epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_set)} images\n")
        print(f"bits/dim (exact)     {test_bpd:9.4f}   uniform baseline 8.0000")
        print(f"bits saved per pixel {8.0 - test_bpd:9.4f}")
        print(f"max inversion error  {inversion_error:9.2e}   (the likelihood needs this small)\n")
        print(f"samples through a {feature_kind} feature network:")
        print(f"  FID       {metrics['fid']:9.3f}   (not comparable with published FID)")
        print(f"  KID       {metrics['kid']:+9.5f} +- {metrics['kid_std']:.5f}")
        print(f"  precision {metrics['precision']:9.3f}")
        print(f"  recall    {metrics['recall']:9.3f}")
        print("\nan excellent likelihood with mediocre samples is the expected result:")
        print("likelihood and perceptual quality are different objectives.")
        print(f"\nfigures -> {out_dir}/samples.png, invertibility.png, temperature.png")
        print(f"metrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a normalizing flow.")
    parser.add_argument("--checkpoint", default="outputs/fashion-mnist_h64/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASETS))
    parser.add_argument("--temperature", type=float, default=1.0, help="prior scale for sampling")
    parser.add_argument("--fid-samples", type=int, default=2000)
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
    run_evaluation(args)


if __name__ == "__main__":
    main()
