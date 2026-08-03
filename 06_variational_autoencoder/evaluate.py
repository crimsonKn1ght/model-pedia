"""Evaluate a VAE: the bound, the latent, and the samples.

    python evaluate.py --checkpoint outputs/fashion-mnist_beta1_z16/best.pt

A VAE has two jobs and they need separate measurements.

**As a density model** it reports -ELBO in nats, which is an *upper bound* on the
negative log likelihood, not the likelihood itself. The bound is loose by exactly
the KL between the approximate posterior and the true one, and nothing here can
tell you by how much. Project 11 (normalizing flows) computes an exact likelihood
for the same datasets, which is the honest point of comparison - and the reason the
bits-per-dimension figure is reported here in the same units.

**As a generator** it is scored on samples drawn from the prior, with FID, KID and
generative precision/recall. Prior sampling is the test a plain autoencoder fails:
its latent has empty pockets between training points, so decoding a random code
gives noise. A VAE's does not, because the KL term forces every posterior to be a
whole Gaussian.

``active_units`` is the diagnostic to watch. A latent dimension whose KL is near
zero has posterior equal to prior - the encoder ignores it and the decoder cannot
read it. At high beta most of them switch off, and a model with 2 active units out
of 16 can still post a respectable -ELBO while having thrown most of the latent away.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from data import DATASETS, dataset_info, feature_net_cache, get_splits, make_loader
from model import active_units, build_model
from train import evaluate_split
from utils import (
    AverageMeter,
    features_from_loader,
    features_from_sampler,
    generative_metrics,
    get_device,
    load_or_train_feature_net,
    plot_bars,
    plot_image_grid,
    plot_image_rows,
    plot_latent_scatter,
    psnr,
    save_json,
    set_seed,
)


@torch.no_grad()
def reconstruction_quality(model, loader, device) -> dict:
    """Pixel-space reconstruction, separate from the ELBO's nats."""
    model.eval()
    psnr_meter, mse_meter = AverageMeter(), AverageMeter()
    for images, _ in loader:
        images = images.to(device)
        reconstruction = model.reconstruct_from_logits(model(images)["logits"])
        psnr_meter.update(psnr(reconstruction, images).mean().item(), images.size(0))
        mse_meter.update(
            (reconstruction - images).pow(2).flatten(1).mean(dim=1).mean().item(), images.size(0)
        )
    return {"psnr_db": psnr_meter.avg, "mse": mse_meter.avg}


@torch.no_grad()
def collect_latents(model, loader, device, limit: int = 3000):
    """Posterior means and labels, for the latent-space figure."""
    model.eval()
    latents, labels = [], []
    for images, targets in loader:
        mu, _ = model.encode(images.to(device))
        latents.append(mu.cpu())
        labels.append(targets)
        if sum(t.size(0) for t in latents) >= limit:
            break
    return torch.cat(latents)[:limit], torch.cat(labels)[:limit]


@torch.no_grad()
def save_latent_figures(model, loader, device, kl_per_dimension, out_dir: Path) -> None:
    """Interpolations, and traversals of the dimensions that carry the most information."""
    images, _ = next(iter(loader))
    images = images.to(device)

    rows = []
    for pair in range(min(3, images.size(0) // 2)):
        line = model.interpolate(images[2 * pair], images[2 * pair + 1], steps=8)
        rows.append((f"pair {pair + 1}", line.cpu()))
    plot_image_rows(rows, out_dir / "interpolation.png",
                    title="decoding a straight line between two posterior means")

    order = torch.tensor(kl_per_dimension).argsort(descending=True)[:4].tolist()
    rows = [
        (f"z[{dimension}] KL {kl_per_dimension[dimension]:.2f}",
         model.traverse(images[0], dimension, steps=8).cpu())
        for dimension in order
    ]
    plot_image_rows(rows, out_dir / "traversals.png",
                    title="varying one latent dimension at a time (most informative first)")


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)

    train_set, _, test_set, info = get_splits(
        ckpt["dataset"],
        root=args.data_root,
        image_size=ckpt["image_size"],
        seed=args.seed,
        synthetic=args.smoke_test,
    )
    train_loader = make_loader(train_set, args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, num_workers=args.num_workers)

    model = build_model(
        ckpt["in_channels"], ckpt["image_size"], ckpt["latent_dim"],
        ckpt["base_channels"], ckpt["beta"], ckpt["likelihood"],
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test = evaluate_split(model, test_loader, dev)
    quality = reconstruction_quality(model, test_loader, dev)
    pixels = ckpt["in_channels"] * ckpt["image_size"] ** 2

    # The FID ruler: a small classifier on this dataset, trained once and cached.
    feature_net, feature_kind = load_or_train_feature_net(
        None if args.smoke_test else feature_net_cache(args.data_root, ckpt["dataset"],
                                                       ckpt["image_size"]),
        train_loader,
        ckpt["in_channels"],
        info["classes"],
        dev,
        epochs=1 if args.smoke_test else 2,
        verbose=verbose,
    )
    real_features = features_from_loader(feature_net, test_loader, dev, limit=args.fid_samples)
    fake_features = features_from_sampler(
        feature_net, lambda n: model.sample(n, dev), real_features.size(0), args.batch_size, dev
    )
    sample_metrics = generative_metrics(
        real_features, fake_features, kid_subset_size=min(500, real_features.size(0))
    )

    out_dir = Path(args.checkpoint).parent
    plot_image_grid(model.sample(64, dev).cpu(), out_dir / "samples.png", columns=8,
                    title=f"prior samples ({ckpt['dataset']}, beta {ckpt['beta']:g})")
    latents, labels = collect_latents(model, test_loader, dev)
    plot_latent_scatter(
        latents, labels, out_dir / "latent_space.png",
        class_names=info["class_names"],
        title=f"posterior means, latent {ckpt['latent_dim']}d "
              f"({'PCA to 2D' if ckpt['latent_dim'] > 2 else 'exact'})",
    )
    save_latent_figures(model, test_loader, dev, test["kl_per_dimension"], out_dir)
    plot_bars(
        [f"z[{i}]" for i in range(ckpt["latent_dim"])],
        {"KL, nats": test["kl_per_dimension"]},
        out_dir / "kl_per_dimension.png",
        ylabel="nats",
        title=f"information per latent dimension ({test['active_units']} active)",
    )

    result = {
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "beta": ckpt["beta"],
        "latent_dim": ckpt["latent_dim"],
        "likelihood": ckpt["likelihood"],
        "test_images": len(test_set),
        "neg_elbo_nats": test["neg_elbo"],
        "reconstruction_nats": test["reconstruction"],
        "kl_nats": test["kl"],
        # The ELBO is an upper bound on the NLL, so this is an upper bound on bits/dim.
        "bits_per_dimension_bound": test["neg_elbo"] / (pixels * math.log(2)),
        "active_units": test["active_units"],
        "kl_per_dimension": test["kl_per_dimension"],
        "reconstruction_psnr_db": quality["psnr_db"],
        "reconstruction_mse": quality["mse"],
        "feature_net": feature_kind,
        "samples": sample_metrics,
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"\nmodel    : beta {ckpt['beta']:g}, latent {ckpt['latent_dim']}, "
              f"{ckpt['likelihood']} likelihood, epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_set)} images\n")
        print(f"-ELBO                {result['neg_elbo_nats']:9.2f} nats  "
              f"(reconstruction {result['reconstruction_nats']:.2f} + KL {result['kl_nats']:.2f})")
        print(f"bits/dim (bound)     {result['bits_per_dimension_bound']:9.4f}")
        print(f"active latent units  {result['active_units']:9d} / {ckpt['latent_dim']}")
        print(f"reconstruction PSNR  {result['reconstruction_psnr_db']:9.2f} dB\n")
        print(f"prior samples scored through a {feature_kind} feature network:")
        print(f"  FID       {sample_metrics['fid']:9.3f}   (not comparable with published FID)")
        print(f"  KID       {sample_metrics['kid']:+9.5f} +- {sample_metrics['kid_std']:.5f}")
        print(f"  precision {sample_metrics['precision']:9.3f}   (are the samples plausible)")
        print(f"  recall    {sample_metrics['recall']:9.3f}   (do they cover the data)")
        print(f"\nfigures -> {out_dir}/samples.png, latent_space.png, interpolation.png,")
        print(f"           traversals.png, kl_per_dimension.png")
        print(f"metrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a VAE.")
    parser.add_argument("--checkpoint", default="outputs/fashion-mnist_beta1_z16/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASETS),
                        help="override the dataset recorded in the checkpoint")
    parser.add_argument("--fid-samples", type=int, default=2000,
                        help="images per side of the FID/KID comparison")
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
