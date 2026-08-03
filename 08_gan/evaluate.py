"""Evaluate a GAN: sample quality, mode coverage, and a memorisation check.

    python evaluate.py --checkpoint outputs/fashion-mnist_bce/best.pt

A GAN has no likelihood, so everything here is measured from samples.

**FID and KID** summarise how far the generated distribution is from the real one in
feature space. Necessary, and not sufficient: a single number cannot distinguish "the
samples look wrong" from "the samples look right but there are only three of them".

**Precision and recall** do separate those. Precision is the fraction of samples
inside the real data's k-NN manifold - are they plausible. Recall is the fraction of
real images inside the samples' manifold - do they cover the data. **Mode collapse is
exactly low recall at high precision**, and it is the characteristic GAN failure, so
this is the pair to read first.

**The nearest-neighbour check** answers a different worry. A generator that memorised
its training set would score beautifully on all of the above. For each sample, the
closest training image in feature space is found; if those distances are far smaller
than the distance between two random real images, the model is copying. The figure
puts them side by side so you can judge for yourself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import DATASETS, feature_net_cache, get_splits, make_loader
from model import build_models
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


@torch.no_grad()
def nearest_neighbours(
    feature_net, generator, train_loader, device, latent_dim: int, n: int = 8
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Closest training image to each sample, in feature space.

    Returns ``(samples, neighbours, mean sample distance, mean real-real distance)``.
    The second distance is the yardstick: two different real images are that far apart,
    so samples much closer than it to a training image are copies.
    """
    samples = generator(torch.randn(n, latent_dim, device=device))
    sample_features = feature_net.features(samples)

    train_images, train_features = [], []
    for images, _ in train_loader:
        images = images.to(device)
        train_images.append(images.cpu())
        train_features.append(feature_net.features(images).cpu())
        if sum(t.size(0) for t in train_images) >= 4000:
            break
    train_images = torch.cat(train_images)
    train_features = torch.cat(train_features)

    distances = torch.cdist(sample_features.cpu(), train_features)
    closest = distances.argmin(dim=1)
    sample_distance = float(distances.min(dim=1).values.mean())

    # How far apart are two *different* real images, on average?
    subset = train_features[: min(1000, train_features.size(0))]
    real_distances = torch.cdist(subset, subset)
    real_distances.fill_diagonal_(float("inf"))
    real_distance = float(real_distances.min(dim=1).values.mean())

    return samples.cpu(), train_images[closest], sample_distance, real_distance


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

    generator, _ = build_models(
        ckpt["latent_dim"], ckpt["in_channels"], ckpt["image_size"], ckpt["width"]
    )
    generator = generator.to(dev)
    generator.load_state_dict(ckpt["generator_state"])
    generator.eval()

    feature_net, feature_kind = load_or_train_feature_net(
        None if args.smoke_test
        else feature_net_cache(args.data_root, ckpt["dataset"], ckpt["image_size"]),
        train_loader, ckpt["in_channels"], info["classes"], dev,
        epochs=1 if args.smoke_test else 2, verbose=verbose,
    )
    real_features = features_from_loader(feature_net, test_loader, dev, limit=args.fid_samples)
    fake_features = features_from_sampler(
        feature_net, lambda n: generator.sample(n, dev), real_features.size(0), args.batch_size, dev
    )
    metrics = generative_metrics(
        real_features, fake_features, kid_subset_size=min(500, real_features.size(0))
    )

    out_dir = Path(args.checkpoint).parent
    plot_image_grid(
        generator.sample(64, dev).cpu(), out_dir / "samples.png", columns=8,
        title=f"{ckpt['dataset']} samples (epoch {ckpt['epoch']}, FID {metrics['fid']:.1f})",
    )

    # Latent interpolation: a generator that has learned structure interpolates
    # smoothly; one that memorised jumps between training images.
    with torch.no_grad():
        start = torch.randn(4, ckpt["latent_dim"], device=dev)
        end = torch.randn(4, ckpt["latent_dim"], device=dev)
        rows = []
        for index in range(4):
            weights = torch.linspace(0, 1, 8, device=dev).view(-1, 1)
            latents = (1 - weights) * start[index] + weights * end[index]
            rows.append((f"pair {index + 1}", generator(latents).cpu()))
    plot_image_rows(rows, out_dir / "interpolation.png",
                    title="straight lines through the latent space")

    samples, neighbours, sample_distance, real_distance = nearest_neighbours(
        feature_net, generator, train_loader, dev, ckpt["latent_dim"]
    )
    plot_image_rows(
        [("sample", samples), ("closest train", neighbours)],
        out_dir / "nearest_neighbours.png",
        title=f"nearest training image in feature space "
              f"(sample {sample_distance:.2f} vs real-real {real_distance:.2f})",
    )

    result = {
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "loss": ckpt["loss"],
        "latent_dim": ckpt["latent_dim"],
        "test_images": len(test_set),
        "feature_net": feature_kind,
        "samples": metrics,
        "memorisation": {
            "mean_sample_to_train_distance": sample_distance,
            "mean_real_to_real_distance": real_distance,
            "ratio": sample_distance / max(real_distance, 1e-9),
        },
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"\nmodel    : {ckpt['loss']} loss, latent {ckpt['latent_dim']}, epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_set)} images, "
              f"{metrics['fake_samples']} samples\n")
        print(f"FID       {metrics['fid']:9.3f}   (not comparable with published FID)")
        print(f"KID       {metrics['kid']:+9.5f} +- {metrics['kid_std']:.5f}")
        print(f"precision {metrics['precision']:9.3f}   are the samples plausible")
        print(f"recall    {metrics['recall']:9.3f}   do they cover the data (mode collapse)")
        memo = result["memorisation"]
        print(f"\nmemorisation check: samples sit {memo['mean_sample_to_train_distance']:.2f} from "
              f"their closest training image;")
        print(f"two different real images sit {memo['mean_real_to_real_distance']:.2f} apart "
              f"(ratio {memo['ratio']:.2f}).")
        print("A ratio well below 1 would mean the generator is copying.")
        print(f"\nfigures -> {out_dir}/samples.png, interpolation.png, nearest_neighbours.png")
        print(f"metrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a GAN from its samples.")
    parser.add_argument("--checkpoint", default="outputs/fashion-mnist_bce/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASETS))
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
