"""Evaluate a conditional GAN: overall quality, per-class quality, and obedience.

    python evaluate.py --checkpoint outputs/fashion-mnist_acgan/best.pt

Conditioning gives this project a measurement the unconditional GAN cannot have.
Generate images for a known label, ask an independent classifier what it sees, and
count agreement. That is **class accuracy**, and it answers "did the conditioning
work" directly rather than by inspection.

**FID by class** then asks the harder question. An overall FID hides which classes the
generator is bad at, and conditional models are usually uneven - the classes with the
most distinctive shape come out first. Per-class FID is computed against that class's
real test images only, so the comparison is apples to apples.

The trap this project exists to show: **class accuracy and diversity pull in opposite
directions.** A generator can score near-perfect class accuracy by producing one
over-typical example per class, which is why recall is reported alongside. An ACGAN
run with a large ``--aux-weight`` is the fastest way to see it happen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import DATASETS, feature_net_cache, get_splits, make_loader
from model import build_models
from utils import (
    features_from_loader,
    fid_score,
    generative_metrics,
    get_device,
    load_or_train_feature_net,
    plot_bars,
    plot_image_grid,
    plot_image_rows,
    save_json,
    set_seed,
)


@torch.no_grad()
def real_features_by_class(feature_net, loader, device, num_classes: int, limit: int):
    """Test features grouped by label, plus everything pooled."""
    per_class = [[] for _ in range(num_classes)]
    pooled, seen = [], 0
    for images, labels in loader:
        features = feature_net.features(images.to(device)).cpu()
        pooled.append(features)
        for cls in range(num_classes):
            mask = labels == cls
            if mask.any():
                per_class[cls].append(features[mask])
        seen += images.size(0)
        if seen >= limit:
            break
    return (
        torch.cat(pooled)[:limit],
        [torch.cat(chunks) if chunks else torch.zeros(0) for chunks in per_class],
    )


@torch.no_grad()
def generate_by_class(generator, feature_net, device, num_classes: int, per_class: int, batch: int):
    """Features and classifier predictions for a balanced batch of each class."""
    features = [[] for _ in range(num_classes)]
    predictions = [[] for _ in range(num_classes)]
    for cls in range(num_classes):
        remaining = per_class
        while remaining > 0:
            n = min(batch, remaining)
            labels = torch.full((n,), cls, device=device, dtype=torch.long)
            images = generator(torch.randn(n, generator.latent_dim, device=device), labels)
            features[cls].append(feature_net.features(images).cpu())
            predictions[cls].append(feature_net(images).argmax(dim=1).cpu())
            remaining -= n
    return (
        [torch.cat(chunk) for chunk in features],
        [torch.cat(chunk) for chunk in predictions],
    )


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    num_classes = ckpt["num_classes"]
    class_names = ckpt["class_names"] or [str(i) for i in range(num_classes)]

    train_set, _, test_set, info = get_splits(
        ckpt["dataset"], root=args.data_root, image_size=ckpt["image_size"],
        seed=args.seed, synthetic=args.smoke_test,
    )
    train_loader = make_loader(train_set, args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, num_workers=args.num_workers)

    generator, _ = build_models(
        ckpt["latent_dim"], num_classes, ckpt["in_channels"], ckpt["image_size"],
        ckpt["width"], ckpt["mode"],
    )
    generator = generator.to(dev)
    generator.load_state_dict(ckpt["generator_state"])
    generator.eval()

    feature_net, feature_kind = load_or_train_feature_net(
        None if args.smoke_test
        else feature_net_cache(args.data_root, ckpt["dataset"], ckpt["image_size"]),
        train_loader, ckpt["in_channels"], num_classes, dev,
        epochs=1 if args.smoke_test else 2, verbose=verbose,
    )

    pooled_real, real_by_class = real_features_by_class(
        feature_net, test_loader, dev, num_classes, args.fid_samples
    )
    per_class = max(1, args.fid_samples // num_classes)
    fake_by_class, predicted_by_class = generate_by_class(
        generator, feature_net, dev, num_classes, per_class, args.batch_size
    )
    pooled_fake = torch.cat(fake_by_class)[: pooled_real.size(0)]

    overall = generative_metrics(
        pooled_real, pooled_fake, kid_subset_size=min(500, pooled_real.size(0))
    )
    rows = []
    for cls in range(num_classes):
        accuracy = float((predicted_by_class[cls] == cls).double().mean())
        class_fid = (
            fid_score(real_by_class[cls], fake_by_class[cls])
            if real_by_class[cls].size(0) > 1 else float("nan")
        )
        rows.append({"class": class_names[cls], "fid": class_fid, "accuracy": accuracy,
                     "real_images": int(real_by_class[cls].size(0))})
    class_accuracy = float(np.mean([row["accuracy"] for row in rows]))

    out_dir = Path(args.checkpoint).parent
    labels = torch.arange(num_classes, device=dev).repeat_interleave(8)
    with torch.no_grad():
        grid = generator(torch.randn(labels.numel(), ckpt["latent_dim"], device=dev), labels)
    plot_image_grid(grid.cpu(), out_dir / "samples_by_class.png", columns=8,
                    title=f"one class per row ({ckpt['mode']}, class accuracy {class_accuracy:.2f})")

    # The same latent under every label: what the model thinks "style" is.
    with torch.no_grad():
        shared = torch.randn(1, ckpt["latent_dim"], device=dev).repeat(num_classes, 1)
        sweep = generator(shared, torch.arange(num_classes, device=dev))
    plot_image_rows([("one latent, every label", sweep.cpu())], out_dir / "latent_sweep.png",
                    title="the same noise vector conditioned on each class in turn")
    plot_bars(
        class_names,
        {"FID": [row["fid"] for row in rows],
         "class accuracy x 100": [row["accuracy"] * 100 for row in rows]},
        out_dir / "per_class.png", ylabel="score", title="quality and obedience per class",
    )

    result = {
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "mode": ckpt["mode"],
        "aux_weight": ckpt["aux_weight"],
        "feature_net": feature_kind,
        "test_images": len(test_set),
        "samples_per_class": per_class,
        "class_accuracy": class_accuracy,
        "samples": overall,
        "per_class": rows,
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"\nmodel    : {ckpt['mode']}, aux weight {ckpt['aux_weight']:g}, "
              f"epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_set)} images, "
              f"{per_class} samples per class\n")
        print(f"FID           {overall['fid']:9.3f}   (not comparable with published FID)")
        print(f"KID           {overall['kid']:+9.5f} +- {overall['kid_std']:.5f}")
        print(f"precision     {overall['precision']:9.3f}")
        print(f"recall        {overall['recall']:9.3f}   low here with high accuracy = "
              f"over-typical samples")
        print(f"class accuracy{class_accuracy:9.3f}   did it draw what it was asked for\n")

        pad = max(len(name) for name in class_names)
        header = f"{'class'.ljust(pad)} {'FID':>8s} {'accuracy':>9s} {'real':>6s}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(f"{row['class'].ljust(pad)} {row['fid']:8.2f} {row['accuracy']:9.3f} "
                  f"{row['real_images']:6d}")
        print(f"\nfigures -> {out_dir}/samples_by_class.png, latent_sweep.png, per_class.png")
        print(f"metrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a conditional GAN.")
    parser.add_argument("--checkpoint", default="outputs/fashion-mnist_acgan/best.pt")
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
