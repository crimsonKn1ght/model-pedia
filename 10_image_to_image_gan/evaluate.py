"""Evaluate Pix2Pix: pixel metrics, distribution metrics, and pictures.

    python evaluate.py --checkpoint outputs/shapes_l1100/best.pt

Paired translation is the one generative task in this repository with a ground-truth answer
per input, so it gets both kinds of metric, and they disagree in a way worth understanding.

**Pixel metrics** - L1, PSNR, SSIM - compare the output with its target directly. They are
unambiguous and they reward blur: when several outputs are plausible, the pixel-optimal
answer is the average of them, and an average of sharp images is soft.

**FID** compares the *distribution* of outputs with the distribution of real targets, so it
notices blur that PSNR forgives. A model trained with L1 alone typically wins on PSNR and
loses on FID, which is exactly why Pix2Pix adds an adversarial term.

LPIPS would be the third metric here and is deliberately absent: it needs pretrained
ImageNet weights, which would be a heavier dependency than anything else in this
repository. SSIM plus FID covers the same ground with code you can read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import DATASETS, get_splits, make_loader
from model import build_models
from train import evaluate_split
from utils import (
    features_from_loader,
    generative_metrics,
    get_device,
    load_or_train_feature_net,
    plot_image_rows,
    save_json,
    set_seed,
)


@torch.no_grad()
def translated_features(feature_net, generator, loader, device, limit: int):
    features, seen = [], 0
    for source, _ in loader:
        generated = generator(source.to(device))
        features.append(feature_net.features(generated).cpu())
        seen += source.size(0)
        if seen >= limit:
            break
    return torch.cat(features)[:limit]


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)

    train_set, _, test_set, info = get_splits(
        ckpt["dataset"], root=args.data_root, image_size=ckpt["image_size"],
        test_size=args.test_size, seed=args.seed, synthetic=args.smoke_test,
    )
    size = info["size"]
    train_loader = make_loader(train_set, size, args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(test_set, size, args.batch_size, num_workers=args.num_workers)

    generator, _ = build_models(
        ckpt["in_channels"], ckpt["in_channels"], ckpt["width"], ckpt["levels"],
        ckpt["skips"], ckpt["patch_layers"], ckpt["whole_image"],
    )
    generator = generator.to(dev)
    generator.load_state_dict(ckpt["generator_state"])
    generator.eval()

    pixel = evaluate_split(generator, test_loader, dev)

    # The feature network measures the *targets*, which are the real images here. They
    # carry no class labels, so this falls back to fixed-seed random features - recorded
    # in metrics.json as feature_net, and comparable within this project only.
    target_batches = [(target, torch.zeros(target.size(0), dtype=torch.long))
                      for _, target in train_loader]
    feature_net, feature_kind = load_or_train_feature_net(
        None, target_batches, ckpt["in_channels"], None, dev, verbose=False
    )
    real_features = features_from_loader(
        feature_net,
        [(target, None) for _, target in test_loader],
        dev, limit=args.fid_samples,
    )
    fake_features = translated_features(
        feature_net, generator, test_loader, dev, real_features.size(0)
    )
    distribution = generative_metrics(
        real_features, fake_features, kid_subset_size=min(500, real_features.size(0))
    )

    out_dir = Path(args.checkpoint).parent
    source, target = next(iter(test_loader))
    source, target = source[:8].to(dev), target[:8].to(dev)
    with torch.no_grad():
        generated = generator(source)
    plot_image_rows(
        [("input", source.cpu()), ("generated", generated.cpu()), ("target", target.cpu())],
        out_dir / "examples.png",
        title=f"test translations (L1 weight {ckpt['l1_weight']:g}, "
              f"SSIM {pixel['ssim']:.3f}, FID {distribution['fid']:.2f})",
    )

    result = {
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "l1_weight": ckpt["l1_weight"],
        "skips": ckpt["skips"],
        "discriminator": "whole image" if ckpt["whole_image"]
        else f"patchgan-{ckpt['patch_layers']}",
        "test_pairs": len(test_set),
        "feature_net": feature_kind,
        "pixel": pixel,
        "distribution": distribution,
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"\nmodel    : L1 weight {ckpt['l1_weight']:g}, {result['discriminator']}"
              f"{'' if ckpt['skips'] else ', no skips'}, epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_set)} pairs\n")
        print(f"L1                  {pixel['l1']:9.4f}   (lower is better)")
        print(f"PSNR                {pixel['psnr']:9.2f} dB")
        print(f"SSIM                {pixel['ssim']:9.4f}")
        print(f"FID                 {distribution['fid']:9.3f}   "
              f"(notices blur that PSNR forgives)")
        print(f"KID                 {distribution['kid']:+9.5f} +- {distribution['kid_std']:.5f}")
        print(f"precision / recall  {distribution['precision']:9.3f} / "
              f"{distribution['recall']:.3f}")
        print(f"\nfeature network: {feature_kind} (the targets carry no labels here)")
        print(f"figures -> {out_dir / 'examples.png'}")
        print(f"metrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Pix2Pix model.")
    parser.add_argument("--checkpoint", default="outputs/shapes_l1100/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASETS))
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--fid-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on generated pairs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_workers = 0
        args.fid_samples = 24
    run_evaluation(args)


if __name__ == "__main__":
    main()
