"""Evaluate a DDPM: sample quality, the reverse trajectory, and what sampling costs.

    python evaluate.py --checkpoint outputs/fashion-mnist_cosine_t200/best.pt

Three things are reported.

**Sample quality** - FID, KID and generative precision/recall, the same measurements
project 08 applies to the GAN and through the same feature network, so the two are
directly comparable on the same dataset. That comparison is the point: diffusion
usually wins on *recall* by a wide margin, because it has no discriminator to satisfy
and therefore no incentive to abandon the difficult parts of the distribution.

**The trajectory** - `trajectory.png` shows the same sample at several points on the
way back from noise. Structure appears early and detail late, which is the practical
reason step-skipping works at all and the setup for project 13.

**Cost** - wall-clock seconds per image and network evaluations per image. A GAN needs
one forward pass; this needs ``T``. That single number is what all of the fast-sampling
literature exists to attack, and project 13 measures the first and most useful attack
on it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from data import DATASETS, feature_net_cache, get_splits, make_loader
from model import build_diffusion, build_model
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

    model = build_model(ckpt["in_channels"], ckpt["base_channels"], ckpt["attention"]).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    diffusion = build_diffusion(ckpt["steps"], ckpt["schedule"], ckpt["prediction"]).to(dev)

    shape = (None, ckpt["in_channels"], ckpt["image_size"], ckpt["image_size"])

    def sample(n: int) -> torch.Tensor:
        return diffusion.sample(model, (n, *shape[1:]), dev)

    feature_net, feature_kind = load_or_train_feature_net(
        None if args.smoke_test
        else feature_net_cache(args.data_root, ckpt["dataset"], ckpt["image_size"]),
        train_loader, ckpt["in_channels"], info["classes"], dev,
        epochs=1 if args.smoke_test else 2, verbose=verbose,
    )
    real_features = features_from_loader(feature_net, test_loader, dev, limit=args.fid_samples)

    started = time.time()
    fake_features = features_from_sampler(
        feature_net, sample, real_features.size(0), args.sample_batch, dev
    )
    sampling_seconds = time.time() - started
    metrics = generative_metrics(
        real_features, fake_features, kid_subset_size=min(500, real_features.size(0))
    )

    out_dir = Path(args.checkpoint).parent
    plot_image_grid(sample(64).cpu(), out_dir / "samples.png", columns=8,
                    title=f"{ckpt['dataset']}, {ckpt['steps']} steps, FID {metrics['fid']:.1f}")

    _, trajectory = diffusion.sample(model, (8, *shape[1:]), dev, record=6)
    plot_image_rows(
        [(f"step {int(i)}", frame) for i, frame in
         zip(torch.linspace(ckpt["steps"], 0, len(trajectory)), trajectory)],
        out_dir / "trajectory.png",
        title="the reverse process, from noise (top) to image (bottom)",
    )

    result = {
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "steps": ckpt["steps"],
        "schedule": ckpt["schedule"],
        "prediction": ckpt["prediction"],
        "feature_net": feature_kind,
        "test_images": len(test_set),
        "samples": metrics,
        "cost": {
            "sampling_seconds": round(sampling_seconds, 1),
            "images": int(real_features.size(0)),
            "seconds_per_image": sampling_seconds / max(real_features.size(0), 1),
            "network_evaluations_per_image": ckpt["steps"],
        },
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        cost = result["cost"]
        print(f"\nmodel    : {ckpt['steps']} steps, {ckpt['schedule']} schedule, "
              f"predicting {ckpt['prediction']}, epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_set)} images\n")
        print(f"FID       {metrics['fid']:9.3f}   (not comparable with published FID)")
        print(f"KID       {metrics['kid']:+9.5f} +- {metrics['kid_std']:.5f}")
        print(f"precision {metrics['precision']:9.3f}")
        print(f"recall    {metrics['recall']:9.3f}   diffusion's usual advantage over a GAN")
        print(f"\nsampling  {cost['images']} images in {cost['sampling_seconds']:.1f}s "
              f"= {cost['seconds_per_image'] * 1000:.0f} ms/image")
        print(f"          {cost['network_evaluations_per_image']} network evaluations per image "
              f"(a GAN needs 1)")
        print(f"\nfigures -> {out_dir}/samples.png, trajectory.png")
        print(f"metrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a DDPM.")
    parser.add_argument("--checkpoint", default="outputs/fashion-mnist_cosine_t200/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASETS))
    parser.add_argument("--fid-samples", type=int, default=1000,
                        help="sampling is the expensive part, so this is lower than elsewhere")
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
