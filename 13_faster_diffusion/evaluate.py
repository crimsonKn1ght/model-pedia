"""Price the quality/speed trade of DDIM against the DDPM it was trained as.

    python evaluate.py --checkpoint ../12_diffusion/outputs/fashion-mnist_cosine_t200/best.pt
    python evaluate.py --checkpoint ... --steps 5 10 20 50 100 200

Nothing is trained here. One set of weights from project 12 is sampled several times
with different numbers of network evaluations, and each run is scored with the same FID,
KID and precision/recall used everywhere else in this repository. The output is a curve,
and the curve is the result: **quality against cost, for a model that is already
trained.**

The full-length DDPM ancestral sampler is included as the reference row, so the table
answers two questions at once - how much does skipping steps cost, and does DDIM at full
length differ from DDPM at full length (it does, because ``eta=0`` removes the sampling
noise).

``--interpolate`` adds a figure that only a deterministic sampler can produce: two
latents, and the images along the straight line between them. With ``eta=0`` the mapping
from latent to image is a function, so this is a genuine traversal rather than a
sequence of unrelated samples.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

import _paths  # noqa: F401
from data import feature_net_cache, get_splits, make_loader
from sampler import DDIMSampler, load_checkpoint
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


def plot_quality_curve(rows: list[dict], path: Path) -> None:
    """FID against network evaluations, and against wall-clock seconds per image."""
    ddim = [row for row in rows if row["sampler"] == "ddim"]
    ddpm = [row for row in rows if row["sampler"] == "ddpm"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r["steps"] for r in ddim], [r["fid"] for r in ddim], marker="o", label="DDIM")
    axes[1].plot([r["ms_per_image"] for r in ddim], [r["fid"] for r in ddim], marker="o",
                 label="DDIM")
    for row in ddpm:
        axes[0].axhline(row["fid"], ls="--", color="tab:red",
                        label=f"DDPM, {row['steps']} steps")
        axes[1].scatter([row["ms_per_image"]], [row["fid"]], color="tab:red", marker="s",
                        label=f"DDPM, {row['steps']} steps", zorder=5)

    axes[0].set_xlabel("network evaluations per image")
    axes[0].set_xscale("log")
    axes[1].set_xlabel("milliseconds per image")
    for ax in axes:
        ax.set_ylabel("FID")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("what skipping steps costs")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    model, diffusion, ckpt = load_checkpoint(args.checkpoint, dev)

    train_set, _, test_set, info = get_splits(
        ckpt["dataset"], root=args.data_root, image_size=ckpt["image_size"],
        seed=args.seed, synthetic=args.smoke_test,
    )
    train_loader = make_loader(train_set, 256, num_workers=args.num_workers)
    test_loader = make_loader(test_set, 256, num_workers=args.num_workers)

    feature_net, feature_kind = load_or_train_feature_net(
        None if args.smoke_test
        else feature_net_cache(args.data_root, ckpt["dataset"], ckpt["image_size"]),
        train_loader, ckpt["in_channels"], info["classes"], dev,
        epochs=1 if args.smoke_test else 2, verbose=verbose,
    )
    real_features = features_from_loader(feature_net, test_loader, dev, limit=args.fid_samples)
    shape = (ckpt["in_channels"], ckpt["image_size"], ckpt["image_size"])
    sampler = DDIMSampler(diffusion, eta=args.eta)

    def score(sample_fn, steps: int, name: str) -> dict:
        started = time.time()
        fake_features = features_from_sampler(
            feature_net, sample_fn, real_features.size(0), args.sample_batch, dev
        )
        seconds = time.time() - started
        metrics = generative_metrics(
            real_features, fake_features, kid_subset_size=min(500, real_features.size(0))
        )
        return {
            "sampler": name,
            "steps": steps,
            "fid": metrics["fid"],
            "kid": metrics["kid"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "seconds": round(seconds, 1),
            "ms_per_image": 1000 * seconds / max(real_features.size(0), 1),
        }

    rows = []
    for steps in args.steps:
        if steps > diffusion.steps:
            continue
        rows.append(
            score(
                lambda n, s=steps: sampler.sample(model, (n, *shape), dev, steps=s),
                steps, "ddim",
            )
        )
    if not args.skip_ddpm:
        rows.append(
            score(lambda n: diffusion.sample(model, (n, *shape), dev), diffusion.steps, "ddpm")
        )

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / (
        f"{ckpt['dataset']}_eta{args.eta:g}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_quality_curve(rows, out_dir / "quality_vs_steps.png")

    # The same latent decoded at several step counts. With eta=0 this is the same
    # function evaluated at different accuracies, so the images should agree in layout
    # and differ in polish.
    latent = torch.randn(8, *shape, device=dev)
    grid_rows = []
    for steps in [s for s in args.steps if s <= diffusion.steps][:4]:
        images = sampler.sample(model, (8, *shape), dev, steps=steps, latent=latent)
        grid_rows.append((f"{steps} steps", images.cpu()))
    plot_image_rows(grid_rows, out_dir / "same_latent.png",
                    title=f"one latent, decoded with fewer and fewer steps (eta {args.eta:g})")

    best = min(rows, key=lambda row: row["fid"])
    plot_image_grid(
        sampler.sample(model, (64, *shape), dev, steps=max(args.steps)).cpu(),
        out_dir / "samples.png", columns=8,
        title=f"DDIM samples, {max(args.steps)} steps, eta {args.eta:g}",
    )

    if args.interpolate:
        start, end = torch.randn(1, *shape, device=dev), torch.randn(1, *shape, device=dev)
        weights = torch.linspace(0, 1, 8, device=dev).view(-1, 1, 1, 1)
        # Spherical interpolation: a straight line through Gaussian noise shrinks the
        # norm in the middle and produces washed-out images.
        latents = ((1 - weights) * start + weights * end)
        latents = latents / latents.flatten(1).norm(dim=1).view(-1, 1, 1, 1) * (
            start.flatten().norm()
        )
        images = sampler.sample(model, (8, *shape), dev, steps=max(args.steps), latent=latents)
        plot_image_rows([("interpolation", images.cpu())], out_dir / "interpolation.png",
                        title="latent interpolation, only meaningful because eta=0 is deterministic")

    result = {
        "checkpoint": str(args.checkpoint),
        "dataset": ckpt["dataset"],
        "trained_steps": diffusion.steps,
        "schedule": ckpt["schedule"],
        "eta": args.eta,
        "feature_net": feature_kind,
        "fid_samples": int(real_features.size(0)),
        "rows": rows,
        "best": best,
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"\nmodel    : project 12 checkpoint, {diffusion.steps} trained steps, "
              f"{ckpt['schedule']} schedule")
        print(f"test set : {ckpt['dataset']}  {real_features.size(0)} images per comparison")
        print(f"eta      : {args.eta:g} "
              f"({'deterministic' if args.eta == 0 else 'partly stochastic'})\n")
        header = (
            f"{'sampler':>8s} {'steps':>6s} {'FID':>8s} {'KID':>9s} {'prec':>6s} {'rec':>6s} "
            f"{'ms/img':>8s} {'speedup':>8s}"
        )
        print(header)
        print("-" * len(header))
        reference = next((row for row in rows if row["sampler"] == "ddpm"), None)
        for row in rows:
            speedup = (
                f"{reference['ms_per_image'] / row['ms_per_image']:7.1f}x"
                if reference else "       -"
            )
            print(
                f"{row['sampler']:>8s} {row['steps']:6d} {row['fid']:8.2f} {row['kid']:+9.5f} "
                f"{row['precision']:6.3f} {row['recall']:6.3f} {row['ms_per_image']:8.0f} {speedup}"
            )
        print(f"\nbest FID {best['fid']:.2f} at {best['steps']} {best['sampler'].upper()} steps")
        print(f"figures -> {out_dir}/quality_vs_steps.png, same_latent.png, samples.png")
        print(f"metrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare DDIM step counts against DDPM.")
    parser.add_argument(
        "--checkpoint",
        default="../12_diffusion/outputs/fashion-mnist_cosine_t200/best.pt",
        help="a checkpoint trained by project 12",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20, 50, 100, 200])
    parser.add_argument("--eta", type=float, default=0.0,
                        help="0 is deterministic DDIM; 1 recovers DDPM-like noise")
    parser.add_argument("--skip-ddpm", action="store_true", help="omit the full-length reference")
    parser.add_argument("--interpolate", action="store_true", help="add a latent interpolation figure")
    parser.add_argument("--data-root", default="../12_diffusion/data")
    parser.add_argument("--fid-samples", type=int, default=500)
    parser.add_argument("--sample-batch", type=int, default=125)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on generated data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_workers = 0
        args.fid_samples = 64
        args.sample_batch = 32
        args.steps = [2, 5]
    run_evaluation(args)


if __name__ == "__main__":
    main()
