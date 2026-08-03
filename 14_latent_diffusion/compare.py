"""Controlled comparisons for latent diffusion.

    python compare.py --study downsample   # how far to compress before diffusing
    python compare.py --study channels     # how wide the latent should be

Same data, same schedule, same seed. Each arm trains both stages, so an arm is a complete
latent-diffusion model rather than a shared first stage with different diffusion heads -
which matters, because the two stages interact: a more aggressive first stage makes
diffusion cheaper *and* lowers the ceiling it is working under.

That is why every row reports the ceiling next to the FID. A row whose FID sits close to
its ceiling is limited by the autoencoder, and more diffusion training will not help it; a
row whose FID is far above its ceiling has room left. Reading only the FID column would
make those two situations look identical.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import _paths  # noqa: F401
from data import DATASETS
from evaluate import run_evaluation
from train import run_training as run_diffusion
from train_autoencoder import run_training as run_autoencoder
from utils import plot_bars, save_json

STUDIES = {
    "downsample": [
        ("2x (16x16 latent)", {"levels": 1}),
        ("4x (8x8 latent)", {"levels": 2}),
    ],
    "channels": [
        ("2 channels", {"latent_channels": 2}),
        ("4 channels", {"latent_channels": 4}),
        ("8 channels", {"latent_channels": 8}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled latent-diffusion comparison.")
    parser.add_argument("--study", default="downsample", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--autoencoder-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--fid-samples", type=int, default=500)
    parser.add_argument("--pixel-checkpoint", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke-test", action="store_true", help="run on generated data")
    args = parser.parse_args()

    workers = 0 if args.smoke_test else args.num_workers
    out_root = "outputs/smoke" if args.smoke_test else f"outputs/{args.study}"
    rows = []

    for label, overrides in STUDIES[args.study]:
        print(f"\n=== {args.study}: {label} ===")
        out_dir = f"{out_root}/{label.split(' ')[0].replace('.', '')}"
        stage_one = run_autoencoder(
            dataset=args.dataset,
            latent_channels=overrides.get("latent_channels", 4),
            levels=1 if args.smoke_test else overrides.get("levels", 2),
            epochs=1 if args.smoke_test else args.autoencoder_epochs,
            batch_size=32 if args.smoke_test else args.batch_size,
            train_subset=args.train_subset or None,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=out_dir,
            synthetic=args.smoke_test,
            verbose=False,
        )
        stage_two = run_diffusion(
            autoencoder=stage_one["checkpoint"],
            steps=20 if args.smoke_test else args.steps,
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=32 if args.smoke_test else args.batch_size,
            train_subset=args.train_subset or None,
            fid_every=1 if args.smoke_test else max(args.epochs // 3, 1),
            fid_track_samples=64 if args.smoke_test else args.fid_samples,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=out_dir,
            synthetic=args.smoke_test,
            verbose=False,
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=stage_two["checkpoint"],
                pixel_checkpoint=args.pixel_checkpoint,
                data_root=args.data_root,
                fid_samples=64 if args.smoke_test else args.fid_samples,
                sample_batch=32 if args.smoke_test else 125,
                batch_size=256,
                num_workers=workers,
                device=args.device,
                seed=args.seed,
                smoke_test=args.smoke_test,
            ),
            verbose=False,
        )
        rows.append(
            {
                "label": label,
                "latent": result["latent"],
                "compression": result["compression"],
                "autoencoder_seconds": stage_one["train_seconds"],
                "diffusion_seconds": stage_two["train_seconds"],
                "ceiling_fid": result["reconstruction_ceiling"]["fid"],
                "ceiling_psnr": result["reconstruction_ceiling"]["psnr_db"],
                "fid": result["samples"]["fid"],
                "recall": result["samples"]["recall"],
                "ms_per_image": result["cost"]["latent"]["seconds_per_image"] * 1000,
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'latent':>10s} {'compress':>9s} {'stage1 s':>9s} "
        f"{'stage2 s':>9s} {'ceiling':>8s} {'FID':>8s} {'headroom':>9s} {'ms/img':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['latent']:>10s} "
            f"{row['compression']:8.1f}x {row['autoencoder_seconds']:9.1f} "
            f"{row['diffusion_seconds']:9.1f} {row['ceiling_fid']:8.2f} {row['fid']:8.2f} "
            f"{row['fid'] - row['ceiling_fid']:9.2f} {row['ms_per_image']:8.0f}"
        )
    print("\n'headroom' is FID minus the ceiling: near zero means the first stage is the")
    print("limit, and training the diffusion model longer will not help.")

    plot_bars(
        [row["label"] for row in rows],
        {"FID": [row["fid"] for row in rows],
         "autoencoder ceiling": [row["ceiling_fid"] for row in rows]},
        f"{out_root}/{args.study}.png",
        ylabel="FID",
        title=f"{args.study}: quality against the ceiling",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
