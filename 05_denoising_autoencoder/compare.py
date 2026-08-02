"""Ablations for the denoiser.

    python compare.py --study skips      # U-Net vs the same net without skips
    python compare.py --study target     # predict the clean image vs the noise
    python compare.py --study loss       # L1 vs L2

Same data, same schedule, same seed in every arm.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from train import run_training

STUDIES = {
    "skips": [
        ("no skips", {"model_name": "unet_noskip"}),
        ("U-Net", {"model_name": "unet"}),
    ],
    "target": [
        ("predict image", {"predict_residual": False}),
        ("predict noise", {"predict_residual": True}),
    ],
    "loss": [
        ("L2", {"loss": "l2"}),
        ("L1", {"loss": "l1"}),
    ],
}

REPORT_SIGMA = 0.15


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled denoising comparison.")
    parser.add_argument("--study", default="skips", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sigma-range", type=float, nargs=2, default=[0.05, 0.25])
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke-test", action="store_true", help="run on random tensors")
    args = parser.parse_args()

    workers = 0 if args.smoke_test else args.num_workers
    rows = []
    for label, overrides in STUDIES[args.study]:
        print(f"\n=== {args.study}: {label} ===")
        summary = run_training(
            dataset=args.dataset,
            sigma_range=tuple(args.sigma_range),
            val_sigma=REPORT_SIGMA,
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_subset=args.train_subset or None,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=(
                f"outputs/smoke/{label.replace(' ', '_')}"
                if args.smoke_test
                else f"outputs/{args.study}/{label.replace(' ', '_')}"
            ),
            synthetic=args.smoke_test,
            **overrides,
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                sigmas=[REPORT_SIGMA],
                test_dir=None,
                data_root=args.data_root,
                batch_size=256,
                num_workers=workers,
                device=args.device,
                seed=args.seed,
                smoke_test=args.smoke_test,
            ),
            verbose=False,
        )
        at_sigma = result["by_sigma"][0]
        rows.append(
            {
                "label": label,
                "params": summary["parameters"],
                "seconds": summary["train_seconds"],
                "noisy_psnr": at_sigma["noisy_psnr_db"],
                "psnr": at_sigma["denoised_psnr_db"],
                "gain": at_sigma["psnr_gain_db"],
                "ssim": at_sigma["denoised_ssim"],
            }
        )

    print(f"\n=== {args.study} on {args.dataset}, test sigma {REPORT_SIGMA} ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'params':>10s} {'train s':>8s} "
        f"{'PSNR dB':>8s} {'gain':>7s} {'SSIM':>7s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['params']:10,d} {row['seconds']:8.1f} "
            f"{row['psnr']:8.2f} {row['gain']:+7.2f} {row['ssim']:7.4f}"
        )
    print(f"\nnoisy input baseline: {rows[0]['noisy_psnr']:.2f} dB")


if __name__ == "__main__":
    main()
