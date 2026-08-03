"""Controlled comparisons for the VQ-VAE.

    python compare.py --study codebook     # 32 / 128 / 512 entries
    python compare.py --study codebook_rule  # EMA vs a loss term
    python compare.py --study commitment   # how hard to pull the encoder to its code

Same data, same schedule, same seed in every arm. Stage one only - add
``--with-prior`` to train the autoregressive prior in each arm and score samples,
which roughly doubles the runtime.

``codebook`` is the study to run. Reconstruction improves as the codebook grows, but
not proportionally, and the perplexity column explains why: past a point the extra
entries are not used. A 512-entry codebook at perplexity 40 is a 40-entry codebook
that cost 512 entries of memory, and only the perplexity number makes that visible.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from train import run_training
from train_prior import run_prior_training
from utils import plot_bars, save_json

STUDIES = {
    "codebook": [
        ("32 codes", {"num_codes": 32}),
        ("128 codes", {"num_codes": 128}),
        ("512 codes", {"num_codes": 512}),
    ],
    "codebook_rule": [
        ("loss-based", {"ema": False}),
        ("EMA", {"ema": True}),
    ],
    "commitment": [
        ("commit 0.1", {"commitment": 0.1}),
        ("commit 0.25", {"commitment": 0.25}),
        ("commit 1.0", {"commitment": 1.0}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled VQ-VAE comparison.")
    parser.add_argument("--study", default="codebook", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--num-codes", type=int, default=128)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--prior-epochs", type=int, default=15)
    parser.add_argument("--with-prior", action="store_true", help="also train and score a prior")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--fid-samples", type=int, default=2000)
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
        summary = run_training(
            dataset=args.dataset,
            num_codes=overrides.get("num_codes", args.num_codes),
            commitment=overrides.get("commitment", args.commitment),
            ema=overrides.get("ema", True),
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=32 if args.smoke_test else args.batch_size,
            lr=args.lr,
            train_subset=args.train_subset or None,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_').replace('.', '')}",
            synthetic=args.smoke_test,
            verbose=False,
        )
        if args.with_prior:
            run_prior_training(
                checkpoint=summary["checkpoint"],
                epochs=1 if args.smoke_test else args.prior_epochs,
                batch_size=32 if args.smoke_test else args.batch_size,
                train_subset=args.train_subset or None,
                num_workers=workers,
                seed=args.seed,
                device=args.device,
                data_root=args.data_root,
                synthetic=args.smoke_test,
                verbose=False,
            )

        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                prior=None,
                data_root=args.data_root,
                dataset=None,
                temperature=1.0,
                fid_samples=64 if args.smoke_test else args.fid_samples,
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
                "seconds": summary["train_seconds"],
                "num_codes": result["num_codes"],
                "psnr": result["reconstruction_psnr_db"],
                "ssim": result["reconstruction_ssim"],
                "codes_used": result["codes_used"],
                "perplexity": result["perplexity"],
                "fid": result.get("samples", {}).get("fid"),
                "precision": result.get("samples", {}).get("precision"),
                "recall": result.get("samples", {}).get("recall"),
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'train s':>8s} {'PSNR':>7s} {'SSIM':>7s} "
        f"{'used':>9s} {'perplexity':>11s}"
    )
    if args.with_prior:
        header += f" {'FID':>8s} {'prec':>6s} {'rec':>6s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{row['label'].ljust(width)} {row['seconds']:8.1f} {row['psnr']:7.2f} "
            f"{row['ssim']:7.4f} {row['codes_used']:4d}/{row['num_codes']:<4d} "
            f"{row['perplexity']:11.1f}"
        )
        if args.with_prior:
            line += f" {row['fid']:8.2f} {row['precision']:6.3f} {row['recall']:6.3f}"
        print(line)
    print("\nperplexity is the effective codebook size. Compare it with the nominal one.")

    series = {
        "PSNR / 30 dB": [row["psnr"] / 30 for row in rows],
        "SSIM": [row["ssim"] for row in rows],
        "used fraction": [row["codes_used"] / row["num_codes"] for row in rows],
    }
    plot_bars([row["label"] for row in rows], series, f"{out_root}/{args.study}.png",
              ylabel="score (rescaled)", title=f"{args.study} study on {args.dataset}")
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
