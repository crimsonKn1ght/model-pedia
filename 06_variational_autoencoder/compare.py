"""Controlled comparisons for the VAE.

    python compare.py --study beta         # the reconstruction / KL exchange rate
    python compare.py --study latent       # how much latent is enough
    python compare.py --study likelihood   # Bernoulli vs Gaussian decoder

Same data, same schedule, same seed in every arm.

``beta`` is the one to run. Every arm is scored on **-ELBO with beta set back to 1**,
so the numbers are comparable even though the training objectives were not. Raising
beta buys a tidier latent and costs reconstruction, and the interesting part is where
the cost lands: not spread evenly, but concentrated in latent dimensions switching
off one at a time. Watch the ``active`` column.

``latent`` shows the other side of the same coin. Adding capacity past what the data
needs does not hurt the ELBO, because the KL term simply switches the surplus
dimensions off - a VAE prunes its own latent, which is one of the more elegant
consequences of the objective.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from model import LIKELIHOODS
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "beta": [
        ("beta 0.5", {"beta": 0.5}),
        ("beta 1", {"beta": 1.0}),
        ("beta 4", {"beta": 4.0}),
        ("beta 8", {"beta": 8.0}),
    ],
    "latent": [
        ("latent 2", {"latent_dim": 2}),
        ("latent 8", {"latent_dim": 8}),
        ("latent 32", {"latent_dim": 32}),
    ],
    "likelihood": [
        ("Bernoulli", {"likelihood": "bernoulli"}),
        ("Gaussian", {"likelihood": "gaussian"}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled VAE comparison.")
    parser.add_argument("--study", default="beta", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--likelihood", default="bernoulli", choices=LIKELIHOODS)
    parser.add_argument("--epochs", type=int, default=12)
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
            latent_dim=overrides.get("latent_dim", args.latent_dim),
            beta=overrides.get("beta", args.beta),
            likelihood=overrides.get("likelihood", args.likelihood),
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
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                data_root=args.data_root,
                dataset=None,
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
                "neg_elbo": result["neg_elbo_nats"],
                "reconstruction": result["reconstruction_nats"],
                "kl": result["kl_nats"],
                "active": result["active_units"],
                "latent_dim": result["latent_dim"],
                "psnr": result["reconstruction_psnr_db"],
                "fid": result["samples"]["fid"],
                "precision": result["samples"]["precision"],
                "recall": result["samples"]["recall"],
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'train s':>8s} {'-ELBO':>9s} {'recon':>9s} {'KL':>7s} "
        f"{'active':>7s} {'PSNR':>7s} {'FID':>8s} {'prec':>6s} {'rec':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['seconds']:8.1f} {row['neg_elbo']:9.2f} "
            f"{row['reconstruction']:9.2f} {row['kl']:7.2f} "
            f"{row['active']:3d}/{row['latent_dim']:<3d} {row['psnr']:7.2f} "
            f"{row['fid']:8.2f} {row['precision']:6.3f} {row['recall']:6.3f}"
        )
    print("\n-ELBO is reported with beta set back to 1, so the arms are comparable.")
    print("FID is measured through a small classifier trained on this dataset, not")
    print("InceptionV3, so compare these values only with each other.")

    plot_bars(
        [row["label"] for row in rows],
        {
            "reconstruction, nats": [row["reconstruction"] for row in rows],
            "KL, nats x 10": [row["kl"] * 10 for row in rows],
        },
        f"{out_root}/{args.study}_elbo.png",
        ylabel="nats",
        title=f"{args.study}: where the ELBO goes",
    )
    plot_bars(
        [row["label"] for row in rows],
        {
            "FID": [row["fid"] for row in rows],
            "precision x 100": [row["precision"] * 100 for row in rows],
            "recall x 100": [row["recall"] * 100 for row in rows],
        },
        f"{out_root}/{args.study}_samples.png",
        ylabel="score",
        title=f"{args.study}: sample quality",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigures -> {out_root}/{args.study}_elbo.png, {args.study}_samples.png")


if __name__ == "__main__":
    main()
