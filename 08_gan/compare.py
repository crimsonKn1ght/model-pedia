"""Controlled comparisons for the GAN.

    python compare.py --study loss         # BCE vs hinge
    python compare.py --study smoothing    # one-sided label smoothing
    python compare.py --study balance      # discriminator steps per generator step

Same data, same schedule, same seed in every arm, and every arm is scored on FID, KID
and precision/recall rather than on its own loss - which is the only way to compare
GAN training tricks at all, since the losses are not measured in the same units
between arms and are not a quality signal within one.

``balance`` is the instructive one. The theory says the discriminator should be trained
to optimality between generator steps; the practice is that a discriminator which wins
too clearly stops producing gradient, and the run stalls. Watch ``D(real)`` and
``D(fake)`` in the per-epoch output as you raise ``--d-steps``.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from model import LOSSES
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "loss": [
        ("BCE", {"loss": "bce"}),
        ("hinge", {"loss": "hinge"}),
    ],
    "smoothing": [
        ("no smoothing", {"label_smoothing": 0.0}),
        ("smoothing 0.1", {"label_smoothing": 0.1}),
    ],
    "balance": [
        ("1 D step", {"d_steps": 1}),
        ("2 D steps", {"d_steps": 2}),
    ],
    "latent": [
        ("latent 16", {"latent_dim": 16}),
        ("latent 64", {"latent_dim": 64}),
        ("latent 128", {"latent_dim": 128}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled GAN comparison.")
    parser.add_argument("--study", default="loss", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--loss", default="bce", choices=LOSSES)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
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
            loss=overrides.get("loss", args.loss),
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=32 if args.smoke_test else args.batch_size,
            lr=args.lr,
            train_subset=args.train_subset or None,
            fid_track_samples=64 if args.smoke_test else 1000,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_').replace('.', '')}",
            synthetic=args.smoke_test,
            verbose=False,
            **{key: value for key, value in overrides.items() if key != "loss"},
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
        history = summary["history"]
        rows.append(
            {
                "label": label,
                "seconds": summary["train_seconds"],
                "best_epoch": summary["best_epoch"],
                "fid": result["samples"]["fid"],
                "kid": result["samples"]["kid"],
                "precision": result["samples"]["precision"],
                "recall": result["samples"]["recall"],
                "d_real": history["train_d_real"][-1] if history["train_d_real"] else float("nan"),
                "d_fake": history["train_d_fake"][-1] if history["train_d_fake"] else float("nan"),
                "memorisation_ratio": result["memorisation"]["ratio"],
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'train s':>8s} {'best ep':>8s} {'FID':>8s} {'KID':>9s} "
        f"{'prec':>6s} {'rec':>6s} {'D(real)':>8s} {'D(fake)':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['seconds']:8.1f} {row['best_epoch']:8d} "
            f"{row['fid']:8.2f} {row['kid']:+9.5f} {row['precision']:6.3f} {row['recall']:6.3f} "
            f"{row['d_real']:8.3f} {row['d_fake']:8.3f}"
        )
    print("\nlow recall at high precision is mode collapse. D(real) near 1 with D(fake)")
    print("near 0 means the discriminator won and the generator stopped learning.")

    plot_bars(
        [row["label"] for row in rows],
        {
            "FID": [row["fid"] for row in rows],
            "precision x 100": [row["precision"] * 100 for row in rows],
            "recall x 100": [row["recall"] * 100 for row in rows],
        },
        f"{out_root}/{args.study}.png",
        ylabel="score",
        title=f"{args.study} study on {args.dataset}",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
