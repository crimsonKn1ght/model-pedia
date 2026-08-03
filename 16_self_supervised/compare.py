"""Controlled comparisons for self-supervised pretraining.

    python compare.py --study method     # SimCLR vs BYOL
    python compare.py --study augment    # what the augmentations are worth

Same encoder, same data, same schedule, same seed in every arm - only the named
thing changes. Each arm is pretrained, then probed against its own untrained
control, so the reported gain is always a within-arm difference.

The ``augment`` study is the one to run first. Removing colour jitter leaves the
model able to match views on average colour, and removing augmentation entirely
makes the two views identical, at which point the loss goes to almost nothing and
the features are worth almost nothing. That is the clearest demonstration in the
repository that in self-supervised learning the augmentation *is* the supervision.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "method": [
        ("SimCLR", {"method": "simclr"}),
        ("BYOL", {"method": "byol"}),
    ],
    "augment": [
        ("no augment", {"augment": "none"}),
        ("crop only", {"augment": "crop"}),
        ("crop+colour", {"augment": "full"}),
    ],
    "width": [
        ("width 16", {"width": 16}),
        ("width 32", {"width": 32}),
    ],
}

REPORT_FRACTION = "1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled self-supervised comparison.")
    parser.add_argument("--study", default="augment", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--method", default="simclr", help="held fixed unless the study varies it")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--probe-subset", type=int, default=10000)
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke-test", action="store_true", help="run on random tensors")
    args = parser.parse_args()

    workers = 0 if args.smoke_test else args.num_workers
    out_root = "outputs/smoke" if args.smoke_test else f"outputs/{args.study}"
    rows = []

    for label, overrides in STUDIES[args.study]:
        print(f"\n=== {args.study}: {label} ===")
        summary = run_training(
            method=overrides.get("method", args.method),
            dataset=args.dataset,
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=32 if args.smoke_test else args.batch_size,
            lr=args.lr,
            train_subset=args.train_subset or None,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_').replace('+', '_')}",
            synthetic=args.smoke_test,
            **{key: value for key, value in overrides.items() if key != "method"},
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                data_root=args.data_root,
                probe_subset=args.probe_subset or None,
                knn_k=args.knn_k,
                probe_epochs=5 if args.smoke_test else args.probe_epochs,
                label_fractions=[1.0],
                batch_size=256,
                num_workers=workers,
                device=args.device,
                seed=args.seed,
                smoke_test=args.smoke_test,
            ),
            verbose=False,
        )
        pretrained = result["arms"]["pretrained"]
        control = result["arms"]["random init"]
        rows.append(
            {
                "label": label,
                "method": summary["method"],
                "seconds": summary["train_seconds"],
                "final_loss": summary["history"]["train_loss"][-1]
                if summary["history"]["train_loss"]
                else float("nan"),
                "knn": pretrained["knn_accuracy"],
                "knn_control": control["knn_accuracy"],
                "linear": pretrained["linear_probe"][REPORT_FRACTION]["test_accuracy"],
                "linear_control": control["linear_probe"][REPORT_FRACTION]["test_accuracy"],
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} pretrain epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'train s':>8s} {'SSL loss':>9s} "
        f"{'k-NN':>7s} {'ctrl':>7s} {'linear':>7s} {'ctrl':>7s} {'gain':>7s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['seconds']:8.1f} {row['final_loss']:9.4f} "
            f"{row['knn']:7.4f} {row['knn_control']:7.4f} "
            f"{row['linear']:7.4f} {row['linear_control']:7.4f} "
            f"{row['linear'] - row['linear_control']:+7.4f}"
        )
    print("\n'ctrl' is the same architecture with random weights - the gain over it")
    print("is the only part of the accuracy that pretraining is responsible for.")

    labels = [row["label"] for row in rows]
    plot_bars(
        labels,
        {
            "k-NN": [row["knn"] for row in rows],
            "linear probe": [row["linear"] for row in rows],
            "random-init control": [row["linear_control"] for row in rows],
        },
        f"{out_root}/{args.study}.png",
        ylabel="test accuracy",
        title=f"{args.study} study on {args.dataset}",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
