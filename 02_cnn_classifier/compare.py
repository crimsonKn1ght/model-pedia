"""Ablations that isolate what actually makes a CNN classifier work.

    python compare.py --study residual     # resnet20 vs the same net without skips
    python compare.py --study depth        # resnet20 vs resnet32
    python compare.py --study augment      # crop/flip on vs off

Every arm sees the same data, optimiser and seed, so the difference in the
final table belongs to the one thing that changed.
"""

from __future__ import annotations

import argparse

from data import DATASETS
from evaluate import run_evaluation
from train import run_training

STUDIES = {
    # name -> list of (label, overrides)
    "residual": [
        ("plain20", {"arch": "plain20"}),
        ("resnet20", {"arch": "resnet20"}),
        ("plain32", {"arch": "plain32"}),
        ("resnet32", {"arch": "resnet32"}),
    ],
    "depth": [
        ("resnet20", {"arch": "resnet20"}),
        ("resnet32", {"arch": "resnet32"}),
    ],
    "augment": [
        ("no augment", {"arch": "resnet20", "augment": False}),
        ("crop + flip", {"arch": "resnet20", "augment": True}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled comparison.")
    parser.add_argument("--study", default="residual", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--train-subset", type=int, default=15000)
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
            checkpoint=summary["checkpoint"],
            data_root=args.data_root,
            num_workers=workers,
            device=args.device,
            seed=args.seed,
            synthetic=args.smoke_test,
            verbose=False,
        )
        rows.append(
            {
                "label": label,
                "params": summary["parameters"],
                "seconds": summary["train_seconds"],
                "val_acc": summary["best_val_acc"],
                "top1": result["top1_accuracy"],
                "top5": result["top5_accuracy"],
            }
        )

    print(
        f"\n=== summary: {args.study} on {args.dataset}, {args.epochs} epochs, seed {args.seed} ==="
    )
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'params':>10s} {'train s':>8s} "
        f"{'val acc':>8s} {'top-1':>8s} {'top-5':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['params']:10,d} {row['seconds']:8.1f} "
            f"{row['val_acc']:8.4f} {row['top1']:8.4f} {row['top5']:8.4f}"
        )


if __name__ == "__main__":
    main()
