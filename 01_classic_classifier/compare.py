"""Train every model on the same data and print one comparison table.

    python compare.py --dataset fashion-mnist --epochs 5

This is the point of the project: identical data, identical optimiser, only the
architecture changes, so the accuracy difference belongs to the architecture.
"""

from __future__ import annotations

import argparse

from data import DATASETS
from evaluate import run_evaluation
from model import MODELS
from train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the MLP and LeNet side by side.")
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=sorted(MODELS), choices=sorted(MODELS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true", help="run on random tensors")
    args = parser.parse_args()

    workers = 0 if args.smoke_test else args.num_workers
    rows = []
    for model_name in args.models:
        print(f"\n=== {model_name} on {args.dataset} ===")
        summary = run_training(
            model_name=model_name,
            dataset=args.dataset,
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            num_workers=workers,
            out_dir=f"outputs/smoke/{model_name}" if args.smoke_test else None,
            synthetic=args.smoke_test,
        )
        result = run_evaluation(
            checkpoint=summary["checkpoint"],
            data_root=args.data_root,
            device=args.device,
            num_workers=workers,
            seed=args.seed,
            synthetic=args.smoke_test,
            verbose=False,
        )
        rows.append(
            {
                "model": model_name,
                "params": summary["parameters"],
                "seconds": summary["train_seconds"],
                "val_acc": summary["best_val_acc"],
                "test_acc": result["test_accuracy"],
                "macro_f1": result["macro_f1"],
            }
        )

    print(f"\n=== summary: {args.dataset}, {args.epochs} epochs, seed {args.seed} ===")
    header = f"{'model':8s} {'params':>10s} {'train s':>8s} {'val acc':>8s} {'test acc':>9s} {'macro F1':>9s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['model']:8s} {row['params']:10,d} {row['seconds']:8.1f} "
            f"{row['val_acc']:8.4f} {row['test_acc']:9.4f} {row['macro_f1']:9.4f}"
        )


if __name__ == "__main__":
    main()
