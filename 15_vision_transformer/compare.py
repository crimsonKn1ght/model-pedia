"""Controlled comparisons for the Vision Transformer.

    python compare.py --study arch      # ViT vs ResNet at matched parameters and time
    python compare.py --study augment   # how much augmentation closes the gap
    python compare.py --study patch     # patch size, i.e. how many tokens

Same data, same schedule, same seed, same augmentation in every arm unless it is the
variable.

``arch`` is the headline. The two models are within 3% of each other on parameter count,
so the table also reports images per second and accuracy per second of training - because
a comparison at matched size while quietly spending different amounts of compute is not a
comparison. On CIFAR-10 at this scale the ResNet is expected to win; the interesting
question is by how much, and what closes it.

``augment`` answers that second question. A convolution has locality and translation
equivariance built in and a transformer has to learn them, so the transformer is the one
that gains from having more views of the data.

``patch`` is the token-count dial. Halving the patch size quadruples the tokens, which
quadruples the attention cost - and the accuracy gain is much smaller than the cost.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import AUGMENTATIONS, DATASETS
from evaluate import run_evaluation
from model import ARCHITECTURES
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "arch": [
        ("ResNet", {"architecture": "resnet"}),
        ("ViT", {"architecture": "vit"}),
    ],
    "augment": [
        ("no augment", {"augment": "none"}),
        ("basic", {"augment": "basic"}),
        ("strong", {"augment": "strong"}),
    ],
    "patch": [
        ("patch 2", {"patch_size": 2}),
        ("patch 4", {"patch_size": 4}),
        ("patch 8", {"patch_size": 8}),
    ],
    "depth": [
        ("depth 4", {"depth": 4}),
        ("depth 6", {"depth": 6}),
        ("depth 8", {"depth": 8}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled ViT comparison.")
    parser.add_argument("--study", default="arch", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--arch", default="vit", choices=ARCHITECTURES)
    parser.add_argument("--augment", default="basic", choices=AUGMENTATIONS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-subset", type=int, default=20000)
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
            architecture=overrides.get("architecture", args.arch),
            dataset=args.dataset,
            augment=overrides.get("augment", args.augment),
            patch_size=2 if args.smoke_test else overrides.get("patch_size", 4),
            depth=overrides.get("depth", 6),
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=32 if args.smoke_test else args.batch_size,
            lr=args.lr,
            train_subset=args.train_subset or None,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_')}",
            synthetic=args.smoke_test,
            verbose=False,
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                data_root=args.data_root,
                dataset=None,
                batch_size=32 if args.smoke_test else 128,
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
                "architecture": result["architecture"],
                "parameters": result["parameters"],
                "seconds": summary["train_seconds"],
                "accuracy": result["test_accuracy"],
                "train_images_per_second": result["throughput"]["train_images_per_second"],
                "accuracy_per_second": result["test_accuracy"] / max(summary["train_seconds"], 1e-9),
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'params':>11s} {'train s':>8s} {'img/s':>8s} "
        f"{'test acc':>9s} {'acc/1000s':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['parameters']:11,d} {row['seconds']:8.1f} "
            f"{row['train_images_per_second']:8.0f} {row['accuracy']:9.4f} "
            f"{row['accuracy_per_second'] * 1000:10.4f}"
        )
    print("\nmatched parameters is only half a control; the img/s column is the other half.")

    plot_bars(
        [row["label"] for row in rows],
        {"test accuracy": [row["accuracy"] for row in rows]},
        f"{out_root}/{args.study}.png",
        ylabel="top-1 accuracy",
        title=f"{args.study} study on {args.dataset}",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
