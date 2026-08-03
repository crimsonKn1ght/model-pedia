"""Controlled comparisons for segmentation.

    python compare.py --study arch     # U-Net vs no skips vs full-resolution dilated
    python compare.py --study loss     # cross entropy vs Dice vs both

Same data, same schedule, same seed in every arm; one thing changes.

``arch`` isolates the skip connections. Cutting them barely moves pixel accuracy -
the background is still easy - and takes a visible bite out of mean IoU, because
what the skips carry is the high-resolution detail the thin classes are made of.

``loss`` isolates class imbalance. Cross entropy is a per-pixel average, so a class
covering 5% of the pixels gets 5% of the gradient. Dice normalises per class, and
the combination usually gives the best mIoU while keeping cross entropy's stable
early gradients.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from train import LOSSES, run_training
from utils import plot_bars, save_json

STUDIES = {
    "arch": [
        ("dilated", {"model_name": "dilated"}),
        ("no skips", {"model_name": "unet_noskip"}),
        ("U-Net", {"model_name": "unet"}),
    ],
    "loss": [
        ("cross entropy", {"loss": "ce"}),
        ("Dice", {"loss": "dice"}),
        ("CE + Dice", {"loss": "ce+dice"}),
    ],
    "capacity": [
        ("base 8", {"base_channels": 8}),
        ("base 16", {"base_channels": 16}),
        ("base 32", {"base_channels": 32}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled segmentation comparison.")
    parser.add_argument("--study", default="arch", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="shapes", choices=sorted(DATASETS))
    parser.add_argument("--model", default="unet", help="held fixed unless the study varies it")
    parser.add_argument("--loss", default="ce", choices=LOSSES)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--train-subset", type=int, default=4000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke-test", action="store_true", help="run on generated 32x32 data")
    args = parser.parse_args()

    workers = 0 if args.smoke_test else args.num_workers
    out_root = "outputs/smoke" if args.smoke_test else f"outputs/{args.study}"
    rows = []

    for label, overrides in STUDIES[args.study]:
        print(f"\n=== {args.study}: {label} ===")
        summary = run_training(
            model_name=overrides.get("model_name", args.model),
            dataset=args.dataset,
            image_size=args.image_size,
            loss=overrides.get("loss", args.loss),
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=8 if args.smoke_test else args.batch_size,
            lr=args.lr,
            train_subset=args.train_subset or None,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_').replace('+', 'and')}",
            synthetic=args.smoke_test,
            verbose=False,
            **{k: v for k, v in overrides.items() if k not in ("model_name", "loss")},
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                data_root=args.data_root,
                image_size=None,
                batch_size=args.batch_size,
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
                "parameters": summary["parameters"],
                "seconds": summary["train_seconds"],
                "miou": result["mean_iou"],
                "dice": result["mean_dice"],
                "pixel_accuracy": result["pixel_accuracy"],
                "per_class_iou": {row["class"]: row["iou"] for row in result["per_class"]},
                "baseline_miou": result["majority_class_baseline"]["miou"],
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'params':>10s} {'train s':>8s} "
        f"{'mIoU':>7s} {'Dice':>7s} {'pixel acc':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['parameters']:10,d} {row['seconds']:8.1f} "
            f"{row['miou']:7.4f} {row['dice']:7.4f} {row['pixel_accuracy']:10.4f}"
        )
    print(f"\nmajority-class baseline: mIoU {rows[0]['baseline_miou']:.4f}")

    class_names = list(rows[0]["per_class_iou"])
    print("\nper-class IoU:")
    header = "arm".ljust(width) + "".join(f" {name[:10]:>10s}" for name in class_names)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = row["label"].ljust(width)
        for name in class_names:
            line += f" {row['per_class_iou'][name]:10.4f}"
        print(line)

    labels = [row["label"] for row in rows]
    plot_bars(
        labels,
        {
            "mean IoU": [row["miou"] for row in rows],
            "mean Dice": [row["dice"] for row in rows],
            "pixel accuracy": [row["pixel_accuracy"] for row in rows],
        },
        f"{out_root}/{args.study}.png",
        ylabel="test score",
        title=f"{args.study} study on {args.dataset}",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
