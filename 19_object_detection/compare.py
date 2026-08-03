"""Controlled comparisons for detection.

    python compare.py --study stride     # grid resolution, and what it costs to be coarse
    python compare.py --study box_loss   # CIoU vs L1 on the corners
    python compare.py --study crowd      # more objects per image, same detector

Same data, same schedule, same seed in every arm.

``stride`` is the structural one. This detector predicts one box per grid cell, so
the stride is a hard ceiling on how close two object centres can be and still both
be found. Halving it multiplies the cells by four - more capacity for crowded
scenes, more empty cells to learn to ignore, and a slower epoch.

``crowd`` attacks the same limit from the other side: keep the grid fixed and put
more objects in each image. mAP falls, and the fall is not the network getting worse
at recognition - it is the assignment rule running out of cells.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "stride": [
        ("stride 16", {"stride": 16}),
        ("stride 8", {"stride": 8}),
        ("stride 4", {"stride": 4}),
    ],
    "box_loss": [
        ("L1 corners", {"box_loss": "l1"}),
        ("CIoU", {"box_loss": "ciou"}),
    ],
    "crowd": [
        ("1 object", {"max_objects": 1}),
        ("3 objects", {"max_objects": 3}),
        ("5 objects", {"max_objects": 5}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled detection comparison.")
    parser.add_argument("--study", default="stride", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="digits", choices=sorted(DATASETS))
    parser.add_argument("--stride", type=int, default=8, choices=[2, 4, 8, 16])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke-test", action="store_true", help="run on generated shapes")
    args = parser.parse_args()

    workers = 0 if args.smoke_test else args.num_workers
    out_root = "outputs/smoke" if args.smoke_test else f"outputs/{args.study}"
    conf = 0.0 if args.smoke_test else args.conf_threshold
    rows = []

    for label, overrides in STUDIES[args.study]:
        print(f"\n=== {args.study}: {label} ===")
        summary = run_training(
            dataset=args.dataset,
            image_size=args.image_size,
            stride=overrides.get("stride", args.stride),
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=8 if args.smoke_test else args.batch_size,
            lr=args.lr,
            conf_threshold=conf,
            nms_iou=args.nms_iou,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_')}",
            synthetic=args.smoke_test,
            verbose=False,
            **{key: value for key, value in overrides.items() if key != "stride"},
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                data_root=args.data_root,
                test_size=args.test_size,
                conf_threshold=conf,
                nms_iou=args.nms_iou,
                nms_sweep=None,
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
                "cells": summary["grid"] ** 2,
                "seconds": summary["train_seconds"],
                "map_50": result["map_50"],
                "map_75": result["map_75"],
                "map_50_95": result["map_50_95"],
                "objects": result["ground_truth_objects"],
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'params':>10s} {'cells':>6s} {'train s':>8s} "
        f"{'objects':>8s} {'mAP@0.5':>9s} {'mAP@0.75':>9s} {'mAP@.5:.95':>11s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['parameters']:10,d} {row['cells']:6d} "
            f"{row['seconds']:8.1f} {row['objects']:8d} {row['map_50']:9.4f} "
            f"{row['map_75']:9.4f} {row['map_50_95']:11.4f}"
        )
    print("\nmAP@0.5 asks whether the object was found; the 0.75 and 0.5:0.95 columns")
    print("ask how well the box fits, and drop much faster when the grid is coarse.")

    plot_bars(
        [row["label"] for row in rows],
        {
            "mAP@0.5": [row["map_50"] for row in rows],
            "mAP@0.75": [row["map_75"] for row in rows],
            "mAP@0.5:0.95": [row["map_50_95"] for row in rows],
        },
        f"{out_root}/{args.study}.png",
        ylabel="mAP",
        title=f"{args.study} study on {args.dataset}",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
