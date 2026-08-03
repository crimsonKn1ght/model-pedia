"""Score a detector on the test split: mAP, per-class AP, PR curves, pictures.

    python evaluate.py --checkpoint outputs/digits_stride8/best.pt
    python evaluate.py --checkpoint ... --nms-sweep 0.3 0.5 0.7 0.9

Two numbers, and the distance between them is the interesting part:

* **mAP@0.5** - a detection counts if it overlaps the object by half. This mostly
  asks *did you find it*.
* **mAP@0.5:0.95** - the average over ten thresholds from 0.5 to 0.95. Boxes that
  are roughly right score at 0.5 and nothing at 0.9, so this mostly asks *how well
  does the box fit*. It is always the lower of the two, often by a lot, and a large
  gap says the detector is finding objects but drawing sloppy boxes.

``--nms-sweep`` re-scores the same predictions under several NMS thresholds. Too
strict and neighbouring objects suppress each other, costing recall; too loose and
duplicates survive as false positives. Nothing about the trained weights changes,
which makes it a clean demonstration that a chunk of a detector's reported accuracy
lives in its post-processing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import get_splits, make_loader
from model import build_detector
from train import collect_detections
from utils import (
    COCO_THRESHOLDS,
    evaluate_detections,
    get_device,
    plot_bars,
    plot_detections,
    plot_pr_curves,
    save_json,
    set_seed,
)


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)

    _, _, test_set, info = get_splits(
        ckpt["dataset"],
        root=args.data_root,
        image_size=ckpt["image_size"],
        test_size=args.test_size,
        seed=args.seed,
        synthetic=args.smoke_test,
    )
    size = info["size"]
    class_names = ckpt["class_names"]
    num_classes = ckpt["num_classes"]

    test_loader = make_loader(test_set, args.batch_size, num_workers=args.num_workers)
    model = build_detector(
        ckpt["in_channels"], num_classes, ckpt["base_channels"], ckpt["stride"]
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])

    predictions, targets, images, (example_predictions, example_targets) = collect_detections(
        model, test_loader, dev, size, args.conf_threshold, args.nms_iou
    )
    scored = evaluate_detections(predictions, targets, num_classes, thresholds=COCO_THRESHOLDS)

    out_dir = Path(args.checkpoint).parent
    plot_pr_curves(
        scored["curves"], scored["per_class"], class_names, out_dir / "pr_curves.png",
        title=f"{ckpt['dataset']} test, IoU 0.5 (stride {ckpt['stride']})",
    )
    plot_detections(
        images[:8], example_predictions[:8], example_targets[:8], class_names,
        out_dir / "detections.png", title="test detections",
    )

    result = {
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "stride": ckpt["stride"],
        "grid": size // ckpt["stride"],
        "test_images": len(test_set),
        "conf_threshold": args.conf_threshold,
        "nms_iou": args.nms_iou,
        "map_50": scored["map_50"],
        "map_75": scored["map_75"],
        "map_50_95": scored["map_50_95"],
        "predictions_kept": sum(int(p["boxes"].size(0)) for p in predictions),
        "ground_truth_objects": sum(int(t["labels"].numel()) for t in targets),
        "per_class": {
            (class_names[cls] if class_names and cls < len(class_names) else str(cls)): entry
            for cls, entry in scored["per_class"].items()
        },
    }

    if args.nms_sweep:
        sweep = []
        for threshold in args.nms_sweep:
            swept, swept_targets, _, _ = collect_detections(
                model, test_loader, dev, size, args.conf_threshold, threshold
            )
            scores = evaluate_detections(swept, swept_targets, num_classes, thresholds=(0.5, 0.75))
            sweep.append(
                {
                    "nms_iou": threshold,
                    "map_50": scores["map_50"],
                    "map_75": scores["map_75"],
                    "predictions_kept": sum(int(p["boxes"].size(0)) for p in swept),
                }
            )
        result["nms_sweep"] = sweep
        plot_bars(
            [f"NMS {row['nms_iou']:g}" for row in sweep],
            {
                "mAP@0.5": [row["map_50"] for row in sweep],
                "mAP@0.75": [row["map_75"] for row in sweep],
            },
            out_dir / "nms_sweep.png",
            ylabel="mAP",
            title="the same weights, different non-maximum suppression",
        )

    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"model    : stride {ckpt['stride']} ({result['grid']}x{result['grid']} grid), "
              f"epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_set)} images, "
              f"{result['ground_truth_objects']} objects")
        print(f"decoding : conf > {args.conf_threshold}, NMS IoU {args.nms_iou} -> "
              f"{result['predictions_kept']} detections\n")

        print(f"mAP@0.5      {result['map_50']:.4f}")
        print(f"mAP@0.75     {result['map_75']:.4f}")
        print(f"mAP@0.5:0.95 {result['map_50_95']:.4f}\n")

        pad = max(len(name) for name in result["per_class"]) if result["per_class"] else 5
        header = f"{'class'.ljust(pad)} {'AP@0.5':>8s} {'AP@0.5:0.95':>12s} {'objects':>8s}"
        print(header)
        print("-" * len(header))
        for name, entry in result["per_class"].items():
            print(
                f"{name.ljust(pad)} {entry['ap_50']:8.4f} {entry['ap_50_95']:12.4f} "
                f"{entry['num_ground_truth']:8d}"
            )

        if args.nms_sweep:
            print()
            header = f"{'NMS IoU':>8s} {'kept':>7s} {'mAP@0.5':>9s} {'mAP@0.75':>9s}"
            print(header)
            print("-" * len(header))
            for row in result["nms_sweep"]:
                print(
                    f"{row['nms_iou']:8.2f} {row['predictions_kept']:7d} "
                    f"{row['map_50']:9.4f} {row['map_75']:9.4f}"
                )

        print(f"\nPR curves  -> {out_dir / 'pr_curves.png'}")
        print(f"detections -> {out_dir / 'detections.png'}")
        print(f"metrics    -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a detector with mAP.")
    parser.add_argument("--checkpoint", default="outputs/digits_stride8/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument(
        "--conf-threshold", type=float, default=0.05,
        help="keep low: mAP rewards ranked low-confidence guesses rather than punishing them",
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--nms-sweep", type=float, nargs="*", default=None,
        help="re-score the same weights at several NMS thresholds",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on generated shapes")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_workers = 0
        args.conf_threshold = 0.0  # an untrained model is below any real threshold
    run_evaluation(args)


if __name__ == "__main__":
    main()
