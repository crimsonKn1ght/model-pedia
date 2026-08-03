"""Score a segmentation checkpoint on the test split.

    python evaluate.py --checkpoint outputs/shapes_unet/best.pt
    python evaluate.py --checkpoint ... --image-size 128    # fully convolutional, so this works

Reports mean IoU, per-class IoU and Dice, and pixel accuracy, each next to the
``predict the majority class everywhere`` baseline. The baseline is the point of
the table: on a background-dominated dataset it scores a pixel accuracy that looks
like a working model, and a mean IoU that shows it is not one. When the two
metrics disagree, mIoU is the one telling the truth.

The row-normalised confusion matrix is where to look next. It says *what* the
errors are rather than how many: on Oxford Pets the border class mostly leaks into
pet and background, which is a statement about resolution, not about recognition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import get_splits, make_loader
from model import build_model
from train import majority_class_baseline
from utils import (
    ConfusionMatrix,
    get_device,
    plot_confusion_matrix,
    plot_segmentation_rows,
    save_json,
    set_seed,
)


@torch.no_grad()
def score_split(model, loader, device, num_classes: int) -> ConfusionMatrix:
    model.eval()
    confusion = ConfusionMatrix(num_classes)
    for images, masks in loader:
        predictions = model(images.to(device)).argmax(dim=1)
        confusion.update(predictions, masks.to(device))
    return confusion


@torch.no_grad()
def save_examples(model, loader, device, num_classes: int, path, n: int = 8) -> None:
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images[:n].to(device), masks[:n].to(device)
    predictions = model(images).argmax(dim=1)
    plot_segmentation_rows(
        images.cpu(), masks.cpu(), predictions.cpu(), num_classes, path,
        title="test predictions",
    )


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)

    _, _, test_base, info = get_splits(
        ckpt["dataset"], root=args.data_root, seed=args.seed, synthetic=args.smoke_test
    )
    size = args.image_size or ckpt["image_size"]
    num_classes = ckpt["num_classes"]
    class_names = ckpt["class_names"] or [str(i) for i in range(num_classes)]

    test_loader = make_loader(
        test_base, info, size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = build_model(
        ckpt["model"], ckpt["in_channels"], num_classes, ckpt["base_channels"]
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])

    confusion = score_split(model, test_loader, dev, num_classes)
    baseline = majority_class_baseline(confusion)

    out_dir = Path(args.checkpoint).parent
    suffix = "" if size == ckpt["image_size"] else f"_{size}px"
    plot_confusion_matrix(
        confusion.matrix,
        class_names,
        out_dir / f"confusion{suffix}.png",
        title=f"{ckpt['model']} on {ckpt['dataset']} test ({size}x{size})",
    )
    save_examples(model, test_loader, dev, num_classes, out_dir / f"examples{suffix}.png")

    result = {
        "model": ckpt["model"],
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "trained_at_size": ckpt["image_size"],
        "evaluated_at_size": size,
        "test_images": len(test_base),
        **confusion.summary(class_names),
        "majority_class_baseline": {
            "class": class_names[baseline["class"]],
            "miou": baseline["miou"],
            "pixel_accuracy": baseline["pixel_accuracy"],
        },
    }
    save_json(result, out_dir / f"metrics{suffix}.json")

    if verbose:
        note = "" if size == ckpt["image_size"] else f"  (trained at {ckpt['image_size']})"
        print(f"model    : {ckpt['model']} from epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_base)} images at {size}x{size}{note}\n")

        pad = max(len(name) for name in class_names)
        header = f"{'class'.ljust(pad)} {'IoU':>7s} {'Dice':>7s} {'pixels':>8s}"
        print(header)
        print("-" * len(header))
        for row in result["per_class"]:
            print(
                f"{row['class'].ljust(pad)} {row['iou']:7.4f} {row['dice']:7.4f} "
                f"{row['pixel_share']:7.2%}"
            )
        print("-" * len(header))
        print(f"{'mean'.ljust(pad)} {result['mean_iou']:7.4f} {result['mean_dice']:7.4f}")
        print(f"\npixel accuracy : {result['pixel_accuracy']:.4f}")
        print(
            f"baseline       : always predict '{result['majority_class_baseline']['class']}' -> "
            f"mIoU {result['majority_class_baseline']['miou']:.4f}, "
            f"pixel accuracy {result['majority_class_baseline']['pixel_accuracy']:.4f}"
        )
        print(f"\nconfusion -> {out_dir / f'confusion{suffix}.png'}")
        print(f"examples  -> {out_dir / f'examples{suffix}.png'}")
        print(f"metrics   -> {out_dir / f'metrics{suffix}.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/shapes_unet/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="evaluate at a different resolution than training; the nets are fully convolutional",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on generated 32x32 data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_workers = 0
        args.image_size = None
    run_evaluation(args)


if __name__ == "__main__":
    main()
