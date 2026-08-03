"""Train the detector.

    python train.py --dataset digits --epochs 10
    python train.py --dataset digits --stride 4     # finer grid, more findable objects

The training loss is a sum of three things that are not measured in the same units
- objectness, class and box - so it is a poor progress signal on its own. Each
epoch therefore also computes **validation mAP@0.5**, and that is what selects the
checkpoint. It costs a decode plus NMS plus the matching loop over the validation
split, which is a real fraction of an epoch and worth every bit of it: the loss can
fall for a while purely by getting better at predicting "empty" on the cells that
are empty.

The printed breakdown separates ``obj`` (positives) from ``noobj`` (everything
else) for the same reason. Early on ``noobj`` collapses almost immediately, because
guessing "nothing here" is right for a few hundred cells out of a few hundred.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import DATASETS, get_splits, make_loader
from model import build_detector, detection_loss
from utils import (
    AverageMeter,
    count_parameters,
    evaluate_detections,
    get_device,
    plot_curves,
    plot_detections,
    save_json,
    set_seed,
)


@torch.no_grad()
def collect_detections(model, loader, device, image_size: int, conf_threshold: float, nms_iou: float):
    """Run the detector over a split, returning ``(predictions, targets, images, batch)``."""
    model.eval()
    predictions, targets = [], []
    first_images, first_predictions, first_targets = None, None, None

    for images, batch_targets in loader:
        prediction = model(images.to(device))
        decoded = model.detect(prediction, image_size, conf_threshold, nms_iou)
        predictions.extend(decoded)
        targets.extend(batch_targets)
        if first_images is None:
            first_images = images.cpu()
            first_predictions = decoded
            first_targets = batch_targets

    return predictions, targets, first_images, (first_predictions, first_targets)


def run_training(
    dataset: str = "digits",
    image_size: int | None = None,
    stride: int = 8,
    base_channels: int = 32,
    box_loss: str = "ciou",
    box_weight: float = 5.0,
    noobj_weight: float = 1.0,
    conf_threshold: float = 0.05,
    nms_iou: float = 0.5,
    max_objects: int = 3,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-3,
    weight_decay: float = 5e-4,
    train_size: int = 4000,
    val_size: int = 500,
    test_size: int = 1000,
    num_workers: int = 2,
    seed: int = 0,
    device: str = "auto",
    data_root: str = "data",
    out_dir: str | None = None,
    synthetic: bool = False,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    dev = get_device(device)
    out_path = Path(out_dir or f"outputs/{dataset}_stride{stride}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        dataset,
        root=data_root,
        image_size=image_size,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        max_objects=max_objects,
        seed=seed,
        synthetic=synthetic,
    )
    size = info["size"]
    train_loader = make_loader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_set, batch_size, num_workers=num_workers)

    model = build_detector(info["channels"], info["classes"], base_channels, stride).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    config = {
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": info["classes"],
        "class_names": info["class_names"],
        "image_size": size,
        "stride": stride,
        "base_channels": base_channels,
        "box_loss": box_loss,
    }

    if verbose:
        grid = size // stride
        print(f"device   : {dev}")
        print(f"model    : stride {stride}, {grid}x{grid} = {grid * grid} cells, "
              f"{count_parameters(model):,} parameters")
        print(f"data     : {len(train_set)} train / {len(val_set)} val "
              f"at {info['channels']}x{size}x{size}")
        print(f"classes  : {info['classes']}  box loss {box_loss}\n")

    history = {"train_loss": [], "train_obj": [], "train_box": [], "val_map_50": []}
    best_map, best_epoch = -float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        meters = {key: AverageMeter() for key in ("loss", "obj", "noobj", "cls", "box")}
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for images, targets in batches:
            images = images.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = detection_loss(
                model,
                model(images),
                targets,
                box_loss=box_loss,
                box_weight=box_weight,
                noobj_weight=noobj_weight,
            )
            loss.backward()
            optimizer.step()
            for key, meter in meters.items():
                meter.update(parts[key], images.size(0))
            batches.set_postfix(loss=f"{meters['loss'].avg:.3f}")
        scheduler.step()

        predictions, targets, _, _ = collect_detections(
            model, val_loader, dev, size, conf_threshold, nms_iou
        )
        val = evaluate_detections(predictions, targets, info["classes"], thresholds=(0.5,))

        history["train_loss"].append(meters["loss"].avg)
        history["train_obj"].append(meters["obj"].avg)
        history["train_box"].append(meters["box"].avg)
        history["val_map_50"].append(val["map_50"])

        if val["map_50"] >= best_map:
            best_map, best_epoch = val["map_50"], epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_map_50": val["map_50"],
                 **config},
                ckpt_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  loss {meters['loss'].avg:.3f} "
                f"(obj {meters['obj'].avg:.3f} noobj {meters['noobj'].avg:.3f} "
                f"cls {meters['cls'].avg:.3f} box {meters['box'].avg:.3f})  "
                f"val mAP@0.5 {val['map_50']:.4f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss", "map_50"))
        _, _, images, (predictions, targets) = collect_detections(
            model, val_loader, dev, size, conf_threshold, nms_iou
        )
        plot_detections(
            images[:8], predictions[:8], targets[:8], info["class_names"],
            out_path / "detections_val.png", title="validation detections",
        )

    summary = {
        **config,
        "parameters": count_parameters(model),
        "grid": size // stride,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_map_50": best_map,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose and epochs:
        print(f"\nbest val mAP@0.5 {best_map:.4f} at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the tiny detector.")
    parser.add_argument("--dataset", default="digits", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--stride", type=int, default=8, choices=[2, 4, 8, 16],
        help="grid stride; one box per cell, so this caps how close two objects can be",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--box-loss", default="ciou", choices=["ciou", "l1"])
    parser.add_argument("--box-weight", type=float, default=5.0)
    parser.add_argument("--noobj-weight", type=float, default=1.0)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-objects", type=int, default=3, help="digits dataset only")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch on generated shapes")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        dataset=args.dataset,
        image_size=args.image_size,
        stride=args.stride,
        base_channels=args.base_channels,
        box_loss=args.box_loss,
        box_weight=args.box_weight,
        noobj_weight=args.noobj_weight,
        # A one-epoch smoke run never clears the confidence threshold, so drop it
        # and exercise the decode, NMS and matching code that mAP depends on.
        conf_threshold=0.0 if args.smoke_test else args.conf_threshold,
        nms_iou=args.nms_iou,
        max_objects=args.max_objects,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=8 if args.smoke_test else args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out_dir,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
