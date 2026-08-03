"""Evaluate a ViT or ResNet checkpoint, and look inside the ViT.

    python evaluate.py --checkpoint outputs/cifar10_vit/best.pt

Reports top-1 accuracy, per-class accuracy and a confusion matrix on the test split,
plus the measured throughput of the model - so an accuracy number can always be read
next to what it cost.

For a ViT it also writes ``attention.png``: the attention rollout for a handful of test
images, overlaid on the input. Rollout composes the attention matrices through the depth
of the network, with the residual path added in as an identity, to show which input
patches the class token at the top actually depends on. It is the interpretability a
convolutional network does not hand you, and it is worth looking at even when the ViT is
losing on accuracy - a model attending to the object is failing differently from one
attending to the background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import DATASETS, denormalize, get_splits, make_loader
from model import build_model
from utils import (
    accuracy_and_confusion,
    count_parameters,
    get_device,
    measure_throughput,
    per_class_accuracy,
    plot_attention_maps,
    plot_confusion_matrix,
    save_json,
    set_seed,
)


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    num_classes = ckpt["num_classes"]
    class_names = ckpt["class_names"] or [str(i) for i in range(num_classes)]

    _, _, test_set, info = get_splits(
        ckpt["dataset"], root=args.data_root, image_size=ckpt["image_size"],
        augment="none", seed=args.seed, synthetic=args.smoke_test,
    )
    test_loader = make_loader(test_set, args.batch_size, num_workers=args.num_workers)

    model = build_model(
        ckpt["architecture"], num_classes, ckpt["image_size"], ckpt["in_channels"],
        ckpt["patch_size"], ckpt["dim"], ckpt["depth"], ckpt["heads"], 0.0, ckpt["width"],
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    accuracy, matrix = accuracy_and_confusion(model, test_loader, dev, num_classes)
    throughput = measure_throughput(
        model, (args.batch_size, ckpt["in_channels"], ckpt["image_size"], ckpt["image_size"]),
        dev, steps=2 if args.smoke_test else 5,
    )
    class_accuracy = per_class_accuracy(matrix)

    out_dir = Path(args.checkpoint).parent
    plot_confusion_matrix(
        matrix, class_names, out_dir / "confusion_matrix.png",
        title=f"{ckpt['architecture']} on {ckpt['dataset']} test (accuracy {accuracy:.3f})",
    )

    if ckpt["architecture"] == "vit":
        images, targets = next(iter(test_loader))
        images = images[:8].to(dev)
        rollout = model.attention_rollout(images)
        with torch.no_grad():
            predictions = model(images).argmax(dim=1).cpu()
        labels = [
            f"{class_names[int(t)]} -> {class_names[int(p)]}"
            for t, p in zip(targets[:8], predictions)
        ]
        plot_attention_maps(
            denormalize(images, info).cpu(), rollout, labels, out_dir / "attention.png",
            title="attention rollout: which patches the class token depends on",
        )

    result = {
        "architecture": ckpt["architecture"],
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "parameters": count_parameters(model),
        "test_images": len(test_set),
        "test_accuracy": accuracy,
        "per_class_accuracy": dict(zip(class_names, class_accuracy)),
        "throughput": throughput,
        "augment_trained_with": ckpt["augment"],
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"\nmodel    : {ckpt['architecture']} with {result['parameters']:,} parameters, "
              f"epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_set)} images\n")
        print(f"top-1 accuracy       {accuracy:.4f}")
        print(f"train images/second  {throughput['train_images_per_second']:.0f}")
        print(f"infer images/second  {throughput['inference_images_per_second']:.0f}")
        print(f"trained with augment {ckpt['augment']}\n")

        pad = max(len(name) for name in class_names)
        worst = sorted(zip(class_names, class_accuracy), key=lambda pair: pair[1])[:5]
        print("five weakest classes:")
        for name, value in worst:
            print(f"  {name.ljust(pad)} {value:.4f}")
        print(f"\nfigures -> {out_dir}/confusion_matrix.png"
              f"{', attention.png' if ckpt['architecture'] == 'vit' else ''}")
        print(f"metrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a ViT or ResNet checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/cifar10_vit/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASETS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on generated data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_workers = 0
        args.batch_size = 32
    run_evaluation(args)


if __name__ == "__main__":
    main()
