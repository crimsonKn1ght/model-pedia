"""Score a pretrained encoder on the test split, against an untrained control.

    python evaluate.py --checkpoint outputs/cifar10_simclr/best.pt

The encoder is frozen. Two probes read it out:

* **k-NN** - no fitting at all, so there is nothing to tune and nothing to leak.
* **linear probe** - one linear layer on cached features. The standard measure of
  representation quality, because a linear boundary can only work if the encoder
  has already done the non-linear part.

Both are also run on a **randomly initialised encoder of the identical
architecture**. That control is not optional. A random convolutional network is a
surprisingly decent feature extractor - random filters still compute edges - so
without it a linear probe number sounds impressive when much of it came from the
architecture rather than from pretraining. Every table here prints both columns
and the gap between them.

``--label-fractions`` repeats the linear probe with 1%, 10% and 100% of the
labels. Label efficiency is the practical reason to pretrain at all, and it is
where the gap over the control is widest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset

from data import dataset_info, get_splits, make_eval_loader
from model import build_encoder
from utils import (
    extract_features,
    get_device,
    knn_accuracy,
    linear_probe,
    plot_bars,
    plot_feature_scatter,
    save_json,
    set_seed,
    subsample_per_class,
)


def probe_encoder(
    encoder,
    train_loader,
    test_loader,
    device,
    num_classes: int,
    knn_k: int,
    label_fractions: list[float],
    probe_epochs: int,
    seed: int,
) -> dict:
    """Everything measured from one frozen encoder."""
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)

    result = {
        "knn_accuracy": knn_accuracy(
            train_features, train_labels, test_features, test_labels, num_classes, k=knn_k
        ),
        "linear_probe": {},
    }
    for fraction in label_fractions:
        if fraction >= 1.0:
            idx = torch.arange(train_features.size(0))
        else:
            idx = subsample_per_class(train_labels, fraction, num_classes, seed=seed)
        probe = linear_probe(
            train_features[idx],
            train_labels[idx],
            test_features,
            test_labels,
            num_classes,
            epochs=probe_epochs,
            device=device,
            seed=seed,
        )
        result["linear_probe"][f"{fraction:g}"] = {
            "labels_used": int(idx.numel()),
            "test_accuracy": probe["test_accuracy"],
            "train_accuracy": probe["train_accuracy"],
        }

    result["features"] = test_features
    result["labels"] = test_labels
    return result


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)

    info = dataset_info(ckpt["dataset"])
    train_base, _, test_base, _ = get_splits(
        ckpt["dataset"],
        root=args.data_root,
        seed=args.seed,
        synthetic=args.smoke_test,
    )
    if args.probe_subset:
        train_base = Subset(train_base, range(min(args.probe_subset, len(train_base))))

    size = ckpt["image_size"]
    train_loader = make_eval_loader(
        train_base, info, size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = make_eval_loader(
        test_base, info, size, batch_size=args.batch_size, num_workers=args.num_workers
    )

    def fresh_encoder():
        return build_encoder(
            ckpt["in_channels"], ckpt["width"], ckpt["blocks_per_stage"]
        ).to(dev)

    pretrained = fresh_encoder()
    pretrained.load_state_dict(ckpt["encoder_state"])
    control = fresh_encoder()  # same architecture, never trained

    num_classes = ckpt["num_classes"]
    arms = {}
    for name, encoder in (("pretrained", pretrained), ("random init", control)):
        arms[name] = probe_encoder(
            encoder,
            train_loader,
            test_loader,
            dev,
            num_classes,
            args.knn_k,
            args.label_fractions,
            args.probe_epochs,
            args.seed,
        )

    out_dir = Path(args.checkpoint).parent
    plot_feature_scatter(
        [(name, arm["features"], arm["labels"]) for name, arm in arms.items()],
        out_dir / "features_pca.png",
        class_names=info["class_names"],
        title=f"test features, {ckpt['method']} on {ckpt['dataset']}",
    )
    fractions = [f"{f:g}" for f in args.label_fractions]
    plot_bars(
        [f"{f} labels" for f in fractions] + [f"{args.knn_k}-NN"],
        {
            name: [arm["linear_probe"][f]["test_accuracy"] for f in fractions]
            + [arm["knn_accuracy"]]
            for name, arm in arms.items()
        },
        out_dir / "probe_accuracy.png",
        ylabel="test accuracy",
        title=f"frozen-feature probes: {ckpt['method']} vs an untrained encoder",
    )

    result = {
        "method": ckpt["method"],
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "pretrain_epoch": ckpt["epoch"],
        "augment": ckpt.get("augment", "full"),
        "test_images": len(test_base),
        "probe_images": len(train_base),
        "knn_k": args.knn_k,
        "arms": {
            name: {"knn_accuracy": arm["knn_accuracy"], "linear_probe": arm["linear_probe"]}
            for name, arm in arms.items()
        },
    }
    result["knn_gain"] = arms["pretrained"]["knn_accuracy"] - arms["random init"]["knn_accuracy"]
    result["linear_gain"] = (
        arms["pretrained"]["linear_probe"][fractions[-1]]["test_accuracy"]
        - arms["random init"]["linear_probe"][fractions[-1]]["test_accuracy"]
    )
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"method     : {ckpt['method']} (epoch {ckpt['epoch']}, augment={result['augment']})")
        print(f"dataset    : {ckpt['dataset']}  {len(test_base)} test images, "
              f"{len(train_base)} used to fit the probes\n")
        rows = [(f"{args.knn_k}-NN", "knn_accuracy", None)]
        rows += [(f"linear, {f} of labels", "linear_probe", f) for f in fractions]
        pad = max(len(label) for label, _, _ in rows)
        header = f"{'probe'.rjust(pad)} {'pretrained':>12s} {'random init':>12s} {'gain':>8s}"
        print(header)
        print("-" * len(header))
        for label, key, fraction in rows:
            values = []
            for name in ("pretrained", "random init"):
                arm = arms[name]
                values.append(
                    arm[key] if fraction is None else arm[key][fraction]["test_accuracy"]
                )
            print(
                f"{label.rjust(pad)} {values[0]:12.4f} {values[1]:12.4f} "
                f"{values[0] - values[1]:+8.4f}"
            )
        print("\nfeatures -> " + str(out_dir / "features_pca.png"))
        print("probes   -> " + str(out_dir / "probe_accuracy.png"))
        print("metrics  -> " + str(out_dir / "metrics.json"))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe a frozen self-supervised encoder.")
    parser.add_argument("--checkpoint", default="outputs/cifar10_simclr/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--probe-subset",
        type=int,
        default=10000,
        help="training images used to fit the probes (0 = all)",
    )
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--probe-epochs", type=int, default=40, help="epochs on cached features")
    parser.add_argument(
        "--label-fractions",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0],
        help="fractions of the probe labels to fit on, for the label-efficiency table",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on random tensors")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_workers = 0
        args.probe_epochs = min(args.probe_epochs, 5)
    args.probe_subset = args.probe_subset or None
    run_evaluation(args)


if __name__ == "__main__":
    main()
