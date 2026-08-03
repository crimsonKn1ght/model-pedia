"""Controlled comparisons for captioning.

    python compare.py --study capacity     # decoder depth
    python compare.py --study smoothing    # what label smoothing is worth
    python compare.py --study encoder      # trained-from-scratch CNN vs frozen ImageNet

Same data, same schedule, same seed in every arm, and every arm is scored with the
same decoding so the comparison is about the model rather than the search.

``smoothing`` is the instructive one. Turning label smoothing off *improves* the
validation cross entropy and usually makes the captions worse, because an image has
several correct captions and a loss that demands all the probability on one
particular next word is optimising for something untrue. It is the cleanest example
in this repository of the loss and the metric disagreeing, and of the metric being
right.

``encoder`` needs the torchvision ImageNet weights, so it wants a network connection
on first run. On ``shapes`` the from-scratch CNN wins - the images contain nothing
ImageNet knows about. On ``flickr8k`` it is not close.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from model import ENCODERS
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "capacity": [
        ("depth 1", {"depth": 1}),
        ("depth 3", {"depth": 3}),
        ("depth 5", {"depth": 5}),
    ],
    "smoothing": [
        ("no smoothing", {"label_smoothing": 0.0}),
        ("smoothing 0.1", {"label_smoothing": 0.1}),
    ],
    "encoder": [
        ("cnn from scratch", {"encoder_name": "cnn"}),
        ("frozen resnet18", {"encoder_name": "resnet18"}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled captioning comparison.")
    parser.add_argument("--study", default="capacity", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="shapes", choices=sorted(DATASETS))
    parser.add_argument("--encoder", default="cnn", choices=sorted(ENCODERS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke-test", action="store_true", help="run on generated scenes")
    args = parser.parse_args()

    workers = 0 if args.smoke_test else args.num_workers
    out_root = "outputs/smoke" if args.smoke_test else f"outputs/{args.study}"
    rows = []

    for label, overrides in STUDIES[args.study]:
        print(f"\n=== {args.study}: {label} ===")
        summary = run_training(
            dataset=args.dataset,
            encoder_name=overrides.get("encoder_name", args.encoder),
            image_size=args.image_size,
            dim=args.dim,
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=16 if args.smoke_test else args.batch_size,
            lr=args.lr,
            train_size=args.train_size or None,
            val_size=args.val_size,
            test_size=args.test_size,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_').replace('.', '')}",
            synthetic=args.smoke_test,
            verbose=False,
            **{key: value for key, value in overrides.items() if key != "encoder_name"},
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                data_root=args.data_root,
                test_size=args.test_size,
                beam_sizes=[1 if args.smoke_test else args.beam_size],
                batch_size=args.batch_size,
                num_workers=workers,
                device=args.device,
                seed=args.seed,
                smoke_test=args.smoke_test,
            ),
            verbose=False,
        )
        beam = next(iter(result["by_beam"]))
        scores = result["by_beam"][beam]
        rows.append(
            {
                "label": label,
                "parameters": summary["parameters"],
                "seconds": summary["train_seconds"],
                "val_loss": summary["history"]["val_loss"][-1]
                if summary["history"]["val_loss"] else float("nan"),
                "bleu_4": scores["bleu_4"],
                "meteor_exact": scores["meteor_exact"],
                "cider_d": scores["cider_d"],
                "distinct": scores["distinct_captions"],
                "length": scores["mean_length"],
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs, beam {args.beam_size}) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'params':>10s} {'train s':>8s} {'val loss':>9s} "
        f"{'BLEU-4':>8s} {'METEOR':>8s} {'CIDEr-D':>8s} {'distinct':>9s} {'len':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['parameters']:10,d} {row['seconds']:8.1f} "
            f"{row['val_loss']:9.4f} {row['bleu_4']:8.4f} {row['meteor_exact']:8.4f} "
            f"{row['cider_d']:8.3f} {row['distinct']:9.4f} {row['length']:6.1f}"
        )
    print("\n'distinct' is the fraction of test images given a caption no other image got.")
    print("A low value with a decent BLEU means one safe sentence is being reused.")

    plot_bars(
        [row["label"] for row in rows],
        {
            "BLEU-4": [row["bleu_4"] for row in rows],
            "METEOR (exact)": [row["meteor_exact"] for row in rows],
            "CIDEr-D / 10": [row["cider_d"] / 10 for row in rows],
            "distinct captions": [row["distinct"] for row in rows],
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
