"""Controlled comparisons for masked image modelling.

    python compare.py --study mask_ratio   # how much can you hide?
    python compare.py --study decoder      # how small can the decoder be?

Same encoder, same data, same schedule, same seed in every arm.

``mask_ratio`` is the study worth running first, because the answer is
counter-intuitive: reconstruction gets *harder* as the ratio rises, yet the
representation keeps improving well past the point where the pictures start to
look like guesses. Hiding a little is a copy-the-neighbour task; hiding most of
the image is not.

``decoder`` checks the claim that the decoder is disposable. Its depth changes
reconstruction quality noticeably and downstream accuracy barely at all, which is
why MAE can afford a decoder a fraction of the encoder's size.

Add ``--tasks recon probe finetune`` for the full picture; the default leaves out
fine-tuning, which is by far the slowest part.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import TASKS, run_evaluation
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "mask_ratio": [
        ("mask 0.25", {"mask_ratio": 0.25}),
        ("mask 0.50", {"mask_ratio": 0.50}),
        ("mask 0.75", {"mask_ratio": 0.75}),
        ("mask 0.90", {"mask_ratio": 0.90}),
    ],
    "decoder": [
        ("decoder x1", {"decoder_depth": 1}),
        ("decoder x2", {"decoder_depth": 2}),
        ("decoder x4", {"decoder_depth": 4}),
    ],
    "target": [
        ("raw pixels", {"norm_pixel_loss": False}),
        ("per-patch normalised", {"norm_pixel_loss": True}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled MAE comparison.")
    parser.add_argument("--study", default="mask_ratio", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument(
        "--tasks", nargs="+", default=["recon", "probe"], choices=TASKS,
        help="measurements per arm; add 'finetune' for the slow but decisive one",
    )
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke-test", action="store_true", help="run on random tensors")
    args = parser.parse_args()

    workers = 0 if args.smoke_test else args.num_workers
    out_root = "outputs/smoke" if args.smoke_test else f"outputs/{args.study}"
    rows = []

    for label, overrides in STUDIES[args.study]:
        print(f"\n=== {args.study}: {label} ===")
        summary = run_training(
            dataset=args.dataset,
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=32 if args.smoke_test else args.batch_size,
            lr=args.lr,
            train_subset=args.train_subset or None,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_').replace('.', '')}",
            synthetic=args.smoke_test,
            verbose=False,
            **overrides,
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                data_root=args.data_root,
                tasks=args.tasks,
                mask_ratios=[summary["mask_ratio"]],
                train_subset=args.train_subset or None,
                probe_epochs=5 if args.smoke_test else args.probe_epochs,
                finetune_epochs=1 if args.smoke_test else args.finetune_epochs,
                finetune_lr=5e-4,
                finetune_batch_size=32 if args.smoke_test else 128,
                batch_size=256,
                num_workers=workers,
                device=args.device,
                seed=args.seed,
                smoke_test=args.smoke_test,
            ),
            verbose=False,
        )

        row = {
            "label": label,
            "seconds": summary["train_seconds"],
            "visible_patches": summary["visible_patches"],
            "psnr": result["reconstruction"][0]["masked_psnr_db"] if "recon" in args.tasks else None,
        }
        classification = result.get("classification", {})
        for key, short in (("linear_probe", "probe"), ("finetune", "finetune")):
            if key in classification:
                row[short] = classification[key]["pretrained"]["test_accuracy"]
                row[f"{short}_control"] = classification[key]["random init"]["test_accuracy"]
        rows.append(row)

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} pretrain epochs) ===")
    width = max(len(row["label"]) for row in rows)
    columns = [("train s", "seconds", "{:8.1f}", 8), ("visible", "visible_patches", "{:8d}", 8)]
    if "recon" in args.tasks:
        columns.append(("PSNR dB", "psnr", "{:8.2f}", 8))
    if "probe" in args.tasks:
        columns += [("probe", "probe", "{:8.4f}", 8), ("ctrl", "probe_control", "{:8.4f}", 8)]
    if "finetune" in args.tasks:
        columns += [("f-tune", "finetune", "{:8.4f}", 8), ("ctrl", "finetune_control", "{:8.4f}", 8)]

    header = "arm".ljust(width) + "".join(f" {name:>{size}s}" for name, _, _, size in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = row["label"].ljust(width)
        for _, key, fmt, _ in columns:
            line += " " + fmt.format(row[key])
        print(line)
    print("\n'ctrl' is the identical ViT trained from scratch under the same budget.")

    labels = [row["label"] for row in rows]
    series = {}
    if "recon" in args.tasks:
        series["masked PSNR / 30 dB"] = [row["psnr"] / 30.0 for row in rows]
    for key, pretty in (("probe", "linear probe"), ("finetune", "fine-tune")):
        if key in args.tasks or (key == "probe" and "probe" in args.tasks):
            if key in rows[0]:
                series[pretty] = [row[key] for row in rows]
                series[f"{pretty} control"] = [row[f"{key}_control"] for row in rows]
    plot_bars(
        labels,
        series,
        f"{out_root}/{args.study}.png",
        ylabel="test accuracy (PSNR rescaled)",
        title=f"{args.study} study on {args.dataset}",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
