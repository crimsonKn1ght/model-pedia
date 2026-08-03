"""Controlled comparisons for paired image-to-image translation.

    python compare.py --study l1_weight      # the ablation the method is built on
    python compare.py --study discriminator  # PatchGAN receptive field
    python compare.py --study skips          # what the U-Net's skips are worth here

Same data, same schedule, same seed.

``l1_weight`` is the study to run, because the two metric families move in opposite
directions and the table shows it in one place:

* **L1 only** (``0`` adversarial weight is not offered; ``--l1-weight`` large enough
  dominates) - best PSNR, worst FID. Correct on average and soft.
* **GAN only** (``--l1-weight 0``) - worst PSNR, and free to produce something sharp that
  does not match the input at all.
* **both** - the compromise the paper ships.

Reading only PSNR would pick the blurriest model in the table. Reading only FID would pick
one that ignores its input. That is the entire argument for a composite objective, and it is
measurable rather than aesthetic.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "l1_weight": [
        ("GAN only", {"l1_weight": 0.0}),
        ("L1 x 10", {"l1_weight": 10.0}),
        ("L1 x 100", {"l1_weight": 100.0}),
        ("L1 x 1000", {"l1_weight": 1000.0}),
    ],
    "discriminator": [
        ("patchgan 1", {"patch_layers": 1}),
        ("patchgan 3", {"patch_layers": 3}),
        ("whole image", {"whole_image": True}),
    ],
    "skips": [
        ("no skips", {"skips": False}),
        ("U-Net skips", {"skips": True}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled Pix2Pix comparison.")
    parser.add_argument("--study", default="l1_weight", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="shapes", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--fid-samples", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke-test", action="store_true", help="run on generated pairs")
    args = parser.parse_args()

    workers = 0 if args.smoke_test else args.num_workers
    out_root = "outputs/smoke" if args.smoke_test else f"outputs/{args.study}"
    rows = []

    for label, overrides in STUDIES[args.study]:
        print(f"\n=== {args.study}: {label} ===")
        summary = run_training(
            dataset=args.dataset,
            width=16 if args.smoke_test else 64,
            levels=2 if args.smoke_test else 4,
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=8 if args.smoke_test else args.batch_size,
            lr=args.lr,
            train_size=args.train_size,
            test_size=args.test_size,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_')}",
            synthetic=args.smoke_test,
            verbose=False,
            **overrides,
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                data_root=args.data_root,
                dataset=None,
                test_size=args.test_size,
                fid_samples=24 if args.smoke_test else args.fid_samples,
                batch_size=32,
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
                "seconds": summary["train_seconds"],
                "l1": result["pixel"]["l1"],
                "psnr": result["pixel"]["psnr"],
                "ssim": result["pixel"]["ssim"],
                "fid": result["distribution"]["fid"],
                "precision": result["distribution"]["precision"],
                "recall": result["distribution"]["recall"],
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'train s':>8s} {'L1':>8s} {'PSNR':>7s} {'SSIM':>7s} "
        f"{'FID':>8s} {'prec':>6s} {'rec':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['seconds']:8.1f} {row['l1']:8.4f} "
            f"{row['psnr']:7.2f} {row['ssim']:7.4f} {row['fid']:8.2f} "
            f"{row['precision']:6.3f} {row['recall']:6.3f}"
        )
    best_psnr = max(rows, key=lambda row: row["psnr"])["label"]
    best_fid = min(rows, key=lambda row: row["fid"])["label"]
    print(f"\nbest PSNR: {best_psnr}    best FID: {best_fid}")
    print("when those disagree, neither metric alone is enough to pick a model.")

    plot_bars(
        [row["label"] for row in rows],
        {"SSIM": [row["ssim"] for row in rows],
         "PSNR / 30": [row["psnr"] / 30 for row in rows],
         "FID / 10": [row["fid"] / 10 for row in rows]},
        f"{out_root}/{args.study}.png",
        ylabel="score (rescaled)",
        title=f"{args.study} study on {args.dataset}",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
