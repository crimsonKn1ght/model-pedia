"""Controlled comparisons for the flow.

    python compare.py --study capacity        # coupling network width
    python compare.py --study depth           # coupling layers per stage
    python compare.py --study dequantisation  # why the preprocessing matters

Same data, same schedule, same seed.

``dequantisation`` is the study to run, and it is a warning rather than a tuning
exercise. Pixels are discrete; a continuous density can place unbounded mass on a finite
set of points, so a flow trained without dequantisation noise reports a bits-per-dimension
that keeps improving without limit and describes nothing. The arm exists so you can watch
the number go somewhere impossible and know what that looks like.

Both arms are *evaluated* the same way, so the difference in the table is entirely a
difference in what they were trained on.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "capacity": [
        ("hidden 32", {"hidden": 32}),
        ("hidden 64", {"hidden": 64}),
        ("hidden 128", {"hidden": 128}),
    ],
    "depth": [
        ("2 per stage", {"couplings_per_stage": 2}),
        ("3 per stage", {"couplings_per_stage": 3}),
        ("5 per stage", {"couplings_per_stage": 5}),
    ],
    "dequantisation": [
        ("no dequantisation", {"dequantise": False}),
        ("uniform dequantisation", {"dequantise": True}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled flow comparison.")
    parser.add_argument("--study", default="capacity", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--fid-samples", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke-test", action="store_true", help="run on generated data")
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
                temperature=1.0,
                fid_samples=64 if args.smoke_test else args.fid_samples,
                batch_size=256,
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
                "test_bpd": result["test_bits_per_dimension"],
                "inversion_error": result["max_inversion_error"],
                "fid": result["samples"]["fid"],
                "precision": result["samples"]["precision"],
                "recall": result["samples"]["recall"],
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'params':>10s} {'train s':>8s} {'bits/dim':>9s} "
        f"{'inv err':>9s} {'FID':>8s} {'prec':>6s} {'rec':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['parameters']:10,d} {row['seconds']:8.1f} "
            f"{row['test_bpd']:9.4f} {row['inversion_error']:9.1e} {row['fid']:8.2f} "
            f"{row['precision']:6.3f} {row['recall']:6.3f}"
        )
    print("\nbits/dim is exact and comparable with published numbers; 8.0 is the uniform")
    print("baseline. FID here is not comparable outside this repository.")

    plot_bars(
        [row["label"] for row in rows],
        {"bits/dim": [row["test_bpd"] for row in rows],
         "uniform baseline": [8.0 for _ in rows]},
        f"{out_root}/{args.study}.png",
        ylabel="bits per dimension (lower is better)",
        title=f"{args.study} study on {args.dataset}",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
