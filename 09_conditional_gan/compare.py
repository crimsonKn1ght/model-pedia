"""Controlled comparisons for conditional GANs.

    python compare.py --study mode         # cGAN vs ACGAN
    python compare.py --study aux_weight   # how hard to push the ACGAN classifier

Same data, same schedule, same seed in every arm.

``aux_weight`` is the study that matters, because it makes a trade visible that is easy
to miss. Turning up the auxiliary classifier weight raises class accuracy - the
generator is being paid to produce recognisable examples - and past a point it lowers
recall, because the surest way to be recognisable is to draw the most typical member of
the class every time. Reporting accuracy alone would call that an improvement.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from model import MODES
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "mode": [
        ("cGAN", {"mode": "cgan"}),
        ("ACGAN", {"mode": "acgan"}),
    ],
    "aux_weight": [
        ("aux 0.2", {"aux_weight": 0.2}),
        ("aux 1.0", {"aux_weight": 1.0}),
        ("aux 5.0", {"aux_weight": 5.0}),
    ],
    "latent": [
        ("latent 16", {"latent_dim": 16}),
        ("latent 64", {"latent_dim": 64}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled conditional-GAN comparison.")
    parser.add_argument("--study", default="mode", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--mode", default="acgan", choices=MODES)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
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
            mode=overrides.get("mode", args.mode),
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=32 if args.smoke_test else args.batch_size,
            lr=args.lr,
            train_subset=args.train_subset or None,
            fid_track_samples=64 if args.smoke_test else 1000,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_').replace('.', '')}",
            synthetic=args.smoke_test,
            verbose=False,
            **{key: value for key, value in overrides.items() if key != "mode"},
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                data_root=args.data_root,
                dataset=None,
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
                "seconds": summary["train_seconds"],
                "fid": result["samples"]["fid"],
                "precision": result["samples"]["precision"],
                "recall": result["samples"]["recall"],
                "class_accuracy": result["class_accuracy"],
                "worst_class": min(result["per_class"], key=lambda row: row["accuracy"])["class"],
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'train s':>8s} {'FID':>8s} {'prec':>6s} {'rec':>6s} "
        f"{'class acc':>10s} {'worst class':>12s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['seconds']:8.1f} {row['fid']:8.2f} "
            f"{row['precision']:6.3f} {row['recall']:6.3f} {row['class_accuracy']:10.3f} "
            f"{row['worst_class']:>12s}"
        )
    print("\nread class accuracy and recall together: accuracy rising while recall falls")
    print("means the generator is drawing the most typical member of each class.")

    plot_bars(
        [row["label"] for row in rows],
        {
            "class accuracy": [row["class_accuracy"] for row in rows],
            "recall": [row["recall"] for row in rows],
            "precision": [row["precision"] for row in rows],
        },
        f"{out_root}/{args.study}.png",
        ylabel="score",
        title=f"{args.study} study on {args.dataset}",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
