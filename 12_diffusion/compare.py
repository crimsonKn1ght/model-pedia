"""Controlled comparisons for the DDPM.

    python compare.py --study schedule     # cosine vs linear betas
    python compare.py --study prediction   # predict the noise or the clean image
    python compare.py --study steps        # 100 / 200 / 400 timesteps

Same data, same schedule length where it is not the variable, same seed.

``prediction`` is the one that best repays running. Predicting the noise and predicting
the clean image are algebraically equivalent - either one determines the other - and
they train very differently, because the noise target has the same scale at every
timestep while the clean image does not. This is a good example of a reparameterisation
that changes nothing mathematically and everything practically.

``steps`` is the cost/quality dial. More timesteps means each denoising step is easier
and sampling is proportionally slower. Note that the training loss barely moves across
these arms while FID does - more evidence that the loss is not the metric.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from data import DATASETS
from evaluate import run_evaluation
from model import PREDICTIONS, SCHEDULES
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "schedule": [
        ("linear", {"schedule": "linear"}),
        ("cosine", {"schedule": "cosine"}),
    ],
    "prediction": [
        ("predict x0", {"prediction": "x0"}),
        ("predict noise", {"prediction": "noise"}),
    ],
    "steps": [
        ("100 steps", {"steps": 100}),
        ("200 steps", {"steps": 200}),
        ("400 steps", {"steps": 400}),
    ],
    "attention": [
        ("no attention", {"attention": False}),
        ("with attention", {"attention": True}),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled DDPM comparison.")
    parser.add_argument("--study", default="prediction", choices=sorted(STUDIES))
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--schedule", default="cosine", choices=SCHEDULES)
    parser.add_argument("--prediction", default="noise", choices=PREDICTIONS)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--fid-samples", type=int, default=500)
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
            schedule=overrides.get("schedule", args.schedule),
            prediction=overrides.get("prediction", args.prediction),
            steps=20 if args.smoke_test else overrides.get("steps", args.steps),
            attention=overrides.get("attention", True),
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=32 if args.smoke_test else args.batch_size,
            lr=args.lr,
            train_subset=args.train_subset or None,
            fid_every=1 if args.smoke_test else max(args.epochs // 3, 1),
            fid_track_samples=64 if args.smoke_test else 500,
            num_workers=workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=f"{out_root}/{label.replace(' ', '_')}",
            synthetic=args.smoke_test,
            verbose=False,
        )
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=summary["checkpoint"],
                data_root=args.data_root,
                dataset=None,
                fid_samples=64 if args.smoke_test else args.fid_samples,
                sample_batch=32 if args.smoke_test else 125,
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
                "train_loss": summary["history"]["train_loss"][-1],
                "val_loss": summary["history"]["val_loss"][-1],
                "steps": result["steps"],
                "fid": result["samples"]["fid"],
                "precision": result["samples"]["precision"],
                "recall": result["samples"]["recall"],
                "ms_per_image": result["cost"]["seconds_per_image"] * 1000,
            }
        )

    print(f"\n=== {args.study} on {args.dataset} ({args.epochs} epochs) ===")
    width = max(len(row["label"]) for row in rows)
    header = (
        f"{'arm'.ljust(width)} {'train s':>8s} {'val loss':>9s} {'T':>5s} {'FID':>8s} "
        f"{'prec':>6s} {'rec':>6s} {'ms/img':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label'].ljust(width)} {row['seconds']:8.1f} {row['val_loss']:9.4f} "
            f"{row['steps']:5d} {row['fid']:8.2f} {row['precision']:6.3f} {row['recall']:6.3f} "
            f"{row['ms_per_image']:8.0f}"
        )
    print("\nnote how little the validation loss separates the arms compared with FID.")

    plot_bars(
        [row["label"] for row in rows],
        {"FID": [row["fid"] for row in rows],
         "precision x 100": [row["precision"] * 100 for row in rows],
         "recall x 100": [row["recall"] * 100 for row in rows]},
        f"{out_root}/{args.study}.png",
        ylabel="score",
        title=f"{args.study} study on {args.dataset}",
    )
    save_json({"study": args.study, "dataset": args.dataset, "rows": rows},
              f"{out_root}/{args.study}.json")
    print(f"\nfigure -> {out_root}/{args.study}.png")


if __name__ == "__main__":
    main()
