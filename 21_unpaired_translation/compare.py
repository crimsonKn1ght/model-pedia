"""Controlled studies: what each part of the CycleGAN objective is actually doing.

    python compare.py --study cycle          # the important one
    python compare.py --study identity
    python compare.py --study pool

One thing changes per arm; everything else - data, seed, architecture, epoch count - is
held fixed, and every arm is scored the same way on the same test split.

``cycle`` is the study worth running first, because it is the one that decides whether the
method works at all. At ``lambda_cycle=0`` the objective is pure adversarial loss in both
directions, which is satisfied by any mapping whose outputs land in the target domain -
including one that throws its input away. The FID column can stay respectable while the
cycle-SSIM column collapses, and the figure shows translations that have stopped
corresponding to their inputs. That is the failure cycle consistency exists to prevent, and
seeing FID fail to notice it is the point.

``identity`` and ``pool`` are smaller effects: the identity term stops the generators from
shifting colours they have no reason to touch, and the image buffer damps the oscillation
the generator/discriminator pair falls into when the discriminator only ever sees the
newest fake.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import run_evaluation
from train import run_training
from utils import plot_bars, save_json

STUDIES = {
    "cycle": {
        "label": "lambda_cycle",
        "arms": [
            {"name": "0 (no cycle loss)", "lambda_cycle": 0.0},
            {"name": "1", "lambda_cycle": 1.0},
            {"name": "10 (default)", "lambda_cycle": 10.0},
        ],
    },
    "identity": {
        "label": "lambda_identity",
        "arms": [
            {"name": "0 (off)", "lambda_identity": 0.0},
            {"name": "0.5 (default)", "lambda_identity": 0.5},
        ],
    },
    "pool": {
        "label": "image buffer",
        "arms": [
            {"name": "0 (newest fake only)", "pool_size": 0},
            {"name": "50 (default)", "pool_size": 50},
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one part of the objective.")
    parser.add_argument("--study", default="cycle", choices=sorted(STUDIES))
    parser.add_argument("--task", default="shapes")
    parser.add_argument("--dataset", default="fashion-mnist")
    parser.add_argument("--domain-a", default="5")
    parser.add_argument("--domain-b", default="7")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--fid-samples", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="generated domains, 1 epoch")
    args = parser.parse_args()

    study = STUDIES[args.study]
    arms = study["arms"][:2] if args.smoke_test else study["arms"]
    root = Path(args.out_dir or f"outputs/study_{args.study}")
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm in arms:
        settings = {key: value for key, value in arm.items() if key != "name"}
        print(f"\n=== {study['label']} = {arm['name']} ===")
        trained = run_training(
            task=args.task,
            dataset=args.dataset,
            domain_a=args.domain_a,
            domain_b=args.domain_b,
            base_channels=8 if args.smoke_test else 24,
            res_blocks=1 if args.smoke_test else 3,
            patch_layers=2 if args.smoke_test else 3,
            epochs=1 if args.smoke_test else args.epochs,
            batch_size=8 if args.smoke_test else 32,
            train_size=args.train_size,
            fid_samples=16 if args.smoke_test else args.fid_samples,
            num_workers=0 if args.smoke_test else args.num_workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            out_dir=str(root / arm["name"].split()[0].replace(".", "p")),
            synthetic=args.smoke_test,
            verbose=False,
            **settings,
        )
        scored = run_evaluation(
            checkpoint=trained["checkpoint"],
            batch_size=8 if args.smoke_test else 32,
            fid_samples=16 if args.smoke_test else args.fid_samples,
            num_workers=0 if args.smoke_test else args.num_workers,
            seed=args.seed,
            device=args.device,
            data_root=args.data_root,
            synthetic=args.smoke_test,
            verbose=False,
        )
        forward = [key for key in scored if key.startswith("fid_") and "identity" not in key][0]
        direction = forward[len("fid_"):]
        rows.append({
            "arm": arm["name"],
            "fid": scored[f"fid_{direction}"],
            "fid_identity": scored[f"fid_identity_{direction}"],
            "cycle_ssim": scored[f"cycle_ssim_{direction}"],
            "precision": scored[f"precision_{direction}"],
            "recall": scored[f"recall_{direction}"],
            "best_epoch": trained["best_epoch"],
            "seconds": trained["seconds"],
        })
        print(f"  FID {rows[-1]['fid']:.3f}  (identity baseline {rows[-1]['fid_identity']:.3f})"
              f"  cycle SSIM {rows[-1]['cycle_ssim']:.4f}")

    header = (f"{study['label']:>22s} {'FID':>9s} {'baseline':>9s} {'cycle SSIM':>11s} "
              f"{'prec':>7s} {'recall':>7s} {'s':>7s}")
    print(f"\n=== {args.study} study, test split ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['arm']:>22s} {row['fid']:9.3f} {row['fid_identity']:9.3f} "
              f"{row['cycle_ssim']:11.4f} {row['precision']:7.3f} {row['recall']:7.3f} "
              f"{row['seconds']:7.1f}")

    if args.study == "cycle":
        print("\nRead the two middle columns together. FID says whether the output looks like")
        print("the target domain; cycle SSIM says whether anything of the input survived.")
        print("Removing the cycle term is free according to the first and ruinous according")
        print("to the second, which is why the method needs it.")

    plot_bars(
        [row["arm"] for row in rows],
        {"FID": [row["fid"] for row in rows],
         "identity baseline": [row["fid_identity"] for row in rows]},
        root / "fid.png",
        ylabel="FID (lower is better)",
        title=f"{args.study}: did the output reach the target domain",
        log=True,
    )
    plot_bars(
        [row["arm"] for row in rows],
        {"cycle SSIM": [row["cycle_ssim"] for row in rows]},
        root / "cycle.png",
        ylabel="cycle SSIM (1.0 = translated nothing)",
        title=f"{args.study}: did the content survive the round trip",
    )
    save_json({"study": args.study, "rows": rows}, root / "results.json")
    print(f"\nfigures -> {root}")


if __name__ == "__main__":
    main()
