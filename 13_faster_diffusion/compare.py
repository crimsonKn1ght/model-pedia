"""How much stochasticity should a fast sampler keep?

    python compare.py --checkpoint ../12_diffusion/outputs/fashion-mnist_cosine_t200/best.pt

``eta`` interpolates between two samplers built from the same trained network. At
``eta=0`` the reverse step is deterministic: one latent, one image, and interpolation in
latent space becomes meaningful. At ``eta=1`` the step re-injects DDPM's noise and the
sampler is stochastic again.

The reason to compare them at *several step counts* rather than one is that they do not
rank the same way everywhere. Injected noise costs little when there are many steps to
absorb it and hurts when there are few, so the best ``eta`` depends on the budget - and a
comparison run at one step count would report whichever answer that budget happened to
favour.

Nothing is retrained: every row is the same weights, sampled differently.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import _paths  # noqa: F401
from evaluate import run_evaluation
from train import run_training
from utils import plot_bars, save_json

ETAS = (0.0, 0.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DDIM eta values across step counts.")
    parser.add_argument(
        "--checkpoint", default="../12_diffusion/outputs/fashion-mnist_cosine_t200/best.pt"
    )
    parser.add_argument("--etas", type=float, nargs="+", default=list(ETAS))
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--data-root", default="../12_diffusion/data")
    parser.add_argument("--fid-samples", type=int, default=500)
    parser.add_argument("--sample-batch", type=int, default=125)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="run on generated data")
    args = parser.parse_args()

    steps = [2, 5] if args.smoke_test else args.steps
    etas = args.etas[:2] if args.smoke_test else args.etas
    out_root = Path("outputs/smoke" if args.smoke_test else "outputs/eta")

    checkpoint = args.checkpoint
    if not Path(checkpoint).exists():
        if not args.smoke_test:
            raise SystemExit(
                f"no checkpoint at {checkpoint}. Train one in project 12 first:\n"
                "  cd ../12_diffusion && python train.py --dataset fashion-mnist"
            )
        # A smoke run has nothing to sample from, so make a throwaway model first.
        print("no checkpoint given; training a one-epoch synthetic model to sample from")
        checkpoint = run_training(
            epochs=1, steps=20, batch_size=32, num_workers=0,
            out_dir=str(out_root / "model"), synthetic=True, verbose=False,
        )["checkpoint"]
    by_eta = {}

    for eta in etas:
        print(f"\n=== eta {eta:g} ===")
        result = run_evaluation(
            SimpleNamespace(
                checkpoint=checkpoint,
                steps=steps,
                eta=eta,
                skip_ddpm=True,
                interpolate=False,
                data_root=args.data_root,
                fid_samples=64 if args.smoke_test else args.fid_samples,
                sample_batch=32 if args.smoke_test else args.sample_batch,
                num_workers=0 if args.smoke_test else args.num_workers,
                device=args.device,
                seed=args.seed,
                out_dir=str(out_root / f"eta{eta:g}"),
                smoke_test=args.smoke_test,
            ),
            verbose=False,
        )
        by_eta[eta] = {row["steps"]: row for row in result["rows"]}

    print("\n=== FID by eta and step count (same weights throughout) ===")
    header = f"{'eta':>6s}" + "".join(f" {f'{s} steps':>10s}" for s in steps)
    print(header)
    print("-" * len(header))
    for eta in etas:
        line = f"{eta:6.2f}"
        for step in steps:
            row = by_eta[eta].get(step)
            line += f" {row['fid']:10.2f}" if row else f" {'-':>10s}"
        print(line)

    print("\n=== recall by eta and step count ===")
    print(header)
    print("-" * len(header))
    for eta in etas:
        line = f"{eta:6.2f}"
        for step in steps:
            row = by_eta[eta].get(step)
            line += f" {row['recall']:10.3f}" if row else f" {'-':>10s}"
        print(line)

    print("\neta=0 is the deterministic sampler; only it supports latent interpolation.")

    plot_bars(
        [f"{s} steps" for s in steps],
        {f"eta {eta:g}": [by_eta[eta][s]["fid"] if s in by_eta[eta] else 0.0 for s in steps]
         for eta in etas},
        out_root / "eta.png",
        ylabel="FID",
        title="stochasticity against step budget",
    )
    save_json({"checkpoint": str(checkpoint), "steps": steps,
               "by_eta": {str(k): list(v.values()) for k, v in by_eta.items()}},
              out_root / "eta.json")
    print(f"\nfigure -> {out_root / 'eta.png'}")


if __name__ == "__main__":
    main()
