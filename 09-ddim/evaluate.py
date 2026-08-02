"""Compare DDIM against DDPM: quality versus number of sampling steps.

    python evaluate.py
    python evaluate.py --steps 5 10 20 50 100 --eta 0.0

This is the experiment the project exists for.  The same trained network is
sampled with:

* DDPM ancestral sampling using every timestep (the baseline), and
* DDIM with progressively fewer steps.

For each setting it records FID/KID, wall-clock seconds per image, and the
number of network evaluations, then plots FID against step count.  The expected
shape of that curve -- flat over a wide range, rising sharply only at very few
steps -- is the practical claim DDIM makes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import metrics as metrics_mod
from common import utils, viz
from ddpm_bridge import GaussianDiffusion, UNet
from model import DDIMSampler

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--checkpoint", default=str(HERE / "outputs" / "ddpm.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-eval-samples", type=int, default=512)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--eta", type=float, default=0.0, help="0 = deterministic DDIM")
    parser.add_argument("--feature-extractor", default="small-cnn", choices=["small-cnn", "inception"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)

    ckpt = utils.load_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    out_dir = utils.resolve_out_dir(args, str(Path(args.checkpoint).parent))
    eval_dir = utils.ensure_dir(out_dir / "evaluation")

    unet = UNet(channels=cfg["channels"], base=cfg["base_channels"]).to(device)
    unet.load_state_dict(ckpt["ema_model"])
    unet.eval()
    diffusion = GaussianDiffusion(unet, timesteps=cfg["timesteps"], schedule=cfg["schedule"]).to(device)
    sampler = DDIMSampler(diffusion)

    shape = (cfg["channels"], cfg["image_size"], cfg["image_size"])
    test_loader = data_mod.get_dataloader(
        cfg["dataset"],
        root=args.data_root,
        image_size=cfg["image_size"],
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    n_samples = 64 if args.quick else args.num_eval_samples
    step_counts = [5, 10] if args.quick else args.steps

    extractor = metrics_mod.get_feature_extractor(
        cfg["dataset"],
        root=args.data_root,
        image_size=cfg["image_size"],
        device=device,
        kind=args.feature_extractor,
        num_workers=args.num_workers,
    )
    real_features = metrics_mod.features_from_loader(
        extractor, test_loader, device, max_samples=n_samples
    )

    rows = {}
    table = []

    def measure(label: str, sample_fn, evaluations: int):
        with utils.Timer() as timer:
            features = metrics_mod.features_from_sampler(
                extractor, sample_fn, device, num_samples=n_samples, batch_size=args.batch_size
            )
        m = metrics_mod.generative_metrics(real_features, features, with_precision_recall=False)
        entry = {
            "sampler": label,
            "network_evaluations": evaluations,
            "fid": m["fid"],
            "kid_mean": m["kid_mean"],
            "seconds_per_image": timer.elapsed / n_samples,
        }
        table.append(entry)
        rows[label] = sample_fn(8)
        print(
            f"{label:<18} evals {evaluations:>4}  FID {m['fid']:8.3f}  "
            f"KID {m['kid_mean']:8.4f}  {entry['seconds_per_image']:.3f} s/image"
        )
        return entry

    print(f"\nsampling {n_samples} images per setting on {device}\n")

    # Baseline: full DDPM ancestral sampling.
    baseline = measure(
        f"DDPM ({cfg['timesteps']})",
        lambda n: diffusion.sample(n, shape, device),
        cfg["timesteps"],
    )

    for steps in step_counts:
        measure(
            f"DDIM ({steps})",
            lambda n, s=steps: sampler.sample(n, shape, device, num_steps=s, eta=args.eta),
            steps,
        )

    ddim_entries = [e for e in table if e["sampler"].startswith("DDIM")]
    viz.plot_line(
        [e["network_evaluations"] for e in ddim_entries],
        [e["fid"] for e in ddim_entries],
        eval_dir / "fid-vs-steps.png",
        title=f"DDIM: FID against sampling steps (DDPM baseline = {baseline['fid']:.1f})",
        xlabel="network evaluations per image",
        ylabel="FID",
        series_label="DDIM",
    )
    viz.plot_line(
        [e["network_evaluations"] for e in ddim_entries],
        [e["seconds_per_image"] for e in ddim_entries],
        eval_dir / "speed-vs-steps.png",
        title="sampling cost against steps",
        xlabel="network evaluations per image",
        ylabel="seconds per image",
        series_label="DDIM",
    )
    viz.save_comparison_grid(rows, eval_dir / "samples-by-step-count.png")

    if not args.quick:
        viz.save_image_grid(
            sampler.interpolate(shape, device, rows=4, steps=8, num_steps=max(step_counts)),
            eval_dir / "latent-interpolation.png",
            nrow=8,
        )

    speedup = baseline["seconds_per_image"] / min(e["seconds_per_image"] for e in ddim_entries)
    results = {
        "baseline_fid": baseline["fid"],
        "baseline_seconds_per_image": baseline["seconds_per_image"],
        "max_speedup": speedup,
        "eta": args.eta,
        "settings": table,
    }
    utils.save_json(results, eval_dir / "metrics.json")
    print(f"\nfastest DDIM setting is {speedup:.1f}x faster per image than full DDPM sampling")
    print(f"figures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
