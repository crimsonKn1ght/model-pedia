"""Evaluate a trained DDPM.

    python evaluate.py

Produces:

* FID/KID of ancestral samples against real test images;
* the **denoising trajectory** -- the same latent shown at a spread of
  timesteps, from pure noise on the left to a finished image on the right,
  which is the clearest single picture of what diffusion actually does;
* a **forward-process** strip, showing real images being destroyed by the
  schedule (the process the model learns to reverse);
* measured sampling throughput, which is the number project 14 improves on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import metrics as metrics_mod
from common import utils, viz
from model import GaussianDiffusion, UNet

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--checkpoint", default=str(HERE / "outputs" / "ddpm.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-eval-samples", type=int, default=512)
    parser.add_argument("--feature-extractor", default="small-cnn", choices=["small-cnn", "inception"])
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
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
    unet.load_state_dict(ckpt["ema_model" if args.weights == "ema" else "model"])
    unet.eval()
    diffusion = GaussianDiffusion(unet, timesteps=cfg["timesteps"], schedule=cfg["schedule"]).to(device)

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

    # -- 1. the forward process, for reference ------------------------------- #
    real = next(iter(test_loader))[0][:1].to(device)
    forward_frames = []
    for frac in [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]:
        step = min(cfg["timesteps"] - 1, int(frac * (cfg["timesteps"] - 1)))
        t = torch.full((1,), step, device=device, dtype=torch.long)
        forward_frames.append(diffusion.q_sample(real, t)[0])
    viz.save_animation_strip(forward_frames, eval_dir / "forward-process.png")

    # -- 2. the reverse process (what the model learned) --------------------- #
    record_every = max(1, cfg["timesteps"] // 10)
    _, trajectory = diffusion.sample(4, shape, device, record_every=record_every)
    viz.save_animation_strip(trajectory, eval_dir / "denoising-trajectory.png")

    # -- 3. sampling speed --------------------------------------------------- #
    with utils.Timer() as timer:
        grid = diffusion.sample(16, shape, device)
    seconds_per_image = timer.elapsed / 16
    viz.save_image_grid(grid, eval_dir / "samples.png", nrow=4)

    # -- 4. FID / KID --------------------------------------------------------- #
    n_samples = 64 if args.quick else args.num_eval_samples
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
    fake_features = metrics_mod.features_from_sampler(
        extractor,
        lambda n: diffusion.sample(n, shape, device),
        device,
        num_samples=n_samples,
        batch_size=args.batch_size,
    )
    results = metrics_mod.generative_metrics(real_features, fake_features)
    results["timesteps"] = float(cfg["timesteps"])
    results["seconds_per_image"] = seconds_per_image
    results["network_evaluations_per_image"] = float(cfg["timesteps"])

    utils.save_json(results, eval_dir / "metrics.json")
    print("\nDDPM evaluation")
    print(metrics_mod.format_metrics(results))
    print(f"\nfigures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
