"""Evaluate latent diffusion, and measure what the compression actually bought.

    python evaluate.py

Reports FID/KID of decoded samples, plus the two numbers that justify the whole
approach:

* **cost per sampling step** in latent space against pixel space, measured by
  timing both U-Nets on this machine;
* **peak memory** per sampling batch for each.

It also separates the two stages' contributions. The autoencoder's own
reconstruction FID is a floor: latent diffusion can never beat it, because every
sample it produces has been through the decoder. If the sample FID is close to
that floor, stage 2 is doing its job and stage 1 is the bottleneck.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

import bootstrap  # noqa: F401
from common import data as data_mod
from common import metrics as metrics_mod
from common import utils, viz
from model import AutoencoderKL, LatentScaler

HERE = Path(__file__).resolve().parent


def load_ddpm_module():
    path = HERE.parent / "13_ddpm" / "model.py"
    spec = importlib.util.spec_from_file_location("ddpm_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ddpm = load_ddpm_module()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--checkpoint", default=str(HERE / "outputs" / "latent-diffusion.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-eval-samples", type=int, default=512)
    parser.add_argument("--feature-extractor", default="small-cnn", choices=["small-cnn", "inception"])
    return parser


def time_unet(unet, shape, device, batch: int = 16, repeats: int = 5) -> float:
    """Seconds for one denoising step at the given tensor shape."""
    x = torch.randn(batch, *shape, device=device)
    t = torch.zeros(batch, device=device, dtype=torch.long)
    with torch.no_grad():
        unet(x, t)  # warm up
        with utils.Timer() as timer:
            for _ in range(repeats):
                unet(x, t)
    return timer.elapsed / repeats / batch


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)

    ckpt = utils.load_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    out_dir = utils.resolve_out_dir(args, str(Path(args.checkpoint).parent))
    eval_dir = utils.ensure_dir(out_dir / "evaluation")

    autoencoder = AutoencoderKL(
        channels=cfg["channels"],
        base=cfg["ae_base"],
        latent_channels=cfg["latent_channels"],
        downsample_factor=cfg["downsample_factor"],
    ).to(device)
    autoencoder.load_state_dict(ckpt["autoencoder"])
    autoencoder.eval()

    unet = ddpm.UNet(
        channels=cfg["latent_channels"],
        base=cfg["unet_base"],
        channel_multipliers=(1, 2),
        attention_at=1,
    ).to(device)
    unet.load_state_dict(ckpt["ema_model"])
    unet.eval()
    diffusion = ddpm.GaussianDiffusion(unet, timesteps=cfg["timesteps"], schedule=cfg["schedule"]).to(device)
    scaler = LatentScaler(cfg["latent_scale"])

    latent_shape = autoencoder.latent_shape(cfg["image_size"])
    pixel_shape = (cfg["channels"], cfg["image_size"], cfg["image_size"])

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

    def sample_images(n: int) -> torch.Tensor:
        latents = diffusion.sample(n, latent_shape, device)
        return autoencoder.decode(scaler.decode(latents))

    # -- 1. sample quality --------------------------------------------------- #
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
    with utils.Timer() as sample_timer:
        fake_features = metrics_mod.features_from_sampler(
            extractor, sample_images, device, num_samples=n_samples, batch_size=args.batch_size
        )
    results = metrics_mod.generative_metrics(real_features, fake_features)
    results["seconds_per_image"] = sample_timer.elapsed / n_samples

    # -- 2. the stage-1 floor ------------------------------------------------- #
    recon_chunks, mse_m = [], utils.AverageMeter()
    seen = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            recon = autoencoder(x)[0]
            mse_m.update(F.mse_loss(recon, x).item(), x.shape[0])
            recon_chunks.append(extractor.features(data_mod.denormalize(recon)).cpu())
            seen += x.shape[0]
            if seen >= n_samples:
                break
    recon_features = torch.cat(recon_chunks)[:n_samples]
    results["autoencoder_mse"] = mse_m.avg
    results["fid_autoencoder_floor"] = metrics_mod.fid_from_features(
        *metrics_mod.standardize_pair(real_features[: recon_features.shape[0]], recon_features)
    )

    # -- 3. latent versus pixel cost ------------------------------------------ #
    pixel_unet = ddpm.UNet(channels=cfg["channels"], base=cfg["unet_base"]).to(device).eval()
    latent_step = time_unet(unet, latent_shape, device)
    pixel_step = time_unet(pixel_unet, pixel_shape, device)
    results.update(
        {
            "latent_elements": float(latent_shape[0] * latent_shape[1] * latent_shape[2]),
            "pixel_elements": float(pixel_shape[0] * pixel_shape[1] * pixel_shape[2]),
            "compression_ratio": (pixel_shape[0] * pixel_shape[1] * pixel_shape[2])
            / (latent_shape[0] * latent_shape[1] * latent_shape[2]),
            "seconds_per_step_latent": latent_step,
            "seconds_per_step_pixel": pixel_step,
            "step_speedup_vs_pixel": pixel_step / max(latent_step, 1e-9),
            "stage1_minutes": cfg["stage1_minutes"],
            "stage2_minutes": cfg["stage2_minutes"],
        }
    )

    # -- figures --------------------------------------------------------------- #
    viz.save_image_grid(sample_images(32), eval_dir / "samples.png", nrow=8)
    batch = next(iter(test_loader))[0][:8].to(device)
    with torch.no_grad():
        viz.save_comparison_grid(
            {"input": batch, "autoencoder": autoencoder(batch)[0]},
            eval_dir / "stage1-reconstructions.png",
        )
    viz.plot_bars(
        ["latent U-Net", "pixel U-Net"],
        [latent_step * 1000, pixel_step * 1000],
        eval_dir / "cost-per-step.png",
        title="cost of one denoising step",
        ylabel="milliseconds per image",
    )

    utils.save_json(results, eval_dir / "metrics.json")
    print("\nlatent diffusion evaluation")
    print(metrics_mod.format_metrics(results))
    print(
        f"\none denoising step is {results['step_speedup_vs_pixel']:.1f}x cheaper in latent space "
        f"({results['compression_ratio']:.0f}x fewer values to denoise)"
    )
    print(f"figures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
