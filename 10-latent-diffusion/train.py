"""Train a latent diffusion model: autoencoder first, then diffusion on latents.

    python train.py                                  # Fashion-MNIST
    python train.py --dataset celeba64 --image-size 64
    python train.py --downsample-factor 2            # a milder compression

Stage 1 trains the autoencoder and freezes it.  Stage 2 trains a DDPM U-Net on
the encoded latents.  Both stages report their own metrics; the interesting
comparison is stage 2's cost against pixel-space DDPM in project 08, which
``evaluate.py`` quantifies.

The diffusion U-Net is imported from project 08 unchanged -- the architecture
does not care whether its input is pixels or latents, which is precisely the
observation latent diffusion is built on.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from model import AutoencoderKL, LatentScaler

HERE = Path(__file__).resolve().parent


def load_ddpm_module():
    """Reuse the U-Net and diffusion process from project 08."""
    path = HERE.parent / "08-ddpm" / "model.py"
    spec = importlib.util.spec_from_file_location("ddpm_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ddpm = load_ddpm_module()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--dataset", default="fashion-mnist", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--downsample-factor", type=int, default=4, help="the f of the paper")
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--ae-base", type=int, default=32)
    parser.add_argument("--ae-epochs", type=int, default=3)
    parser.add_argument("--ae-lr", type=float, default=2e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--unet-base", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--diffusion-steps", type=int, default=2000, help="optimizer steps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    return parser


@torch.no_grad()
def update_ema(ema_model, model, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    out_dir = utils.resolve_out_dir(args, str(HERE / "outputs"))
    samples_dir = utils.ensure_dir(out_dir / "samples")

    meta = data_mod.info(args.dataset)
    loader = data_mod.get_dataloader(
        args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    history = utils.History()

    # ------------------------------------------------------------------ #
    # Stage 1: perceptual compression
    # ------------------------------------------------------------------ #
    autoencoder = AutoencoderKL(
        channels=meta.channels,
        base=args.ae_base,
        latent_channels=args.latent_channels,
        downsample_factor=args.downsample_factor,
    ).to(device)
    latent_shape = autoencoder.latent_shape(args.image_size)
    print(utils.describe_model("stage 1 autoencoder", autoencoder))
    print(
        f"compression: {meta.channels}x{args.image_size}x{args.image_size} -> "
        f"{latent_shape[0]}x{latent_shape[1]}x{latent_shape[2]}  "
        f"({meta.channels * args.image_size**2 / (latent_shape[0] * latent_shape[1] * latent_shape[2]):.1f}x fewer values)"
    )

    opt_ae = torch.optim.Adam(autoencoder.parameters(), lr=args.ae_lr)
    ae_epochs = 1 if args.quick else args.ae_epochs
    fixed = next(iter(loader))[0][:8].to(device)

    with utils.Timer() as ae_timer:
        for epoch in range(1, ae_epochs + 1):
            autoencoder.train()
            rec_m = utils.AverageMeter()
            for x, _ in utils.progress(
                utils.maybe_limit(loader, args), desc=f"stage 1 epoch {epoch}/{ae_epochs}",
                total=utils.quick_len(loader, args),
            ):
                x = x.to(device)
                loss, rec, _ = autoencoder.loss(x, kl_weight=args.kl_weight)
                opt_ae.zero_grad(set_to_none=True)
                loss.backward()
                opt_ae.step()
                rec_m.update(rec.item(), x.shape[0])
            history.add(autoencoder_mse=rec_m.avg)
            print(f"stage 1 epoch {epoch:>3}  reconstruction MSE {rec_m.avg:.5f}")

            autoencoder.eval()
            with torch.no_grad():
                viz.save_comparison_grid(
                    {"input": fixed, "reconstruction": autoencoder(fixed)[0]},
                    samples_dir / f"stage1-reconstructions-epoch{epoch:03d}.png",
                )

    # Freeze stage 1: from here on it is a fixed codec, not a trained model.
    autoencoder.eval().requires_grad_(False)
    scaler = LatentScaler.fit(autoencoder, loader, device)
    print(f"stage 1 done in {ae_timer.elapsed / 60:.1f} min; latent scale {scaler.scale:.4f}")

    # ------------------------------------------------------------------ #
    # Stage 2: diffusion in the frozen latent space
    # ------------------------------------------------------------------ #
    unet = ddpm.UNet(
        channels=args.latent_channels,
        base=args.unet_base,
        channel_multipliers=(1, 2),
        attention_at=1,
    ).to(device)
    diffusion = ddpm.GaussianDiffusion(unet, timesteps=args.timesteps, schedule=args.schedule).to(device)
    ema_unet = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_diffusion = ddpm.GaussianDiffusion(
        ema_unet, timesteps=args.timesteps, schedule=args.schedule
    ).to(device)
    print(utils.describe_model("stage 2 diffusion U-Net", unet))

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    max_steps = 8 if args.quick else args.diffusion_steps
    step, window = 0, utils.AverageMeter()

    with utils.Timer() as diff_timer:
        while step < max_steps:
            for x, _ in loader:
                if step >= max_steps:
                    break
                with torch.no_grad():
                    mu, _ = autoencoder.encode(x.to(device))
                    z = scaler.encode(mu)
                loss = diffusion.loss(z)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                update_ema(ema_unet, unet, args.ema_decay)

                window.update(loss.item(), x.shape[0])
                step += 1

                if step % max(1, max_steps // 10) == 0 or step == max_steps:
                    history.add(latent_noise_mse=window.avg)
                    print(f"stage 2 step {step:>6}/{max_steps}  noise MSE {window.avg:.5f}")
                    window.reset()

                if step % max(1, max_steps // 4) == 0 or step == max_steps:
                    latents = ema_diffusion.sample(16, latent_shape, device)
                    images = autoencoder.decode(scaler.decode(latents))
                    viz.save_image_grid(images, samples_dir / f"samples-step{step:06d}.png", nrow=4)

    print(f"stage 2 done in {diff_timer.elapsed / 60:.1f} min")

    utils.save_checkpoint(
        out_dir / "latent-diffusion.pt",
        autoencoder=autoencoder,
        model=unet,
        ema_model=ema_unet,
        config={
            "dataset": args.dataset,
            "image_size": args.image_size,
            "channels": meta.channels,
            "latent_channels": args.latent_channels,
            "downsample_factor": args.downsample_factor,
            "ae_base": args.ae_base,
            "unet_base": args.unet_base,
            "timesteps": args.timesteps,
            "schedule": args.schedule,
            "latent_scale": scaler.scale,
            "diffusion_steps": max_steps,
            "stage1_minutes": ae_timer.elapsed / 60,
            "stage2_minutes": diff_timer.elapsed / 60,
        },
    )
    history.save(out_dir / "history.json")
    viz.plot_curves(history.records, out_dir / "training-curves.png", title="latent diffusion training")
    print(f"\nsaved checkpoint and figures to {out_dir}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
