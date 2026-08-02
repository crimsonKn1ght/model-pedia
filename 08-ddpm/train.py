"""Train a DDPM (U-Net noise predictor) on images.

    python train.py                              # MNIST, ~8 min on 4 CPU cores
    python train.py --dataset cifar10 --max-steps 6000
    python train.py --schedule linear            # the original DDPM schedule

Training is budgeted in **optimizer steps** rather than epochs, because that is
what actually determines diffusion sample quality and it makes the runtime
predictable on any machine.

An exponential moving average of the weights is kept alongside the live model
and is what gets sampled from.  For diffusion this is not a nicety: the EMA
weights produce visibly cleaner samples at a given step count, which matters a
great deal when the budget is small.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from model import GaussianDiffusion, UNet

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--dataset", default="mnist", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--max-steps", type=int, default=2000, help="total optimizer steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sample-every", type=int, default=500)
    return parser


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
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
        subset=args.train_subset,
    )

    unet = UNet(channels=meta.channels, base=args.base_channels).to(device)
    diffusion = GaussianDiffusion(unet, timesteps=args.timesteps, schedule=args.schedule).to(device)
    ema_unet = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_diffusion = GaussianDiffusion(
        ema_unet, timesteps=args.timesteps, schedule=args.schedule
    ).to(device)

    print(utils.describe_model("U-Net", unet))
    print(
        f"dataset={args.dataset} timesteps={args.timesteps} schedule={args.schedule} "
        f"steps={args.max_steps} device={device}"
    )

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    history = utils.History()
    max_steps = 8 if args.quick else args.max_steps
    sample_every = 4 if args.quick else args.sample_every
    shape = (meta.channels, args.image_size, args.image_size)

    step = 0
    window = utils.AverageMeter()
    timer = utils.Timer()
    with timer:
        while step < max_steps:
            for x, _ in loader:
                if step >= max_steps:
                    break
                x = x.to(device)
                loss = diffusion.loss(x)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                update_ema(ema_unet, unet, args.ema_decay)

                window.update(loss.item(), x.shape[0])
                step += 1

                if step % max(1, sample_every // 5) == 0 or step == max_steps:
                    history.add(step=step, noise_mse=window.avg)
                    print(f"step {step:>6}/{max_steps}  noise MSE {window.avg:.5f}")
                    window.reset()

                if step % sample_every == 0 or step == max_steps:
                    grid = ema_diffusion.sample(16, shape, device)
                    viz.save_image_grid(grid, samples_dir / f"samples-step{step:06d}.png", nrow=4)

    print(f"\ntrained {max_steps} steps in {timer.elapsed / 60:.1f} min")

    utils.save_checkpoint(
        out_dir / "ddpm.pt",
        model=unet,
        ema_model=ema_unet,
        config={
            "dataset": args.dataset,
            "image_size": args.image_size,
            "channels": meta.channels,
            "base_channels": args.base_channels,
            "timesteps": args.timesteps,
            "schedule": args.schedule,
            "max_steps": max_steps,
        },
    )
    history.save(out_dir / "history.json")
    viz.plot_curves(
        {k: v for k, v in history.records.items() if k != "step"},
        out_dir / "training-curves.png",
        title="DDPM training",
        xlabel="logged interval",
    )
    print(f"saved checkpoint and figures to {out_dir}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
