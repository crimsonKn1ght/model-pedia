"""Obtain the DDPM that DDIM will sample from.

    python train.py                  # reuse project 08's checkpoint if present
    python train.py --force-train    # train a fresh one here instead

DDIM introduces **no new parameters and no new training objective** -- it is a
different way of running the reverse process of an already-trained DDPM. So this
script does not train anything if project 08 has already produced a checkpoint;
it reuses it, which is the setup the comparison in ``evaluate.py`` assumes
("same trained DDPM, different sampler").

If project 08 has not been run, this trains an equivalent DDPM here so the
project still works standalone.
"""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from ddpm_bridge import GaussianDiffusion, UNet

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / "08-ddpm" / "outputs" / "ddpm.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--source-checkpoint", default=str(DEFAULT_SOURCE))
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--dataset", default="mnist", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    return parser


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def train_from_scratch(args, out_dir: Path) -> None:
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    meta = data_mod.info(args.dataset)
    loader = data_mod.get_dataloader(
        args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    unet = UNet(channels=meta.channels, base=args.base_channels).to(device)
    diffusion = GaussianDiffusion(unet, timesteps=args.timesteps, schedule=args.schedule).to(device)
    ema_unet = copy.deepcopy(unet).eval().requires_grad_(False)
    print(utils.describe_model("U-Net", unet))

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    history = utils.History()
    max_steps = 8 if args.quick else args.max_steps
    step = 0
    window = utils.AverageMeter()

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
            if step % max(1, max_steps // 10) == 0 or step == max_steps:
                history.add(noise_mse=window.avg)
                print(f"step {step:>6}/{max_steps}  noise MSE {window.avg:.5f}")
                window.reset()

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
    viz.plot_curves(history.records, out_dir / "training-curves.png", title="DDPM training (for DDIM)")


def main() -> None:
    args = build_parser().parse_args()
    out_dir = utils.resolve_out_dir(args, str(HERE / "outputs"))
    target = out_dir / "ddpm.pt"
    source = Path(args.source_checkpoint)

    if not args.force_train and source.exists():
        shutil.copyfile(source, target)
        cfg = utils.load_checkpoint(target)["config"]
        print(f"reused the DDPM trained by project 08: {source}")
        print(
            f"  dataset={cfg['dataset']} timesteps={cfg['timesteps']} "
            f"trained for {cfg['max_steps']} steps"
        )
        print("DDIM introduces no new parameters, so no training was needed.")
    else:
        if not source.exists():
            print(f"{source} not found; training an equivalent DDPM here instead")
        train_from_scratch(args, out_dir)

    print(f"\ncheckpoint ready at {target}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
