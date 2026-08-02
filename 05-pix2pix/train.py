"""Train Pix2Pix on a paired translation task.

    python train.py                              # edges -> photo from CIFAR-10
    python train.py --dataset celeba64 --image-size 64
    python train.py --task facades               # the original pix2pix dataset
    python train.py --lambda-l1 0                # ablation: no L1 term

The L1 weight is the knob worth playing with.  At ``lambda_l1 = 100`` the
adversarial loss only sharpens what L1 has already placed; at ``0`` the model
produces confident texture that ignores the input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from data_pairs import build_loader
from model import PatchDiscriminator, UNetGenerator

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--task", default="edges", choices=["edges", "facades"])
    parser.add_argument("--dataset", default="cifar10", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4, help="U-Net downsampling stages")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    out_dir = utils.resolve_out_dir(args, str(HERE / "outputs"))
    samples_dir = utils.ensure_dir(out_dir / "samples")

    loader, in_ch, out_ch = build_loader(
        args.task, args.dataset, args.data_root, args.image_size,
        args.batch_size, train=True, num_workers=args.num_workers,
    )

    generator = UNetGenerator(in_ch, out_ch, base=args.base_channels, depth=args.depth).to(device)
    discriminator = PatchDiscriminator(in_ch + out_ch, base=args.base_channels).to(device)
    print(utils.describe_model("U-Net generator", generator))
    print(utils.describe_model("PatchGAN discriminator", discriminator))
    print(f"task={args.task} image_size={args.image_size} lambda_l1={args.lambda_l1} device={device}")

    adversarial = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    history = utils.History()
    epochs = 1 if args.quick else args.epochs
    fixed_source, fixed_target = next(iter(loader))
    fixed_source, fixed_target = fixed_source[:8].to(device), fixed_target[:8].to(device)

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        d_m, g_m, l1_m = (utils.AverageMeter() for _ in range(3))

        for source, target in utils.progress(
            utils.maybe_limit(loader, args), desc=f"epoch {epoch}/{epochs}",
            total=utils.quick_len(loader, args),
        ):
            source, target = source.to(device), target.to(device)
            n = source.shape[0]
            generated = generator(source)

            # -- discriminator: real pair vs generated pair ------------------ #
            d_real = discriminator(source, target)
            d_fake = discriminator(source, generated.detach())
            loss_d = 0.5 * (
                adversarial(d_real, torch.ones_like(d_real))
                + adversarial(d_fake, torch.zeros_like(d_fake))
            )
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # -- generator: fool D, and stay close to the target in L1 -------- #
            d_fake_for_g = discriminator(source, generated)
            loss_gan = adversarial(d_fake_for_g, torch.ones_like(d_fake_for_g))
            loss_l1 = l1(generated, target)
            loss_g = loss_gan + args.lambda_l1 * loss_l1

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            d_m.update(loss_d.item(), n)
            g_m.update(loss_gan.item(), n)
            l1_m.update(loss_l1.item(), n)

        history.add(discriminator_loss=d_m.avg, generator_gan_loss=g_m.avg, l1_loss=l1_m.avg)
        print(f"epoch {epoch:>3}  loss_D {d_m.avg:.4f}  loss_G(adv) {g_m.avg:.4f}  L1 {l1_m.avg:.4f}")

        generator.eval()
        with torch.no_grad():
            viz.save_comparison_grid(
                {
                    "input": fixed_source.repeat(1, 3, 1, 1) if in_ch == 1 else fixed_source,
                    "generated": generator(fixed_source),
                    "target": fixed_target,
                },
                samples_dir / f"translation-epoch{epoch:03d}.png",
            )

    utils.save_checkpoint(
        out_dir / "pix2pix.pt",
        generator=generator,
        discriminator=discriminator,
        config={
            "task": args.task,
            "dataset": args.dataset,
            "image_size": args.image_size,
            "in_channels": in_ch,
            "out_channels": out_ch,
            "base_channels": args.base_channels,
            "depth": args.depth,
            "lambda_l1": args.lambda_l1,
        },
    )
    history.save(out_dir / "history.json")
    viz.plot_curves(history.records, out_dir / "training-curves.png", title="Pix2Pix training")
    print(f"\nsaved checkpoint and figures to {out_dir}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
