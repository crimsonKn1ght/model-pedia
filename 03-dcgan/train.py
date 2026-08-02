"""Train a DCGAN for unconditional image generation.

    python train.py                                  # MNIST
    python train.py --dataset cifar10 --epochs 12
    python train.py --dataset celeba64 --image-size 64

Watch the two losses.  Healthy adversarial training keeps them in tension: the
discriminator loss hovering somewhere around 0.5-1.0 and D(G(z)) creeping up
towards 0.5.  A discriminator loss that crashes to zero means it has won and the
generator has stopped receiving useful gradients.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from model import Discriminator, Generator

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--dataset", default="mnist", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1; 0.5 per the DCGAN paper")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="train D against 1-eps instead of 1; damps an over-confident discriminator",
    )
    return parser


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
        augment=args.dataset in {"celeba64", "cifar10"},
        num_workers=args.num_workers,
    )

    generator = Generator(args.latent_dim, meta.channels, args.image_size, args.base_channels).to(device)
    discriminator = Discriminator(meta.channels, args.image_size, args.base_channels).to(device)
    print(utils.describe_model("generator", generator))
    print(utils.describe_model("discriminator", discriminator))
    print(f"dataset={args.dataset} image_size={args.image_size} device={device}")

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    history = utils.History()
    epochs = 1 if args.quick else args.epochs
    fixed_z = torch.randn(64, args.latent_dim, device=device)

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        d_meter, g_meter, dx_meter, dgz_meter = (utils.AverageMeter() for _ in range(4))

        for x, _ in utils.progress(
            utils.maybe_limit(loader, args), desc=f"epoch {epoch}/{epochs}",
            total=utils.quick_len(loader, args),
        ):
            x = x.to(device)
            n = x.shape[0]
            real_target = torch.full((n,), 1.0 - args.label_smoothing, device=device)
            fake_target = torch.zeros(n, device=device)

            # -- discriminator: real should score high, fake should score low -- #
            z = torch.randn(n, args.latent_dim, device=device)
            fake = generator(z)
            d_real = discriminator(x)
            d_fake = discriminator(fake.detach())  # detach: no generator gradients here
            loss_d = criterion(d_real, real_target) + criterion(d_fake, fake_target)

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # -- generator: non-saturating loss, fakes should be called real --- #
            d_fake_for_g = discriminator(fake)
            loss_g = criterion(d_fake_for_g, torch.ones(n, device=device))

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            d_meter.update(loss_d.item(), n)
            g_meter.update(loss_g.item(), n)
            dx_meter.update(torch.sigmoid(d_real).mean().item(), n)
            dgz_meter.update(torch.sigmoid(d_fake_for_g).mean().item(), n)

        history.add(
            discriminator_loss=d_meter.avg,
            generator_loss=g_meter.avg,
            d_of_real=dx_meter.avg,
            d_of_fake=dgz_meter.avg,
        )
        print(
            f"epoch {epoch:>3}  loss_D {d_meter.avg:.4f}  loss_G {g_meter.avg:.4f}  "
            f"D(x) {dx_meter.avg:.3f}  D(G(z)) {dgz_meter.avg:.3f}"
        )

        generator.eval()
        with torch.no_grad():
            viz.save_image_grid(
                generator(fixed_z), samples_dir / f"samples-epoch{epoch:03d}.png", nrow=8
            )

    utils.save_checkpoint(
        out_dir / "dcgan.pt",
        generator=generator,
        discriminator=discriminator,
        config={
            "dataset": args.dataset,
            "image_size": args.image_size,
            "channels": meta.channels,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
        },
    )
    history.save(out_dir / "history.json")
    viz.plot_curves(history.records, out_dir / "training-curves.png", title="DCGAN training")
    print(f"\nsaved checkpoint and figures to {out_dir}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
