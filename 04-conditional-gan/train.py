"""Train a class-conditional GAN (cGAN or ACGAN).

    python train.py                          # ACGAN on MNIST
    python train.py --mode cgan              # the concatenation-style cGAN
    python train.py --dataset cifar10 --epochs 15

Each epoch writes a class grid: row ``k`` contains samples the generator was
asked to make of class ``k``.  Reading down the rows is the fastest way to see
whether conditioning actually took hold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from model import ConditionalGenerator, ProjectionDiscriminator

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--dataset", default="mnist", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--mode", default="acgan", choices=["cgan", "acgan"])
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument(
        "--aux-weight", type=float, default=1.0, help="ACGAN classification loss weight"
    )
    parser.add_argument(
        "--aux-on-fake",
        action="store_true",
        help="also train the discriminator's classifier on generated images, as in the "
             "original ACGAN; this reliably collapses intra-class diversity",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    out_dir = utils.resolve_out_dir(args, str(HERE / "outputs"))
    samples_dir = utils.ensure_dir(out_dir / "samples")

    meta = data_mod.info(args.dataset)
    if meta.num_classes < 2:
        raise SystemExit(f"{args.dataset} has no class labels; pick a labelled dataset")

    loader = data_mod.get_dataloader(
        args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.train_subset,
    )

    generator = ConditionalGenerator(
        args.latent_dim, meta.num_classes, meta.channels, args.image_size, args.base_channels
    ).to(device)
    discriminator = ProjectionDiscriminator(
        meta.num_classes, meta.channels, args.image_size, args.base_channels, mode=args.mode
    ).to(device)

    print(utils.describe_model("generator", generator))
    print(utils.describe_model("discriminator", discriminator))
    print(f"mode={args.mode} dataset={args.dataset} classes={meta.num_classes} device={device}")

    adversarial = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    history = utils.History()
    epochs = 1 if args.quick else args.epochs

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        d_m, g_m, acc_m = (utils.AverageMeter() for _ in range(3))

        for x, y in utils.progress(
            utils.maybe_limit(loader, args), desc=f"epoch {epoch}/{epochs}",
            total=utils.quick_len(loader, args),
        ):
            x, y = x.to(device), y.to(device)
            n = x.shape[0]
            ones = torch.ones(n, device=device)
            zeros = torch.zeros(n, device=device)

            fake_y = torch.randint(0, meta.num_classes, (n,), device=device)
            z = torch.randn(n, args.latent_dim, device=device)
            fake = generator(z, fake_y)

            # -- discriminator ------------------------------------------------ #
            real_validity, real_classes = discriminator(x, y)
            fake_validity, fake_classes = discriminator(fake.detach(), fake_y)
            loss_d = adversarial(real_validity, ones * 0.9) + adversarial(fake_validity, zeros)
            if args.mode == "acgan":
                # The classifier head trains on real images by default.
                #
                # The original ACGAN also trains it on generated images. That
                # creates a feedback loop: the generator collapses towards one
                # maximally-classifiable prototype per class, the discriminator's
                # classifier is then trained on those prototypes with their
                # labels, and it rewards the generator for collapsing further.
                # In practice this drives intra-class diversity to zero -- every
                # sample of a class comes out identical, recall goes to 0, and
                # no adversarial signal is strong enough to pull it back.
                # Keeping the classifier grounded on real data breaks the loop.
                loss_d = loss_d + args.aux_weight * F.cross_entropy(real_classes, y)
                if args.aux_on_fake:
                    loss_d = loss_d + args.aux_weight * F.cross_entropy(fake_classes, fake_y)

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # -- generator ---------------------------------------------------- #
            fake_validity, fake_classes = discriminator(fake, fake_y)
            loss_g = adversarial(fake_validity, ones)
            if args.mode == "acgan":
                loss_g = loss_g + args.aux_weight * F.cross_entropy(fake_classes, fake_y)

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            d_m.update(loss_d.item(), n)
            g_m.update(loss_g.item(), n)
            if args.mode == "acgan":
                acc_m.update((real_classes.argmax(1) == y).float().mean().item(), n)

        entries = {"discriminator_loss": d_m.avg, "generator_loss": g_m.avg}
        line = f"epoch {epoch:>3}  loss_D {d_m.avg:.4f}  loss_G {g_m.avg:.4f}"
        if args.mode == "acgan":
            entries["discriminator_class_accuracy"] = acc_m.avg
            line += f"  D class acc {acc_m.avg:.3f}"
        history.add(**entries)
        print(line)

        generator.eval()
        viz.save_image_grid(
            generator.sample_class_grid(8, device),
            samples_dir / f"class-grid-epoch{epoch:03d}.png",
            nrow=8,
        )

    utils.save_checkpoint(
        out_dir / "conditional-gan.pt",
        generator=generator,
        discriminator=discriminator,
        config={
            "dataset": args.dataset,
            "mode": args.mode,
            "image_size": args.image_size,
            "channels": meta.channels,
            "num_classes": meta.num_classes,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
            "aux_on_fake": args.aux_on_fake,
        },
    )
    history.save(out_dir / "history.json")
    viz.plot_curves(history.records, out_dir / "training-curves.png", title=f"{args.mode} training")
    print(f"\nsaved checkpoint and figures to {out_dir}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
