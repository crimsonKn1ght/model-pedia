"""Train CycleGAN between two unpaired image collections.

    python train.py                                   # CIFAR-10 horse <-> deer
    python train.py --domain-a 7 --domain-b 9         # horse <-> truck
    python train.py --task datasets --domain-a mnist --domain-b fashion-mnist
    python train.py --task horse2zebra --image-size 64

Four losses are combined:

* adversarial (least squares) for each direction,
* cycle consistency in both directions, weighted by ``--lambda-cycle``,
* identity, weighted by ``--lambda-identity``.

Setting ``--lambda-cycle 0`` is the instructive ablation: the adversarial losses
keep falling while the translations stop having anything to do with the inputs.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn as nn

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from domains import build_domains, unpaired_loader
from model import ImagePool, PatchDiscriminator, ResnetGenerator

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--task", default="classes", choices=["classes", "datasets", "horse2zebra"])
    parser.add_argument("--dataset", default="cifar10", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--domain-a", default="7", help="class index, or dataset name for --task datasets")
    parser.add_argument("--domain-b", default="4")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--res-blocks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=0.5)
    parser.add_argument("--pool-size", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    out_dir = utils.resolve_out_dir(args, str(HERE / "outputs"))
    samples_dir = utils.ensure_dir(out_dir / "samples")

    set_a, set_b, channels, name_a, name_b = build_domains(
        args.task, args.dataset, args.data_root, args.image_size, True, args.domain_a,
        args.domain_b, subset=args.train_subset,
    )
    loader = unpaired_loader(set_a, set_b, args.batch_size, True, args.num_workers, args.seed)

    # G: A -> B and F: B -> A
    g_ab = ResnetGenerator(channels, args.base_channels, args.res_blocks).to(device)
    g_ba = ResnetGenerator(channels, args.base_channels, args.res_blocks).to(device)
    d_a = PatchDiscriminator(channels, args.base_channels).to(device)
    d_b = PatchDiscriminator(channels, args.base_channels).to(device)

    print(utils.describe_model("generator A->B", g_ab))
    print(utils.describe_model("discriminator B", d_b))
    print(
        f"domains: {name_a} ({len(set_a)} images)  <->  {name_b} ({len(set_b)} images)  "
        f"device={device}"
    )

    # Least-squares GAN loss: MSE against 1 for real, 0 for fake.
    gan_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    opt_g = torch.optim.Adam(
        itertools.chain(g_ab.parameters(), g_ba.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )
    opt_d = torch.optim.Adam(
        itertools.chain(d_a.parameters(), d_b.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )
    pool_a, pool_b = ImagePool(args.pool_size), ImagePool(args.pool_size)

    history = utils.History()
    epochs = 1 if args.quick else args.epochs
    fixed_a, fixed_b = next(iter(loader))
    fixed_a, fixed_b = fixed_a[:8].to(device), fixed_b[:8].to(device)

    for epoch in range(1, epochs + 1):
        for net in (g_ab, g_ba, d_a, d_b):
            net.train()
        g_m, d_m, cyc_m = (utils.AverageMeter() for _ in range(3))

        for real_a, real_b in utils.progress(
            utils.maybe_limit(loader, args), desc=f"epoch {epoch}/{epochs}",
            total=utils.quick_len(loader, args),
        ):
            real_a, real_b = real_a.to(device), real_b.to(device)
            n = real_a.shape[0]

            # -- generators ---------------------------------------------------- #
            fake_b = g_ab(real_a)
            fake_a = g_ba(real_b)
            pred_fake_b = d_b(fake_b)
            pred_fake_a = d_a(fake_a)
            loss_gan = gan_loss(pred_fake_b, torch.ones_like(pred_fake_b)) + gan_loss(
                pred_fake_a, torch.ones_like(pred_fake_a)
            )

            # Cycle consistency: translate back and compare with the original.
            recovered_a = g_ba(fake_b)
            recovered_b = g_ab(fake_a)
            loss_cycle = cycle_loss(recovered_a, real_a) + cycle_loss(recovered_b, real_b)

            # Identity: feeding a generator an image already in its target
            # domain should change nothing.
            loss_identity = torch.zeros((), device=device)
            if args.lambda_identity > 0:
                loss_identity = identity_loss(g_ab(real_b), real_b) + identity_loss(
                    g_ba(real_a), real_a
                )

            loss_g = (
                loss_gan
                + args.lambda_cycle * loss_cycle
                + args.lambda_cycle * args.lambda_identity * loss_identity
            )
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            # -- discriminators (against pooled history) ----------------------- #
            pred_real_a = d_a(real_a)
            pred_pool_a = d_a(pool_a.query(fake_a).detach())
            loss_d_a = 0.5 * (
                gan_loss(pred_real_a, torch.ones_like(pred_real_a))
                + gan_loss(pred_pool_a, torch.zeros_like(pred_pool_a))
            )
            pred_real_b = d_b(real_b)
            pred_pool_b = d_b(pool_b.query(fake_b).detach())
            loss_d_b = 0.5 * (
                gan_loss(pred_real_b, torch.ones_like(pred_real_b))
                + gan_loss(pred_pool_b, torch.zeros_like(pred_pool_b))
            )
            loss_d = loss_d_a + loss_d_b

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            g_m.update(loss_gan.item(), n)
            d_m.update(loss_d.item(), n)
            cyc_m.update(loss_cycle.item(), n)

        history.add(generator_gan_loss=g_m.avg, discriminator_loss=d_m.avg, cycle_loss=cyc_m.avg)
        print(
            f"epoch {epoch:>3}  loss_G(adv) {g_m.avg:.4f}  loss_D {d_m.avg:.4f}  "
            f"cycle {cyc_m.avg:.4f}"
        )

        for net in (g_ab, g_ba):
            net.eval()
        with torch.no_grad():
            viz.save_comparison_grid(
                {
                    f"real {name_a}": fixed_a,
                    f"-> {name_b}": g_ab(fixed_a),
                    f"cycled {name_a}": g_ba(g_ab(fixed_a)),
                    f"real {name_b}": fixed_b,
                    f"-> {name_a}": g_ba(fixed_b),
                },
                samples_dir / f"translations-epoch{epoch:03d}.png",
            )

    utils.save_checkpoint(
        out_dir / "cyclegan.pt",
        g_ab=g_ab,
        g_ba=g_ba,
        d_a=d_a,
        d_b=d_b,
        config={
            "task": args.task,
            "dataset": args.dataset,
            "domain_a": args.domain_a,
            "domain_b": args.domain_b,
            "name_a": name_a,
            "name_b": name_b,
            "channels": channels,
            "image_size": args.image_size,
            "base_channels": args.base_channels,
            "res_blocks": args.res_blocks,
            "lambda_cycle": args.lambda_cycle,
        },
    )
    history.save(out_dir / "history.json")
    viz.plot_curves(history.records, out_dir / "training-curves.png", title="CycleGAN training")
    print(f"\nsaved checkpoint and figures to {out_dir}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
