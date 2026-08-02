"""Train a RealNVP flow by directly maximising the exact log-likelihood.

    python train.py                          # MNIST
    python train.py --dataset cifar10 --epochs 12

There is no adversarial game and no variational bound here: the loss *is* the
negative log-likelihood, reported in bits per dimension.  Uniform noise
dequantisation means the printed value is a proper upper bound on the
discrete-data bits/dim, directly comparable to numbers in the flow literature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from model import RealNVP

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--dataset", default="mnist", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-scales", type=int, default=2)
    parser.add_argument("--couplings-per-scale", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=50.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    out_dir = utils.resolve_out_dir(args, str(HERE / "outputs"))
    samples_dir = utils.ensure_dir(out_dir / "samples")

    meta = data_mod.info(args.dataset)
    # Flows model the raw pixel distribution, so no [-1, 1] rescaling here.
    loader = data_mod.get_dataloader(
        args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        train=True,
        batch_size=args.batch_size,
        normalize="unit",
        num_workers=args.num_workers,
    )

    model = RealNVP(
        channels=meta.channels,
        image_size=args.image_size,
        hidden=args.hidden,
        num_scales=args.num_scales,
        couplings_per_scale=args.couplings_per_scale,
    ).to(device)
    print(utils.describe_model("RealNVP", model))
    print(
        f"dataset={args.dataset}  latent {model.final_channels}x{model.final_size}x"
        f"{model.final_size}  device={device}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = utils.History()
    epochs = 1 if args.quick else args.epochs

    for epoch in range(1, epochs + 1):
        model.train()
        bpd_meter = utils.AverageMeter()
        for x, _ in utils.progress(
            utils.maybe_limit(loader, args), desc=f"epoch {epoch}/{epochs}",
            total=utils.quick_len(loader, args),
        ):
            x = x.to(device)
            loss = model.bits_per_dim(x).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Flows are prone to occasional exploding gradients; clip firmly.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            bpd_meter.update(loss.item(), x.shape[0])

        history.add(train_bits_per_dim=bpd_meter.avg)
        print(f"epoch {epoch:>3}  train bits/dim {bpd_meter.avg:.4f}")

        model.eval()
        viz.save_image_grid(
            model.sample(64, device, temperature=0.8),
            samples_dir / f"samples-epoch{epoch:03d}.png",
            nrow=8,
            normalize="unit",
        )

    utils.save_checkpoint(
        out_dir / "realnvp.pt",
        model=model,
        config={
            "dataset": args.dataset,
            "image_size": args.image_size,
            "channels": meta.channels,
            "hidden": args.hidden,
            "num_scales": args.num_scales,
            "couplings_per_scale": args.couplings_per_scale,
        },
    )
    history.save(out_dir / "history.json")
    viz.plot_curves(history.records, out_dir / "training-curves.png", title="RealNVP training")
    print(f"\nsaved checkpoint and figures to {out_dir}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
