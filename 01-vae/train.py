"""Train a convolutional VAE / beta-VAE.

    python train.py                                  # MNIST, beta = 1
    python train.py --dataset fashion-mnist --epochs 8
    python train.py --beta 4 --out-dir outputs/beta4  # the beta-VAE comparison
    python train.py --dataset celeba64 --image-size 64 --likelihood gaussian

Each epoch writes a sample grid and a reconstruction grid so the model can be
watched as it learns.  Everything lands in ``--out-dir`` (default ``outputs/``
inside this project folder).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from model import VAE, bits_per_dim, vae_loss

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--dataset", default="mnist", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight; >1 gives a beta-VAE")
    parser.add_argument(
        "--likelihood",
        default="bernoulli",
        choices=["bernoulli", "gaussian"],
        help="bernoulli suits MNIST-like data, gaussian suits natural images",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    out_dir = utils.resolve_out_dir(args, str(HERE / "outputs"))
    samples_dir = utils.ensure_dir(out_dir / "samples")

    meta = data_mod.info(args.dataset)
    # The VAE works in [0, 1] because both likelihoods are defined there.
    loader = data_mod.get_dataloader(
        args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        train=True,
        batch_size=args.batch_size,
        normalize="unit",
        num_workers=args.num_workers,
    )

    model = VAE(
        channels=meta.channels,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        base=args.base_channels,
        likelihood=args.likelihood,
    ).to(device)
    print(utils.describe_model("VAE", model))
    print(f"dataset={args.dataset} beta={args.beta} likelihood={args.likelihood} device={device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = utils.History()
    epochs = 1 if args.quick else args.epochs

    # A fixed noise vector so the per-epoch sample grids are directly comparable.
    fixed_z = torch.randn(64, args.latent_dim, device=device)
    fixed_batch = next(iter(loader))[0][:16].to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        elbo_meter, recon_meter, kl_meter = (utils.AverageMeter() for _ in range(3))

        for x, _ in utils.progress(
            utils.maybe_limit(loader, args), desc=f"epoch {epoch}/{epochs}",
            total=utils.quick_len(loader, args),
        ):
            x = x.to(device)
            recon_logits, mu, logvar = model(x)
            loss, recon, kl = vae_loss(
                recon_logits, x, mu, logvar, beta=args.beta, likelihood=args.likelihood
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            n = x.shape[0]
            elbo_meter.update(loss.item(), n)
            recon_meter.update(recon.item(), n)
            kl_meter.update(kl.item(), n)

        bpd = bits_per_dim(recon_meter.avg + kl_meter.avg, meta.channels, args.image_size)
        history.add(
            neg_elbo=elbo_meter.avg,
            reconstruction=recon_meter.avg,
            kl=kl_meter.avg,
            bits_per_dim=bpd,
        )
        print(
            f"epoch {epoch:>3}  -ELBO {elbo_meter.avg:9.3f}  "
            f"recon {recon_meter.avg:9.3f}  KL {kl_meter.avg:8.3f}  bpd {bpd:6.4f}"
        )

        model.eval()
        with torch.no_grad():
            viz.save_image_grid(
                model.decode_to_image(fixed_z), samples_dir / f"prior-samples-epoch{epoch:03d}.png",
                nrow=8, normalize="unit",
            )
            viz.save_comparison_grid(
                {"input": fixed_batch, "reconstruction": model.reconstruct(fixed_batch)},
                samples_dir / f"reconstructions-epoch{epoch:03d}.png",
                normalize="unit",
            )

    utils.save_checkpoint(
        out_dir / "vae.pt",
        model=model,
        config={
            "dataset": args.dataset,
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
            "likelihood": args.likelihood,
            "beta": args.beta,
            "channels": meta.channels,
        },
    )
    history.save(out_dir / "history.json")
    viz.plot_curves(history.records, out_dir / "training-curves.png", title="VAE training")
    print(f"\nsaved checkpoint and figures to {out_dir}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
