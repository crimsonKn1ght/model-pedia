"""Train a VQ-VAE, then train a PixelCNN prior over its discrete codes.

    python train.py                                # CIFAR-10
    python train.py --dataset celeba64 --image-size 64
    python train.py --prior-epochs 0               # compressor only, no sampling

Stage 1 learns the encoder, codebook and decoder by reconstruction.  Stage 2
freezes those and fits an autoregressive prior to the index grids, which is what
makes it possible to *generate* rather than only reconstruct.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import bootstrap  # noqa: F401
from common import data as data_mod
from common import utils, viz
from model import PixelCNNPrior, VQVAE

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--dataset", default="cifar10", choices=data_mod.DATASET_NAMES)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--num-embeddings", type=int, default=256, help="codebook size K")
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99, help="0 disables the EMA codebook")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--prior-epochs", type=int, default=4)
    parser.add_argument("--prior-hidden", type=int, default=96)
    parser.add_argument("--prior-layers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    return parser


def train_vqvae(args, model, loader, device, out_dir, history) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    samples_dir = utils.ensure_dir(out_dir / "samples")
    fixed = next(iter(loader))[0][:8].to(device)
    epochs = 1 if args.quick else args.epochs

    for epoch in range(1, epochs + 1):
        model.train()
        recon_m, vq_m, ppl_m = (utils.AverageMeter() for _ in range(3))
        for x, _ in utils.progress(
            utils.maybe_limit(loader, args), desc=f"vq-vae epoch {epoch}/{epochs}",
            total=utils.quick_len(loader, args),
        ):
            x = x.to(device)
            recon, vq_loss, _, perplexity = model(x)
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + vq_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = x.shape[0]
            recon_m.update(recon_loss.item(), n)
            vq_m.update(vq_loss.item(), n)
            ppl_m.update(perplexity.item(), n)

        history.add(reconstruction_mse=recon_m.avg, vq_loss=vq_m.avg, codebook_perplexity=ppl_m.avg)
        print(
            f"vq-vae epoch {epoch:>3}  recon {recon_m.avg:.5f}  vq {vq_m.avg:.5f}  "
            f"perplexity {ppl_m.avg:7.2f} / {args.num_embeddings}"
        )

        model.eval()
        with torch.no_grad():
            viz.save_comparison_grid(
                {"input": fixed, "reconstruction": model(fixed)[0]},
                samples_dir / f"reconstructions-epoch{epoch:03d}.png",
            )


def train_prior(args, model, prior, loader, device, out_dir, history) -> None:
    """Fit the PixelCNN to the index grids produced by the frozen VQ-VAE."""
    optimizer = torch.optim.Adam(prior.parameters(), lr=1e-3)
    samples_dir = utils.ensure_dir(out_dir / "samples")
    epochs = 1 if args.quick else args.prior_epochs
    model.eval()

    for epoch in range(1, epochs + 1):
        prior.train()
        loss_m, acc_m = utils.AverageMeter(), utils.AverageMeter()
        for x, _ in utils.progress(
            utils.maybe_limit(loader, args), desc=f"prior epoch {epoch}/{epochs}",
            total=utils.quick_len(loader, args),
        ):
            x = x.to(device)
            with torch.no_grad():
                indices = model.encode_indices(x)
            logits = prior(indices)
            loss = F.cross_entropy(logits, indices)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = x.shape[0]
            loss_m.update(loss.item(), n)
            acc_m.update((logits.argmax(dim=1) == indices).float().mean().item(), n)

        history.add(prior_nll=loss_m.avg, prior_accuracy=acc_m.avg)
        print(f"prior  epoch {epoch:>3}  nll {loss_m.avg:.4f} nats/code  next-code acc {acc_m.avg:.3f}")

        latent_hw = args.image_size // 4
        grids = prior.sample(16, latent_hw, latent_hw, device)
        viz.save_image_grid(
            model.decode_indices(grids), samples_dir / f"prior-samples-epoch{epoch:03d}.png", nrow=4
        )


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    out_dir = utils.resolve_out_dir(args, str(HERE / "outputs"))

    meta = data_mod.info(args.dataset)
    loader = data_mod.get_dataloader(
        args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = VQVAE(
        channels=meta.channels,
        hidden=args.hidden,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
    ).to(device)
    prior = PixelCNNPrior(args.num_embeddings, args.prior_hidden, args.prior_layers).to(device)

    print(utils.describe_model("VQ-VAE", model))
    print(utils.describe_model("PixelCNN prior", prior))
    latent_hw = args.image_size // 4
    print(
        f"dataset={args.dataset}  latent grid {latent_hw}x{latent_hw} of "
        f"{args.num_embeddings} symbols  device={device}"
    )

    history = utils.History()
    train_vqvae(args, model, loader, device, out_dir, history)
    if args.prior_epochs > 0:
        train_prior(args, model, prior, loader, device, out_dir, history)

    utils.save_checkpoint(
        out_dir / "vqvae.pt",
        model=model,
        prior=prior,
        config={
            "dataset": args.dataset,
            "image_size": args.image_size,
            "channels": meta.channels,
            "hidden": args.hidden,
            "embedding_dim": args.embedding_dim,
            "num_embeddings": args.num_embeddings,
            "commitment_cost": args.commitment_cost,
            "decay": args.decay,
            "prior_hidden": args.prior_hidden,
            "prior_layers": args.prior_layers,
            "prior_trained": args.prior_epochs > 0,
        },
    )
    history.save(out_dir / "history.json")
    viz.plot_curves(history.records, out_dir / "training-curves.png", title="VQ-VAE training")
    print(f"\nsaved checkpoint and figures to {out_dir}")
    print("next:  python evaluate.py")


if __name__ == "__main__":
    main()
