"""Stage two: train an autoregressive prior over the codes, so the VQ-VAE can generate.

    python train_prior.py --checkpoint outputs/fashion-mnist_k128/best.pt

Stage one gave a frozen mapping from images to an 8x8 grid of integers. This stage
models the distribution of those grids with a small causal Transformer in raster
order, and then sampling an image becomes: sample 64 integers from the prior, look
them up in the codebook, run the decoder.

The whole point is that stage two is now an ordinary sequence-modelling problem. The
prior never sees a pixel. Its cross entropy is a genuine likelihood over the discrete
latent - no bound, unlike the VAE's ELBO - but it is a likelihood over *codes*, so it
cannot be compared with bits-per-dimension figures on images.

The codes are extracted once with the encoder in eval mode and held in memory, which
is why this stage runs an order of magnitude faster than stage one: no convolutions
are involved at all after the first pass.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import get_splits, make_loader
from model import build_model, build_prior
from utils import (
    AverageMeter,
    count_parameters,
    get_device,
    plot_curves,
    plot_image_grid,
    save_json,
    set_seed,
)


@torch.no_grad()
def extract_codes(model, loader, device) -> torch.Tensor:
    """Encode a split to ``(N, grid*grid)`` integers, once."""
    model.eval()
    codes = [model.encode_indices(images.to(device)).flatten(1).cpu() for images, _ in loader]
    return torch.cat(codes)


@torch.no_grad()
def evaluate_prior(prior, loader, device) -> float:
    prior.eval()
    meter = AverageMeter()
    for (codes,) in loader:
        codes = codes.to(device)
        meter.update(prior.loss(codes).item(), codes.size(0))
    return meter.avg


def run_prior_training(
    checkpoint: str,
    dim: int = 128,
    depth: int = 4,
    heads: int = 4,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    train_subset: int | None = 20000,
    num_workers: int = 2,
    seed: int = 0,
    device: str = "auto",
    data_root: str = "data",
    synthetic: bool = False,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    dev = get_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=True)
    out_path = Path(checkpoint).parent

    vqvae = build_model(
        ckpt["in_channels"], ckpt["image_size"], ckpt["base_channels"], ckpt["code_dim"],
        ckpt["num_codes"], ckpt["commitment"], ckpt["ema"],
    ).to(dev)
    vqvae.load_state_dict(ckpt["model_state"])
    vqvae.eval()
    for parameter in vqvae.parameters():
        parameter.requires_grad_(False)

    train_set, val_set, _, _ = get_splits(
        ckpt["dataset"], root=data_root, image_size=ckpt["image_size"],
        train_subset=train_subset, seed=seed, synthetic=synthetic,
    )
    train_codes = extract_codes(
        vqvae, make_loader(train_set, 256, num_workers=num_workers), dev
    )
    val_codes = extract_codes(vqvae, make_loader(val_set, 256, num_workers=num_workers), dev)

    code_loader = DataLoader(TensorDataset(train_codes), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_codes), batch_size=batch_size)

    prior = build_prior(ckpt["num_codes"], vqvae.grid**2, dim, depth, heads).to(dev)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    if verbose:
        print(f"device     : {dev}")
        print(f"vq-vae     : {ckpt['num_codes']} codes on a {vqvae.grid}x{vqvae.grid} grid "
              f"(frozen, epoch {ckpt['epoch']})")
        print(f"prior      : {count_parameters(prior):,} parameters over "
              f"{vqvae.grid ** 2} positions")
        print(f"codes      : {train_codes.size(0)} train / {val_codes.size(0)} val grids\n")

    history = {"train_loss": [], "val_loss": []}
    best_loss, best_epoch = float("inf"), 0
    prior_path = out_path / "prior.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        prior.train()
        meter = AverageMeter()
        batches = tqdm(code_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for (codes,) in batches:
            codes = codes.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = prior.loss(codes)
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), codes.size(0))
            batches.set_postfix(loss=f"{meter.avg:.4f}")
        scheduler.step()

        val_loss = evaluate_prior(prior, val_loader, dev)
        history["train_loss"].append(meter.avg)
        history["val_loss"].append(val_loss)

        if val_loss <= best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(
                {
                    "prior_state": prior.state_dict(),
                    "dim": dim, "depth": depth, "heads": heads,
                    "num_codes": ckpt["num_codes"], "sequence_length": vqvae.grid**2,
                    "epoch": epoch, "val_loss": val_loss,
                },
                prior_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  train {meter.avg:.4f}  val {val_loss:.4f} nats/code"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "prior_curves.png", keys=("loss",))
        sampled = prior.sample(32, dev).view(-1, vqvae.grid, vqvae.grid)
        plot_image_grid(
            vqvae.decode_indices(sampled).cpu(), out_path / "prior_samples.png", columns=8,
            title="codes sampled from the prior, decoded by the VQ-VAE",
        )

    summary = {
        "checkpoint": str(prior_path),
        "vqvae_checkpoint": str(checkpoint),
        "parameters": count_parameters(prior),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "uniform_baseline_nats": float(torch.tensor(float(ckpt["num_codes"])).log()),
        "train_seconds": round(elapsed, 1),
        "history": history,
    }
    save_json(summary, out_path / "prior_history.json")

    if verbose and epochs:
        uniform = summary["uniform_baseline_nats"]
        print(f"\nbest val {best_loss:.4f} nats/code at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"uniform-code baseline: {uniform:.4f} nats/code - the prior is worth "
              f"{uniform - best_loss:.4f} nats per position")
        print(f"prior      -> {prior_path}")
        print(f"next       : python evaluate.py --checkpoint {checkpoint}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an autoregressive prior over VQ codes.")
    parser.add_argument("--checkpoint", default="outputs/fashion-mnist_k128/best.pt")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch on generated data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_prior_training(
        checkpoint=args.checkpoint,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=32 if args.smoke_test else args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_subset=args.train_subset or None,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
