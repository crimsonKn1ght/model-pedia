"""Train a VAE.

    python train.py --dataset fashion-mnist --epochs 12
    python train.py --beta 4 --latent-dim 16          # a beta-VAE

Three numbers are printed every epoch and they say different things:

* **-ELBO** is the objective with ``beta`` set back to 1. It is the only number that
  can be compared between arms of the beta study, because the training loss itself
  changes units when beta changes.
* **KL** is how many nats the latent is actually carrying. Watch it fall as beta
  rises.
* **active** counts latent dimensions with more than 0.01 nats of KL. This is the
  posterior-collapse counter, and it is more informative than the KL total: the
  latent does not fade smoothly, dimensions switch off one at a time.

Checkpoints are selected on validation -ELBO. In eval mode the encoder returns its
posterior mean rather than a sample, so validation numbers are not noisy.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import DATASETS, get_splits, make_loader
from model import LIKELIHOODS, active_units, build_model
from utils import (
    AverageMeter,
    count_parameters,
    get_device,
    plot_curves,
    plot_image_grid,
    plot_image_rows,
    save_json,
    set_seed,
)


@torch.no_grad()
def evaluate_split(model, loader, device) -> dict:
    """-ELBO and its parts over a whole split, with the deterministic encoder."""
    model.eval()
    meters = {key: AverageMeter() for key in ("loss", "reconstruction", "kl", "neg_elbo")}
    kl_sum, batches = None, 0

    for images, _ in loader:
        images = images.to(device)
        outputs = model(images)
        losses = model.loss(images, outputs)
        for key, meter in meters.items():
            meter.update(losses[key].item(), images.size(0))
        kl_sum = losses["kl_per_dimension"] if kl_sum is None else kl_sum + losses["kl_per_dimension"]
        batches += 1

    kl_per_dimension = (kl_sum / max(batches, 1)).cpu()
    return {
        **{key: meter.avg for key, meter in meters.items()},
        "kl_per_dimension": kl_per_dimension.tolist(),
        "active_units": active_units(kl_per_dimension),
    }


@torch.no_grad()
def save_figures(model, loader, device, out_path: Path, n: int = 8) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)
    outputs = model(images)
    reconstructions = model.reconstruct_from_logits(outputs["logits"])
    plot_image_rows(
        [("input", images.cpu()), ("reconstruction", reconstructions.cpu())],
        out_path / "reconstructions_val.png",
        title="validation reconstructions (posterior mean, no sampling)",
    )
    plot_image_grid(
        model.sample(32, device).cpu(),
        out_path / "samples.png",
        columns=8,
        title="samples decoded from the prior",
    )


def run_training(
    dataset: str = "fashion-mnist",
    image_size: int | None = None,
    latent_dim: int = 16,
    base_channels: int = 32,
    beta: float = 1.0,
    likelihood: str = "bernoulli",
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    val_split: float = 0.05,
    train_subset: int | None = 20000,
    num_workers: int = 2,
    seed: int = 0,
    device: str = "auto",
    data_root: str = "data",
    out_dir: str | None = None,
    synthetic: bool = False,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    dev = get_device(device)
    out_path = Path(out_dir or f"outputs/{dataset}_beta{beta:g}_z{latent_dim}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        dataset,
        root=data_root,
        image_size=image_size,
        val_split=val_split,
        train_subset=train_subset,
        seed=seed,
        synthetic=synthetic,
    )
    train_loader = make_loader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_set, batch_size, num_workers=num_workers)

    model = build_model(
        info["channels"], info["size"], latent_dim, base_channels, beta, likelihood
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    config = {
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": info["classes"],
        "image_size": info["size"],
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "beta": beta,
        "likelihood": likelihood,
    }

    if verbose:
        print(f"device     : {dev}")
        print(f"model      : {count_parameters(model):,} parameters, "
              f"latent {latent_dim}, beta {beta:g}, {likelihood} likelihood")
        print(f"data       : {len(train_set)} train / {len(val_set)} val at "
              f"{info['channels']}x{info['size']}x{info['size']}\n")

    history = {
        "train_loss": [], "train_neg_elbo": [], "train_kl": [],
        "val_loss": [], "val_neg_elbo": [], "val_kl": [], "val_active": [],
    }
    best_elbo, best_epoch = float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        meters = {key: AverageMeter() for key in ("loss", "neg_elbo", "kl")}
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for images, _ in batches:
            images = images.to(dev)
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(images, model(images))
            losses["loss"].backward()
            optimizer.step()
            for key, meter in meters.items():
                meter.update(losses[key].item(), images.size(0))
            batches.set_postfix(loss=f"{meters['loss'].avg:.1f}")
        scheduler.step()

        val = evaluate_split(model, val_loader, dev)
        history["train_loss"].append(meters["loss"].avg)
        history["train_neg_elbo"].append(meters["neg_elbo"].avg)
        history["train_kl"].append(meters["kl"].avg)
        history["val_loss"].append(val["loss"])
        history["val_neg_elbo"].append(val["neg_elbo"])
        history["val_kl"].append(val["kl"])
        history["val_active"].append(val["active_units"])

        if val["neg_elbo"] <= best_elbo:
            best_elbo, best_epoch = val["neg_elbo"], epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch,
                 "val_neg_elbo": val["neg_elbo"], **config},
                ckpt_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  train -ELBO {meters['neg_elbo'].avg:8.2f}  |  "
                f"val -ELBO {val['neg_elbo']:8.2f}  recon {val['reconstruction']:8.2f}  "
                f"KL {val['kl']:6.2f}  active {val['active_units']:2d}/{latent_dim}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss", "neg_elbo", "kl"))
        save_figures(model, val_loader, dev, out_path)

    summary = {
        **config,
        "parameters": count_parameters(model),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_neg_elbo": best_elbo,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose and epochs:
        print(f"\nbest val -ELBO {best_elbo:.2f} nats at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a convolutional VAE.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="weight on the KL term; 1 is a VAE, higher is a beta-VAE",
    )
    parser.add_argument("--likelihood", default="bernoulli", choices=LIKELIHOODS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--train-subset", type=int, default=20000, help="cap images (0 = all)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch on generated data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        dataset=args.dataset,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        beta=args.beta,
        likelihood=args.likelihood,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=32 if args.smoke_test else args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        train_subset=args.train_subset or None,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out_dir,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
