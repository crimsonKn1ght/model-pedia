"""Train a RealNVP flow by maximising exact likelihood.

    python train.py --dataset fashion-mnist --epochs 15

The loss *is* the metric here, which is the first time that has been true in this half of
the repository. Bits per dimension is the exact negative log likelihood of the data under
the model, in a unit that is comparable across papers and across projects - unlike the
FID numbers elsewhere, which are measured through a feature network of this repository's
own making.

Two things to watch:

* **the number to beat is 8.0 bits/dim**, which is what a uniform distribution over 256
  grey levels scores. Anything above that means the model is worse than assuming nothing.
* **dequantisation noise is on during training and off during evaluation.** The train and
  validation numbers are therefore not measuring quite the same thing, and validation is
  the honest one - see ``model.preprocess``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import DATASETS, get_splits, make_loader
from model import build_model
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
def evaluate_split(model, loader, device) -> float:
    """Exact bits/dim, with dequantisation noise off."""
    model.eval()
    meter = AverageMeter()
    for images, _ in loader:
        images = images.to(device)
        meter.update(model.bits_per_dimension(images, dequantise=False).mean().item(),
                     images.size(0))
    return meter.avg


def run_training(
    dataset: str = "fashion-mnist",
    image_size: int | None = None,
    hidden: int = 64,
    couplings_per_stage: int = 3,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    grad_clip: float = 5.0,
    dequantise: bool = True,
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
    out_path = Path(out_dir or f"outputs/{dataset}_h{hidden}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        dataset, root=data_root, image_size=image_size, train_subset=train_subset,
        seed=seed, synthetic=synthetic,
    )
    train_loader = make_loader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_set, 256, num_workers=num_workers)

    model = build_model(info["channels"], info["size"], hidden, couplings_per_stage).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    config = {
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": info["classes"],
        "image_size": info["size"],
        "hidden": hidden,
        "couplings_per_stage": couplings_per_stage,
        "dequantise": dequantise,
    }
    if verbose:
        print(f"device     : {dev}")
        print(f"flow       : {count_parameters(model):,} parameters, "
              f"{sum(1 for k in model.stage_kinds if k == 'coupling')} coupling layers")
        print(f"data       : {len(train_set)} train / {len(val_set)} val at "
              f"{info['channels']}x{info['size']}x{info['size']}")
        print("uniform baseline: 8.000 bits/dim\n")

    history = {"train_bpd": [], "val_bpd": []}
    best_bpd, best_epoch = float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False,
                       disable=not verbose)
        for images, _ in batches:
            images = images.to(dev)
            optimizer.zero_grad(set_to_none=True)
            bpd = model.bits_per_dimension(images, dequantise=dequantise).mean()
            bpd.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            meter.update(bpd.item(), images.size(0))
            batches.set_postfix(bpd=f"{meter.avg:.4f}")
        scheduler.step()

        val_bpd = evaluate_split(model, val_loader, dev)
        history["train_bpd"].append(meter.avg)
        history["val_bpd"].append(val_bpd)

        if val_bpd <= best_bpd:
            best_bpd, best_epoch = val_bpd, epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_bpd": val_bpd, **config}, ckpt_path)

        if verbose:
            print(f"epoch {epoch:2d}/{epochs}  train {meter.avg:.4f}  "
                  f"val {val_bpd:.4f} bits/dim"
                  f"{'  <- best' if epoch == best_epoch else ''}")

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("bpd",))
        plot_image_grid(model.sample(32, dev).cpu(), out_path / "samples.png", columns=8,
                        title="samples: a Gaussian pushed backwards through the flow")

    summary = {
        **config,
        "parameters": count_parameters(model),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_bpd": best_bpd,
        "uniform_baseline_bpd": 8.0,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose and epochs:
        print(f"\nbest val {best_bpd:.4f} bits/dim at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"uniform baseline is 8.000, so the model is worth "
              f"{8.0 - best_bpd:.4f} bits per pixel")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a RealNVP flow.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=64, help="coupling network width")
    parser.add_argument("--couplings-per-stage", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--no-dequantise", action="store_true",
        help="train on quantised pixels; the bits/dim it reports will be meaningless",
    )
    parser.add_argument("--train-subset", type=int, default=20000)
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
        hidden=args.hidden,
        couplings_per_stage=args.couplings_per_stage,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=32 if args.smoke_test else args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        dequantise=not args.no_dequantise,
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
