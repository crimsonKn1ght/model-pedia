"""Train a DDPM.

    python train.py --dataset fashion-mnist --epochs 20
    python train.py --schedule linear --prediction x0

The training loss here is a real, meaningful quantity - mean squared error on predicted
noise - and it still is not a quality measure. It averages over timesteps, so a model
that is excellent at ``t=5`` and useless at ``t=180`` can post the same loss as one that
is mediocre everywhere, and only the second makes good samples. So FID is computed every
``--fid-every`` epochs and selects the checkpoint.

Sampling is the expensive part: one network evaluation per timestep, per image. With the
default 200 steps, scoring a thousand samples costs 200 forward passes over a thousand
images, which is why FID is not computed every epoch and why project 13 (DDIM) exists.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import DATASETS, feature_net_cache, get_splits, make_loader
from model import PREDICTIONS, SCHEDULES, build_diffusion, build_model
from utils import (
    AverageMeter,
    count_parameters,
    features_from_loader,
    features_from_sampler,
    fid_score,
    get_device,
    load_or_train_feature_net,
    plot_curves,
    plot_image_grid,
    save_json,
    set_seed,
)


@torch.no_grad()
def validation_loss(model, diffusion, loader, device, seed: int = 1234) -> float:
    """Averaged over a fixed set of timesteps and noise, so epochs are comparable."""
    model.eval()
    meter = AverageMeter()
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        for images, _ in loader:
            images = images.to(device)
            meter.update(diffusion.loss(model, images).item(), images.size(0))
    return meter.avg


def run_training(
    dataset: str = "fashion-mnist",
    image_size: int | None = None,
    base_channels: int = 32,
    attention: bool = True,
    steps: int = 200,
    schedule: str = "cosine",
    prediction: str = "noise",
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 2e-4,
    train_subset: int | None = 20000,
    fid_every: int = 4,
    fid_track_samples: int = 500,
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
    out_path = Path(out_dir or f"outputs/{dataset}_{schedule}_t{steps}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        dataset, root=data_root, image_size=image_size, train_subset=train_subset,
        augment=True, seed=seed, synthetic=synthetic,
    )
    train_loader = make_loader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_set, 256, num_workers=num_workers)

    model = build_model(info["channels"], base_channels, attention).to(dev)
    diffusion = build_diffusion(steps, schedule, prediction).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    feature_net, feature_kind = load_or_train_feature_net(
        None if synthetic else feature_net_cache(data_root, dataset, info["size"]),
        make_loader(train_set, 256, num_workers=num_workers),
        info["channels"], info["classes"], dev, epochs=1 if synthetic else 2, verbose=False,
    )
    real_features = features_from_loader(feature_net, val_loader, dev, limit=fid_track_samples)

    config = {
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": info["classes"],
        "image_size": info["size"],
        "base_channels": base_channels,
        "attention": attention,
        "steps": steps,
        "schedule": schedule,
        "prediction": prediction,
    }

    if verbose:
        print(f"device     : {dev}")
        print(f"U-Net      : {count_parameters(model):,} parameters"
              f"{', with attention' if attention else ''}")
        print(f"diffusion  : {steps} steps, {schedule} schedule, predicting {prediction}")
        print(f"data       : {len(train_set)} train at "
              f"{info['channels']}x{info['size']}x{info['size']}")
        print(f"FID        : every {fid_every} epochs on {real_features.size(0)} samples "
              f"({feature_kind} features)\n")

    history = {"train_loss": [], "val_loss": [], "val_fid": [], "fid_epochs": []}
    best_fid, best_epoch = float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for images, _ in batches:
            images = images.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.loss(model, images)
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), images.size(0))
            batches.set_postfix(loss=f"{meter.avg:.4f}")
        scheduler.step()

        val = validation_loss(model, diffusion, val_loader, dev)
        history["train_loss"].append(meter.avg)
        history["val_loss"].append(val)

        fid = None
        if epoch % fid_every == 0 or epoch == epochs:
            fake_features = features_from_sampler(
                feature_net,
                lambda n: diffusion.sample(model, (n, info["channels"], info["size"],
                                                   info["size"]), dev),
                real_features.size(0), 125, dev,
            )
            fid = fid_score(real_features, fake_features)
            history["val_fid"].append(fid)
            history["fid_epochs"].append(epoch)
            if fid <= best_fid:
                best_fid, best_epoch = fid, epoch
                torch.save(
                    {"model_state": model.state_dict(), "epoch": epoch, "val_fid": fid, **config},
                    ckpt_path,
                )

        if verbose:
            marker = "  <- best" if fid is not None and epoch == best_epoch else ""
            fid_text = f"  FID {fid:8.3f}" if fid is not None else ""
            print(
                f"epoch {epoch:2d}/{epochs}  train {meter.avg:.4f}  val {val:.4f}{fid_text}{marker}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss",))
        samples = diffusion.sample(model, (32, info["channels"], info["size"], info["size"]), dev)
        plot_image_grid(samples.cpu(), out_path / "samples_final.png", columns=8,
                        title=f"samples after {epochs} epochs, {steps} sampling steps")

    summary = {
        **config,
        "parameters": count_parameters(model),
        "feature_net": feature_kind,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_fid": best_fid,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose and epochs:
        print(f"\nbest FID {best_fid:.3f} at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DDPM.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--steps", type=int, default=200, help="diffusion timesteps T")
    parser.add_argument("--schedule", default="cosine", choices=SCHEDULES)
    parser.add_argument("--prediction", default="noise", choices=PREDICTIONS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--fid-every", type=int, default=4, help="epochs between FID evaluations")
    parser.add_argument("--fid-track-samples", type=int, default=500)
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
        base_channels=args.base_channels,
        attention=not args.no_attention,
        steps=20 if args.smoke_test else args.steps,
        schedule=args.schedule,
        prediction=args.prediction,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=32 if args.smoke_test else args.batch_size,
        lr=args.lr,
        train_subset=args.train_subset or None,
        fid_every=1 if args.smoke_test else args.fid_every,
        fid_track_samples=64 if args.smoke_test else args.fid_track_samples,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out_dir,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
