"""Stage two: diffusion inside the frozen latent space.

    python train.py --autoencoder outputs/fashion-mnist_z4x2/autoencoder.pt

Identical to project 12's training loop with one substitution: the batch is encoded first,
and the U-Net denoises the latent instead of the image. That single change is latent
diffusion, and everything it buys follows from the size of the tensor being denoised.

The encoder is frozen and used with no gradient, so this stage costs a cheap encode per
batch plus a diffusion step on a tensor several times smaller. The saving compounds at
*sampling* time, where the diffusion model runs once per timestep and the decoder runs
once in total.

Latents are divided by the scale measured in stage one. The noise schedule assumes roughly
unit variance, and a latent whose scale drifted would break it quietly - the loss would
still fall and the samples would be wrong.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

import _paths  # noqa: F401
from autoencoder import load_autoencoder
from data import feature_net_cache, get_splits, make_loader
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


def run_training(
    autoencoder: str,
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
    first_stage, ae_config = load_autoencoder(autoencoder, dev)
    latent_scale = ae_config["latent_scale"]
    out_path = Path(out_dir or Path(autoencoder).parent)
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        ae_config["dataset"], root=data_root, image_size=ae_config["image_size"],
        train_subset=train_subset, augment=True, seed=seed, synthetic=synthetic,
    )
    train_loader = make_loader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_set, 256, num_workers=num_workers)

    # The U-Net now works on the latent, so its input channels are the latent's.
    model = build_model(ae_config["latent_channels"], base_channels, attention).to(dev)
    diffusion = build_diffusion(steps, schedule, prediction).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    grid = first_stage.grid
    latent_shape = (ae_config["latent_channels"], grid, grid)

    @torch.no_grad()
    def encode(images: torch.Tensor) -> torch.Tensor:
        mean, _ = first_stage.encode(images)
        return mean / latent_scale

    @torch.no_grad()
    def sample_images(n: int) -> torch.Tensor:
        latents = diffusion.sample(model, (n, *latent_shape), dev)
        # diffusion.sample returns [0, 1]; undo that to get back to latent units.
        latents = (latents * 2.0 - 1.0) * latent_scale
        return first_stage.decode(latents)

    feature_net, feature_kind = load_or_train_feature_net(
        None if synthetic else feature_net_cache(data_root, ae_config["dataset"],
                                                 ae_config["image_size"]),
        make_loader(train_set, 256, num_workers=num_workers),
        ae_config["in_channels"], info["classes"], dev,
        epochs=1 if synthetic else 2, verbose=False,
    )
    real_features = features_from_loader(feature_net, val_loader, dev, limit=fid_track_samples)

    config = {
        "autoencoder": str(autoencoder),
        "dataset": ae_config["dataset"],
        "in_channels": ae_config["in_channels"],
        "num_classes": info["classes"],
        "image_size": ae_config["image_size"],
        "latent_channels": ae_config["latent_channels"],
        "latent_grid": grid,
        "latent_scale": latent_scale,
        "base_channels": base_channels,
        "attention": attention,
        "steps": steps,
        "schedule": schedule,
        "prediction": prediction,
    }
    if verbose:
        print(f"device     : {dev}")
        print(f"first stage: frozen, latent {ae_config['latent_channels']}x{grid}x{grid} "
              f"({first_stage.compression:.0f}x fewer values), scale {latent_scale:.4f}")
        print(f"U-Net      : {count_parameters(model):,} parameters on the latent")
        print(f"diffusion  : {steps} steps, {schedule} schedule\n")

    history = {"train_loss": [], "val_loss": [], "val_fid": [], "fid_epochs": []}
    best_fid, best_epoch = float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False,
                       disable=not verbose)
        for images, _ in batches:
            latents = encode(images.to(dev))
            optimizer.zero_grad(set_to_none=True)
            # diffusion.loss rescales [0,1] -> [-1,1]; the latent is already centred,
            # so it is mapped back first to keep the two conventions consistent.
            loss = diffusion.loss(model, (latents + 1.0) / 2.0)
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), images.size(0))
            batches.set_postfix(loss=f"{meter.avg:.4f}")
        scheduler.step()

        model.eval()
        val_meter = AverageMeter()
        with torch.no_grad(), torch.random.fork_rng(
            devices=[dev] if dev.type == "cuda" else []
        ):
            torch.manual_seed(1234)
            for images, _ in val_loader:
                latents = encode(images.to(dev))
                val_meter.update(
                    diffusion.loss(model, (latents + 1.0) / 2.0).item(), images.size(0)
                )

        history["train_loss"].append(meter.avg)
        history["val_loss"].append(val_meter.avg)

        fid = None
        if epoch % fid_every == 0 or epoch == epochs:
            fake_features = features_from_sampler(
                feature_net, sample_images, real_features.size(0), 125, dev
            )
            fid = fid_score(real_features, fake_features)
            history["val_fid"].append(fid)
            history["fid_epochs"].append(epoch)
            if fid <= best_fid:
                best_fid, best_epoch = fid, epoch
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "val_fid": fid, **config}, ckpt_path)

        if verbose:
            marker = "  <- best" if fid is not None and epoch == best_epoch else ""
            fid_text = f"  FID {fid:8.3f}" if fid is not None else ""
            print(f"epoch {epoch:2d}/{epochs}  train {meter.avg:.4f}  "
                  f"val {val_meter.avg:.4f}{fid_text}{marker}")

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss",))
        plot_image_grid(sample_images(32).cpu(), out_path / "samples_final.png", columns=8,
                        title=f"latent diffusion samples after {epochs} epochs")

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
    parser = argparse.ArgumentParser(description="Train diffusion in a frozen latent space.")
    parser.add_argument("--autoencoder", default="outputs/fashion-mnist_z4x2/autoencoder.pt")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--schedule", default="cosine", choices=SCHEDULES)
    parser.add_argument("--prediction", default="noise", choices=PREDICTIONS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--fid-every", type=int, default=4)
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
        autoencoder=args.autoencoder,
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
