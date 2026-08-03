"""Train a DCGAN.

    python train.py --dataset fashion-mnist --epochs 20
    python train.py --loss hinge --label-smoothing 0.1

**The loss curves do not tell you whether it is working.** A GAN's losses measure who
is currently winning a game whose equilibrium is both networks being confused, so a
falling generator loss can mean better samples or a collapsing discriminator, and
there is no way to tell from the number. This is not a subtlety to keep in mind, it is
the central practical fact about training GANs.

So this script computes **FID every epoch** against a fixed set of real features and
selects the checkpoint on that. It costs one pass over a thousand generated images
per epoch and it is the only honest progress signal available. Watch `curves.png`:
the losses wander, the FID descends.

Two diagnostics are printed beside them, and they are what you read when a run goes
wrong:

* ``D(real)`` and ``D(fake)`` - the discriminator's mean probabilities. Both near 0.5
  is a healthy game. ``D(real)`` at 1.0 and ``D(fake)`` at 0.0 means the
  discriminator has won and the generator has no gradient left to follow.
* ``recall`` in ``evaluate.py`` - the mode-collapse detector. A collapsed generator
  can post a decent FID and near-zero recall.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import DATASETS, feature_net_cache, get_splits, make_loader
from model import (
    LOSSES,
    build_models,
    discriminator_loss,
    generator_loss,
)
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
    dataset: str = "fashion-mnist",
    image_size: int | None = None,
    latent_dim: int = 64,
    width: int = 64,
    loss: str = "bce",
    label_smoothing: float = 0.0,
    d_steps: int = 1,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 2e-4,
    beta1: float = 0.5,
    train_subset: int | None = 20000,
    fid_track_samples: int = 1000,
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
    out_path = Path(out_dir or f"outputs/{dataset}_{loss}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        dataset, root=data_root, image_size=image_size, train_subset=train_subset,
        augment=True, seed=seed, synthetic=synthetic,
    )
    train_loader = make_loader(
        train_set, batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = make_loader(val_set, 256, num_workers=num_workers)

    generator, discriminator = build_models(latent_dim, info["channels"], info["size"], width)
    generator, discriminator = generator.to(dev), discriminator.to(dev)
    # beta1=0.5 rather than the usual 0.9: the DCGAN paper's fix for oscillation, and
    # one of the few hyperparameters here that genuinely matters.
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # The FID ruler, and the real features to measure against. Built once.
    feature_net, feature_kind = load_or_train_feature_net(
        None if synthetic else feature_net_cache(data_root, dataset, info["size"]),
        make_loader(train_set, 256, num_workers=num_workers),
        info["channels"], info["classes"], dev,
        epochs=1 if synthetic else 2, verbose=False,
    )
    real_features = features_from_loader(feature_net, val_loader, dev, limit=fid_track_samples)
    fixed_latent = torch.randn(64, latent_dim, device=dev)

    config = {
        "dataset": dataset,
        "in_channels": info["channels"],
        "num_classes": info["classes"],
        "image_size": info["size"],
        "latent_dim": latent_dim,
        "width": width,
        "loss": loss,
        "label_smoothing": label_smoothing,
        "d_steps": d_steps,
    }

    if verbose:
        print(f"device     : {dev}")
        print(f"generator  : {count_parameters(generator):,} parameters, latent {latent_dim}")
        print(f"discrimin. : {count_parameters(discriminator):,} parameters, {loss} loss")
        print(f"data       : {len(train_set)} train images at "
              f"{info['channels']}x{info['size']}x{info['size']}")
        print(f"FID        : every epoch on {real_features.size(0)} samples, "
              f"{feature_kind} features\n")

    history = {
        "train_d_loss": [], "train_g_loss": [],
        "train_d_real": [], "train_d_fake": [], "val_fid": [],
    }
    best_fid, best_epoch = float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        meters = {key: AverageMeter() for key in ("d_loss", "g_loss", "d_real", "d_fake")}
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)

        for images, _ in batches:
            images = images.to(dev)
            n = images.size(0)

            for _ in range(d_steps):
                latent = torch.randn(n, latent_dim, device=dev)
                with torch.no_grad():
                    fake = generator(latent)
                real_logits = discriminator(images)
                fake_logits = discriminator(fake)
                d_loss = discriminator_loss(real_logits, fake_logits, loss, label_smoothing)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            latent = torch.randn(n, latent_dim, device=dev)
            fake_logits = discriminator(generator(latent))
            g_loss = generator_loss(fake_logits, loss)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            meters["d_loss"].update(d_loss.item(), n)
            meters["g_loss"].update(g_loss.item(), n)
            meters["d_real"].update(real_logits.detach().sigmoid().mean().item(), n)
            meters["d_fake"].update(fake_logits.detach().sigmoid().mean().item(), n)
            batches.set_postfix(d=f"{meters['d_loss'].avg:.3f}", g=f"{meters['g_loss'].avg:.3f}")

        generator.eval()
        fake_features = features_from_sampler(
            feature_net, lambda n: generator.sample(n, dev), real_features.size(0), 256, dev
        )
        fid = fid_score(real_features, fake_features)

        history["train_d_loss"].append(meters["d_loss"].avg)
        history["train_g_loss"].append(meters["g_loss"].avg)
        history["train_d_real"].append(meters["d_real"].avg)
        history["train_d_fake"].append(meters["d_fake"].avg)
        history["val_fid"].append(fid)

        if fid <= best_fid:
            best_fid, best_epoch = fid, epoch
            torch.save(
                {
                    "generator_state": generator.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "epoch": epoch, "val_fid": fid, **config,
                },
                ckpt_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  D {meters['d_loss'].avg:.3f}  G {meters['g_loss'].avg:.3f}  "
                f"D(real) {meters['d_real'].avg:.3f}  D(fake) {meters['d_fake'].avg:.3f}  "
                f"FID {fid:8.3f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("d_loss", "g_loss", "fid"))
        generator.eval()
        with torch.no_grad():
            plot_image_grid(
                generator(fixed_latent).cpu(), out_path / "samples_final.png", columns=8,
                title=f"samples after {epochs} epochs (fixed latents)",
            )

    summary = {
        **config,
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
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
        print("the loss curves are not a quality signal; the FID column is.")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DCGAN.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--loss", default="bce", choices=LOSSES)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="target 1-eps for real images; stops the discriminator being certain")
    parser.add_argument("--d-steps", type=int, default=1, help="discriminator steps per generator step")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1; 0.5 is the DCGAN fix")
    parser.add_argument("--train-subset", type=int, default=20000, help="cap images (0 = all)")
    parser.add_argument("--fid-track-samples", type=int, default=1000)
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
        width=args.width,
        loss=args.loss,
        label_smoothing=args.label_smoothing,
        d_steps=args.d_steps,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=32 if args.smoke_test else args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        train_subset=args.train_subset or None,
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
