"""Train a conditional GAN.

    python train.py --dataset fashion-mnist --mode acgan --epochs 20
    python train.py --mode cgan

Two numbers are tracked every epoch, and conditioning is what makes the second one
possible:

* **FID**, as in project 08, because the losses are not a quality signal.
* **class accuracy** - generate images for known labels, ask an independent classifier
  what it sees, and count agreement. This is a direct measure of whether the
  conditioning works at all, and it is available only because the model was told what
  to draw.

The classifier is the same small network the FID features come from, trained on real
data before the GAN starts and never updated afterwards. Using the discriminator's own
auxiliary head instead would be circular - it is part of what is being trained.

Checkpoints are selected on FID, not on accuracy: a generator can reach high class
accuracy by producing one over-typical example per class, which is exactly the
diversity failure ``evaluate.py`` looks for.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import DATASETS, feature_net_cache, get_splits, make_loader
from model import MODES, build_models, discriminator_loss, generator_loss
from utils import (
    AverageMeter,
    count_parameters,
    features_from_loader,
    fid_score,
    get_device,
    load_or_train_feature_net,
    plot_curves,
    plot_image_grid,
    save_json,
    set_seed,
)


@torch.no_grad()
def sample_features_and_accuracy(generator, feature_net, device, total: int, batch_size: int):
    """Features for FID plus the fraction of samples an independent classifier agrees with."""
    generator.eval()
    features, correct, seen = [], 0, 0
    while seen < total:
        n = min(batch_size, total - seen)
        images, labels = generator.sample(n, device)
        logits = feature_net(images)
        features.append(feature_net.features(images).cpu())
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += n
    return torch.cat(features), correct / max(seen, 1)


def run_training(
    dataset: str = "fashion-mnist",
    image_size: int | None = None,
    mode: str = "acgan",
    latent_dim: int = 64,
    width: int = 64,
    aux_weight: float = 1.0,
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
    out_path = Path(out_dir or f"outputs/{dataset}_{mode}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, info = get_splits(
        dataset, root=data_root, image_size=image_size, train_subset=train_subset,
        augment=True, seed=seed, synthetic=synthetic,
    )
    if info["classes"] is None:
        raise ValueError(
            f"{dataset} has no labels, so it cannot train a conditional GAN. "
            "Use mnist, fashion-mnist or cifar10."
        )
    train_loader = make_loader(
        train_set, batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = make_loader(val_set, 256, num_workers=num_workers)

    generator, discriminator = build_models(
        latent_dim, info["classes"], info["channels"], info["size"], width, mode
    )
    generator, discriminator = generator.to(dev), discriminator.to(dev)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

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
        "class_names": info["class_names"],
        "image_size": info["size"],
        "latent_dim": latent_dim,
        "width": width,
        "mode": mode,
        "aux_weight": aux_weight,
    }

    if verbose:
        print(f"device     : {dev}")
        print(f"mode       : {mode} ({'label into the discriminator' if mode == 'cgan' else 'auxiliary classifier head'})")
        print(f"generator  : {count_parameters(generator):,} parameters, latent {latent_dim}")
        print(f"data       : {len(train_set)} train images, {info['classes']} classes at "
              f"{info['channels']}x{info['size']}x{info['size']}")
        print(f"FID + class accuracy every epoch, {feature_kind} features\n")

    history = {
        "train_d_loss": [], "train_g_loss": [], "train_d_real": [], "train_d_fake": [],
        "val_fid": [], "val_class_accuracy": [],
    }
    best_fid, best_epoch = float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        meters = {key: AverageMeter() for key in ("d_loss", "g_loss", "d_real", "d_fake", "aux")}
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)

        for images, labels in batches:
            images, labels = images.to(dev), labels.to(dev)
            n = images.size(0)
            fake_labels = torch.randint(0, info["classes"], (n,), device=dev)

            with torch.no_grad():
                fake = generator(torch.randn(n, latent_dim, device=dev), fake_labels)
            real_out = discriminator(images, labels)
            fake_out = discriminator(fake, fake_labels)
            d_losses = discriminator_loss(real_out, fake_out, labels, fake_labels, aux_weight)
            opt_d.zero_grad(set_to_none=True)
            d_losses["loss"].backward()
            opt_d.step()

            fake_labels = torch.randint(0, info["classes"], (n,), device=dev)
            fake = generator(torch.randn(n, latent_dim, device=dev), fake_labels)
            g_losses = generator_loss(discriminator(fake, fake_labels), fake_labels, aux_weight)
            opt_g.zero_grad(set_to_none=True)
            g_losses["loss"].backward()
            opt_g.step()

            meters["d_loss"].update(d_losses["loss"].item(), n)
            meters["g_loss"].update(g_losses["loss"].item(), n)
            meters["aux"].update(float(g_losses["auxiliary"]), n)
            meters["d_real"].update(real_out[0].detach().sigmoid().mean().item(), n)
            meters["d_fake"].update(fake_out[0].detach().sigmoid().mean().item(), n)
            batches.set_postfix(d=f"{meters['d_loss'].avg:.3f}", g=f"{meters['g_loss'].avg:.3f}")

        fake_features, accuracy = sample_features_and_accuracy(
            generator, feature_net, dev, real_features.size(0), 256
        )
        fid = fid_score(real_features, fake_features)

        history["train_d_loss"].append(meters["d_loss"].avg)
        history["train_g_loss"].append(meters["g_loss"].avg)
        history["train_d_real"].append(meters["d_real"].avg)
        history["train_d_fake"].append(meters["d_fake"].avg)
        history["val_fid"].append(fid)
        history["val_class_accuracy"].append(accuracy)

        if fid <= best_fid:
            best_fid, best_epoch = fid, epoch
            torch.save(
                {
                    "generator_state": generator.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "epoch": epoch, "val_fid": fid, "val_class_accuracy": accuracy, **config,
                },
                ckpt_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  D {meters['d_loss'].avg:.3f}  G {meters['g_loss'].avg:.3f}  "
                f"D(real) {meters['d_real'].avg:.3f}  D(fake) {meters['d_fake'].avg:.3f}  "
                f"FID {fid:8.3f}  class acc {accuracy:.3f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("d_loss", "g_loss", "fid", "class_accuracy"))
        generator.eval()
        labels = torch.arange(info["classes"], device=dev).repeat_interleave(8)[:64]
        with torch.no_grad():
            images = generator(torch.randn(labels.numel(), latent_dim, device=dev), labels)
        plot_image_grid(images.cpu(), out_path / "samples_final.png", columns=8,
                        title="one class per row")

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
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a conditional GAN.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--mode", default="acgan", choices=MODES)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--aux-weight", type=float, default=1.0, help="ACGAN classifier weight")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--train-subset", type=int, default=20000)
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
        mode=args.mode,
        latent_dim=args.latent_dim,
        width=args.width,
        aux_weight=args.aux_weight,
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
