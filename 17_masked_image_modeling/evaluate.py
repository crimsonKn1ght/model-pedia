"""Evaluate a pretrained MAE: reconstruction, then what the encoder is worth.

    python evaluate.py --checkpoint outputs/cifar10_mae/best.pt
    python evaluate.py --checkpoint ... --tasks recon      # skip the classifier runs

Three measurements, in increasing order of what they actually tell you.

1. **Reconstruction** across mask ratios, in masked-patch PSNR. Interesting, and
   almost the least informative of the three: a model can reconstruct pleasantly
   by learning local smoothness and still carry nothing useful about content.
2. **Linear probe** on the frozen class token. Cheap, and the standard read-out.
   MAE is known to probe *worse* than contrastive methods - its features are not
   arranged to be linearly separable - so a modest number here is expected.
3. **Fine-tuning** the whole encoder. This is what MAE is for, and where the
   pretrained weights should show a clear advantage.

Every classification number is paired with the identical ViT trained from random
initialisation under the identical budget. Without that control, "the fine-tuned
MAE reaches X%" is not a claim about pretraining at all - a small ViT trained from
scratch on 10 000 images already reaches something, and the pretraining is only
responsible for the difference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import DATASETS, eval_transform, get_splits, make_loader, train_transform
from model import build_classifier, build_mae, patchify
from train import VAL_MASK_SEED, _fork_rng
from utils import (
    AverageMeter,
    classification_accuracy,
    count_parameters,
    extract_cls_features,
    fit_linear_head,
    get_device,
    masked_psnr,
    plot_bars,
    plot_image_rows,
    plot_ratio_sweep,
    save_json,
    set_seed,
)

TASKS = ("recon", "probe", "finetune")


@torch.no_grad()
def reconstruction_at_ratio(model, loader, device, mask_ratio: float) -> dict:
    """Masked-patch loss and PSNR at one test-time mask ratio."""
    model.eval()
    loss_meter, psnr_meter = AverageMeter(), AverageMeter()
    with _fork_rng(device):
        torch.manual_seed(VAL_MASK_SEED)
        for images, _ in loader:
            images = images.to(device)
            loss, prediction, mask = model(images, mask_ratio)
            _, reconstruction = model.reconstruct(images, prediction, mask)
            patch = model.encoder.patch_size
            loss_meter.update(loss.item(), images.size(0))
            psnr_meter.update(
                masked_psnr(patchify(reconstruction, patch), patchify(images, patch), mask),
                images.size(0),
            )
    return {
        "mask_ratio": mask_ratio,
        "loss": loss_meter.avg,
        "masked_psnr_db": psnr_meter.avg,
    }


@torch.no_grad()
def save_reconstruction_grid(model, loader, device, ratios: list[float], path, n: int = 8) -> None:
    """One row of originals, then masked/reconstructed pairs per mask ratio."""
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)

    rows = [("original", images.cpu())]
    for ratio in ratios:
        with _fork_rng(device):
            torch.manual_seed(VAL_MASK_SEED)
            _, prediction, mask = model(images, ratio)
        masked, reconstruction = model.reconstruct(images, prediction, mask)
        rows.append((f"masked {ratio:g}", masked.cpu()))
        rows.append((f"recon {ratio:g}", reconstruction.cpu()))

    plot_image_rows(rows, path, title="reconstruction as more of the image is hidden")


def finetune(
    encoder_state: dict | None,
    config: dict,
    train_loader,
    test_loader,
    device,
    epochs: int,
    lr: float,
    seed: int,
    verbose: bool,
    label: str,
) -> dict:
    """Train a classifier end to end, optionally from pretrained encoder weights."""
    set_seed(seed)  # both arms start from the identical random head and see the same batches
    model = build_classifier(
        config["num_classes"],
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        dim=config["dim"],
        depth=config["depth"],
        heads=config["heads"],
    ).to(device)
    if encoder_state is not None:
        model.load_encoder(encoder_state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        batches = tqdm(
            train_loader, desc=f"{label} epoch {epoch}/{epochs}", leave=False, disable=not verbose
        )
        for images, targets in batches:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.size(0))
            batches.set_postfix(loss=f"{loss_meter.avg:.4f}")
        scheduler.step()
        history.append(loss_meter.avg)

    return {
        "test_accuracy": classification_accuracy(model, test_loader, device),
        "train_loss": history,
        "parameters": count_parameters(model),
    }


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    tasks = tuple(args.tasks)

    train_base, _, test_base, info = get_splits(
        ckpt["dataset"],
        root=args.data_root,
        train_subset=args.train_subset or None,
        seed=args.seed,
        synthetic=args.smoke_test,
    )
    size, patch = ckpt["image_size"], ckpt["patch_size"]
    config = {key: ckpt[key] for key in
              ("num_classes", "in_channels", "image_size", "patch_size", "dim", "depth", "heads")}

    eval_pipeline = eval_transform(info, size)
    train_eval_loader = make_loader(
        train_base, eval_pipeline, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = make_loader(
        test_base, eval_pipeline, batch_size=args.batch_size, num_workers=args.num_workers
    )
    train_aug_loader = make_loader(
        train_base,
        train_transform(info, size),
        batch_size=args.finetune_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.checkpoint).parent
    result = {
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "pretrain_epoch": ckpt["epoch"],
        "pretrain_mask_ratio": ckpt["mask_ratio"],
        "image": f"{ckpt['in_channels']}x{size}x{size}, patch {patch}",
        "test_images": len(test_base),
        "train_images": len(train_base),
    }

    if "recon" in tasks:
        model = build_mae(
            image_size=size,
            patch_size=patch,
            in_channels=ckpt["in_channels"],
            dim=ckpt["dim"],
            depth=ckpt["depth"],
            heads=ckpt["heads"],
            decoder_dim=ckpt["decoder_dim"],
            decoder_depth=ckpt["decoder_depth"],
            norm_pixel_loss=ckpt["norm_pixel_loss"],
        ).to(dev)
        model.load_state_dict(ckpt["model_state"])

        rows = [reconstruction_at_ratio(model, test_loader, dev, r) for r in args.mask_ratios]
        plot_ratio_sweep(rows, ckpt["mask_ratio"], out_dir / "mask_ratio_sweep.png")
        grid_ratios = [r for r in args.mask_ratios if r in (0.5, 0.75, 0.9)] or args.mask_ratios[:2]
        save_reconstruction_grid(model, test_loader, dev, grid_ratios, out_dir / "examples.png")
        result["reconstruction"] = rows

    if "probe" in tasks or "finetune" in tasks:
        def fresh_encoder():
            return build_classifier(**config).encoder.to(dev)

        pretrained_encoder = fresh_encoder()
        pretrained_encoder.load_state_dict(ckpt["encoder_state"])
        control_encoder = fresh_encoder()

        result["classification"] = {}
        if "probe" in tasks:
            probes = {}
            for name, encoder in (("pretrained", pretrained_encoder), ("random init", control_encoder)):
                train_features, train_labels = extract_cls_features(encoder, train_eval_loader, dev)
                test_features, test_labels = extract_cls_features(encoder, test_loader, dev)
                probes[name] = fit_linear_head(
                    train_features,
                    train_labels,
                    test_features,
                    test_labels,
                    config["num_classes"],
                    epochs=args.probe_epochs,
                    device=dev,
                    seed=args.seed,
                )
            result["classification"]["linear_probe"] = probes

        if "finetune" in tasks:
            result["classification"]["finetune"] = {
                name: finetune(
                    state,
                    config,
                    train_aug_loader,
                    test_loader,
                    dev,
                    args.finetune_epochs,
                    args.finetune_lr,
                    args.seed,
                    verbose,
                    name,
                )
                for name, state in (
                    ("pretrained", ckpt["encoder_state"]),
                    ("random init", None),
                )
            }

        labels, series = [], {"pretrained": [], "random init": []}
        for key, pretty in (("linear_probe", "linear probe"), ("finetune", "fine-tune")):
            if key in result["classification"]:
                labels.append(pretty)
                for name in series:
                    series[name].append(result["classification"][key][name]["test_accuracy"])
        if labels:
            plot_bars(
                labels,
                series,
                out_dir / "downstream_accuracy.png",
                ylabel="test accuracy",
                title=f"MAE pretraining vs the same ViT from scratch ({ckpt['dataset']})",
            )

    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"checkpoint : {args.checkpoint}")
        print(f"pretrained : epoch {ckpt['epoch']} at mask ratio {ckpt['mask_ratio']:g}")
        print(f"dataset    : {ckpt['dataset']}  {len(test_base)} test images\n")

        if "reconstruction" in result:
            header = f"{'mask ratio':>11s} {'objective':>10s} {'masked PSNR':>12s}"
            print(header)
            print("-" * len(header))
            for row in result["reconstruction"]:
                marker = " <- pretrained at" if row["mask_ratio"] == ckpt["mask_ratio"] else ""
                print(
                    f"{row['mask_ratio']:11.2f} {row['loss']:10.4f} "
                    f"{row['masked_psnr_db']:12.2f}{marker}"
                )
            print()

        if "classification" in result:
            header = f"{'read-out':>13s} {'pretrained':>12s} {'random init':>12s} {'gain':>8s}"
            print(header)
            print("-" * len(header))
            for key, pretty in (("linear_probe", "linear probe"), ("finetune", "fine-tune")):
                if key not in result["classification"]:
                    continue
                arms = result["classification"][key]
                a = arms["pretrained"]["test_accuracy"]
                b = arms["random init"]["test_accuracy"]
                print(f"{pretty:>13s} {a:12.4f} {b:12.4f} {a - b:+8.4f}")
            print("\nboth columns use the identical architecture and training budget.")

        print(f"\nmetrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a masked autoencoder.")
    parser.add_argument("--checkpoint", default="outputs/cifar10_mae/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        choices=TASKS,
        help="which measurements to run; 'finetune' is the slow one",
    )
    parser.add_argument(
        "--mask-ratios",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 0.9],
        help="test-time mask ratios for the reconstruction sweep",
    )
    parser.add_argument("--train-subset", type=int, default=10000, help="images for the probes")
    parser.add_argument("--probe-epochs", type=int, default=40, help="epochs on cached features")
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--finetune-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on random tensors")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_workers = 0
        args.probe_epochs = min(args.probe_epochs, 5)
        args.finetune_epochs = 1
    run_evaluation(args)


if __name__ == "__main__":
    main()
