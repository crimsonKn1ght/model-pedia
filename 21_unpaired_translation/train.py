"""Train CycleGAN between two unpaired collections.

    python train.py --task shapes                             # generated, no download
    python train.py --task classes --dataset fashion-mnist --domain-a 5 --domain-b 7
    python train.py --lambda-cycle 0        # the ablation: watch the translations drift

**What selects the checkpoint, and why it is not the obvious thing.** There is no target
image, so there is no reconstruction error to minimise. The two things that *can* be
measured are:

* **FID of the translations** against real images of the target domain - did the output
  land in domain B;
* **cycle-reconstruction error** ``F(G(a))`` against ``a`` - was the content preserved.

Neither works alone. A model that ignores its input and emits one convincing domain-B
image scores well on the first. A model that learns the identity mapping and translates
nothing at all scores *perfectly* on the second. So the checkpoint is selected on mean FID
across both directions, and the cycle error is recorded next to it as the thing that
detects the failure FID cannot see. ``evaluate.py`` puts both against the
identity-mapping baseline, which is what makes the pair of numbers readable.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch
from tqdm import tqdm

from data import (
    DATASETS,
    TASKS,
    domain_classifier_loader,
    feature_net_cache,
    get_domains,
    make_loader,
    single_loader,
)
from model import build_models, discriminator_loss, generator_losses, ImagePool
from utils import (
    AverageMeter,
    count_parameters,
    features_from_loader,
    fid_score,
    get_device,
    load_or_train_feature_net,
    plot_curves,
    plot_image_rows,
    psnr,
    save_json,
    set_seed,
    ssim,
)


@torch.no_grad()
def translate_features(generator, loader, feature_net, device, limit: int | None = None):
    """Features of ``generator(x)`` for real images ``x`` from one domain."""
    generator.eval()
    feature_net.eval()
    collected, seen = [], 0
    for batch in loader:
        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        collected.append(feature_net.features(generator(images)).cpu())
        seen += images.size(0)
        if limit is not None and seen >= limit:
            break
    features = torch.cat(collected)
    return features[:limit] if limit is not None else features


@torch.no_grad()
def cycle_quality(forward, backward, loader, device, limit: int | None = None) -> dict:
    """How well ``backward(forward(x))`` recovers ``x``. Needs no target image."""
    forward.eval()
    backward.eval()
    meters = {key: AverageMeter() for key in ("l1", "psnr", "ssim")}
    seen = 0
    for batch in loader:
        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        recovered = backward(forward(images))
        n = images.size(0)
        meters["l1"].update((recovered - images).abs().mean().item(), n)
        meters["psnr"].update(psnr(recovered, images).mean().item(), n)
        meters["ssim"].update(ssim(recovered, images).mean().item(), n)
        seen += n
        if limit is not None and seen >= limit:
            break
    return {key: meter.avg for key, meter in meters.items()}


def run_training(
    task: str = "shapes",
    dataset: str = "fashion-mnist",
    domain_a: str = "5",
    domain_b: str = "7",
    image_size: int | None = None,
    base_channels: int = 24,
    res_blocks: int = 3,
    patch_layers: int = 3,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 0.5,
    pool_size: int = 50,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-4,
    beta1: float = 0.5,
    train_size: int = 2000,
    val_size: int = 200,
    test_size: int = 500,
    fid_samples: int = 200,
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
    out_path = Path(out_dir or f"outputs/{task}_cycle{lambda_cycle:g}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_a, train_b, val_a, val_b, _, _, info = get_domains(
        task, dataset, domain_a, domain_b, root=data_root, image_size=image_size,
        train_size=train_size, val_size=val_size, test_size=test_size, seed=seed,
        synthetic=synthetic,
    )
    size, channels = info["size"], info["channels"]
    name_a, name_b = info["name_a"], info["name_b"]

    train_loader = make_loader(train_a, train_b, batch_size, shuffle=True,
                              num_workers=num_workers, seed=seed)
    val_loader_a = single_loader(val_a, batch_size, num_workers)
    val_loader_b = single_loader(val_b, batch_size, num_workers)

    g_ab, g_ba, d_a, d_b = build_models(channels, base_channels, res_blocks, patch_layers)
    g_ab, g_ba, d_a, d_b = g_ab.to(dev), g_ba.to(dev), d_a.to(dev), d_b.to(dev)

    opt_g = torch.optim.Adam(
        itertools.chain(g_ab.parameters(), g_ba.parameters()), lr=lr, betas=(beta1, 0.999)
    )
    opt_d = torch.optim.Adam(
        itertools.chain(d_a.parameters(), d_b.parameters()), lr=lr, betas=(beta1, 0.999)
    )
    pool_a, pool_b = ImagePool(pool_size), ImagePool(pool_size)

    # The FID ruler: a two-class domain classifier, trained once and cached.
    feature_loader = domain_classifier_loader(train_a, train_b, 64, num_workers, seed)
    cache = None if synthetic else feature_net_cache(data_root, task, dataset, size)
    feature_net, feature_kind = load_or_train_feature_net(
        cache, feature_loader, channels, 2, dev, epochs=1 if synthetic else 2,
        verbose=verbose,
    )

    config = {
        "task": task, "dataset": dataset, "domain_a": domain_a, "domain_b": domain_b,
        "name_a": name_a, "name_b": name_b, "channels": channels, "image_size": size,
        "base_channels": base_channels, "res_blocks": res_blocks,
        "patch_layers": patch_layers, "lambda_cycle": lambda_cycle,
        "lambda_identity": lambda_identity, "pool_size": pool_size,
        "feature_net": feature_kind,
    }

    if verbose:
        print(f"device        : {dev}")
        print(f"task          : {task}  ({name_a} <-> {name_b}), unpaired")
        print(f"generators    : 2 x {count_parameters(g_ab):,} parameters")
        print(f"discriminators: 2 x {count_parameters(d_a):,} parameters "
              f"({patch_layers}-layer PatchGAN)")
        print(f"objective     : adversarial + {lambda_cycle:g} x cycle"
              f"{f' + {lambda_cycle * lambda_identity:g} x identity' if lambda_identity else ''}")
        print(f"data          : {len(train_a)} + {len(train_b)} train images at "
              f"{channels}x{size}x{size}")
        print(f"feature net   : {feature_kind} (domain classifier)\n")

    # The train_/val_ prefixes are what plot_curves() keys off; without them it finds
    # nothing to draw and writes no figure at all.
    history = {"train_adversarial": [], "train_cycle": [], "train_identity": [],
               "train_discriminator": [], "val_fid_ab": [], "val_fid_ba": [],
               "val_fid_mean": [], "val_cycle_ssim_a": [], "val_cycle_ssim_b": []}
    best_fid, best_epoch = float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    # Real validation features are fixed for the whole run, so the FID column is
    # comparable epoch to epoch.
    real_features_a = features_from_loader(feature_net, val_loader_a, dev, fid_samples)
    real_features_b = features_from_loader(feature_net, val_loader_b, dev, fid_samples)

    fixed_a = next(iter(single_loader(val_a, 8, 0)))
    fixed_b = next(iter(single_loader(val_b, 8, 0)))
    fixed_a = (fixed_a[0] if isinstance(fixed_a, (tuple, list)) else fixed_a).to(dev)
    fixed_b = (fixed_b[0] if isinstance(fixed_b, (tuple, list)) else fixed_b).to(dev)

    for epoch in range(1, epochs + 1):
        # A fresh arbitrary pairing every epoch: no A ever sees the same B twice.
        train_loader.dataset.reshuffle(epoch)
        for net in (g_ab, g_ba, d_a, d_b):
            net.train()
        meters = {key: AverageMeter() for key in
                  ("adversarial", "cycle", "identity", "discriminator")}

        iterator = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False,
                        disable=not verbose)
        for real_a, real_b in iterator:
            real_a, real_b = real_a.to(dev), real_b.to(dev)
            n = real_a.size(0)

            total_g, parts, fake_a, fake_b = generator_losses(
                g_ab, g_ba, d_a, d_b, real_a, real_b, lambda_cycle, lambda_identity
            )
            opt_g.zero_grad(set_to_none=True)
            total_g.backward()
            opt_g.step()

            # The discriminators see a mix of current and past fakes.
            loss_d = (
                discriminator_loss(d_a, real_a, pool_a.query(fake_a))
                + discriminator_loss(d_b, real_b, pool_b.query(fake_b))
            )
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            for key in ("adversarial", "cycle", "identity"):
                meters[key].update(parts[key], n)
            meters["discriminator"].update(float(loss_d.detach()), n)

        # -- validation: where did the translations land, and did content survive -- #
        fake_features_b = translate_features(g_ab, val_loader_a, feature_net, dev, fid_samples)
        fake_features_a = translate_features(g_ba, val_loader_b, feature_net, dev, fid_samples)
        fid_ab = fid_score(real_features_b, fake_features_b)
        fid_ba = fid_score(real_features_a, fake_features_a)
        fid_mean = 0.5 * (fid_ab + fid_ba)
        cycle_a = cycle_quality(g_ab, g_ba, val_loader_a, dev, fid_samples)
        cycle_b = cycle_quality(g_ba, g_ab, val_loader_b, dev, fid_samples)

        for key in ("adversarial", "cycle", "identity", "discriminator"):
            history[f"train_{key}"].append(meters[key].avg)
        history["val_fid_ab"].append(fid_ab)
        history["val_fid_ba"].append(fid_ba)
        history["val_fid_mean"].append(fid_mean)
        history["val_cycle_ssim_a"].append(cycle_a["ssim"])
        history["val_cycle_ssim_b"].append(cycle_b["ssim"])

        if fid_mean <= best_fid:
            best_fid, best_epoch = fid_mean, epoch
            torch.save({"g_ab_state": g_ab.state_dict(), "g_ba_state": g_ba.state_dict(),
                        "d_a_state": d_a.state_dict(), "d_b_state": d_b.state_dict(),
                        "config": config, "epoch": epoch, "val_fid_mean": fid_mean},
                       ckpt_path)

        if verbose:
            print(f"epoch {epoch:>3}  adv {meters['adversarial'].avg:.4f}  "
                  f"cycle {meters['cycle'].avg:.4f}  D {meters['discriminator'].avg:.4f}  "
                  f"| val FID {fid_mean:7.3f}  cycle SSIM {cycle_a['ssim']:.4f}"
                  f"{'  <- best' if epoch == best_epoch else ''}")

        with torch.no_grad():
            g_ab.eval()
            g_ba.eval()
            plot_image_rows(
                [(f"real {name_a}", fixed_a.cpu()),
                 (f"{name_a} -> {name_b}", g_ab(fixed_a).cpu()),
                 (f"cycled back", g_ba(g_ab(fixed_a)).cpu()),
                 (f"real {name_b}", fixed_b.cpu()),
                 (f"{name_b} -> {name_a}", g_ba(fixed_b).cpu())],
                out_path / "translations_val.png",
                title=f"epoch {epoch}: unpaired translation both ways",
            )

    elapsed = time.time() - started
    plot_curves(history, out_path / "curves.png",
                keys=("adversarial", "cycle", "discriminator", "fid_mean"))
    save_json(history, out_path / "history.json")

    result = {
        "checkpoint": str(ckpt_path), "out_dir": str(out_path), "config": config,
        "best_epoch": best_epoch, "best_val_fid_mean": best_fid,
        "final_val_cycle_ssim_a": (
            history["val_cycle_ssim_a"][-1] if history["val_cycle_ssim_a"] else None
        ),
        "seconds": elapsed, "epochs": epochs,
    }
    save_json(result, out_path / "train_summary.json")
    if verbose:
        print(f"\nbest mean val FID {best_fid:.3f} at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CycleGAN on two unpaired domains.")
    parser.add_argument("--task", default="shapes", choices=TASKS)
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--domain-a", default="5", help="class index, or dataset name")
    parser.add_argument("--domain-b", default="7")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--res-blocks", type=int, default=3)
    parser.add_argument("--patch-layers", type=int, default=3)
    parser.add_argument("--lambda-cycle", type=float, default=10.0,
                        help="0 removes cycle consistency entirely")
    parser.add_argument("--lambda-identity", type=float, default=0.5)
    parser.add_argument("--pool-size", type=int, default=50, help="0 disables the buffer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--fid-samples", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="1 epoch on generated domains, no download")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        task=args.task,
        dataset=args.dataset,
        domain_a=args.domain_a,
        domain_b=args.domain_b,
        image_size=args.image_size,
        base_channels=8 if args.smoke_test else args.base_channels,
        res_blocks=1 if args.smoke_test else args.res_blocks,
        patch_layers=2 if args.smoke_test else args.patch_layers,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        pool_size=args.pool_size,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=8 if args.smoke_test else args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        fid_samples=16 if args.smoke_test else args.fid_samples,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out_dir,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
