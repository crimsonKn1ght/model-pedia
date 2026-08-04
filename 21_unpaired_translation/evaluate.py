"""Score a trained CycleGAN on the held-out test split of both domains.

    python evaluate.py --checkpoint outputs/shapes_cycle10/best.pt

Unpaired translation has no ground truth, so there is no single number to report. What
this prints instead is two numbers that fail in opposite directions, each next to the
baseline that makes it readable:

**Did the output reach the target domain?** FID, KID and precision/recall of ``G(A)``
against real domain-B test images. The baseline is the **identity mapping** - emit the
input unchanged. Its FID is the distance between the two domains themselves, so it is the
score to beat: anything above it means the translation moved the images the wrong way.

**Did the content survive?** Cycle reconstruction ``F(G(a))`` against ``a``, as L1, PSNR
and SSIM. Here the identity mapping is *perfect* - it reconstructs exactly - so this
number cannot be read on its own. Its job is to catch the opposite failure: a model that
wins on FID by ignoring its input entirely.

A model has to do well on both at once, and reporting either alone hides a way of cheating.
The table prints them together for exactly that reason.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import (
    domain_classifier_loader,
    feature_net_cache,
    get_domains,
    single_loader,
)
from model import ResnetGenerator
from train import cycle_quality, translate_features
from utils import (
    features_from_loader,
    fid_score,
    generative_metrics,
    get_device,
    load_or_train_feature_net,
    plot_bars,
    plot_image_rows,
    save_json,
    set_seed,
)


class Identity(torch.nn.Module):
    """The do-nothing translator, used as the baseline both metrics are read against."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def run_evaluation(
    checkpoint: str,
    batch_size: int = 32,
    test_size: int = 500,
    fid_samples: int = 500,
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
    state = torch.load(checkpoint, map_location=dev, weights_only=False)
    config = state["config"]
    out_path = Path(out_dir or Path(checkpoint).parent)
    out_path.mkdir(parents=True, exist_ok=True)

    size, channels = config["image_size"], config["channels"]
    name_a, name_b = config["name_a"], config["name_b"]

    g_ab = ResnetGenerator(channels, config["base_channels"], config["res_blocks"]).to(dev)
    g_ba = ResnetGenerator(channels, config["base_channels"], config["res_blocks"]).to(dev)
    g_ab.load_state_dict(state["g_ab_state"])
    g_ba.load_state_dict(state["g_ba_state"])
    g_ab.eval()
    g_ba.eval()

    train_a, train_b, _, _, test_a, test_b, info = get_domains(
        config["task"], config["dataset"], config["domain_a"], config["domain_b"],
        root=data_root, image_size=size, test_size=test_size, seed=seed,
        synthetic=synthetic,
    )
    loader_a = single_loader(test_a, batch_size, num_workers)
    loader_b = single_loader(test_b, batch_size, num_workers)

    feature_loader = domain_classifier_loader(train_a, train_b, 64, num_workers, seed)
    cache = None if synthetic else feature_net_cache(
        data_root, config["task"], config["dataset"], size
    )
    feature_net, feature_kind = load_or_train_feature_net(
        cache, feature_loader, channels, 2, dev, epochs=1 if synthetic else 2, verbose=False
    )

    limit = min(fid_samples, len(test_a), len(test_b))
    real_a = features_from_loader(feature_net, loader_a, dev, limit)
    real_b = features_from_loader(feature_net, loader_b, dev, limit)

    identity = Identity()
    results: dict = {
        "checkpoint": str(checkpoint),
        "task": config["task"],
        "domains": f"{name_a} <-> {name_b}",
        "lambda_cycle": config["lambda_cycle"],
        "lambda_identity": config["lambda_identity"],
        "feature_net": feature_kind,
        "test_images_per_domain": limit,
        "epoch": state.get("epoch"),
    }

    # -- did the translations land in the target domain? --------------------- #
    for label, generator, back, loader, real_target, real_source in (
        (f"{name_a}_to_{name_b}", g_ab, g_ba, loader_a, real_b, real_a),
        (f"{name_b}_to_{name_a}", g_ba, g_ab, loader_b, real_a, real_b),
    ):
        fake = translate_features(generator, loader, feature_net, dev, limit)
        metrics = generative_metrics(real_target, fake)
        results[f"fid_{label}"] = metrics["fid"]
        results[f"kid_{label}"] = metrics["kid"]
        results[f"precision_{label}"] = metrics["precision"]
        results[f"recall_{label}"] = metrics["recall"]
        # The identity baseline for this direction: how far apart the domains are. This is
        # symmetric, so both directions report the same number - which is the point, it is
        # a property of the data rather than of the model.
        results[f"fid_identity_{label}"] = fid_score(real_target, real_source)

        cycle = cycle_quality(generator, back, loader, dev, limit)
        results[f"cycle_l1_{label}"] = cycle["l1"]
        results[f"cycle_psnr_{label}"] = cycle["psnr"]
        results[f"cycle_ssim_{label}"] = cycle["ssim"]
        # The identity mapping cycles perfectly, which is the point.
        baseline_cycle = cycle_quality(identity, identity, loader, dev, limit)
        results[f"cycle_ssim_identity_{label}"] = baseline_cycle["ssim"]

    # -- figures ------------------------------------------------------------- #
    batch_a = next(iter(single_loader(test_a, 8, 0)))
    batch_b = next(iter(single_loader(test_b, 8, 0)))
    batch_a = (batch_a[0] if isinstance(batch_a, (tuple, list)) else batch_a).to(dev)
    batch_b = (batch_b[0] if isinstance(batch_b, (tuple, list)) else batch_b).to(dev)
    with torch.no_grad():
        plot_image_rows(
            [(f"real {name_a}", batch_a.cpu()),
             (f"{name_a} -> {name_b}", g_ab(batch_a).cpu()),
             (f"cycled to {name_a}", g_ba(g_ab(batch_a)).cpu()),
             (f"real {name_b}", batch_b.cpu()),
             (f"{name_b} -> {name_a}", g_ba(batch_b).cpu()),
             (f"cycled to {name_b}", g_ab(g_ba(batch_b)).cpu())],
            out_path / "translations.png",
            title=f"test split: {name_a} <-> {name_b}, learned without a single pair",
        )

    forward, backward = f"{name_a}_to_{name_b}", f"{name_b}_to_{name_a}"
    plot_bars(
        [f"{name_a}->{name_b}", f"{name_b}->{name_a}"],
        {
            "CycleGAN": [results[f"fid_{forward}"], results[f"fid_{backward}"]],
            "identity baseline": [results[f"fid_identity_{forward}"],
                                  results[f"fid_identity_{backward}"]],
        },
        out_path / "fid_vs_baseline.png",
        ylabel="FID (lower is better)",
        title="did the translation move the images towards the target domain",
        # Two orders of magnitude between the model and the baseline, so a linear axis
        # would draw the model's bar as a flat line.
        log=True,
    )
    save_json(results, out_path / "metrics.json")

    if verbose:
        print(f"\ntest split, {limit} images per domain, {name_a} <-> {name_b}")
        print(f"feature network: {feature_kind} domain classifier - these FID values are "
              f"comparable within this repository only\n")
        header = f"{'direction':>22s} {'FID':>9s} {'identity':>9s} {'KID':>9s} " \
                 f"{'prec':>7s} {'recall':>7s} {'cycle SSIM':>11s}"
        print(header)
        print("-" * len(header))
        for label in (forward, backward):
            print(f"{label.replace('_', ' '):>22s} "
                  f"{results[f'fid_{label}']:9.3f} "
                  f"{results[f'fid_identity_{label}']:9.3f} "
                  f"{results[f'kid_{label}']:+9.4f} "
                  f"{results[f'precision_{label}']:7.3f} "
                  f"{results[f'recall_{label}']:7.3f} "
                  f"{results[f'cycle_ssim_{label}']:11.4f}")
        print("\nFID below the identity column means the translation moved the images "
              "towards the target domain.")
        print("Cycle SSIM is 1.000 for a model that translates nothing, so it is a "
              "guard rather than a score.")
        print(f"\nfigures and metrics -> {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN on the test split.")
    parser.add_argument("--checkpoint", default="outputs/shapes_cycle10/best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--fid-samples", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="generated domains")
    args = parser.parse_args()

    run_evaluation(
        checkpoint=args.checkpoint,
        batch_size=8 if args.smoke_test else args.batch_size,
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
