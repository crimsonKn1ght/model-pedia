"""Evaluate a RealNVP flow on the held-out split.

    python evaluate.py

The headline number is **test bits per dimension** -- an exact likelihood, not a
bound, and the one metric in this repository that is directly comparable to
published values. Also reported:

* a temperature sweep, since flows sample from a scaled prior and the
  quality/diversity trade-off is very visible;
* an invertibility check: encode test images, decode them again, and measure
  the round-trip error. It should be at the level of floating-point noise. If it
  is not, the flow is not a flow.
* FID/KID, for comparison with the other projects here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import metrics as metrics_mod
from common import utils, viz
from model import RealNVP

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--checkpoint", default=str(HERE / "outputs" / "realnvp.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-eval-samples", type=int, default=1024)
    parser.add_argument("--feature-extractor", default="small-cnn", choices=["small-cnn", "inception"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)

    ckpt = utils.load_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    out_dir = utils.resolve_out_dir(args, str(Path(args.checkpoint).parent))
    eval_dir = utils.ensure_dir(out_dir / "evaluation")

    model = RealNVP(
        channels=cfg["channels"],
        image_size=cfg["image_size"],
        hidden=cfg["hidden"],
        num_scales=cfg["num_scales"],
        couplings_per_scale=cfg["couplings_per_scale"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_loader = data_mod.get_dataloader(
        cfg["dataset"],
        root=args.data_root,
        image_size=cfg["image_size"],
        train=False,
        batch_size=args.batch_size,
        normalize="unit",
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    # -- 1. exact test likelihood ------------------------------------------- #
    bpd = utils.AverageMeter()
    with torch.no_grad():
        for x, _ in utils.progress(utils.maybe_limit(test_loader, args), desc="test bits/dim"):
            x = x.to(device)
            bpd.update(model.bits_per_dim(x).mean().item(), x.shape[0])
    results = {"test_bits_per_dim": bpd.avg}

    # -- 2. invertibility check --------------------------------------------- #
    with torch.no_grad():
        x = next(iter(test_loader))[0][:16].to(device)
        h, _ = model.preprocess(x)
        z = h
        for layers in model.scales:
            for i, layer in enumerate(layers):
                if i == model.couplings_per_scale:
                    from model import squeeze

                    z = squeeze(z)
                z, _ = layer(z)
        recovered = z
        from model import unsqueeze

        for layers in reversed(model.scales):
            for i, layer in enumerate(reversed(list(layers))):
                recovered = layer.inverse(recovered)
                if i == model.couplings_per_scale - 1:
                    recovered = unsqueeze(recovered)
        results["max_roundtrip_error"] = float((recovered - h).abs().max())

    # -- 3. temperature sweep ------------------------------------------------ #
    temperatures = [0.5, 0.7, 0.85, 1.0]
    rows = {}
    for t in temperatures:
        rows[f"T={t}"] = model.sample(8, device, temperature=t)
    viz.save_comparison_grid(rows, eval_dir / "temperature-sweep.png", normalize="unit")
    viz.save_image_grid(
        model.sample(64, device, temperature=0.8), eval_dir / "samples.png", nrow=8, normalize="unit"
    )

    # -- 4. FID/KID, for comparison with the other projects ------------------ #
    n_samples = 256 if args.quick else args.num_eval_samples
    extractor = metrics_mod.get_feature_extractor(
        cfg["dataset"],
        root=args.data_root,
        image_size=cfg["image_size"],
        device=device,
        kind=args.feature_extractor,
        num_workers=args.num_workers,
    )
    real_features = metrics_mod.features_from_loader(
        extractor, test_loader, device, max_samples=n_samples, normalize="unit"
    )
    fake_features = metrics_mod.features_from_sampler(
        extractor,
        lambda n: model.sample(n, device, temperature=0.8),
        device,
        num_samples=n_samples,
        normalize="unit",
    )
    results.update(metrics_mod.generative_metrics(real_features, fake_features))

    utils.save_json(results, eval_dir / "metrics.json")
    print("\nRealNVP evaluation")
    print(metrics_mod.format_metrics(results))
    print(f"\nfigures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
