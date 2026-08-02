"""Evaluate Pix2Pix on the held-out split.

    python evaluate.py

Paired data means the ground truth is known, so *fidelity* can be measured
directly -- L1, PSNR and SSIM against the true target -- alongside the
distribution-level FID/KID.  Both matter: a model can score a good FID by
producing realistic images that are the wrong translation of the input, and the
per-pair metrics are what catch that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import bootstrap  # noqa: F401
from common import data as data_mod
from common import metrics as metrics_mod
from common import utils, viz
from data_pairs import build_loader
from model import UNetGenerator

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--checkpoint", default=str(HERE / "outputs" / "pix2pix.pt"))
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

    generator = UNetGenerator(
        cfg["in_channels"], cfg["out_channels"], base=cfg["base_channels"], depth=cfg["depth"]
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    loader, _, _ = build_loader(
        cfg["task"], cfg["dataset"], args.data_root, cfg["image_size"],
        args.batch_size, train=False, num_workers=args.num_workers,
    )

    extractor = metrics_mod.get_feature_extractor(
        cfg["dataset"],
        root=args.data_root,
        image_size=cfg["image_size"],
        device=device,
        kind=args.feature_extractor,
        num_workers=args.num_workers,
    )

    # -- 1. per-pair fidelity ------------------------------------------------ #
    l1_m, psnr_m, ssim_m, lpips_m = (utils.AverageMeter() for _ in range(4))
    real_chunks, fake_chunks, seen = [], [], 0
    limit = 256 if args.quick else args.num_eval_samples

    with torch.no_grad():
        for source, target in utils.progress(loader, desc="evaluating"):
            source, target = source.to(device), target.to(device)
            generated = generator(source)
            n = source.shape[0]

            l1_m.update(F.l1_loss(generated, target).item(), n)
            psnr_m.update(metrics_mod.psnr(generated, target), n)
            ssim_m.update(metrics_mod.ssim(generated, target), n)
            lpips_m.update(metrics_mod.lpips_proxy(extractor, generated, target), n)

            real_chunks.append(extractor.features(data_mod.denormalize(target)).cpu())
            fake_chunks.append(extractor.features(data_mod.denormalize(generated)).cpu())
            seen += n
            if seen >= limit:
                break

    results = {
        "test_l1": l1_m.avg,
        "test_psnr_db": psnr_m.avg,
        "test_ssim": ssim_m.avg,
        "test_lpips_proxy": lpips_m.avg,
        "lambda_l1": cfg["lambda_l1"],
    }
    results.update(
        metrics_mod.generative_metrics(torch.cat(real_chunks), torch.cat(fake_chunks))
    )

    # -- 2. pictures --------------------------------------------------------- #
    source, target = next(iter(loader))
    source, target = source[:8].to(device), target[:8].to(device)
    with torch.no_grad():
        generated = generator(source)
    viz.save_comparison_grid(
        {
            "input": source.repeat(1, 3, 1, 1) if cfg["in_channels"] == 1 else source,
            "generated": generated,
            "target": target,
        },
        eval_dir / "translations.png",
    )

    utils.save_json(results, eval_dir / "metrics.json")
    print("\nPix2Pix evaluation")
    print(metrics_mod.format_metrics(results))
    print(f"\nfigures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
