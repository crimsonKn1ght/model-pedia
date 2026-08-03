"""Evaluate CycleGAN on held-out images from both domains.

    python evaluate.py

There is no ground-truth translation for unpaired data, so the evaluation has
to answer two different questions:

* **Did the output land in the target domain?**  FID/KID of ``G(A)`` against
  real domain-B test images, and the reverse.
* **Was the content preserved?**  Cycle-reconstruction error ``F(G(a)) vs a``,
  reported as L1, PSNR and SSIM.  A model can score a fine FID by discarding the
  input and emitting a generic domain-B image; the cycle error is what exposes
  that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import bootstrap  # noqa: F401
from common import data as data_mod
from common import metrics as metrics_mod
from common import utils, viz
from domains import build_domains
from model import ResnetGenerator

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--checkpoint", default=str(HERE / "outputs" / "cyclegan.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-eval-samples", type=int, default=512)
    parser.add_argument("--feature-extractor", default="small-cnn", choices=["small-cnn", "inception"])
    return parser


@torch.no_grad()
def translate_and_score(generator, back, loader, extractor, device, limit):
    """Collect translated features plus the cycle-reconstruction error."""
    translated, cycled = [], []
    l1_m, psnr_m, ssim_m = (utils.AverageMeter() for _ in range(3))
    seen = 0
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        y = generator(x)
        x_back = back(y)
        n = x.shape[0]

        translated.append(extractor.features(data_mod.denormalize(y)).cpu())
        cycled.append(x_back.cpu())
        l1_m.update(F.l1_loss(x_back, x).item(), n)
        psnr_m.update(metrics_mod.psnr(x_back, x), n)
        ssim_m.update(metrics_mod.ssim(x_back, x), n)

        seen += n
        if seen >= limit:
            break
    return torch.cat(translated), l1_m.avg, psnr_m.avg, ssim_m.avg


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)

    ckpt = utils.load_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    out_dir = utils.resolve_out_dir(args, str(Path(args.checkpoint).parent))
    eval_dir = utils.ensure_dir(out_dir / "evaluation")

    g_ab = ResnetGenerator(cfg["channels"], cfg["base_channels"], cfg["res_blocks"]).to(device)
    g_ba = ResnetGenerator(cfg["channels"], cfg["base_channels"], cfg["res_blocks"]).to(device)
    g_ab.load_state_dict(ckpt["g_ab"])
    g_ba.load_state_dict(ckpt["g_ba"])
    g_ab.eval()
    g_ba.eval()

    set_a, set_b, _, name_a, name_b = build_domains(
        cfg["task"], cfg["dataset"], args.data_root, cfg["image_size"], False,
        cfg["domain_a"], cfg["domain_b"],
    )
    loader_a = DataLoader(set_a, batch_size=args.batch_size, num_workers=args.num_workers)
    loader_b = DataLoader(set_b, batch_size=args.batch_size, num_workers=args.num_workers)

    metric_dataset = cfg["dataset"] if cfg["task"] == "classes" else (
        cfg["domain_a"] if cfg["task"] == "datasets" else "cifar10"
    )
    extractor = metrics_mod.get_feature_extractor(
        metric_dataset,
        root=args.data_root,
        image_size=cfg["image_size"],
        device=device,
        kind=args.feature_extractor,
        num_workers=args.num_workers,
    )

    limit = 128 if args.quick else args.num_eval_samples
    real_a = metrics_mod.features_from_loader(extractor, loader_a, device, max_samples=limit)
    real_b = metrics_mod.features_from_loader(extractor, loader_b, device, max_samples=limit)

    fake_b, l1_a, psnr_a, ssim_a = translate_and_score(g_ab, g_ba, loader_a, extractor, device, limit)
    fake_a, l1_b, psnr_b, ssim_b = translate_and_score(g_ba, g_ab, loader_b, extractor, device, limit)

    ab = metrics_mod.generative_metrics(real_b, fake_b)
    ba = metrics_mod.generative_metrics(real_a, fake_a)

    results = {
        f"fid_{name_a}_to_{name_b}": ab["fid"],
        f"kid_{name_a}_to_{name_b}": ab["kid_mean"],
        f"fid_{name_b}_to_{name_a}": ba["fid"],
        f"kid_{name_b}_to_{name_a}": ba["kid_mean"],
        f"cycle_l1_{name_a}": l1_a,
        f"cycle_psnr_db_{name_a}": psnr_a,
        f"cycle_ssim_{name_a}": ssim_a,
        f"cycle_l1_{name_b}": l1_b,
        f"cycle_psnr_db_{name_b}": psnr_b,
        f"cycle_ssim_{name_b}": ssim_b,
        "lambda_cycle": cfg["lambda_cycle"],
    }

    batch_a = next(iter(loader_a))[0][:8].to(device)
    batch_b = next(iter(loader_b))[0][:8].to(device)
    with torch.no_grad():
        viz.save_comparison_grid(
            {
                f"real {name_a}": batch_a,
                f"-> {name_b}": g_ab(batch_a),
                f"cycled {name_a}": g_ba(g_ab(batch_a)),
                f"real {name_b}": batch_b,
                f"-> {name_a}": g_ba(batch_b),
                f"cycled {name_b}": g_ab(g_ba(batch_b)),
            },
            eval_dir / "translations.png",
        )

    utils.save_json(results, eval_dir / "metrics.json")
    print(f"\nCycleGAN evaluation ({name_a} <-> {name_b})")
    print(metrics_mod.format_metrics(results))
    print(f"\nfigures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
