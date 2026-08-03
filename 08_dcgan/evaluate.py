"""Evaluate a trained DCGAN: FID/KID, precision-recall, sample grids, interpolation.

    python evaluate.py

Precision and recall are reported separately on purpose.  A single FID number
hides *why* a GAN is bad; the pair does not:

* high precision, low recall  -> sharp samples but mode collapse;
* low precision, high recall  -> broad coverage but poor sample quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import metrics as metrics_mod
from common import utils, viz
from model import Generator

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--checkpoint", default=str(HERE / "outputs" / "dcgan.pt"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--feature-extractor", default="small-cnn", choices=["small-cnn", "inception"])
    return parser


@torch.no_grad()
def interpolation_grid(generator: Generator, device, rows: int = 8, steps: int = 8) -> torch.Tensor:
    """Walk in a straight line between pairs of latent codes.

    Smooth, semantically continuous transitions indicate the generator learned a
    structured mapping rather than memorising a lookup table.
    """
    z_a = torch.randn(rows, generator.latent_dim, device=device)
    z_b = torch.randn(rows, generator.latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps, device=device).view(1, steps, 1)
    z = z_a.unsqueeze(1) * (1 - alphas) + z_b.unsqueeze(1) * alphas
    return generator(z.reshape(rows * steps, -1))


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)

    ckpt = utils.load_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    out_dir = utils.resolve_out_dir(args, str(Path(args.checkpoint).parent))
    eval_dir = utils.ensure_dir(out_dir / "evaluation")

    generator = Generator(
        cfg["latent_dim"], cfg["channels"], cfg["image_size"], cfg["base_channels"]
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    test_loader = data_mod.get_dataloader(
        cfg["dataset"],
        root=args.data_root,
        image_size=cfg["image_size"],
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

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
        extractor, test_loader, device, max_samples=n_samples
    )
    fake_features = metrics_mod.features_from_sampler(
        extractor, lambda n: generator.sample(n, device), device, num_samples=n_samples
    )
    results = metrics_mod.generative_metrics(real_features, fake_features)

    viz.save_image_grid(generator.sample(64, device), eval_dir / "samples.png", nrow=8)
    viz.save_image_grid(interpolation_grid(generator, device), eval_dir / "interpolation.png", nrow=8)
    viz.save_comparison_grid(
        {"real": next(iter(test_loader))[0][:8].to(device), "generated": generator.sample(8, device)},
        eval_dir / "real-vs-generated.png",
    )

    utils.save_json(results, eval_dir / "metrics.json")
    print("\nDCGAN evaluation")
    print(metrics_mod.format_metrics(results))
    print(f"\nfigures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
