"""Evaluate a trained VQ-VAE: reconstruction quality, codebook health, samples.

    python evaluate.py

Reports reconstruction MSE/PSNR/SSIM on the test split, how much of the
codebook is actually used (a collapsed codebook is the classic VQ-VAE failure),
the compression ratio, and FID/KID of images generated from the PixelCNN prior.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import bootstrap  # noqa: F401
from common import data as data_mod
from common import metrics as metrics_mod
from common import utils, viz
from model import PixelCNNPrior, VQVAE

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--checkpoint", default=str(HERE / "outputs" / "vqvae.pt"))
    parser.add_argument("--batch-size", type=int, default=128)
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

    model = VQVAE(
        channels=cfg["channels"],
        hidden=cfg["hidden"],
        embedding_dim=cfg["embedding_dim"],
        num_embeddings=cfg["num_embeddings"],
        commitment_cost=cfg["commitment_cost"],
        decay=cfg["decay"],
        restart_threshold=cfg.get("restart_threshold", 1.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prior = PixelCNNPrior(cfg["num_embeddings"], cfg["prior_hidden"], cfg["prior_layers"]).to(device)
    prior.load_state_dict(ckpt["prior"])
    prior.eval()

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

    # -- 1. reconstruction quality and codebook usage ----------------------- #
    mse_m, psnr_m, ssim_m, ppl_m = (utils.AverageMeter() for _ in range(4))
    code_hits = torch.zeros(cfg["num_embeddings"], dtype=torch.long)
    with torch.no_grad():
        for x, _ in utils.progress(utils.maybe_limit(test_loader, args), desc="reconstruction"):
            x = x.to(device)
            recon, _, indices, perplexity = model(x)
            n = x.shape[0]
            mse_m.update(F.mse_loss(recon, x).item(), n)
            psnr_m.update(metrics_mod.psnr(recon, x), n)
            ssim_m.update(metrics_mod.ssim(recon, x), n)
            ppl_m.update(perplexity.item(), n)
            code_hits += torch.bincount(
                indices.reshape(-1).cpu(), minlength=cfg["num_embeddings"]
            )

    used = int((code_hits > 0).sum())
    latent_hw = cfg["image_size"] // 4
    bits_per_code = math.log2(cfg["num_embeddings"])
    original_bits = cfg["channels"] * cfg["image_size"] ** 2 * 8
    latent_bits = latent_hw**2 * bits_per_code

    results = {
        "test_reconstruction_mse": mse_m.avg,
        "test_psnr_db": psnr_m.avg,
        "test_ssim": ssim_m.avg,
        "codebook_perplexity": ppl_m.avg,
        "codebook_size": float(cfg["num_embeddings"]),
        "codebook_used": float(used),
        "codebook_usage_fraction": used / cfg["num_embeddings"],
        "latent_grid": float(latent_hw),
        "compression_ratio": original_bits / latent_bits,
    }

    # -- 2. pictures -------------------------------------------------------- #
    batch = next(iter(test_loader))[0][:8].to(device)
    with torch.no_grad():
        viz.save_comparison_grid(
            {"input": batch, "reconstruction": model(batch)[0]},
            eval_dir / "reconstructions.png",
        )
    viz.plot_bars(
        [str(i) for i in range(0, cfg["num_embeddings"], max(1, cfg["num_embeddings"] // 32))],
        code_hits[:: max(1, cfg["num_embeddings"] // 32)].float().tolist(),
        eval_dir / "codebook-usage.png",
        title=f"codebook usage ({used}/{cfg['num_embeddings']} codes used)",
        ylabel="times selected",
    )

    # -- 3. generation from the prior --------------------------------------- #
    if cfg.get("prior_trained", True):
        n_samples = 128 if args.quick else args.num_eval_samples

        def sample_images(n: int) -> torch.Tensor:
            grids = prior.sample(n, latent_hw, latent_hw, device)
            return model.decode_indices(grids)

        viz.save_image_grid(sample_images(32), eval_dir / "prior-samples.png", nrow=8)

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
        # PixelCNN sampling is a raster-scan loop, so keep the batches large.
        fake_features = metrics_mod.features_from_sampler(
            extractor, sample_images, device, num_samples=n_samples, batch_size=128
        )
        results.update(metrics_mod.generative_metrics(real_features, fake_features))

        # Reconstruction-only FID isolates the autoencoder from the prior.
        recon_features = metrics_mod.features_from_sampler(
            extractor,
            lambda n: model(next(iter(test_loader))[0][:n].to(device))[0],
            device,
            num_samples=min(n_samples, 512),
            batch_size=args.batch_size,
        )
        results["fid_reconstructions"] = metrics_mod.fid_from_features(
            *metrics_mod.standardize_pair(real_features[: recon_features.shape[0]], recon_features)
        )

    utils.save_json(results, eval_dir / "metrics.json")
    print("\nVQ-VAE evaluation")
    print(metrics_mod.format_metrics(results))
    print(f"\nfigures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
