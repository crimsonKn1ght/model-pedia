"""Evaluate a trained VAE on the held-out test split.

    python evaluate.py
    python evaluate.py --checkpoint outputs/beta4/vae.pt --out-dir outputs/beta4

Produces:

* test ELBO, reconstruction term, KL term and bits/dim (``metrics.json``);
* reconstruction grid and prior-sample grid;
* a 2-D map of the latent space, coloured by class;
* a latent traversal: one row per latent dimension, sweeping it while the
  others stay fixed -- this is where a beta-VAE visibly beats a plain VAE;
* FID/KID of prior samples against real test images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import metrics as metrics_mod
from common import utils, viz
from model import VAE, bits_per_dim, vae_loss

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--checkpoint", default=str(HERE / "outputs" / "vae.pt"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--feature-extractor", default="small-cnn", choices=["small-cnn", "inception"])
    parser.add_argument("--traversal-range", type=float, default=3.0)
    return parser


@torch.no_grad()
def latent_map(model: VAE, loader, device, max_points: int = 2000):
    """Collect posterior means and labels for the latent-space scatter plot."""
    codes, labels = [], []
    seen = 0
    for x, y in loader:
        mu, _ = model.encoder(x.to(device))
        codes.append(mu.cpu())
        labels.append(y)
        seen += x.shape[0]
        if seen >= max_points:
            break
    return torch.cat(codes)[:max_points], torch.cat(labels)[:max_points]


@torch.no_grad()
def latent_traversal(model: VAE, device, span: float, steps: int = 9) -> torch.Tensor:
    """Sweep each latent dimension while holding the others at zero.

    A well-disentangled model changes one interpretable property per row.
    """
    dims = model.latent_dim
    values = torch.linspace(-span, span, steps, device=device)
    base = torch.zeros(1, dims, device=device)
    rows = []
    for d in range(dims):
        z = base.repeat(steps, 1)
        z[:, d] = values
        rows.append(model.decode_to_image(z))
    return torch.cat(rows), steps


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)

    ckpt = utils.load_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    out_dir = utils.resolve_out_dir(args, str(Path(args.checkpoint).parent))
    eval_dir = utils.ensure_dir(out_dir / "evaluation")

    model = VAE(
        channels=cfg["channels"],
        image_size=cfg["image_size"],
        latent_dim=cfg["latent_dim"],
        base=cfg["base_channels"],
        likelihood=cfg["likelihood"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    meta = data_mod.info(cfg["dataset"])
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

    # -- 1. the bound itself ------------------------------------------------ #
    elbo, recon_m, kl_m = (utils.AverageMeter() for _ in range(3))
    with torch.no_grad():
        for x, _ in utils.progress(utils.maybe_limit(test_loader, args), desc="test ELBO"):
            x = x.to(device)
            recon_logits, mu, logvar = model(x)
            loss, recon, kl = vae_loss(
                recon_logits, x, mu, logvar, beta=cfg["beta"], likelihood=cfg["likelihood"]
            )
            elbo.update(loss.item(), x.shape[0])
            recon_m.update(recon.item(), x.shape[0])
            kl_m.update(kl.item(), x.shape[0])

    results = {
        "test_negative_elbo": elbo.avg,
        "test_reconstruction": recon_m.avg,
        "test_kl": kl_m.avg,
        "test_bits_per_dim": bits_per_dim(
            recon_m.avg + kl_m.avg, cfg["channels"], cfg["image_size"]
        ),
        "beta": cfg["beta"],
    }

    # -- 2. pictures -------------------------------------------------------- #
    batch = next(iter(test_loader))[0][:8].to(device)
    viz.save_comparison_grid(
        {"input": batch, "reconstruction": model.reconstruct(batch)},
        eval_dir / "reconstructions.png",
        normalize="unit",
    )
    viz.save_image_grid(
        model.sample(64, device), eval_dir / "prior-samples.png", nrow=8, normalize="unit"
    )

    codes, labels = latent_map(model, test_loader, device)
    viz.plot_latent_scatter(
        codes,
        labels,
        eval_dir / "latent-space.png",
        title=f"VAE latent space ({cfg['dataset']}, beta={cfg['beta']})",
        class_names=meta.class_names,
    )

    traversal, steps = latent_traversal(model, device, args.traversal_range)
    viz.save_image_grid(
        traversal, eval_dir / "latent-traversal.png", nrow=steps, normalize="unit"
    )

    # -- 3. sample quality -------------------------------------------------- #
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
        lambda n: model.sample(n, device),
        device,
        num_samples=n_samples,
        normalize="unit",
    )
    results.update(metrics_mod.generative_metrics(real_features, fake_features))

    utils.save_json(results, eval_dir / "metrics.json")
    print("\nVAE evaluation")
    print(metrics_mod.format_metrics(results))
    print(f"\nfigures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
