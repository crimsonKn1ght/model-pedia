"""Evaluate a VQ-VAE: reconstruction, codebook health, and samples if a prior exists.

    python evaluate.py --checkpoint outputs/fashion-mnist_k128/best.pt

Three groups of numbers.

**Reconstruction** - PSNR, SSIM and MSE on the test split, next to the compression
ratio. That ratio is the honest denominator: a 32x32 greyscale image is 1024 numbers,
an 8x8 code grid is 64 integers, so the representation is 16x shorter *and* discrete.

**Codebook health** - how many entries are ever used, and the perplexity of the usage
histogram. A large codebook running at low perplexity has collapsed onto a handful of
entries; it will reconstruct like the small codebook it has become, and the
reconstruction metrics alone will not say why.

**Samples** - only if ``prior.pt`` sits next to the checkpoint, because a VQ-VAE on
its own cannot generate: there is no distribution over the integers until stage two
provides one. With a prior, FID, KID and generative precision/recall are reported
through the same small-classifier feature network the other generative projects use,
so the numbers compare across this repository but not with published FID.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import DATASETS, feature_net_cache, get_splits, make_loader
from model import build_model, build_prior
from train import evaluate_split
from utils import (
    features_from_loader,
    features_from_sampler,
    generative_metrics,
    get_device,
    load_or_train_feature_net,
    plot_bars,
    plot_image_grid,
    plot_image_rows,
    save_json,
    set_seed,
)


@torch.no_grad()
def save_reconstruction_figure(model, loader, device, out_dir: Path, n: int = 8) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)
    outputs = model(images)
    codes = outputs["indices"].float().unsqueeze(1) / max(model.num_codes - 1, 1)
    codes = torch.nn.functional.interpolate(codes, size=images.shape[-2:], mode="nearest")
    plot_image_rows(
        [
            ("input", images.cpu()),
            (f"codes {model.grid}x{model.grid}", codes.expand(-1, images.size(1), -1, -1).cpu()),
            ("reconstruction", outputs["reconstruction"].cpu()),
        ],
        out_dir / "examples.png",
        title="test reconstructions and the code grid they came from",
    )


def load_prior(path: Path, device):
    if not path.exists():
        return None, None
    state = torch.load(path, map_location=device, weights_only=True)
    prior = build_prior(
        state["num_codes"], state["sequence_length"], state["dim"], state["depth"], state["heads"]
    ).to(device)
    prior.load_state_dict(state["prior_state"])
    prior.eval()
    return prior, state


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)

    train_set, _, test_set, info = get_splits(
        ckpt["dataset"], root=args.data_root, image_size=ckpt["image_size"],
        seed=args.seed, synthetic=args.smoke_test,
    )
    train_loader = make_loader(train_set, args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, num_workers=args.num_workers)

    model = build_model(
        ckpt["in_channels"], ckpt["image_size"], ckpt["base_channels"], ckpt["code_dim"],
        ckpt["num_codes"], ckpt["commitment"], ckpt["ema"],
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test = evaluate_split(model, test_loader, dev)
    out_dir = Path(args.checkpoint).parent
    save_reconstruction_figure(model, test_loader, dev, out_dir)
    plot_bars(
        [str(i) if ckpt["num_codes"] <= 32 else "" for i in range(ckpt["num_codes"])],
        {"times used": test["usage"]["histogram"]},
        out_dir / "codebook_usage.png",
        ylabel="count",
        title=f"test codebook histogram: {test['usage']['codes_used']}/"
              f"{test['usage']['num_codes']} used, perplexity {test['usage']['perplexity']:.1f}",
    )

    result = {
        "dataset": ckpt["dataset"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "num_codes": ckpt["num_codes"],
        "grid": f"{model.grid}x{model.grid}",
        "compression": model.compression,
        "codebook": "ema" if ckpt["ema"] else "loss-based",
        "test_images": len(test_set),
        "reconstruction_mse": test["reconstruction"],
        "reconstruction_psnr_db": test["psnr"],
        "reconstruction_ssim": test["ssim"],
        "codes_used": test["usage"]["codes_used"],
        "usage_fraction": test["usage"]["usage_fraction"],
        "perplexity": test["usage"]["perplexity"],
    }

    prior_path = Path(args.prior) if args.prior else out_dir / "prior.pt"
    prior, prior_state = load_prior(prior_path, dev)
    if prior is not None:
        feature_net, feature_kind = load_or_train_feature_net(
            None if args.smoke_test
            else feature_net_cache(args.data_root, ckpt["dataset"], ckpt["image_size"]),
            train_loader, ckpt["in_channels"], info["classes"], dev,
            epochs=1 if args.smoke_test else 2, verbose=verbose,
        )
        real_features = features_from_loader(feature_net, test_loader, dev, limit=args.fid_samples)

        def sample_images(n: int) -> torch.Tensor:
            codes = prior.sample(n, dev, temperature=args.temperature)
            return model.decode_indices(codes.view(-1, model.grid, model.grid))

        fake_features = features_from_sampler(
            feature_net, sample_images, real_features.size(0), args.batch_size, dev
        )
        result["feature_net"] = feature_kind
        result["prior"] = {
            "checkpoint": str(prior_path),
            "epoch": prior_state["epoch"],
            "val_nats_per_code": prior_state["val_loss"],
            "temperature": args.temperature,
        }
        result["samples"] = generative_metrics(
            real_features, fake_features, kid_subset_size=min(500, real_features.size(0))
        )
        plot_image_grid(
            sample_images(64).cpu(), out_dir / "samples.png", columns=8,
            title=f"prior samples, temperature {args.temperature:g}",
        )

    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"\nmodel    : {ckpt['num_codes']} codes, {result['grid']} grid, "
              f"{result['codebook']} codebook, epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_set)} images\n")
        print(f"compression        {result['compression']:9.0f}x  "
              f"(pixels per code position)")
        print(f"reconstruction     {result['reconstruction_psnr_db']:9.2f} dB PSNR, "
              f"SSIM {result['reconstruction_ssim']:.4f}, MSE {result['reconstruction_mse']:.5f}")
        print(f"codebook used      {result['codes_used']:9d} / {ckpt['num_codes']} entries "
              f"({result['usage_fraction']:.1%})")
        print(f"perplexity         {result['perplexity']:9.1f}  "
              f"(equals the codebook size if usage were uniform)")

        if "samples" in result:
            samples = result["samples"]
            print(f"\nprior: {result['prior']['val_nats_per_code']:.4f} nats/code, "
                  f"samples scored through a {result['feature_net']} feature network:")
            print(f"  FID       {samples['fid']:9.3f}   (not comparable with published FID)")
            print(f"  KID       {samples['kid']:+9.5f} +- {samples['kid_std']:.5f}")
            print(f"  precision {samples['precision']:9.3f}")
            print(f"  recall    {samples['recall']:9.3f}")
        else:
            print(f"\nno prior at {prior_path} - a VQ-VAE cannot sample without one.")
            print(f"run: python train_prior.py --checkpoint {args.checkpoint}")

        print(f"\nmetrics -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a VQ-VAE.")
    parser.add_argument("--checkpoint", default="outputs/fashion-mnist_k128/best.pt")
    parser.add_argument("--prior", default=None, help="defaults to prior.pt beside the checkpoint")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASETS))
    parser.add_argument("--temperature", type=float, default=1.0, help="prior sampling temperature")
    parser.add_argument("--fid-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on generated data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_workers = 0
        args.fid_samples = 64
    run_evaluation(args)


if __name__ == "__main__":
    main()
