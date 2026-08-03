"""Evaluate a conditional GAN: does the label actually control the output?

    python evaluate.py

Two questions matter for a conditional model, and overall FID answers neither:

1. **Is conditioning obeyed?**  An independent classifier -- the same small CNN
   the FID feature space is built from, trained only on real training images --
   is asked to label generated samples.  Its agreement with the requested class
   is the ``classifier_accuracy`` reported here.
2. **Is any class neglected?**  FID is computed per class, against real test
   images of that class only.  A model can post a good overall FID while
   quietly producing garbage for one or two classes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import bootstrap  # noqa: F401
from common import data as data_mod
from common import metrics as metrics_mod
from common import utils, viz
from model import ConditionalGenerator

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--checkpoint", default=str(HERE / "outputs" / "conditional-gan.pt"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-eval-samples", type=int, default=2048)
    parser.add_argument("--samples-per-class", type=int, default=512)
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

    generator = ConditionalGenerator(
        cfg["latent_dim"], cfg["num_classes"], cfg["channels"], cfg["image_size"], cfg["base_channels"]
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    meta = data_mod.info(cfg["dataset"])
    test_set = data_mod.get_dataset(
        cfg["dataset"], root=args.data_root, image_size=cfg["image_size"], train=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    extractor = metrics_mod.get_feature_extractor(
        cfg["dataset"],
        root=args.data_root,
        image_size=cfg["image_size"],
        device=device,
        kind=args.feature_extractor,
        num_workers=args.num_workers,
    )

    n_samples = 256 if args.quick else args.num_eval_samples
    per_class = 64 if args.quick else args.samples_per_class

    # -- overall sample quality --------------------------------------------- #
    real_features = metrics_mod.features_from_loader(
        extractor, test_loader, device, max_samples=n_samples
    )
    fake_features = metrics_mod.features_from_sampler(
        extractor, lambda n: generator.sample(n, device), device, num_samples=n_samples
    )
    results = metrics_mod.generative_metrics(real_features, fake_features)
    results["mode"] = cfg["mode"]

    # -- 1. is the conditioning obeyed? ------------------------------------- #
    if args.feature_extractor == "small-cnn":
        correct, total = 0, 0
        with torch.no_grad():
            for cls in range(cfg["num_classes"]):
                labels = torch.full((per_class,), cls, device=device, dtype=torch.long)
                images = generator.sample(per_class, device, labels=labels)
                acc = metrics_mod.classifier_accuracy(extractor, images, labels, device)
                correct += acc * per_class
                total += per_class
        results["classifier_accuracy"] = correct / total

    # -- 2. per-class FID ---------------------------------------------------- #
    per_class_fid = {}
    for cls in range(cfg["num_classes"]):
        real_cls = data_mod.class_subset(test_set, [cls])
        cls_loader = torch.utils.data.DataLoader(
            real_cls, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        real_f = metrics_mod.features_from_loader(
            extractor, cls_loader, device, max_samples=per_class
        )
        fake_f = metrics_mod.features_from_sampler(
            extractor,
            lambda n, c=cls: generator.sample(
                n, device, labels=torch.full((n,), c, device=device, dtype=torch.long)
            ),
            device,
            num_samples=min(per_class, len(real_cls)),
        )
        real_s, fake_s = metrics_mod.standardize_pair(real_f, fake_f)
        per_class_fid[meta.class_names[cls] if meta.class_names else str(cls)] = (
            metrics_mod.fid_from_features(real_s, fake_s)
        )

    results["fid_per_class_mean"] = float(sum(per_class_fid.values()) / len(per_class_fid))
    results["fid_per_class_worst"] = float(max(per_class_fid.values()))

    # -- figures -------------------------------------------------------------- #
    viz.save_image_grid(
        generator.sample_class_grid(10, device), eval_dir / "class-grid.png", nrow=10
    )
    viz.plot_bars(
        list(per_class_fid),
        list(per_class_fid.values()),
        eval_dir / "fid-per-class.png",
        title="FID by requested class (lower is better)",
        ylabel="FID",
    )

    utils.save_json({**results, "fid_per_class": per_class_fid}, eval_dir / "metrics.json")
    print(f"\nconditional GAN evaluation ({cfg['mode']})")
    print(metrics_mod.format_metrics(results))
    print("\n  FID per class:")
    for name, value in per_class_fid.items():
        print(f"    {name:<14} {value:8.3f}")
    print(f"\nfigures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
