"""Evaluate ViT against the ResNet-18 baseline on the test split.

    python evaluate.py                       # compares whichever checkpoints exist
    python evaluate.py --models vit resnet18

A raw accuracy comparison is not enough to conclude anything, so this also
reports parameter counts, measured inference FLOPs and wall-clock latency. Two
models are only comparable at matched compute; a ViT that loses to a ResNet
while using three times the FLOPs has lost twice.

Per-class accuracy and the confusion matrix are written out as well, plus a ViT
attention map showing which patches the class token attended to.
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
from model import build_model

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_args(parser)
    parser.add_argument("--models", nargs="*", default=["vit", "resnet18"])
    parser.add_argument("--checkpoint-root", default=str(HERE / "outputs"))
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def count_flops(model: torch.nn.Module, shape, device) -> float:
    """Multiply-accumulates for one forward pass, counted with a FLOP hook."""
    total = [0]

    def conv_hook(module, inputs, output):
        total[0] += (
            module.weight.numel() // module.groups * output.shape[-1] * output.shape[-2]
        )

    def linear_hook(module, inputs, output):
        elements = output.numel() // output.shape[-1]
        total[0] += module.weight.numel() * elements

    handles = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    with torch.no_grad():
        model(torch.zeros(1, *shape, device=device))
    for handle in handles:
        handle.remove()
    return float(total[0])


def measure_latency(model, shape, device, batch: int = 64, repeats: int = 5) -> float:
    x = torch.randn(batch, *shape, device=device)
    with torch.no_grad():
        model(x)
        with utils.Timer() as timer:
            for _ in range(repeats):
                model(x)
    return timer.elapsed / repeats / batch


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args.device)
    root = Path(args.checkpoint_root)
    eval_dir = utils.ensure_dir(root / "comparison")

    summary = {}
    per_class_all = {}

    for name in args.models:
        path = root / name / f"{name}.pt"
        if not path.exists():
            print(f"skipping {name}: no checkpoint at {path}  (run: python train.py --model {name})")
            continue

        ckpt = utils.load_checkpoint(path, map_location=device)
        cfg = ckpt["config"]
        model = build_model(
            cfg["model"], cfg["image_size"], cfg["channels"], cfg["num_classes"], **cfg["kwargs"]
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
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
        )

        shape = (cfg["channels"], cfg["image_size"], cfg["image_size"])
        confusion = torch.zeros(cfg["num_classes"], cfg["num_classes"], dtype=torch.long)
        top1, top5, loss_m = (utils.AverageMeter() for _ in range(3))

        with torch.no_grad():
            for x, y in utils.progress(utils.maybe_limit(test_loader, args), desc=f"{name} test"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss_m.update(F.cross_entropy(logits, y).item(), x.shape[0])
                k = min(5, cfg["num_classes"])
                _, pred = logits.topk(k, dim=1)
                correct = pred.eq(y.view(-1, 1))
                top1.update(correct[:, 0].float().mean().item(), x.shape[0])
                top5.update(correct.any(dim=1).float().mean().item(), x.shape[0])
                for t, p in zip(y.cpu(), pred[:, 0].cpu()):
                    confusion[t, p] += 1

        per_class = (confusion.diag().float() / confusion.sum(dim=1).clamp(min=1)).tolist()
        names = meta.class_names or tuple(str(i) for i in range(cfg["num_classes"]))
        per_class_all[name] = dict(zip(names, per_class))

        flops = count_flops(model, shape, device)
        summary[name] = {
            "test_accuracy": top1.avg,
            "test_top5_accuracy": top5.avg,
            "test_loss": loss_m.avg,
            "parameters_millions": utils.count_parameters(model) / 1e6,
            "mega_flops_per_image": flops / 1e6,
            "milliseconds_per_image": measure_latency(model, shape, device) * 1000,
            "train_minutes": cfg["train_minutes"],
            "epochs": float(cfg["epochs"]),
            "worst_class_accuracy": min(per_class),
        }

        viz.plot_bars(
            list(names),
            per_class,
            eval_dir / f"{name}-per-class-accuracy.png",
            title=f"{name}: per-class accuracy",
            ylabel="accuracy",
        )

        if cfg["model"] == "vit":
            batch = next(iter(test_loader))[0][:8].to(device)
            attention = model.attention_rollout(batch)
            attention = F.interpolate(
                attention, size=(cfg["image_size"], cfg["image_size"]), mode="bilinear"
            )
            attention = attention / attention.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
            viz.save_comparison_grid(
                {"input": batch, "attention": attention.repeat(1, cfg["channels"], 1, 1) * 2 - 1},
                eval_dir / "vit-attention.png",
            )

    if not summary:
        raise SystemExit("no checkpoints found; run train.py first")

    if len(summary) > 1:
        viz.plot_bars(
            list(summary),
            [s["test_accuracy"] for s in summary.values()],
            eval_dir / "accuracy-comparison.png",
            title="test accuracy",
            ylabel="accuracy",
        )
        viz.plot_bars(
            list(summary),
            [s["mega_flops_per_image"] for s in summary.values()],
            eval_dir / "compute-comparison.png",
            title="inference cost",
            ylabel="MFLOPs per image",
        )

    utils.save_json({"summary": summary, "per_class": per_class_all}, eval_dir / "metrics.json")

    print("\nclassification comparison")
    header = f"{'model':<12}{'accuracy':>10}{'top-5':>9}{'params(M)':>11}{'MFLOPs':>10}{'ms/img':>9}{'train(min)':>12}"
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(
            f"{name:<12}{s['test_accuracy']:>10.4f}{s['test_top5_accuracy']:>9.4f}"
            f"{s['parameters_millions']:>11.2f}{s['mega_flops_per_image']:>10.1f}"
            f"{s['milliseconds_per_image']:>9.2f}{s['train_minutes']:>12.1f}"
        )
    print(f"\nfigures and metrics written to {eval_dir}")


if __name__ == "__main__":
    main()
