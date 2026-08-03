"""Train the captioner.

    python train.py --dataset shapes
    python train.py --dataset flickr8k --encoder resnet18 --epochs 10

Training is teacher-forced: the decoder is shown the true previous words and asked
for the next one, scored by cross entropy with the padding ignored. That is fast and
stable, and it is *not* the thing the model is judged on - at test time it has to
condition on its own output, and errors compound. The gap between a falling
validation loss and a flat BLEU is the usual first sign of that.

So every epoch also **generates** captions for the validation split with greedy
decoding and scores BLEU-4 against all references, and BLEU-4 selects the
checkpoint. Generation costs one forward pass per token, which is why it is greedy
here and why ``--val-generate`` exists to cap how many images it runs on.

Label smoothing is on by default. Captioning has several right answers per image, so
a loss that demands full confidence in one particular word is asking for something
false; smoothing measurably improves the generated text even as it worsens the loss.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import DATASETS, get_splits, make_eval_loader, make_train_loader
from model import ENCODERS, build_model
from utils import (
    AverageMeter,
    caption_metrics,
    count_parameters,
    get_device,
    plot_caption_examples,
    plot_curves,
    save_json,
    set_seed,
)


@torch.no_grad()
def teacher_forced_loss(model, loader, criterion, device) -> float:
    """Validation cross entropy under teacher forcing."""
    model.eval()
    meter = AverageMeter()
    for images, captions in loader:
        images, captions = images.to(device), captions.to(device)
        logits = model(images, captions[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def generate_split(
    model,
    loader,
    dataset,
    vocabulary,
    device,
    beam_size: int = 1,
    max_images: int | None = None,
):
    """Generate captions for a split; returns ``(candidates, references, images, order)``."""
    model.eval()
    candidates, references, order = [], [], []
    first_images = None

    for images, indices in loader:
        if max_images is not None and len(candidates) >= max_images:
            break
        generated = model.generate(images.to(device), beam_size=beam_size)
        candidates.extend(vocabulary.decode(ids) for ids in generated)
        references.extend(dataset.references[int(index)] for index in indices)
        order.extend(int(index) for index in indices)
        if first_images is None:
            first_images = images.cpu()

    return candidates, references, first_images, order


def run_training(
    dataset: str = "shapes",
    encoder_name: str = "cnn",
    image_size: int | None = None,
    encoder_width: int = 32,
    pretrained: bool = True,
    dim: int = 256,
    depth: int = 3,
    heads: int = 4,
    dropout: float = 0.1,
    max_len: int = 24,
    min_freq: int = 2,
    label_smoothing: float = 0.1,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    train_size: int = 2000,
    val_size: int = 300,
    test_size: int = 500,
    val_generate: int = 200,
    augment: bool = False,
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
    out_path = Path(out_dir or f"outputs/{dataset}_{encoder_name}")
    out_path.mkdir(parents=True, exist_ok=True)

    train_set, val_set, _, vocabulary, info = get_splits(
        dataset,
        root=data_root,
        image_size=image_size,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        max_len=max_len,
        min_freq=min_freq,
        seed=seed,
        synthetic=synthetic,
        augment=augment,
    )
    train_loader = make_train_loader(train_set, batch_size=batch_size, num_workers=num_workers)
    val_train_loader = make_train_loader(
        # A second view of the validation split in train mode, for the teacher-forced
        # loss; the eval-mode loader below is what generation runs on.
        _as_train_mode(val_set), batch_size=batch_size, num_workers=num_workers
    )
    val_loader = make_eval_loader(val_set, batch_size=batch_size, num_workers=num_workers)

    model = build_model(
        len(vocabulary),
        vocabulary.bos_index,
        vocabulary.eos_index,
        encoder_name=encoder_name,
        in_channels=info["channels"],
        encoder_width=encoder_width,
        pretrained=pretrained,
        dim=dim,
        depth=depth,
        heads=heads,
        max_len=max_len,
        dropout=dropout,
    ).to(dev)

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocabulary.pad_index, label_smoothing=label_smoothing
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    config = {
        "dataset": dataset,
        "encoder": encoder_name,
        "in_channels": info["channels"],
        "image_size": info["size"],
        "encoder_width": encoder_width,
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "max_len": max_len,
        "vocabulary": vocabulary.state_dict(),
    }

    if verbose:
        print(f"device     : {dev}")
        print(f"encoder    : {encoder_name} - {ENCODERS[encoder_name]}")
        print(f"model      : {count_parameters(model):,} trainable parameters, "
              f"decoder dim {dim} x depth {depth}")
        print(f"data       : {len(train_set.records)} train images, {len(train_set)} pairs; "
              f"{len(val_set.records)} val images")
        print(f"vocabulary : {len(vocabulary)} tokens, captions capped at {max_len}\n")

    history = {"train_loss": [], "val_loss": [], "val_bleu_4": [], "val_cider_d": []}
    best_bleu, best_epoch = -float("inf"), 0
    ckpt_path = out_path / "best.pt"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        batches = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for images, captions in batches:
            images, captions = images.to(dev), captions.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, captions[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
            loss.backward()
            # Transformer decoders on small data are prone to one bad batch; clipping
            # costs nothing and removes the failure mode.
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            meter.update(loss.item(), images.size(0))
            batches.set_postfix(loss=f"{meter.avg:.4f}")
        scheduler.step()

        val_loss = teacher_forced_loss(model, val_train_loader, criterion, dev)
        candidates, references, _, _ = generate_split(
            model, val_loader, val_set, vocabulary, dev, beam_size=1, max_images=val_generate
        )
        scores = caption_metrics(candidates, references)

        history["train_loss"].append(meter.avg)
        history["val_loss"].append(val_loss)
        history["val_bleu_4"].append(scores["bleu_4"])
        history["val_cider_d"].append(scores["cider_d"])

        if scores["bleu_4"] >= best_bleu:
            best_bleu, best_epoch = scores["bleu_4"], epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_bleu_4": scores["bleu_4"],
                    **config,
                },
                ckpt_path,
            )

        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs}  train loss {meter.avg:.4f}  val loss {val_loss:.4f}  "
                f"BLEU-4 {scores['bleu_4']:.4f}  CIDEr-D {scores['cider_d']:.3f}  "
                f"len {scores['mean_length']:.1f}"
                f"{'  <- best' if epoch == best_epoch else ''}"
            )

    elapsed = time.time() - started
    if epochs > 0:
        plot_curves(history, out_path / "curves.png", keys=("loss", "bleu_4", "cider_d"))
        candidates, references, images, _ = generate_split(
            model, val_loader, val_set, vocabulary, dev, beam_size=1, max_images=8
        )
        plot_caption_examples(
            images[:8],
            [" ".join(caption) for caption in candidates[:8]],
            [" ".join(refs[0]) for refs in references[:8]],
            out_path / "examples_val.png",
            title="validation captions (greedy decoding)",
        )

    summary = {
        **{key: value for key, value in config.items() if key != "vocabulary"},
        "vocabulary_size": len(vocabulary),
        "parameters": count_parameters(model),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_bleu_4": best_bleu,
        "train_seconds": round(elapsed, 1),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    save_json(summary, out_path / "history.json")

    if verbose and epochs:
        print(f"\nbest val BLEU-4 {best_bleu:.4f} at epoch {best_epoch} ({elapsed:.1f}s)")
        print(f"checkpoint -> {ckpt_path}")
        print(f"next       : python evaluate.py --checkpoint {ckpt_path}")
    return summary


def _as_train_mode(dataset):
    """Shallow copy of an eval-mode dataset switched to train mode."""
    import copy

    clone = copy.copy(dataset)
    clone.mode = "train"
    return clone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an image captioner.")
    parser.add_argument("--dataset", default="shapes", choices=sorted(DATASETS))
    parser.add_argument("--encoder", default="cnn", choices=sorted(ENCODERS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--encoder-width", type=int, default=32, help="cnn encoder only")
    parser.add_argument(
        "--no-pretrained", action="store_true", help="random-init the torchvision backbone"
    )
    parser.add_argument("--dim", type=int, default=256, help="decoder width")
    parser.add_argument("--depth", type=int, default=3, help="decoder blocks")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-size", type=int, default=2000, help="images (0 = all)")
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument(
        "--val-generate", type=int, default=200, help="validation images to caption each epoch"
    )
    parser.add_argument("--augment", action="store_true", help="horizontal flips")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch on generated scenes")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        dataset=args.dataset,
        encoder_name=args.encoder,
        image_size=args.image_size,
        encoder_width=args.encoder_width,
        pretrained=not args.no_pretrained,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        max_len=args.max_len,
        min_freq=args.min_freq,
        label_smoothing=args.label_smoothing,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=16 if args.smoke_test else args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_size=args.train_size or None,
        val_size=args.val_size,
        test_size=args.test_size,
        val_generate=args.val_generate,
        augment=args.augment,
        num_workers=0 if args.smoke_test else args.num_workers,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        out_dir=args.out_dir,
        synthetic=args.smoke_test,
    )


if __name__ == "__main__":
    main()
