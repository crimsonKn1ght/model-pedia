"""Score a captioner on the test split, and compare decoding strategies.

    python evaluate.py --checkpoint outputs/shapes_cnn/best.pt
    python evaluate.py --checkpoint ... --beam-sizes 1 3 5

Three metrics, because they measure different failures:

* **BLEU** is precision on n-grams. It is happy with a short, safe, generic caption.
* **METEOR** (exact-match variant here) brings recall in, so leaving things out costs
  something.
* **CIDEr-D** weights n-grams by how rare they are across the reference corpus, so
  the words that actually identify *this* image count for more than the words every
  caption contains.

Two diagnostics sit next to them, and they are the ones that catch the failure modes
a score cannot: ``mean_length``, and ``distinct_captions`` - the fraction of test
images that got a caption no other image got. A model that has learned to emit one
plausible sentence for everything can post a respectable BLEU-1 with
``distinct_captions`` near zero, and no amount of staring at BLEU will tell you.

The same trained weights are then decoded at several beam widths. Beam search is
worth one or two BLEU points and it is not free; ``--beam-sizes`` prices it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import Vocabulary, get_splits, make_eval_loader
from model import build_model
from train import generate_split
from utils import caption_metrics, get_device, plot_bars, plot_caption_examples, save_json, set_seed

REPORT_KEYS = (
    "bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor_exact", "cider_d",
    "mean_length", "distinct_captions",
)


def run_evaluation(args, verbose: bool = True) -> dict:
    set_seed(args.seed)
    dev = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    vocabulary = Vocabulary.from_state_dict(ckpt["vocabulary"])

    _, _, test_set, _, info = get_splits(
        ckpt["dataset"],
        root=args.data_root,
        image_size=ckpt["image_size"],
        test_size=args.test_size,
        max_len=ckpt["max_len"],
        seed=args.seed,
        synthetic=args.smoke_test,
    )
    # The checkpoint's vocabulary is authoritative: rebuilding it here would give a
    # different token order and silently scramble every id in the decoder.
    test_set.vocabulary = vocabulary
    test_loader = make_eval_loader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model(
        len(vocabulary),
        vocabulary.bos_index,
        vocabulary.eos_index,
        encoder_name=ckpt["encoder"],
        in_channels=ckpt["in_channels"],
        encoder_width=ckpt["encoder_width"],
        pretrained=False,  # the trained weights are about to overwrite them anyway
        dim=ckpt["dim"],
        depth=ckpt["depth"],
        heads=ckpt["heads"],
        max_len=ckpt["max_len"],
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])

    by_beam = {}
    examples = None
    for beam_size in args.beam_sizes:
        candidates, references, images, order = generate_split(
            model, test_loader, test_set, vocabulary, dev, beam_size=beam_size
        )
        by_beam[beam_size] = caption_metrics(candidates, references)
        if examples is None or beam_size == max(args.beam_sizes):
            examples = (images, candidates, references)

    out_dir = Path(args.checkpoint).parent
    images, candidates, references = examples
    plot_caption_examples(
        images[:8],
        [" ".join(caption) for caption in candidates[:8]],
        [" ".join(refs[0]) for refs in references[:8]],
        out_dir / "examples.png",
        title=f"test captions, beam {max(args.beam_sizes)} ({ckpt['dataset']}, {ckpt['encoder']})",
    )
    plot_bars(
        [f"beam {beam}" if beam > 1 else "greedy" for beam in args.beam_sizes],
        {
            "BLEU-4": [by_beam[beam]["bleu_4"] for beam in args.beam_sizes],
            "METEOR (exact)": [by_beam[beam]["meteor_exact"] for beam in args.beam_sizes],
            "CIDEr-D / 10": [by_beam[beam]["cider_d"] / 10 for beam in args.beam_sizes],
        },
        out_dir / "decoding.png",
        ylabel="score",
        title="the same weights, different decoding",
    )

    result = {
        "dataset": ckpt["dataset"],
        "encoder": ckpt["encoder"],
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt["epoch"],
        "test_images": len(test_set.records),
        "vocabulary_size": len(vocabulary),
        "by_beam": {str(beam): by_beam[beam] for beam in args.beam_sizes},
        "sample_captions": [
            {"generated": " ".join(caption), "reference": " ".join(refs[0])}
            for caption, refs in zip(candidates[:10], references[:10])
        ],
    }
    save_json(result, out_dir / "metrics.json")

    if verbose:
        print(f"model    : {ckpt['encoder']} encoder, epoch {ckpt['epoch']}")
        print(f"test set : {ckpt['dataset']}  {len(test_set.records)} images, "
              f"{info['captions_per_image']} references each\n")

        header = f"{'decoding':>10s}" + "".join(f" {key:>12s}" for key in REPORT_KEYS)
        print(header)
        print("-" * len(header))
        for beam in args.beam_sizes:
            label = "greedy" if beam == 1 else f"beam {beam}"
            line = f"{label:>10s}"
            for key in REPORT_KEYS:
                line += f" {by_beam[beam][key]:12.4f}"
            print(line)

        print("\nfirst few test captions:")
        for entry in result["sample_captions"][:5]:
            print(f"  model: {entry['generated']}")
            print(f"  ref  : {entry['reference']}")
        print(f"\nexamples -> {out_dir / 'examples.png'}")
        print(f"decoding -> {out_dir / 'decoding.png'}")
        print(f"metrics  -> {out_dir / 'metrics.json'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an image captioner.")
    parser.add_argument("--checkpoint", default="outputs/shapes_cnn/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument(
        "--beam-sizes", type=int, nargs="+", default=[1, 3],
        help="1 is greedy; each extra width costs another decode of the whole split",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate on generated scenes")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        args.num_workers = 0
    run_evaluation(args)


if __name__ == "__main__":
    main()
