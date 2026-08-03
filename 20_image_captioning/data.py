"""Captioning datasets, the vocabulary, and the two collate functions.

    python data.py --dataset shapes            # generated, nothing to download
    python data.py --dataset flickr8k          # ~1.1 GB download, 8091 photographs

``flickr8k`` is the real dataset: 8091 photographs, five human captions each, with
the authors' own train/dev/test split. It is downloaded from a public mirror of the
original release, since the Illinois links have been dead for years.

``shapes`` is generated: one or two coloured shapes per image, described by four
templated captions each. It exists because a captioner trained for two minutes on a
laptop produces word salad on Flickr8k, and word salad teaches you nothing about
whether your BLEU implementation is right. On ``shapes`` a small model learns to say
true sentences, the metrics move for reasons you can read off the picture, and the
failure modes (getting the shape right and the colour wrong) are legible.

Two details that captioning forces on you and classification does not:

**Two datasets per split.** Training wants one row per *(image, caption)* pair -
five captions is five examples. Evaluation wants one row per *image*, with all five
references, because that is what the metrics compare against. ``mode`` switches
between them.

**A vocabulary.** Words below ``--min-freq`` become ``<unk>``, and the count is
taken over the training split only; building it over all splits would leak test
vocabulary into the model's output space.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

FLICKR8K_URLS = {
    "Flickr8k_Dataset.zip": (
        "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    ),
    "Flickr8k_text.zip": (
        "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    ),
}
FLICKR8K_SPLIT_FILES = {
    "train": "Flickr_8k.trainImages.txt",
    "val": "Flickr_8k.devImages.txt",
    "test": "Flickr_8k.testImages.txt",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, keep alphanumeric runs. Deliberately crude and reproducible."""
    return TOKEN_PATTERN.findall(text.lower())


class Vocabulary:
    """Word-level vocabulary with the four special tokens captioning needs."""

    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, words: list[str]) -> None:
        self.itos = [self.PAD, self.BOS, self.EOS, self.UNK] + words
        self.stoi = {word: index for index, word in enumerate(self.itos)}

    @classmethod
    def build(cls, captions: list[list[str]], min_freq: int = 2) -> "Vocabulary":
        counter = Counter(token for caption in captions for token in caption)
        words = sorted(word for word, count in counter.items() if count >= min_freq)
        return cls(words)

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_index(self) -> int:
        return self.stoi[self.PAD]

    @property
    def bos_index(self) -> int:
        return self.stoi[self.BOS]

    @property
    def eos_index(self) -> int:
        return self.stoi[self.EOS]

    def encode(self, tokens: list[str], max_len: int) -> torch.Tensor:
        """``<bos> tokens... <eos>``, truncated to ``max_len`` including both."""
        unknown = self.stoi[self.UNK]
        ids = [self.bos_index]
        ids += [self.stoi.get(token, unknown) for token in tokens[: max_len - 2]]
        ids.append(self.eos_index)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> list[str]:
        """Drop specials, stop at the first ``<eos>``."""
        words = []
        for index in ids:
            index = int(index)
            if index == self.eos_index:
                break
            if index in (self.pad_index, self.bos_index):
                continue
            words.append(self.itos[index])
        return words

    def state_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Vocabulary":
        vocabulary = cls([])
        vocabulary.itos = list(state["itos"])
        vocabulary.stoi = {word: index for index, word in enumerate(vocabulary.itos)}
        return vocabulary


COLOURS = {
    "red": (0.85, 0.20, 0.20),
    "green": (0.25, 0.70, 0.35),
    "blue": (0.25, 0.45, 0.85),
    "yellow": (0.93, 0.83, 0.25),
    "purple": (0.60, 0.35, 0.75),
    "white": (0.95, 0.95, 0.95),
}
SHAPES = ("circle", "square", "triangle")
SIZES = ("small", "large")


def _position_word(centre_x: float, centre_y: float, canvas: int) -> str:
    third = canvas / 3
    if centre_x < third:
        return "left"
    if centre_x > 2 * third:
        return "right"
    if centre_y < third:
        return "top"
    if centre_y > 2 * third:
        return "bottom"
    return "middle"


def _describe(size: str, colour: str, shape: str, position: str, style: int) -> str:
    if style == 0:
        return f"a {size} {colour} {shape}"
    if style == 1:
        return f"a {colour} {shape} on the {position}"
    if style == 2:
        return f"a {size} {colour} {shape} on the {position}"
    return f"a {colour} {shape}"


class ShapesCaptions:
    """Generated scenes with four templated captions each; nothing is downloaded."""

    def __init__(self, n: int, canvas: int = 64, seed: int = 0, max_objects: int = 2) -> None:
        self.n = n
        self.canvas = canvas
        self.seed = seed
        self.max_objects = max_objects

    def __len__(self) -> int:
        return self.n

    def _scene(self, index: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        canvas = self.canvas
        image = np.clip(
            rng.uniform(0.05, 0.25, size=(3, 1, 1)) + rng.normal(0, 0.02, (3, canvas, canvas)),
            0, 1,
        ).astype(np.float32)

        ys, xs = np.mgrid[0:canvas, 0:canvas]
        objects = []
        for _ in range(int(rng.integers(1, self.max_objects + 1))):
            shape = SHAPES[int(rng.integers(0, len(SHAPES)))]
            colour = list(COLOURS)[int(rng.integers(0, len(COLOURS)))]
            size = SIZES[int(rng.integers(0, len(SIZES)))]
            radius = (
                int(rng.integers(canvas // 10, canvas // 7))
                if size == "small"
                else int(rng.integers(canvas // 6, canvas // 4))
            )
            centre_x = int(rng.integers(radius, canvas - radius))
            centre_y = int(rng.integers(radius, canvas - radius))

            if shape == "circle":
                area = (xs - centre_x) ** 2 + (ys - centre_y) ** 2 <= radius**2
            elif shape == "square":
                area = (np.abs(xs - centre_x) <= radius) & (np.abs(ys - centre_y) <= radius)
            else:
                area = (
                    (ys >= centre_y - radius)
                    & (ys <= centre_y + radius)
                    & (np.abs(xs - centre_x) <= (ys - (centre_y - radius)) / 2 + 1)
                )

            rgb = np.array(COLOURS[colour], dtype=np.float32)
            image[:, area] = rgb[:, None]
            objects.append((size, colour, shape, _position_word(centre_x, centre_y, canvas)))

        return image, objects

    def _captions(self, objects) -> list[list[str]]:
        captions = []
        for style, prefix in enumerate(("", "there is ", "the image shows ", "")):
            phrases = [_describe(*obj, style) for obj in objects]
            joiner = " next to " if style == 3 else " and "
            captions.append(tokenize(prefix + joiner.join(phrases)))
        return captions

    def sample(self, index: int):
        image, objects = self._scene(index)
        return image, self._captions(objects)


def _load_flickr8k(root: str) -> dict:
    """Return ``{split: [(image path, [caption token lists]), ...]}``."""
    root_path = Path(root)
    token_file = next(root_path.rglob("Flickr8k.token.txt"), None)
    if token_file is None:
        raise FileNotFoundError(
            f"Flickr8k.token.txt not found under {root_path}. Run "
            "'python data.py --dataset flickr8k' to download it first."
        )

    captions: dict[str, list[list[str]]] = {}
    for line in token_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in line:
            continue
        key, text = line.split("\t", 1)
        name = key.split("#")[0]
        captions.setdefault(name, []).append(tokenize(text))

    images = {
        path.name: path
        for path in root_path.rglob("*.jpg")
        if "__MACOSX" not in path.parts  # the archive ships resource-fork copies
    }

    splits = {}
    for split, filename in FLICKR8K_SPLIT_FILES.items():
        split_file = next(root_path.rglob(filename), None)
        if split_file is None:
            raise FileNotFoundError(f"{filename} not found under {root_path}")
        names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        splits[split] = [
            (images[name], captions[name])
            for name in names
            if name in images and name in captions
        ]
    return splits


def download_flickr8k(root: str = "data") -> None:
    from torchvision.datasets.utils import download_and_extract_archive

    target = Path(root) / "flickr8k"
    target.mkdir(parents=True, exist_ok=True)
    for filename, url in FLICKR8K_URLS.items():
        if (target / filename).exists() or (filename == "Flickr8k_text.zip" and
                                            next(target.rglob("Flickr8k.token.txt"), None)):
            continue
        print(f"downloading {filename} ...")
        download_and_extract_archive(url, download_root=str(target), filename=filename)


class CaptionDataset(Dataset):
    """One row per ``(image, caption)`` pair in train mode, per image in eval mode."""

    def __init__(
        self,
        records: list,
        vocabulary: Vocabulary,
        transform,
        max_len: int = 24,
        mode: str = "train",
        generator: ShapesCaptions | None = None,
    ) -> None:
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        self.records = records
        self.vocabulary = vocabulary
        self.transform = transform
        self.max_len = max_len
        self.mode = mode
        self.generator = generator

        self.references = [captions for _, captions in records]
        self.pairs = [
            (index, caption_index)
            for index, (_, captions) in enumerate(records)
            for caption_index in range(len(captions))
        ]

    def __len__(self) -> int:
        return len(self.pairs) if self.mode == "train" else len(self.records)

    def _image(self, index: int) -> torch.Tensor:
        key, _ = self.records[index]
        if self.generator is not None:
            array, _ = self.generator._scene(int(key))
            return self.transform(Image.fromarray((array.transpose(1, 2, 0) * 255).astype("uint8")))
        return self.transform(Image.open(key).convert("RGB"))

    def __getitem__(self, index: int):
        if self.mode == "train":
            record_index, caption_index = self.pairs[index]
            tokens = self.records[record_index][1][caption_index]
            return self._image(record_index), self.vocabulary.encode(tokens, self.max_len)
        return self._image(index), index


def collate_train(batch, pad_index: int = 0):
    """Stack images, right-pad captions to the longest in the batch."""
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    longest = max(caption.numel() for caption in captions)
    padded = torch.full((len(captions), longest), pad_index, dtype=torch.long)
    for row, caption in enumerate(captions):
        padded[row, : caption.numel()] = caption
    return images, padded


def collate_eval(batch):
    images = torch.stack([item[0] for item in batch])
    indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, indices


DATASETS = {
    "shapes": {"channels": 3, "size": 64, "download_mb": 0, "captions_per_image": 4},
    "flickr8k": {"channels": 3, "size": 128, "download_mb": 1120, "captions_per_image": 5},
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return dict(DATASETS[name])


def build_transform(image_size: int, augment: bool = False):
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def get_splits(
    name: str = "shapes",
    root: str = "data",
    image_size: int | None = None,
    train_size: int = 2000,
    val_size: int = 300,
    test_size: int = 500,
    max_len: int = 24,
    min_freq: int = 2,
    seed: int = 0,
    synthetic: bool = False,
    augment: bool = False,
):
    """Return ``(train_dataset, val_dataset, test_dataset, vocabulary, info)``."""
    info = dataset_info(name)
    size = image_size or info["size"]
    info["size"] = size

    if name == "shapes" or synthetic:
        if synthetic:
            train_size, val_size, test_size = 64, 24, 24
            size = 32
            info["size"] = size
        generators = {
            "train": ShapesCaptions(train_size, size, seed=1),
            "val": ShapesCaptions(val_size, size, seed=2),
            "test": ShapesCaptions(test_size, size, seed=3),
        }
        records = {
            split: [(index, generator.sample(index)[1]) for index in range(len(generator))]
            for split, generator in generators.items()
        }
    else:
        download_flickr8k(root)
        loaded = _load_flickr8k(str(Path(root) / "flickr8k"))
        records = {
            "train": loaded["train"][:train_size] if train_size else loaded["train"],
            "val": loaded["val"][:val_size] if val_size else loaded["val"],
            "test": loaded["test"][:test_size] if test_size else loaded["test"],
        }
        generators = {split: None for split in records}

    vocabulary = Vocabulary.build(
        [caption for _, captions in records["train"] for caption in captions], min_freq=min_freq
    )

    datasets = {}
    for split, rows in records.items():
        datasets[split] = CaptionDataset(
            rows,
            vocabulary,
            build_transform(size, augment=(augment and split == "train")),
            max_len=max_len,
            mode="train" if split == "train" else "eval",
            generator=generators[split],
        )
    return datasets["train"], datasets["val"], datasets["test"], vocabulary, info


def make_train_loader(dataset, batch_size: int = 64, num_workers: int = 2) -> DataLoader:
    pad = dataset.vocabulary.pad_index
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_train(batch, pad),
        pin_memory=torch.cuda.is_available(),
    )


def make_eval_loader(dataset, batch_size: int = 64, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eval,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or inspect a captioning dataset.")
    parser.add_argument("--dataset", default="shapes", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--preview", action="store_true", help="save outputs/data_preview.png")
    args = parser.parse_args()

    train, val, test, vocabulary, info = get_splits(
        args.dataset,
        root=args.root,
        image_size=args.image_size,
        train_size=args.train_size,
        min_freq=args.min_freq,
    )

    print(f"dataset    : {args.dataset}")
    print(f"images     : {len(train.records)} train / {len(val.records)} val / "
          f"{len(test.records)} test")
    print(f"pairs      : {len(train)} (image, caption) training examples")
    print(f"image      : {info['channels']}x{info['size']}x{info['size']} in [0, 1]")
    print(f"vocabulary : {len(vocabulary)} tokens (min frequency {args.min_freq})")

    lengths = [len(caption) for captions in train.references for caption in captions]
    print(f"caption    : {np.mean(lengths):.1f} tokens on average, longest {max(lengths)}")
    print("\nfirst image's references:")
    for caption in train.references[0]:
        print("  " + " ".join(caption))

    if args.preview:
        from utils import plot_caption_examples

        loader = make_eval_loader(val, batch_size=8, num_workers=0)
        images, indices = next(iter(loader))
        plot_caption_examples(
            images,
            [" ".join(val.references[int(i)][0]) for i in indices],
            [" ".join(val.references[int(i)][1]) for i in indices],
            "outputs/data_preview.png",
            title=f"{args.dataset}: two of the reference captions per image",
        )
        print("\npreview -> outputs/data_preview.png")


if __name__ == "__main__":
    main()
