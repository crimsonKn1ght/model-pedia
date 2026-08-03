"""Paired image-to-image datasets, with the transform applied to both halves.

    python data.py --dataset shapes --preview       # generated, no download
    python data.py --dataset facades               # ~30 MB download

``shapes`` is generated: an outline drawing on the left, the filled coloured version on
the right. It is the default because a paired translation task with *exact* targets makes
every metric here unambiguous - there is one right answer per input, so L1 and SSIM mean
what they say, and the failure modes (right shape, wrong colour) are legible in the figure.

``facades`` is the dataset Pix2Pix was published on: architectural label maps paired with
photographs of the buildings. It is the real thing, and it is also a good demonstration of
why paired translation is hard - the label map does not determine the photograph, so there
are many right answers and L1 punishes all but one of them.

Both halves of a pair must get the *same* geometric transform, which is the one thing this
file exists to guarantee. ``JointTransform`` draws the random crop and flip once and
applies them twice.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

FACADES_URL = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"


class JointTransform:
    """Resize, and optionally flip and scale-crop, both images identically."""

    def __init__(self, size: int, augment: bool = False) -> None:
        self.size = size
        self.augment = augment

    def __call__(self, source: Image.Image, target: Image.Image):
        source = TF.resize(source, [self.size, self.size], InterpolationMode.BILINEAR)
        target = TF.resize(target, [self.size, self.size], InterpolationMode.BILINEAR)
        if self.augment:
            if random.random() < 0.5:
                source, target = TF.hflip(source), TF.hflip(target)
            if random.random() < 0.5:
                # Pix2Pix's "resize then random crop" jitter, applied to the pair.
                bigger = int(self.size * 1.15)
                source = TF.resize(source, [bigger, bigger], InterpolationMode.BILINEAR)
                target = TF.resize(target, [bigger, bigger], InterpolationMode.BILINEAR)
                top = random.randint(0, bigger - self.size)
                left = random.randint(0, bigger - self.size)
                source = TF.crop(source, top, left, self.size, self.size)
                target = TF.crop(target, top, left, self.size, self.size)
        return TF.to_tensor(source), TF.to_tensor(target)


class ShapesPairs(Dataset):
    """Outline drawing -> filled colour version, generated deterministically."""

    COLOURS = [
        (0.85, 0.20, 0.20), (0.25, 0.70, 0.35), (0.25, 0.45, 0.85),
        (0.93, 0.83, 0.25), (0.60, 0.35, 0.75),
    ]

    def __init__(self, n: int, canvas: int = 64, seed: int = 0, max_shapes: int = 3) -> None:
        self.n = n
        self.canvas = canvas
        self.seed = seed
        self.max_shapes = max_shapes

    def __len__(self) -> int:
        return self.n

    def _pair(self, index: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        size = self.canvas
        outline = np.zeros((3, size, size), dtype=np.float32)
        filled = np.full((3, size, size), 0.08, dtype=np.float32)
        ys, xs = np.mgrid[0:size, 0:size]

        for _ in range(int(rng.integers(1, self.max_shapes + 1))):
            kind = int(rng.integers(0, 3))
            colour = np.array(self.COLOURS[int(rng.integers(0, len(self.COLOURS)))],
                              dtype=np.float32)
            radius = int(rng.integers(size // 8, size // 4))
            cx = int(rng.integers(radius + 1, size - radius - 1))
            cy = int(rng.integers(radius + 1, size - radius - 1))

            if kind == 0:
                distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
                area = distance <= radius
                edge = np.abs(distance - radius) <= 1.2
            elif kind == 1:
                inside = np.maximum(np.abs(xs - cx), np.abs(ys - cy))
                area = inside <= radius
                edge = np.abs(inside - radius) <= 1.2
            else:
                bary = (ys - (cy - radius)) / 2 + 1
                area = (ys >= cy - radius) & (ys <= cy + radius) & (np.abs(xs - cx) <= bary)
                inner = (ys >= cy - radius + 2) & (ys <= cy + radius - 2) & (
                    np.abs(xs - cx) <= bary - 2
                )
                edge = area & ~inner

            filled[:, area] = colour[:, None]
            # The outline carries shape and position but no colour: the model has to
            # invent the colour, which is exactly where L1 and the GAN loss disagree.
            outline[:, edge] = 1.0

        return (
            Image.fromarray((outline.transpose(1, 2, 0) * 255).astype(np.uint8)),
            Image.fromarray((filled.transpose(1, 2, 0) * 255).astype(np.uint8)),
        )

    def __getitem__(self, index: int):
        return self._pair(index)


class FacadePairs(Dataset):
    """Berkeley's facades set: each file is the two images side by side."""

    def __init__(self, root: str, split: str, download: bool = True) -> None:
        directory = Path(root) / "facades"
        if download and not directory.exists():
            from torchvision.datasets.utils import download_and_extract_archive

            print("downloading facades ...")
            download_and_extract_archive(FACADES_URL, download_root=str(Path(root)))

        folder = next(
            (path for path in [directory / split, directory / ("val" if split == "test" else split)]
             if path.is_dir()),
            None,
        )
        if folder is None:
            raise FileNotFoundError(
                f"no '{split}' directory under {directory}. Expected the standard "
                "facades layout (train/, val/, test/) of side-by-side jpg pairs."
            )
        self.paths = sorted(folder.glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"no .jpg files under {folder}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        combined = Image.open(self.paths[index]).convert("RGB")
        width, height = combined.size
        half = width // 2
        # Pix2Pix's facades files are photograph on the left, labels on the right, and
        # the interesting direction is labels -> photograph.
        photo = combined.crop((0, 0, half, height))
        labels = combined.crop((half, 0, width, height))
        return labels, photo


DATASETS = {
    "shapes": {"channels": 3, "size": 64, "download_mb": 0},
    "facades": {"channels": 3, "size": 128, "download_mb": 30},
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return dict(DATASETS[name])


class PairedDataset(Dataset):
    def __init__(self, base: Dataset, transform: JointTransform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        source, target = self.base[index]
        return self.transform(source, target)


def download(name: str, root: str = "data") -> None:
    if name == "facades":
        FacadePairs(root, "train", download=True)


def get_splits(
    name: str = "shapes",
    root: str = "data",
    image_size: int | None = None,
    train_size: int = 3000,
    val_size: int = 200,
    test_size: int = 500,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train, val, test, info)`` of ``(source PIL, target PIL)`` pairs."""
    info = dataset_info(name)
    if synthetic:
        info["size"] = 32
        return (
            ShapesPairs(48, 32, seed=1), ShapesPairs(24, 32, seed=2),
            ShapesPairs(24, 32, seed=3), info,
        )

    size = image_size or info["size"]
    info["size"] = size
    if name == "shapes":
        return (
            ShapesPairs(train_size, size, seed=1),
            ShapesPairs(val_size, size, seed=2),
            ShapesPairs(test_size, size, seed=3),
            info,
        )

    full_train = FacadePairs(root, "train")
    test_set = FacadePairs(root, "test")
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()
    if train_size:
        train_idx = train_idx[:train_size]
    return Subset(full_train, train_idx), Subset(full_train, val_idx), test_set, info


def make_loader(
    base: Dataset, size: int, batch_size: int = 16, shuffle: bool = False,
    augment: bool = False, num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        PairedDataset(base, JointTransform(size, augment)),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or inspect a paired dataset.")
    parser.add_argument("--dataset", default="shapes", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--preview", action="store_true", help="save outputs/data_preview.png")
    args = parser.parse_args()

    download(args.dataset, args.root)
    train, val, test, info = get_splits(args.dataset, args.root, image_size=args.image_size)
    print(f"dataset : {args.dataset}")
    print(f"train   : {len(train)} pairs")
    print(f"val     : {len(val)} pairs")
    print(f"test    : {len(test)} pairs")
    print(f"image   : {info['channels']}x{info['size']}x{info['size']} in [0, 1]")

    if args.preview:
        from utils import plot_image_rows

        loader = make_loader(val, info["size"], batch_size=8, num_workers=0)
        source, target = next(iter(loader))
        plot_image_rows([("input", source), ("target", target)], "outputs/data_preview.png",
                        title=f"{args.dataset}: paired input and target")
        print("\npreview -> outputs/data_preview.png")


if __name__ == "__main__":
    main()
