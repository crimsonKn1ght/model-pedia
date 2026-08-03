"""Datasets for semantic segmentation, with transforms applied jointly.

    python data.py --dataset shapes          # generated, nothing to download
    python data.py --dataset oxford-pets     # ~810 MB download

Two datasets, for two purposes.

``oxford-pets`` is the real one. Its trimap annotations give three classes -
background, pet, and a border ring around the animal. That border is the whole
reason this dataset is a good teaching example: it is a couple of pixels wide, so
it contributes almost nothing to pixel accuracy and drags mean IoU down hard. A
model can be 90% pixel-accurate and hopeless at the only class that requires it
to know where the animal *ends*.

``shapes`` is generated on the fly - coloured circles, rectangles and triangles on
a textured background, with pixel-exact masks. Nothing is downloaded, a run takes
a couple of minutes, and because the masks are exact there is no annotation noise
between you and the metric.

Every geometric transform has to be applied to the image *and* the mask, with
nearest-neighbour interpolation on the mask - interpolate a label map bilinearly
and you invent classes that do not exist. ``JointTransform`` below is where that
happens.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# Oxford-IIIT Pet trimaps are stored as 1 = pet, 2 = background, 3 = border.
PET_TRIMAP_TO_CLASS = {1: 1, 2: 0, 3: 2}
PET_CLASS_NAMES = ["background", "pet", "border"]
SHAPE_CLASS_NAMES = ["background", "circle", "rectangle", "triangle"]


class JointTransform:
    """Resize, optional flip and scale-crop, applied identically to image and mask."""

    def __init__(self, size: int, augment: bool = False, flip: bool = True) -> None:
        self.size = size
        self.augment = augment
        self.flip = flip

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, [self.size, self.size], InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], InterpolationMode.NEAREST)

        if self.augment:
            if self.flip and random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                # Random scale-crop: take a window of 70-100% and resize it back.
                scale = random.uniform(0.7, 1.0)
                crop = max(8, int(self.size * scale))
                top = random.randint(0, self.size - crop)
                left = random.randint(0, self.size - crop)
                image = TF.resize(
                    TF.crop(image, top, left, crop, crop),
                    [self.size, self.size],
                    InterpolationMode.BILINEAR,
                )
                mask = TF.resize(
                    TF.crop(mask, top, left, crop, crop),
                    [self.size, self.size],
                    InterpolationMode.NEAREST,
                )

        return TF.to_tensor(image), torch.from_numpy(np.array(mask, dtype=np.int64))


class PetSegmentation(Dataset):
    """Oxford-IIIT Pet images with their trimap remapped to 0/1/2."""

    def __init__(self, root: str, split: str, download: bool = True) -> None:
        self.base = datasets.OxfordIIITPet(
            root=root, split=split, target_types="segmentation", download=download
        )
        lookup = np.zeros(256, dtype=np.int64)
        for raw, cls in PET_TRIMAP_TO_CLASS.items():
            lookup[raw] = cls
        self.lookup = lookup

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, trimap = self.base[index]
        mask = Image.fromarray(self.lookup[np.array(trimap)].astype(np.uint8))
        return image.convert("RGB"), mask


class ShapesSegmentation(Dataset):
    """Generated shapes with pixel-exact masks; deterministic given ``seed``.

    Each image is drawn from its own seeded generator, so the dataset is
    reproducible without storing anything on disk. Shapes are allowed to overlap
    and later ones paint over earlier ones, which is what stops the task from
    being solvable by counting colours.
    """

    def __init__(
        self,
        n: int,
        size: int = 64,
        seed: int = 0,
        max_shapes: int = 3,
        num_classes: int = 4,
    ) -> None:
        self.n = n
        self.size = size
        self.seed = seed
        self.max_shapes = max_shapes
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def _canvas(self, rng: np.random.Generator) -> np.ndarray:
        """Low-frequency coloured background plus noise, so the task is not trivial."""
        coarse = rng.uniform(0.15, 0.55, size=(3, 4, 4)).astype(np.float32)
        background = np.array(
            Image.fromarray((coarse.transpose(1, 2, 0) * 255).astype(np.uint8)).resize(
                (self.size, self.size), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        noise = rng.normal(0, 0.03, size=background.shape).astype(np.float32)
        return np.clip(background + noise, 0, 1)

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        size = self.size
        image = self._canvas(rng)
        mask = np.zeros((size, size), dtype=np.int64)

        ys, xs = np.mgrid[0:size, 0:size]
        for _ in range(rng.integers(1, self.max_shapes + 1)):
            kind = int(rng.integers(1, self.num_classes))  # 0 is background
            colour = rng.uniform(0.55, 1.0, size=3).astype(np.float32)
            radius = rng.integers(size // 8, size // 4)
            cx = rng.integers(radius, size - radius)
            cy = rng.integers(radius, size - radius)

            if kind == 1:  # circle
                area = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
            elif kind == 2:  # rectangle
                area = (np.abs(xs - cx) <= radius) & (np.abs(ys - cy) <= radius)
            else:  # upward triangle: inside two slanted edges and above the base
                area = (
                    (ys >= cy - radius)
                    & (ys <= cy + radius)
                    & (np.abs(xs - cx) <= (ys - (cy - radius)) / 2 + 1)
                )

            image[area] = colour
            mask[area] = kind

        return (
            Image.fromarray((image * 255).astype(np.uint8)),
            Image.fromarray(mask.astype(np.uint8)),
        )


DATASETS = {
    "shapes": {
        "channels": 3,
        "size": 64,
        "classes": 4,
        "class_names": SHAPE_CLASS_NAMES,
        "flip": True,
        "download_mb": 0,
    },
    "oxford-pets": {
        "channels": 3,
        "size": 96,
        "classes": 3,
        "class_names": PET_CLASS_NAMES,
        "flip": True,
        "download_mb": 810,
    },
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return dict(DATASETS[name])


class PairedDataset(Dataset):
    """Applies a :class:`JointTransform` to a base yielding ``(image, mask)`` PILs."""

    def __init__(self, base: Dataset, transform: JointTransform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, mask = self.base[index]
        return self.transform(image, mask)


def download(name: str, root: str = "data") -> None:
    if name == "oxford-pets":
        for split in ("trainval", "test"):
            PetSegmentation(root, split, download=True)


def get_splits(
    name: str = "shapes",
    root: str = "data",
    val_split: float = 0.1,
    train_subset: int | None = None,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train_base, val_base, test_base, info)`` of ``(PIL, PIL)`` pairs."""
    info = dataset_info(name)

    if synthetic:
        info["size"] = 32
        return (
            ShapesSegmentation(48, 32, seed=1, num_classes=info["classes"]),
            ShapesSegmentation(24, 32, seed=2, num_classes=info["classes"]),
            ShapesSegmentation(24, 32, seed=3, num_classes=info["classes"]),
            info,
        )

    if name == "shapes":
        # Distinct seeds give genuinely disjoint splits from the same generator.
        n_train = train_subset or 4000
        return (
            ShapesSegmentation(n_train, info["size"], seed=1),
            ShapesSegmentation(500, info["size"], seed=2),
            ShapesSegmentation(1000, info["size"], seed=3),
            info,
        )

    full_train = PetSegmentation(root, "trainval", download=True)
    test_base = PetSegmentation(root, "test", download=True)

    n_val = int(len(full_train) * val_split)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    if train_subset:
        train_idx = train_idx[:train_subset]

    return Subset(full_train, train_idx), Subset(full_train, val_idx), test_base, info


def make_loader(
    base: Dataset,
    info: dict,
    image_size: int,
    batch_size: int = 32,
    shuffle: bool = False,
    augment: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    transform = JointTransform(image_size, augment=augment, flip=info["flip"])
    return DataLoader(
        PairedDataset(base, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def class_pixel_counts(loader, num_classes: int) -> torch.Tensor:
    """Pixels per class over a loader - the class imbalance, in one tensor."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, masks in loader:
        counts += torch.bincount(masks.flatten(), minlength=num_classes)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or inspect a segmentation dataset.")
    parser.add_argument("--dataset", default="shapes", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--preview", action="store_true", help="save outputs/data_preview.png")
    args = parser.parse_args()

    download(args.dataset, args.root)
    train_base, val_base, test_base, info = get_splits(args.dataset, args.root)
    size = args.image_size or info["size"]

    print(f"dataset : {args.dataset}")
    if info["download_mb"]:
        print(f"stored  : {Path(args.root).resolve()}")
    print(f"train   : {len(train_base)}")
    print(f"val     : {len(val_base)}")
    print(f"test    : {len(test_base)}")
    print(f"image   : {info['channels']}x{size}x{size} in [0, 1]")
    print(f"classes : {info['classes']}  {info['class_names']}")

    loader = make_loader(val_base, info, size, batch_size=16, num_workers=0)
    counts = class_pixel_counts(loader, info["classes"])
    share = counts.float() / counts.sum().clamp_min(1)
    print("\npixel share per class (the imbalance every metric has to survive):")
    for name, fraction in zip(info["class_names"], share.tolist()):
        print(f"  {name:12s} {fraction:6.2%}")

    if args.preview:
        from utils import plot_segmentation_rows

        images, masks = next(iter(loader))
        plot_segmentation_rows(
            images[:8], masks[:8], None, info["classes"], "outputs/data_preview.png",
            title=f"{args.dataset}: image, ground-truth mask, overlay",
        )
        print("\npreview -> outputs/data_preview.png")


if __name__ == "__main__":
    main()
