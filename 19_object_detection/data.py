"""Datasets for detection, and the collate function boxes force you to write.

    python data.py --dataset digits          # 12 MB download, builds scenes from MNIST
    python data.py --dataset voc2007         # ~880 MB download, 20 real classes

Detection data is awkward in a way classification data is not: every image has a
*different number* of targets, so the batch cannot be a tensor. ``collate`` below
stacks the images and leaves the annotations as a list of dicts, which is the
convention torchvision's own detection models use.

``digits`` composes scenes from MNIST: one to three digits, each scaled randomly,
pasted at random positions on a textured canvas, with the box taken from the
digit's own non-zero pixels. Two things make it a good first detection dataset -
the boxes are exact rather than hand-drawn, so nothing in the metric is annotation
noise; and it is small enough that a detector reaches a meaningful mAP in a couple
of minutes, which is what makes the mAP implementation worth trusting.

``voc2007`` is the real benchmark. Expect it to need a GPU and a lot more epochs to
produce numbers worth quoting; the code path is here so the pipeline is not a toy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import functional as TF

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


def collate(batch):
    """Stack images, keep annotations as a list - they have ragged lengths."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def _tight_box(patch: torch.Tensor, threshold: float = 0.15):
    """Bounding box of a patch's ink, or ``None`` if it is blank."""
    rows = (patch.max(dim=1).values > threshold).nonzero().flatten()
    cols = (patch.max(dim=0).values > threshold).nonzero().flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    overlap_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = overlap_w * overlap_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / max(union, 1e-9)


class DigitScatter(Dataset):
    """Scenes of scaled MNIST digits on a textured canvas, with exact boxes."""

    def __init__(
        self,
        digits: torch.Tensor,
        labels: torch.Tensor,
        n: int,
        canvas: int = 96,
        seed: int = 0,
        max_objects: int = 3,
        min_size: int = 16,
        max_size: int = 40,
        max_overlap: float = 0.35,
    ) -> None:
        self.digits = digits  # (N, 28, 28) uint8
        self.labels = labels
        self.n = n
        self.canvas = canvas
        self.seed = seed
        self.max_objects = max_objects
        self.min_size = min_size
        self.max_size = max_size
        self.max_overlap = max_overlap

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        canvas = self.canvas
        # Faint texture, so the detector cannot find objects by "anything not black".
        image = torch.from_numpy(
            rng.uniform(0.02, 0.18, size=(1, canvas, canvas)).astype(np.float32)
        )

        boxes, labels = [], []
        for _ in range(int(rng.integers(1, self.max_objects + 1))):
            source = int(rng.integers(0, self.digits.size(0)))
            size = int(rng.integers(self.min_size, self.max_size + 1))
            patch = F.interpolate(
                self.digits[source].float().div(255).view(1, 1, 28, 28),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            local = _tight_box(patch)
            if local is None:
                continue

            placed = False
            for _attempt in range(8):
                top = int(rng.integers(0, canvas - size + 1))
                left = int(rng.integers(0, canvas - size + 1))
                box = (local[0] + left, local[1] + top, local[2] + left, local[3] + top)
                if all(_iou(box, existing) <= self.max_overlap for existing in boxes):
                    placed = True
                    break
            if not placed:
                continue

            region = image[0, top : top + size, left : left + size]
            # Maximum blending, so overlapping digits stay individually visible.
            image[0, top : top + size, left : left + size] = torch.maximum(region, patch)
            boxes.append(box)
            labels.append(int(self.labels[source]))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return image.clamp(0, 1), target


class RandomShapes(Dataset):
    """No-download stand-in for ``--smoke-test``: bright rectangles, exact boxes."""

    def __init__(self, n: int, canvas: int = 48, num_classes: int = 10, seed: int = 0) -> None:
        self.n = n
        self.canvas = canvas
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed * 7919 + index)
        canvas = self.canvas
        image = torch.from_numpy(
            rng.uniform(0.0, 0.15, size=(1, canvas, canvas)).astype(np.float32)
        )
        boxes, labels = [], []
        for _ in range(int(rng.integers(1, 3))):
            width = int(rng.integers(8, canvas // 2))
            height = int(rng.integers(8, canvas // 2))
            left = int(rng.integers(0, canvas - width))
            top = int(rng.integers(0, canvas - height))
            image[0, top : top + height, left : left + width] = float(rng.uniform(0.6, 1.0))
            boxes.append((left, top, left + width, top + height))
            labels.append(int(rng.integers(0, self.num_classes)))
        return image.clamp(0, 1), {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class VOCBoxes(Dataset):
    """Pascal VOC 2007 detection, resized to a square with the boxes rescaled."""

    def __init__(self, root: str, image_set: str, image_size: int = 160, download: bool = True):
        self.base = datasets.VOCDetection(
            root=root, year="2007", image_set=image_set, download=download
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, annotation = self.base[index]
        width, height = image.size
        scale_x = self.image_size / width
        scale_y = self.image_size / height

        boxes, labels = [], []
        objects = annotation["annotation"]["object"]
        if isinstance(objects, dict):  # a single object is not wrapped in a list
            objects = [objects]
        for obj in objects:
            box = obj["bndbox"]
            boxes.append(
                (
                    float(box["xmin"]) * scale_x,
                    float(box["ymin"]) * scale_y,
                    float(box["xmax"]) * scale_x,
                    float(box["ymax"]) * scale_y,
                )
            )
            labels.append(VOC_CLASSES.index(obj["name"]))

        # A plain square resize distorts the aspect ratio. Letterboxing would keep
        # it, at the cost of padding logic in every figure; this is the simpler
        # trade and it is applied identically to images and boxes.
        resized = TF.to_tensor(
            TF.resize(image, [self.image_size, self.image_size]).convert("RGB")
        )
        return resized, {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


DATASETS = {
    "digits": {
        "channels": 1,
        "size": 96,
        "classes": 10,
        "class_names": [str(digit) for digit in range(10)],
        "download_mb": 12,
    },
    "voc2007": {
        "channels": 3,
        "size": 160,
        "classes": 20,
        "class_names": VOC_CLASSES,
        "download_mb": 880,
    },
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return dict(DATASETS[name])


def download(name: str, root: str = "data") -> None:
    if name == "digits":
        for train in (True, False):
            datasets.MNIST(root=root, train=train, download=True)
    elif name == "voc2007":
        for image_set in ("trainval", "test"):
            datasets.VOCDetection(root=root, year="2007", image_set=image_set, download=True)


def get_splits(
    name: str = "digits",
    root: str = "data",
    image_size: int | None = None,
    train_size: int = 4000,
    val_size: int = 500,
    test_size: int = 1000,
    max_objects: int = 3,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train, val, test, info)`` of ``(image, target)`` pairs."""
    info = dataset_info(name)
    if synthetic:
        info["size"] = 48
        return (
            RandomShapes(48, 48, info["classes"], seed=1),
            RandomShapes(24, 48, info["classes"], seed=2),
            RandomShapes(24, 48, info["classes"], seed=3),
            info,
        )

    size = image_size or info["size"]
    info["size"] = size

    if name == "digits":
        train_source = datasets.MNIST(root=root, train=True, download=True)
        test_source = datasets.MNIST(root=root, train=False, download=True)
        common = {"canvas": size, "max_objects": max_objects}
        # Train and val draw digits from the MNIST train split, test from its test
        # split, so no digit image is shared between the two.
        return (
            DigitScatter(train_source.data, train_source.targets, train_size, seed=1, **common),
            DigitScatter(train_source.data, train_source.targets, val_size, seed=2, **common),
            DigitScatter(test_source.data, test_source.targets, test_size, seed=3, **common),
            info,
        )

    full_train = VOCBoxes(root, "trainval", size, download=True)
    test_base = VOCBoxes(root, "test", size, download=True)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()
    if train_size:
        train_idx = train_idx[:train_size]
    if test_size and test_size < len(test_base):
        test_base = Subset(test_base, list(range(test_size)))
    return Subset(full_train, train_idx), Subset(full_train, val_idx), test_base, info


def make_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or inspect a detection dataset.")
    parser.add_argument("--dataset", default="digits", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--preview", action="store_true", help="save outputs/data_preview.png")
    args = parser.parse_args()

    download(args.dataset, args.root)
    train, val, test, info = get_splits(args.dataset, args.root, image_size=args.image_size)

    print(f"dataset : {args.dataset}")
    print(f"stored  : {Path(args.root).resolve()}")
    print(f"train   : {len(train)}")
    print(f"val     : {len(val)}")
    print(f"test    : {len(test)}")
    print(f"image   : {info['channels']}x{info['size']}x{info['size']} in [0, 1]")
    print(f"classes : {info['classes']}")

    loader = make_loader(val, batch_size=16, num_workers=0)
    images, targets = next(iter(loader))
    counts = torch.tensor([target["labels"].numel() for target in targets])
    sizes = torch.cat(
        [target["boxes"][:, 2:] - target["boxes"][:, :2] for target in targets if target["boxes"].numel()]
    )
    print(f"\nobjects per image : {counts.float().mean():.2f} (min {counts.min()}, max {counts.max()})")
    print(f"box size, pixels  : {sizes.mean(dim=0).tolist()} average width/height")

    if args.preview:
        from utils import plot_detections

        empty = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long),
                  "scores": torch.zeros(0)} for _ in targets]
        plot_detections(
            images[:8], empty[:8], targets[:8], info["class_names"], "outputs/data_preview.png",
            title=f"{args.dataset}: ground-truth boxes (bottom row is empty by design)",
        )
        print("\npreview -> outputs/data_preview.png")


if __name__ == "__main__":
    main()
