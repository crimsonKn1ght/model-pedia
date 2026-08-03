"""Datasets for masked image modelling.

    python data.py --dataset cifar10
    python data.py --dataset folder --root path/to/tiny-imagenet-200

Two things differ from the other projects here.

**Images stay in ``[0, 1]``** - no mean/std normalisation. The model predicts
pixels, so the loss, the PSNR and the reconstruction figures are all easier to
read in the units the pixels actually have. MAE normalises *per patch* inside the
loss instead, which is a better idea than a global normalisation anyway.

**Pretraining augmentation is deliberately weak** - a random resized crop and a
flip, nothing else. Masking is already an aggressive corruption; contrastive
methods need heavy colour augmentation because their task would otherwise be
solvable from a colour histogram, but MAE has no such shortcut to close off.

``--dataset folder`` reads any ``ImageFolder`` layout (``train/<class>/*``,
``val/<class>/*``), which is how to run Tiny ImageNet or a dataset of your own.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _train_test_builder(cls):
    def build(root, split, transform):
        return cls(root=root, train=(split == "train"), download=True, transform=transform)

    return build


def _stl10_builder(root, split, transform):
    return datasets.STL10(root=root, split=split, download=True, transform=transform)


def _folder_builder(root, split, transform):
    directory = Path(root) / ("train" if split == "train" else "val")
    if not directory.is_dir():
        raise FileNotFoundError(
            f"expected {directory} to exist. --dataset folder wants an ImageFolder layout: "
            "<root>/train/<class>/*.jpg and <root>/val/<class>/*.jpg"
        )
    return datasets.ImageFolder(str(directory), transform=transform)


CIFAR10_CLASSES = [
    "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
]
FASHION_CLASSES = [
    "tshirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "boot",
]

DATASETS = {
    "cifar10": {
        "builder": _train_test_builder(datasets.CIFAR10),
        "channels": 3,
        "size": 32,
        "patch": 4,
        "classes": 10,
        "class_names": CIFAR10_CLASSES,
        "flip": True,
        "download_mb": 170,
    },
    "cifar100": {
        "builder": _train_test_builder(datasets.CIFAR100),
        "channels": 3,
        "size": 32,
        "patch": 4,
        "classes": 100,
        "class_names": None,
        "flip": True,
        "download_mb": 170,
    },
    "fashion-mnist": {
        "builder": _train_test_builder(datasets.FashionMNIST),
        "channels": 1,
        "size": 28,
        "patch": 4,
        "classes": 10,
        "class_names": FASHION_CLASSES,
        "flip": True,
        "download_mb": 30,
    },
    "mnist": {
        "builder": _train_test_builder(datasets.MNIST),
        "channels": 1,
        "size": 28,
        "patch": 4,
        "classes": 10,
        "class_names": [str(d) for d in range(10)],
        "flip": False,
        "download_mb": 12,
    },
    "stl10": {
        "builder": _stl10_builder,
        "channels": 3,
        "size": 64,
        "patch": 8,
        "classes": 10,
        "class_names": None,
        "flip": True,
        "download_mb": 2600,
    },
    "folder": {
        "builder": _folder_builder,
        "channels": 3,
        "size": 64,
        "patch": 8,
        "classes": None,  # discovered from the directory listing
        "class_names": None,
        "flip": True,
        "download_mb": 0,
    },
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return dict(DATASETS[name])


def pretrain_transform(info: dict, image_size: int):
    """Random resized crop and flip. Nothing photometric - masking is the task."""
    steps = [transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0))]
    if info["flip"]:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def train_transform(info: dict, image_size: int, augment: bool = True):
    """Downstream-classifier training pipeline."""
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        steps.append(transforms.RandomCrop(image_size, padding=image_size // 8))
        if info["flip"]:
            steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def eval_transform(info: dict, image_size: int):
    return transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )


class TransformDataset(Dataset):
    """Applies a transform to a base dataset that yields PIL images."""

    def __init__(self, base: Dataset, transform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        return self.transform(image), int(target)


class SyntheticBase(Dataset):
    """PIL-image stand-in for ``--smoke-test``; no download.

    Each class gets a different low-frequency pattern so that masked
    reconstruction is at least possible in principle.
    """

    def __init__(self, n: int, channels: int, size: int, num_classes: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (n,), generator=generator)
        ramp = torch.linspace(0, 1, size)
        grid = ramp.view(1, -1) + ramp.view(-1, 1)
        images = []
        for label in self.labels.tolist():
            phase = label / max(num_classes, 1)
            pattern = (grid * 3.14159 + phase * 6.28).sin() * 0.4 + 0.5
            noise = torch.rand(channels, size, size, generator=generator) * 0.2
            images.append((pattern.expand(channels, size, size) + noise).clamp(0, 1))
        self.images = torch.stack(images)
        self.to_pil = transforms.ToPILImage()

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, index: int):
        return self.to_pil(self.images[index]), int(self.labels[index])


def download(name: str, root: str = "data") -> None:
    info = dataset_info(name)
    for split in ("train", "test"):
        info["builder"](root, split, None)


def get_splits(
    name: str = "cifar10",
    root: str = "data",
    val_split: float = 0.05,
    train_subset: int | None = None,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train_base, val_base, test_base, info)`` of untransformed images."""
    info = dataset_info(name)
    if synthetic:
        info["classes"] = info["classes"] or 10
        size, patch = 16, 4
        info["size"], info["patch"] = size, patch
        return (
            SyntheticBase(96, info["channels"], size, info["classes"], seed),
            SyntheticBase(48, info["channels"], size, info["classes"], seed + 1),
            SyntheticBase(48, info["channels"], size, info["classes"], seed + 2),
            info,
        )

    full_train = info["builder"](root, "train", None)
    test_base = info["builder"](root, "test", None)
    if info["classes"] is None:
        info["classes"] = len(full_train.classes)
        info["class_names"] = list(full_train.classes)

    n_val = int(len(full_train) * val_split)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    if train_subset:
        train_idx = train_idx[:train_subset]

    return Subset(full_train, train_idx), Subset(full_train, val_idx), test_base, info


def make_loader(
    base: Dataset,
    transform,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        TransformDataset(base, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a dataset for masked image modelling.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    args = parser.parse_args()

    if args.dataset != "folder":
        download(args.dataset, args.root)
    train_base, val_base, test_base, info = get_splits(args.dataset, args.root)

    print(f"dataset  : {args.dataset}")
    print(f"stored   : {Path(args.root).resolve()}")
    print(f"train    : {len(train_base)}  (labels unused during pretraining)")
    print(f"val      : {len(val_base)}")
    print(f"test     : {len(test_base)}")
    print(f"image    : {info['channels']}x{info['size']}x{info['size']} in [0, 1]")
    grid = info["size"] // info["patch"]
    print(f"patches  : {grid}x{grid} = {grid * grid} of size {info['patch']}")
    print(f"classes  : {info['classes']}")


if __name__ == "__main__":
    main()
