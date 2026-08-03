"""Classification datasets for the ViT/ResNet comparison.

    python data.py --dataset cifar10 --augment strong --preview

Augmentation is a first-class knob in this project rather than a detail. A convolution
has locality and translation-equivariance built in; a Vision Transformer has to learn
both from data, and on a dataset the size of CIFAR-10 there is not enough of it. Strong
augmentation is how the gap gets closed in practice, so ``--augment`` has three settings
and ``compare.py --study augment`` measures what each is worth.

| setting | what it does |
|---|---|
| ``none`` | resize only |
| ``basic`` | random resized crop and horizontal flip - the standard recipe |
| ``strong`` | adds colour jitter, random erasing and more aggressive crops |

Images are normalised per dataset, which matters more for the transformer than for the
ResNet: LayerNorm normalises across features within a token, so the input scale
propagates in a way batch normalisation would have absorbed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

AUGMENTATIONS = ("none", "basic", "strong")

CIFAR10_CLASSES = [
    "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
]
FASHION_CLASSES = [
    "tshirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "boot",
]


def _train_test_builder(cls):
    def build(root, split, transform):
        return cls(root=root, train=(split == "train"), download=True, transform=transform)

    return build


def _flowers_builder(root, split, transform):
    return datasets.Flowers102(
        root=root, split="train" if split == "train" else "test", download=True, transform=transform
    )


DATASETS = {
    "cifar10": {
        "builder": _train_test_builder(datasets.CIFAR10),
        "channels": 3, "size": 32, "classes": 10, "class_names": CIFAR10_CLASSES,
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616),
        "flip": True, "download_mb": 170,
    },
    "cifar100": {
        "builder": _train_test_builder(datasets.CIFAR100),
        "channels": 3, "size": 32, "classes": 100, "class_names": None,
        "mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762),
        "flip": True, "download_mb": 170,
    },
    "flowers102": {
        "builder": _flowers_builder,
        "channels": 3, "size": 64, "classes": 102, "class_names": None,
        "mean": (0.4355, 0.3777, 0.2880), "std": (0.2927, 0.2440, 0.2716),
        "flip": True, "download_mb": 345,
    },
    "fashion-mnist": {
        "builder": _train_test_builder(datasets.FashionMNIST),
        "channels": 1, "size": 32, "classes": 10, "class_names": FASHION_CLASSES,
        "mean": (0.2860,), "std": (0.3530,), "flip": True, "download_mb": 30,
    },
    "mnist": {
        "builder": _train_test_builder(datasets.MNIST),
        "channels": 1, "size": 32, "classes": 10,
        "class_names": [str(d) for d in range(10)],
        "mean": (0.1307,), "std": (0.3081,), "flip": False, "download_mb": 12,
    },
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return dict(DATASETS[name])


def build_transform(info: dict, size: int, augment: str = "basic"):
    if augment not in AUGMENTATIONS:
        raise ValueError(f"unknown augmentation {augment!r}, expected {AUGMENTATIONS}")

    steps = [transforms.Resize((size, size))]
    if augment in ("basic", "strong"):
        scale = (0.6, 1.0) if augment == "strong" else (0.8, 1.0)
        steps.append(transforms.RandomResizedCrop(size, scale=scale))
        if info["flip"]:
            steps.append(transforms.RandomHorizontalFlip())
    if augment == "strong" and info["channels"] == 3:
        steps.append(transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.7))

    steps += [transforms.ToTensor(), transforms.Normalize(info["mean"], info["std"])]
    if augment == "strong":
        # Random erasing acts on the tensor, so it has to come after ToTensor.
        steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return transforms.Compose(steps)


def denormalize(images: torch.Tensor, info: dict) -> torch.Tensor:
    mean = torch.tensor(info["mean"]).view(1, -1, 1, 1).to(images.device)
    std = torch.tensor(info["std"]).view(1, -1, 1, 1).to(images.device)
    return (images * std + mean).clamp(0, 1)


def download(name: str, root: str = "data") -> None:
    info = dataset_info(name)
    for split in ("train", "test"):
        info["builder"](root, split, None)


def _synthetic_split(channels: int, size: int, num_classes: int, seed: int):
    def make(n: int, offset: int) -> TensorDataset:
        generator = torch.Generator().manual_seed(seed + offset)
        labels = torch.randint(0, num_classes, (n,), generator=generator)
        ramp = torch.linspace(-1, 1, size)
        grid = ramp.view(1, -1) + ramp.view(-1, 1)
        images = []
        for label in labels.tolist():
            pattern = (grid * 3.1416 + label).sin()
            noise = torch.randn(channels, size, size, generator=generator) * 0.2
            images.append(pattern.expand(channels, size, size) + noise)
        return TensorDataset(torch.stack(images), labels)

    return make(160, 0), make(64, 1), make(64, 2)


def get_splits(
    name: str = "cifar10",
    root: str = "data",
    image_size: int | None = None,
    val_split: float = 0.05,
    train_subset: int | None = None,
    augment: str = "basic",
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train, val, test, info)``; only the train split is augmented."""
    info = dataset_info(name)
    if synthetic:
        info["size"] = 16
        train, val, test = _synthetic_split(info["channels"], 16, info["classes"], seed)
        return train, val, test, info

    size = image_size or info["size"]
    info["size"] = size
    builder = info["builder"]
    full_train = builder(root, "train", build_transform(info, size, augment))
    full_train_eval = builder(root, "train", build_transform(info, size, "none"))
    test_set = builder(root, "test", build_transform(info, size, "none"))

    n_val = int(len(full_train) * val_split)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    if train_subset:
        train_idx = train_idx[:train_subset]

    return Subset(full_train, train_idx), Subset(full_train_eval, val_idx), test_set, info


def make_loader(
    dataset: Dataset, batch_size: int = 128, shuffle: bool = False, num_workers: int = 2
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or inspect a classification dataset.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--augment", default="basic", choices=AUGMENTATIONS)
    parser.add_argument("--preview", action="store_true", help="save outputs/data_preview.png")
    args = parser.parse_args()

    download(args.dataset, args.root)
    train, val, test, info = get_splits(
        args.dataset, args.root, image_size=args.image_size, augment=args.augment
    )
    print(f"dataset : {args.dataset}")
    print(f"stored  : {Path(args.root).resolve()}")
    print(f"train   : {len(train)}")
    print(f"val     : {len(val)}")
    print(f"test    : {len(test)}")
    print(f"image   : {info['channels']}x{info['size']}x{info['size']}, normalised")
    print(f"classes : {info['classes']}")
    print(f"augment : {args.augment}")

    if args.preview:
        from utils import plot_image_grid

        images = torch.stack([train[i][0] for i in range(16)])
        plot_image_grid(denormalize(images, info), "outputs/data_preview.png", columns=8,
                        title=f"{args.dataset} under '{args.augment}' augmentation")
        print("\npreview -> outputs/data_preview.png")


if __name__ == "__main__":
    main()
