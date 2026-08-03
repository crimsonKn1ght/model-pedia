"""Dataset download and loading for MNIST / Fashion-MNIST.

Run this file directly to fetch a dataset before training:

    python data.py --dataset mnist
    python data.py --dataset fashion-mnist
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision import datasets, transforms

DATASETS = {
    "mnist": {
        "builder": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "classes": [str(i) for i in range(10)],
    },
    "fashion-mnist": {
        "builder": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "classes": [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    },
}

IMAGE_SHAPE = (1, 28, 28)


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return DATASETS[name]


def build_transforms(name: str, augment: bool = False):
    """Normalised tensors; optional mild affine jitter for the training split."""
    info = dataset_info(name)
    steps = []
    if augment:
        steps.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    steps += [transforms.ToTensor(), transforms.Normalize(info["mean"], info["std"])]
    return transforms.Compose(steps)


def download(name: str, root: str = "data") -> None:
    """Fetch both splits into ``root`` (a no-op once they are present)."""
    builder = dataset_info(name)["builder"]
    for train in (True, False):
        builder(root=root, train=train, download=True)


def _synthetic_split(num_classes: int, n_train: int, n_val: int, n_test: int, seed: int):
    """Tiny random stand-in used by the --smoke-test flag (no download needed)."""
    generator = torch.Generator().manual_seed(seed)

    def make(n: int) -> TensorDataset:
        x = torch.randn(n, *IMAGE_SHAPE, generator=generator)
        y = torch.randint(0, num_classes, (n,), generator=generator)
        # Give the labels a weak imprint on the images so the loss can move.
        x += y.view(-1, 1, 1, 1).float() * 0.1
        return TensorDataset(x, y)

    return make(n_train), make(n_val), make(n_test)


def get_datasets(
    name: str = "mnist",
    root: str = "data",
    val_split: float = 0.1,
    augment: bool = False,
    train_subset: int | None = None,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train, val, test, class_names)``.

    The validation split is carved out of the official training split, so the
    test split is only ever touched by ``evaluate.py``.
    """
    info = dataset_info(name)
    if synthetic:
        train_ds, val_ds, test_ds = _synthetic_split(len(info["classes"]), 256, 64, 64, seed)
        return train_ds, val_ds, test_ds, info["classes"]

    builder = info["builder"]
    full_train = builder(
        root=root, train=True, download=True, transform=build_transforms(name, augment)
    )
    # A second view of the same files, without augmentation, for validation.
    full_train_eval = builder(
        root=root, train=True, download=True, transform=build_transforms(name, augment=False)
    )
    test_ds = builder(
        root=root, train=False, download=True, transform=build_transforms(name, augment=False)
    )

    n_val = int(len(full_train) * val_split)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    if train_subset is not None:
        train_idx = train_idx[:train_subset]

    return Subset(full_train, train_idx), Subset(full_train_eval, val_idx), test_ds, info["classes"]


def get_dataloaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int = 128,
    eval_batch_size: int | None = None,
    num_workers: int = 2,
):
    eval_batch_size = eval_batch_size or batch_size * 2
    common = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, **common),
        DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, **common),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MNIST / Fashion-MNIST.")
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    args = parser.parse_args()

    download(args.dataset, args.root)
    train_ds, val_ds, test_ds, classes = get_datasets(args.dataset, args.root)
    print(f"dataset : {args.dataset}")
    print(f"stored  : {Path(args.root).resolve()}")
    print(f"train   : {len(train_ds)}")
    print(f"val     : {len(val_ds)}")
    print(f"test    : {len(test_ds)}")
    print(f"classes : {', '.join(classes)}")


if __name__ == "__main__":
    main()
