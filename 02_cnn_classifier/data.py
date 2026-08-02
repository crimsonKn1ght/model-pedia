"""Dataset download and loading for CIFAR-10 / CIFAR-100 / SVHN.

Run this file directly to fetch a dataset before training:

    python data.py --dataset cifar10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _cifar_builder(cls):
    def build(root, train, transform):
        return cls(root=root, train=train, download=True, transform=transform)

    return build


def _svhn_builder(root, train, transform):
    # torchvision's SVHN takes split= rather than train=.
    return datasets.SVHN(
        root=root, split="train" if train else "test", download=True, transform=transform
    )


DATASETS = {
    "cifar10": {
        "builder": _cifar_builder(datasets.CIFAR10),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "classes": CIFAR10_CLASSES,
        "hflip": True,
    },
    "cifar100": {
        "builder": _cifar_builder(datasets.CIFAR100),
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "classes": None,  # filled in from the downloaded metadata
        "hflip": True,
    },
    "svhn": {
        "builder": _svhn_builder,
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
        "classes": [str(i) for i in range(10)],
        # House numbers have a canonical orientation, so no horizontal flip.
        "hflip": False,
    },
}

IMAGE_SHAPE = (3, 32, 32)


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return DATASETS[name]


def build_transforms(name: str, augment: bool):
    """Standard CIFAR recipe: pad-and-crop, optional flip, normalise."""
    info = dataset_info(name)
    steps = []
    if augment:
        steps.append(transforms.RandomCrop(32, padding=4, padding_mode="reflect"))
        if info["hflip"]:
            steps.append(transforms.RandomHorizontalFlip())
    steps += [transforms.ToTensor(), transforms.Normalize(info["mean"], info["std"])]
    return transforms.Compose(steps)


def download(name: str, root: str = "data") -> None:
    builder = dataset_info(name)["builder"]
    for train in (True, False):
        builder(root, train, transforms.ToTensor())


def _class_names(name: str, dataset) -> list[str]:
    declared = dataset_info(name)["classes"]
    if declared is not None:
        return list(declared)
    return list(getattr(dataset, "classes", [str(i) for i in range(100)]))


def _synthetic_split(num_classes: int, n_train: int, n_val: int, n_test: int, seed: int):
    """Tiny random stand-in used by the --smoke-test flag (no download needed)."""
    generator = torch.Generator().manual_seed(seed)

    def make(n: int) -> TensorDataset:
        x = torch.randn(n, *IMAGE_SHAPE, generator=generator)
        y = torch.randint(0, num_classes, (n,), generator=generator)
        x += y.view(-1, 1, 1, 1).float() * 0.05
        return TensorDataset(x, y)

    return make(n_train), make(n_val), make(n_test)


def get_datasets(
    name: str = "cifar10",
    root: str = "data",
    val_split: float = 0.1,
    augment: bool = True,
    train_subset: int | None = None,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train, val, test, class_names)``.

    ``train_subset`` caps the number of training images, which is the main knob
    for keeping a CPU run short. Pass ``None`` or ``0`` to use everything.
    """
    info = dataset_info(name)
    if synthetic:
        n_classes = 100 if name == "cifar100" else 10
        train_ds, val_ds, test_ds = _synthetic_split(n_classes, 256, 64, 64, seed)
        return train_ds, val_ds, test_ds, [str(i) for i in range(n_classes)]

    builder = info["builder"]
    full_train = builder(root, True, build_transforms(name, augment))
    full_train_eval = builder(root, True, build_transforms(name, augment=False))
    test_ds = builder(root, False, build_transforms(name, augment=False))

    n_val = int(len(full_train) * val_split)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    if train_subset:
        train_idx = train_idx[:train_subset]

    return (
        Subset(full_train, train_idx),
        Subset(full_train_eval, val_idx),
        test_ds,
        _class_names(name, test_ds),
    )


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
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, **common),
        DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, **common),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 / CIFAR-100 / SVHN.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    args = parser.parse_args()

    download(args.dataset, args.root)
    train_ds, val_ds, test_ds, classes = get_datasets(args.dataset, args.root)
    print(f"dataset : {args.dataset}")
    print(f"stored  : {Path(args.root).resolve()}")
    print(f"train   : {len(train_ds)}")
    print(f"val     : {len(val_ds)}")
    print(f"test    : {len(test_ds)}")
    print(f"classes : {len(classes)} ({', '.join(classes[:10])}{' ...' if len(classes) > 10 else ''})")


if __name__ == "__main__":
    main()
