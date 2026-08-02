"""Dataset download and loading for the autoencoder.

    python data.py --dataset fashion-mnist

Everything is resized to 32x32 and left in ``[0, 1]``: no mean/std
normalisation, because the decoder ends in a sigmoid and PSNR/SSIM are defined
against a known data range. Labels are loaded too - not to train on, but to
colour the latent-space plot in ``evaluate.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

FASHION_CLASSES = [
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
]
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


def _builder(cls):
    def build(root, train, transform):
        return cls(root=root, train=train, download=True, transform=transform)

    return build


DATASETS = {
    "fashion-mnist": {
        "builder": _builder(datasets.FashionMNIST),
        "channels": 1,
        "classes": FASHION_CLASSES,
    },
    "mnist": {
        "builder": _builder(datasets.MNIST),
        "channels": 1,
        "classes": [str(i) for i in range(10)],
    },
    "cifar10": {
        "builder": _builder(datasets.CIFAR10),
        "channels": 3,
        "classes": CIFAR10_CLASSES,
    },
}

IMAGE_SIZE = 32


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return DATASETS[name]


def build_transforms():
    return transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])


def download(name: str, root: str = "data") -> None:
    builder = dataset_info(name)["builder"]
    for train in (True, False):
        builder(root, train, build_transforms())


def _synthetic_split(channels: int, seed: int):
    """Tiny random stand-in used by the --smoke-test flag (no download needed)."""
    generator = torch.Generator().manual_seed(seed)

    def make(n: int) -> TensorDataset:
        x = torch.rand(n, channels, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
        y = torch.randint(0, 10, (n,), generator=generator)
        return TensorDataset(x, y)

    return make(128), make(64), make(64)


def get_datasets(
    name: str = "fashion-mnist",
    root: str = "data",
    val_split: float = 0.1,
    train_subset: int | None = None,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train, val, test, info)`` where ``info`` carries channels and class names."""
    info = dataset_info(name)
    if synthetic:
        train_ds, val_ds, test_ds = _synthetic_split(info["channels"], seed)
        return train_ds, val_ds, test_ds, info

    builder = info["builder"]
    full_train = builder(root, True, build_transforms())
    test_ds = builder(root, False, build_transforms())

    n_val = int(len(full_train) * val_split)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    if train_subset:
        train_idx = train_idx[:train_subset]

    return Subset(full_train, train_idx), Subset(full_train, val_idx), test_ds, info


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
    parser = argparse.ArgumentParser(description="Download a dataset for the autoencoder.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    args = parser.parse_args()

    download(args.dataset, args.root)
    train_ds, val_ds, test_ds, info = get_datasets(args.dataset, args.root)
    print(f"dataset  : {args.dataset}")
    print(f"stored   : {Path(args.root).resolve()}")
    print(f"train    : {len(train_ds)}")
    print(f"val      : {len(val_ds)}")
    print(f"test     : {len(test_ds)}")
    print(f"image    : {info['channels']}x{IMAGE_SIZE}x{IMAGE_SIZE} in [0, 1]")


if __name__ == "__main__":
    main()
