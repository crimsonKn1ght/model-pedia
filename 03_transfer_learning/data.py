"""Dataset download and loading for Oxford Flowers-102 and Oxford-IIIT Pets.

    python data.py --dataset flowers102

Both are small fine-grained datasets, which is exactly the setting transfer
learning is for: 102 flower species with **ten training images each** is
hopeless from scratch and straightforward from pretrained features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from model import IMAGENET_MEAN, IMAGENET_STD

DATASETS = {
    "flowers102": {
        "num_classes": 102,
        # 1020 train / 1020 val / 6149 test, 10 training images per class.
        "download_mb": 345,
    },
    "pets": {
        "num_classes": 37,
        # 3680 trainval / 3669 test.
        "download_mb": 800,
    },
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return DATASETS[name]


def build_transforms(image_size: int = 224, augment: bool = False):
    """ImageNet-style preprocessing; the normalisation must match the backbone."""
    if augment:
        steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        steps = [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def _flowers(root, split, transform):
    return datasets.Flowers102(root=root, split=split, download=True, transform=transform)


def _pets(root, split, transform):
    return datasets.OxfordIIITPet(
        root=root, split=split, target_types="category", download=True, transform=transform
    )


def download(name: str, root: str = "data") -> None:
    plain = transforms.ToTensor()
    if name == "flowers102":
        for split in ("train", "val", "test"):
            _flowers(root, split, plain)
    else:
        for split in ("trainval", "test"):
            _pets(root, split, plain)


def _class_names(name: str, dataset) -> list[str]:
    classes = getattr(dataset, "classes", None)
    if classes:
        return list(classes)
    # torchvision ships no species names for Flowers-102.
    return [f"class_{i:03d}" for i in range(dataset_info(name)["num_classes"])]


def _synthetic_split(num_classes: int, image_size: int, seed: int):
    """Tiny random stand-in used by the --smoke-test flag (no download needed)."""
    generator = torch.Generator().manual_seed(seed)

    def make(n: int) -> TensorDataset:
        x = torch.randn(n, 3, image_size, image_size, generator=generator)
        y = torch.randint(0, num_classes, (n,), generator=generator)
        return TensorDataset(x, y)

    return make(48), make(24), make(24)


def get_datasets(
    name: str = "flowers102",
    root: str = "data",
    image_size: int = 224,
    augment: bool = True,
    val_split: float = 0.2,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train, val, test, class_names)``.

    Flowers-102 ships an official validation split, so it is used as-is. Pets
    only has trainval/test, so a validation slice is carved out of trainval.
    """
    if synthetic:
        n_classes = min(dataset_info(name)["num_classes"], 10)
        train_ds, val_ds, test_ds = _synthetic_split(n_classes, image_size, seed)
        return train_ds, val_ds, test_ds, [f"class_{i:03d}" for i in range(n_classes)]

    train_tf = build_transforms(image_size, augment=augment)
    eval_tf = build_transforms(image_size, augment=False)

    if name == "flowers102":
        train_ds = _flowers(root, "train", train_tf)
        val_ds = _flowers(root, "val", eval_tf)
        test_ds = _flowers(root, "test", eval_tf)
        return train_ds, val_ds, test_ds, _class_names(name, test_ds)

    trainval = _pets(root, "trainval", train_tf)
    trainval_eval = _pets(root, "trainval", eval_tf)
    test_ds = _pets(root, "test", eval_tf)

    n_val = int(len(trainval) * val_split)
    indices = torch.randperm(len(trainval), generator=torch.Generator().manual_seed(seed))
    val_ds = Subset(trainval_eval, indices[:n_val].tolist())
    train_ds = Subset(trainval, indices[n_val:].tolist())
    return train_ds, val_ds, test_ds, _class_names(name, test_ds)


def get_dataloaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    num_workers: int = 2,
    shuffle_train: bool = True,
):
    eval_batch_size = eval_batch_size or batch_size * 2
    common = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, **common),
        DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, **common),
        DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, **common),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Flowers-102 / Oxford-IIIT Pets.")
    parser.add_argument("--dataset", default="flowers102", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    args = parser.parse_args()

    info = dataset_info(args.dataset)
    print(f"downloading {args.dataset} (~{info['download_mb']} MB), this takes a while ...")
    download(args.dataset, args.root)
    train_ds, val_ds, test_ds, classes = get_datasets(args.dataset, args.root)
    print(f"stored  : {Path(args.root).resolve()}")
    print(f"train   : {len(train_ds)}  ({len(train_ds) / len(classes):.1f} images per class)")
    print(f"val     : {len(val_ds)}")
    print(f"test    : {len(test_ds)}")
    print(f"classes : {len(classes)}")


if __name__ == "__main__":
    main()
