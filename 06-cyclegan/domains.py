"""Builds the two unpaired domains that CycleGAN translates between."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from common import data as data_mod


def build_domains(
    task: str,
    dataset: str,
    data_root: str,
    image_size: int,
    train: bool,
    domain_a: str,
    domain_b: str,
    subset: int = 0,
) -> Tuple[Dataset, Dataset, int, str, str]:
    """Return ``(dataset_a, dataset_b, channels, label_a, label_b)``.

    ``task="classes"``  -- two class subsets of one labelled dataset, e.g. the
                           horse and deer classes of CIFAR-10.
    ``task="datasets"`` -- two different datasets entirely, e.g. MNIST digits
                           and Fashion-MNIST garments.
    ``task="horse2zebra"`` -- the original CycleGAN dataset, if downloaded.
    """
    if task == "classes":
        meta = data_mod.info(dataset)
        base = data_mod.get_dataset(dataset, root=data_root, image_size=image_size, train=train)
        idx_a, idx_b = int(domain_a), int(domain_b)
        set_a = data_mod.class_subset(base, [idx_a])
        set_b = data_mod.class_subset(base, [idx_b])
        names = meta.class_names or tuple(str(i) for i in range(meta.num_classes))
        return limit(set_a, subset), limit(set_b, subset), meta.channels, names[idx_a], names[idx_b]

    if task == "datasets":
        set_a = data_mod.get_dataset(domain_a, root=data_root, image_size=image_size, train=train)
        set_b = data_mod.get_dataset(domain_b, root=data_root, image_size=image_size, train=train)
        channels = data_mod.info(domain_a).channels
        if channels != data_mod.info(domain_b).channels:
            raise ValueError(
                f"{domain_a} has {channels} channels but {domain_b} has "
                f"{data_mod.info(domain_b).channels}; pick two datasets that match"
            )
        return limit(set_a, subset), limit(set_b, subset), channels, domain_a, domain_b

    if task == "horse2zebra":
        split = "train" if train else "test"
        root = Path(data_root) / "horse2zebra"
        dir_a, dir_b = root / f"{split}A", root / f"{split}B"
        if not dir_a.is_dir():
            raise FileNotFoundError(
                f"{dir_a} not found. Run: python download_data.py --horse2zebra"
            )
        return (
            limit(data_mod.ImageFolderFlat(dir_a, image_size), subset),
            limit(data_mod.ImageFolderFlat(dir_b, image_size), subset),
            3,
            "horse",
            "zebra",
        )

    raise ValueError(f"unknown task {task!r}")


def limit(dataset: Dataset, subset: int) -> Dataset:
    """Evenly spread subset of a domain, so runtimes stay predictable."""
    if not subset or subset <= 0 or subset >= len(dataset):
        return dataset
    stride = len(dataset) / subset
    return Subset(dataset, [int(i * stride) for i in range(subset)])


def unpaired_loader(
    set_a: Dataset,
    set_b: Dataset,
    batch_size: int,
    train: bool,
    num_workers: int = 2,
    seed: int = 0,
) -> DataLoader:
    """Loader yielding ``(image_from_A, image_from_B)`` with no correspondence."""
    paired = data_mod.UnpairedDataset(set_a, set_b, seed=seed)
    return DataLoader(
        paired,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
