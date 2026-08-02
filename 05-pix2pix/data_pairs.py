"""Builds the paired dataloaders used by ``train.py`` and ``evaluate.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset

from common import data as data_mod


def build_loader(
    task: str,
    dataset: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    train: bool,
    num_workers: int = 2,
    subset: int = 0,
) -> Tuple[DataLoader, int, int]:
    """Return ``(loader, in_channels, out_channels)`` for the requested task.

    ``task="edges"`` derives exact pairs from any image dataset.
    ``task="facades"`` reads the original side-by-side pix2pix layout.
    """
    if task == "edges":
        base = data_mod.get_dataset(
            dataset, root=data_root, image_size=image_size, train=train
        )
        paired = data_mod.EdgesToPhoto(base)
        channels = data_mod.info(dataset).channels
        in_ch, out_ch = 1, channels
    elif task == "facades":
        split = "train" if train else "val"
        directory = Path(data_root) / "facades" / split
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} not found. Run: python download_data.py --facades"
            )
        paired = data_mod.SideBySideDataset(directory, image_size=image_size, input_right=True)
        in_ch, out_ch = 3, 3
    else:
        raise ValueError(f"unknown task {task!r}")

    if subset and 0 < subset < len(paired):
        stride = len(paired) / subset
        paired = Subset(paired, [int(i * stride) for i in range(subset)])

    loader = DataLoader(
        paired,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, in_ch, out_ch
