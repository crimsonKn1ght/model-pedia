"""Datasets and the noise model for denoising.

    python data.py --dataset cifar10

The clean images come from torchvision; the noise is added on the fly in the
training loop, not baked into the dataset. That matters: every epoch sees a
fresh noise realisation of the same image, so the network cannot memorise a
particular corruption and has to learn the structure of the noise instead.

``--test-dir`` in ``evaluate.py`` accepts any folder of images, which is how to
run a classical benchmark set such as BSD68 or Set12 through a model trained
on CIFAR patches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.io import ImageReadMode, read_image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}


def _builder(cls):
    def build(root, train, transform):
        return cls(root=root, train=train, download=True, transform=transform)

    return build


DATASETS = {
    "cifar10": {"builder": _builder(datasets.CIFAR10), "channels": 3, "size": 32},
    "fashion-mnist": {"builder": _builder(datasets.FashionMNIST), "channels": 1, "size": 32},
    "mnist": {"builder": _builder(datasets.MNIST), "channels": 1, "size": 32},
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return DATASETS[name]


def build_transforms(size: int = 32, augment: bool = False):
    steps = [transforms.Resize(size)]
    if augment:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def add_gaussian_noise(
    images: torch.Tensor,
    sigma: float | tuple[float, float],
    generator: torch.Generator | None = None,
    clamp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add white Gaussian noise, returning ``(noisy, sigma_per_image)``.

    ``sigma`` may be a single level or a ``(low, high)`` range. Training over a
    range gives a *blind* denoiser - one that does not need to be told the
    noise level at test time, which is the only realistic setting.

    Sigma is in the same units as the pixels, i.e. ``0.1`` means 10% of the
    full ``[0, 1]`` range, or roughly 25 on the usual 0-255 scale.
    """
    if isinstance(sigma, (int, float)):
        levels = torch.full((images.size(0),), float(sigma), device=images.device)
    else:
        low, high = sigma
        levels = (
            torch.rand(images.size(0), device=images.device, generator=generator) * (high - low)
            + low
        )

    noise = torch.randn(images.shape, device=images.device, generator=generator)
    noisy = images + noise * levels.view(-1, 1, 1, 1)
    # Real sensors deliver values in range; clamping keeps the input honest.
    return (noisy.clamp(0, 1) if clamp else noisy), levels


class ImageDirectory(Dataset):
    """Every image in a folder, cropped to a size the U-Net can halve twice.

    Used for classical benchmark sets (BSD68, Set12, Kodak) that do not ship
    with torchvision.
    """

    def __init__(self, root: str, channels: int = 3, divisor: int = 4, max_size: int | None = 512):
        self.paths = sorted(
            path
            for path in Path(root).rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
        )
        if not self.paths:
            raise FileNotFoundError(f"no images found under {root}")
        self.mode = ImageReadMode.GRAY if channels == 1 else ImageReadMode.RGB
        self.divisor = divisor
        self.max_size = max_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = read_image(str(self.paths[index]), mode=self.mode).float() / 255.0
        if self.max_size:
            height, width = image.shape[-2:]
            if max(height, width) > self.max_size:
                scale = self.max_size / max(height, width)
                image = transforms.functional.resize(
                    image, [max(int(height * scale), 1), max(int(width * scale), 1)], antialias=True
                )
        height, width = image.shape[-2:]
        height -= height % self.divisor
        width -= width % self.divisor
        return image[..., :height, :width], 0


def download(name: str, root: str = "data") -> None:
    builder = dataset_info(name)["builder"]
    for train in (True, False):
        builder(root, train, build_transforms())


def _synthetic_split(channels: int, size: int, seed: int):
    """Tiny random stand-in used by the --smoke-test flag (no download needed)."""
    generator = torch.Generator().manual_seed(seed)

    def make(n: int) -> TensorDataset:
        x = torch.rand(n, channels, size, size, generator=generator)
        return TensorDataset(x, torch.zeros(n, dtype=torch.long))

    return make(128), make(64), make(64)


def get_datasets(
    name: str = "cifar10",
    root: str = "data",
    val_split: float = 0.1,
    augment: bool = True,
    train_subset: int | None = None,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train, val, test, info)``."""
    info = dataset_info(name)
    if synthetic:
        train_ds, val_ds, test_ds = _synthetic_split(info["channels"], info["size"], seed)
        return train_ds, val_ds, test_ds, info

    builder = info["builder"]
    full_train = builder(root, True, build_transforms(info["size"], augment))
    full_train_eval = builder(root, True, build_transforms(info["size"], augment=False))
    test_ds = builder(root, False, build_transforms(info["size"], augment=False))

    n_val = int(len(full_train) * val_split)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    if train_subset:
        train_idx = train_idx[:train_subset]

    return Subset(full_train, train_idx), Subset(full_train_eval, val_idx), test_ds, info


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
    parser = argparse.ArgumentParser(description="Download a dataset for denoising.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    args = parser.parse_args()

    download(args.dataset, args.root)
    train_ds, val_ds, test_ds, info = get_datasets(args.dataset, args.root)
    print(f"dataset : {args.dataset}")
    print(f"stored  : {Path(args.root).resolve()}")
    print(f"train   : {len(train_ds)}")
    print(f"val     : {len(val_ds)}")
    print(f"test    : {len(test_ds)}")
    print(f"image   : {info['channels']}x{info['size']}x{info['size']} in [0, 1]")


if __name__ == "__main__":
    main()
