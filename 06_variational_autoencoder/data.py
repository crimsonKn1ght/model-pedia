"""Datasets for the generative projects.

    python data.py --dataset fashion-mnist
    python data.py --dataset folder --root path/to/images    # any 64x64 image set

Images are kept in ``[0, 1]`` with no mean/std normalisation. Generative models
predict pixels, so every loss, metric and figure is easier to read in the units the
pixels have - and a Bernoulli likelihood needs ``[0, 1]`` anyway.

Everything is resized to a power of two (28x28 MNIST becomes 32x32) so that strided
convolutions halve cleanly all the way down and the latent grid arithmetic stays
exact.

``classes`` is ``None`` for datasets that ship no usable labels. That is not
cosmetic: the FID feature network is a classifier, so an unlabelled dataset falls
back to fixed-seed random features, and ``evaluate.py`` says so in its output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

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


def _celeba_builder(root, split, transform):
    # CelebA is served from Google Drive and the download fails often; the error is
    # re-raised with the manual route rather than a stack trace about quotas.
    try:
        return datasets.CelebA(
            root=root,
            split="train" if split == "train" else "test",
            target_type="identity",
            download=True,
            transform=transform,
        )
    except Exception as error:  # noqa: BLE001 - the cause is the download, always
        raise RuntimeError(
            f"could not obtain CelebA ({error}).\n"
            "torchvision fetches it from Google Drive, which rate-limits hard. Download "
            "img_align_celeba.zip by hand into <root>/celeba/, or use "
            "--dataset folder --root <a directory of images>."
        ) from error


def _folder_builder(root, split, transform):
    directory = Path(root)
    for candidate in (directory / ("train" if split == "train" else "val"), directory):
        if candidate.is_dir() and any(candidate.rglob("*")):
            return datasets.ImageFolder(str(candidate), transform=transform)
    raise FileNotFoundError(
        f"no images found under {directory}. --dataset folder wants either an "
        "ImageFolder layout (<root>/train/<class>/*) or a single directory of images."
    )


DATASETS = {
    "mnist": {
        "builder": _train_test_builder(datasets.MNIST),
        "channels": 1, "size": 32, "classes": 10,
        "class_names": [str(d) for d in range(10)], "download_mb": 12,
    },
    "fashion-mnist": {
        "builder": _train_test_builder(datasets.FashionMNIST),
        "channels": 1, "size": 32, "classes": 10,
        "class_names": FASHION_CLASSES, "download_mb": 30,
    },
    "cifar10": {
        "builder": _train_test_builder(datasets.CIFAR10),
        "channels": 3, "size": 32, "classes": 10,
        "class_names": CIFAR10_CLASSES, "download_mb": 170,
    },
    "celeba": {
        "builder": _celeba_builder,
        "channels": 3, "size": 64, "classes": None,
        "class_names": None, "download_mb": 1400,
    },
    "folder": {
        "builder": _folder_builder,
        "channels": 3, "size": 64, "classes": None,
        "class_names": None, "download_mb": 0,
    },
}


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return dict(DATASETS[name])


def build_transform(info: dict, size: int, augment: bool = False):
    steps = []
    if info["channels"] == 3 and info["size"] >= 64:
        steps.append(transforms.CenterCrop(min(info["size"] * 2, 178)))  # CelebA faces
    steps.append(transforms.Resize((size, size)))
    if augment:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    if info["channels"] == 1:
        steps.append(transforms.Grayscale(num_output_channels=1))
    return transforms.Compose(steps)


def download(name: str, root: str = "data") -> None:
    info = dataset_info(name)
    for split in ("train", "test"):
        info["builder"](root, split, None)


def _synthetic_split(channels: int, size: int, num_classes: int, seed: int):
    """Structured random stand-in for ``--smoke-test``; no download."""
    def make(n: int, offset: int) -> TensorDataset:
        generator = torch.Generator().manual_seed(seed + offset)
        labels = torch.randint(0, num_classes, (n,), generator=generator)
        ramp = torch.linspace(0, 1, size)
        grid = ramp.view(1, -1) + ramp.view(-1, 1)
        images = []
        for label in labels.tolist():
            pattern = (grid * 3.1416 + label).sin() * 0.4 + 0.5
            noise = torch.rand(channels, size, size, generator=generator) * 0.15
            images.append((pattern.expand(channels, size, size) + noise).clamp(0, 1))
        return TensorDataset(torch.stack(images), labels)

    return make(160, 0), make(64, 1), make(64, 2)


def get_splits(
    name: str = "fashion-mnist",
    root: str = "data",
    image_size: int | None = None,
    val_split: float = 0.05,
    train_subset: int | None = None,
    augment: bool = False,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train, val, test, info)`` of ``(image, label)`` pairs in ``[0, 1]``."""
    info = dataset_info(name)
    if synthetic:
        info["size"] = 16
        info["classes"] = info["classes"] or 4
        train, val, test = _synthetic_split(info["channels"], 16, info["classes"], seed)
        return train, val, test, info

    size = image_size or info["size"]
    info["size"] = size
    builder = info["builder"]
    full_train = builder(root, "train", build_transform(info, size, augment))
    full_train_eval = builder(root, "train", build_transform(info, size, augment=False))
    test_set = builder(root, "test", build_transform(info, size, augment=False))

    n_val = int(len(full_train) * val_split)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    if train_subset:
        train_idx = train_idx[:train_subset]

    return (
        Subset(full_train, train_idx),
        Subset(full_train_eval, val_idx),
        test_set,
        info,
    )


def make_loader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def feature_net_cache(root: str, name: str, size: int) -> str:
    """Where the FID feature network for this dataset is kept between runs."""
    return str(Path(root) / f"fid_net_{name}_{size}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or inspect a dataset.")
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--preview", action="store_true", help="save outputs/data_preview.png")
    args = parser.parse_args()

    if args.dataset not in ("folder",):
        download(args.dataset, args.root)
    train, val, test, info = get_splits(args.dataset, args.root, image_size=args.image_size)

    print(f"dataset : {args.dataset}")
    print(f"stored  : {Path(args.root).resolve()}")
    print(f"train   : {len(train)}")
    print(f"val     : {len(val)}")
    print(f"test    : {len(test)}")
    print(f"image   : {info['channels']}x{info['size']}x{info['size']} in [0, 1]")
    print(f"labels  : {info['classes']} classes"
          if info["classes"] else "labels  : none (FID falls back to random features)")

    if args.preview:
        from utils import plot_image_grid

        images = torch.stack([train[i][0] for i in range(16)])
        plot_image_grid(images, "outputs/data_preview.png", columns=8,
                        title=f"{args.dataset} training images")
        print("\npreview -> outputs/data_preview.png")


if __name__ == "__main__":
    main()
