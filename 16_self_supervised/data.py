"""Datasets and the augmentation pipeline for self-supervised pretraining.

    python data.py --dataset cifar10
    python data.py --dataset cifar10 --preview   # save a figure of the two views

Labels are downloaded but deliberately unused during pretraining: the training
loop only ever sees ``(view1, view2)``. They come back out for the frozen-feature
probes in ``evaluate.py``, which is the only place accuracy can be measured.

The augmentation pipeline is not a detail here, it *is* the task. A contrastive
model is asked to recognise that two augmentations of one image belong together,
so whatever the augmentations destroy is what the representation learns to
ignore. Random-resized crop forces the model to relate a part to the whole;
colour jitter and grayscale stop it from solving the puzzle with an average
colour histogram, which is the shortcut it takes first if you let it. Run
``compare.py --study augment`` to watch the accuracy collapse as they are
removed.
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


CIFAR10_CLASSES = [
    "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
]
FASHION_CLASSES = [
    "tshirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "boot",
]
STL10_CLASSES = [
    "plane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck",
]

# ``flip`` is off where a mirrored image is a different thing (digits), because
# the model would be taught an invariance the downstream task does not want.
DATASETS = {
    "cifar10": {
        "builder": _train_test_builder(datasets.CIFAR10),
        "channels": 3,
        "size": 32,
        "classes": 10,
        "class_names": CIFAR10_CLASSES,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "flip": True,
        "download_mb": 170,
    },
    "cifar100": {
        "builder": _train_test_builder(datasets.CIFAR100),
        "channels": 3,
        "size": 32,
        "classes": 100,
        "class_names": None,
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "flip": True,
        "download_mb": 170,
    },
    "stl10": {
        "builder": _stl10_builder,
        "channels": 3,
        "size": 96,
        "classes": 10,
        "class_names": STL10_CLASSES,
        "mean": (0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
        "flip": True,
        "download_mb": 2600,
    },
    "fashion-mnist": {
        "builder": _train_test_builder(datasets.FashionMNIST),
        "channels": 1,
        "size": 28,
        "classes": 10,
        "class_names": FASHION_CLASSES,
        "mean": (0.2860,),
        "std": (0.3530,),
        "flip": True,
        "download_mb": 30,
    },
    "mnist": {
        "builder": _train_test_builder(datasets.MNIST),
        "channels": 1,
        "size": 28,
        "classes": 10,
        "class_names": [str(d) for d in range(10)],
        "mean": (0.1307,),
        "std": (0.3081,),
        "flip": False,
        "download_mb": 12,
    },
}

AUGMENTATIONS = ("full", "crop", "none")


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return DATASETS[name]


def default_image_size(name: str) -> int:
    """Native resolution, except STL-10 which is downsized to stay laptop-sized."""
    return 64 if name == "stl10" else dataset_info(name)["size"]


def ssl_transform(info: dict, image_size: int, strength: str = "full"):
    """The two-view augmentation. ``strength`` is what ``compare.py`` varies."""
    if strength not in AUGMENTATIONS:
        raise ValueError(f"unknown augmentation {strength!r}, expected one of {AUGMENTATIONS}")

    if strength == "none":
        # Both views are then identical, and the contrastive task is trivially
        # solvable by any function that is not constant. This arm exists to make
        # that failure visible rather than theoretical.
        steps = [transforms.Resize((image_size, image_size))]
    else:
        steps = [transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0))]
        if info["flip"]:
            steps.append(transforms.RandomHorizontalFlip())
        if strength == "full":
            if info["channels"] == 3:
                jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                steps.append(transforms.RandomApply([jitter], p=0.8))
                steps.append(transforms.RandomGrayscale(p=0.2))
            else:
                # Hue and saturation are undefined for one channel.
                jitter = transforms.ColorJitter(0.4, 0.4)
                steps.append(transforms.RandomApply([jitter], p=0.8))

    steps += [transforms.ToTensor(), transforms.Normalize(info["mean"], info["std"])]
    return transforms.Compose(steps)


def eval_transform(info: dict, image_size: int):
    """Deterministic pipeline for the probes - no augmentation anywhere."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(info["mean"], info["std"]),
        ]
    )


def denormalize(images: torch.Tensor, info: dict) -> torch.Tensor:
    """Undo ``Normalize`` so a batch can be plotted."""
    mean = torch.tensor(info["mean"]).view(1, -1, 1, 1).to(images.device)
    std = torch.tensor(info["std"]).view(1, -1, 1, 1).to(images.device)
    return (images * std + mean).clamp(0, 1)


class TwoViewDataset(Dataset):
    """Wraps a dataset of PIL images, returning two independent augmentations.

    The label is passed through untouched so the same wrapper can feed the k-NN
    monitor during training, but the training loss never receives it.
    """

    def __init__(self, base: Dataset, transform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        return self.transform(image), self.transform(image), int(target)


class TransformDataset(Dataset):
    """Single deterministic view, for feature extraction."""

    def __init__(self, base: Dataset, transform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        return self.transform(image), int(target)


class SyntheticBase(Dataset):
    """Tiny PIL-image stand-in used by ``--smoke-test``; nothing is downloaded.

    Each class gets a constant brightness offset so the probes have something
    learnable and their numbers are not pure noise.
    """

    def __init__(self, n: int, channels: int, size: int, num_classes: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (n,), generator=generator)
        noise = torch.rand(n, channels, size, size, generator=generator)
        offset = self.labels.view(-1, 1, 1, 1).float() / max(num_classes - 1, 1)
        self.images = (0.5 * noise + 0.5 * offset).clamp(0, 1)
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
        size = min(info["size"], 32)
        return (
            SyntheticBase(96, info["channels"], size, info["classes"], seed),
            SyntheticBase(48, info["channels"], size, info["classes"], seed + 1),
            SyntheticBase(48, info["channels"], size, info["classes"], seed + 2),
            info,
        )

    full_train = info["builder"](root, "train", None)
    test_base = info["builder"](root, "test", None)

    n_val = int(len(full_train) * val_split)
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    if train_subset:
        train_idx = train_idx[:train_subset]

    return Subset(full_train, train_idx), Subset(full_train, val_idx), test_base, info


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def make_ssl_loader(
    base: Dataset,
    info: dict,
    image_size: int,
    batch_size: int = 256,
    strength: str = "full",
    num_workers: int = 2,
) -> DataLoader:
    """Two-view loader. ``drop_last`` because NT-Xent needs a full batch of negatives."""
    dataset = TwoViewDataset(base, ssl_transform(info, image_size, strength))
    drop_last = len(dataset) > batch_size
    return _loader(dataset, batch_size, True, num_workers, drop_last)


def make_eval_loader(
    base: Dataset,
    info: dict,
    image_size: int,
    batch_size: int = 512,
    num_workers: int = 2,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TransformDataset(base, eval_transform(info, image_size))
    return _loader(dataset, batch_size, shuffle, num_workers, False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a dataset for self-supervised learning.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="save data_views.png: the same images under the two-view augmentation",
    )
    args = parser.parse_args()

    download(args.dataset, args.root)
    train_base, val_base, test_base, info = get_splits(args.dataset, args.root)
    image_size = args.image_size or default_image_size(args.dataset)

    print(f"dataset    : {args.dataset}")
    print(f"stored     : {Path(args.root).resolve()}")
    print(f"train      : {len(train_base)}  (labels unused during pretraining)")
    print(f"val        : {len(val_base)}")
    print(f"test       : {len(test_base)}")
    print(f"image      : {info['channels']}x{image_size}x{image_size}, {info['classes']} classes")

    if args.preview:
        from utils import plot_image_rows  # local import: only needed for the figure

        loader = make_ssl_loader(train_base, info, image_size, batch_size=8, num_workers=0)
        view1, view2, _ = next(iter(loader))
        plot_image_rows(
            [("view 1", denormalize(view1, info)), ("view 2", denormalize(view2, info))],
            "outputs/data_views.png",
            title="the two views the model is asked to match",
        )
        print("preview    -> outputs/data_views.png")


if __name__ == "__main__":
    main()
