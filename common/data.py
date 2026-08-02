"""Dataset registry, download helpers and the loaders used by every project.

All image datasets are returned as ``(image, label)`` pairs where ``image`` is a
float tensor in ``[-1, 1]`` with shape ``(C, H, W)``.  Working in ``[-1, 1]``
keeps the GAN/diffusion projects consistent with a ``tanh`` generator output;
projects that prefer ``[0, 1]`` (the VAE with a Bernoulli likelihood, the flow)
ask for ``normalize="unit"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# --------------------------------------------------------------------------- #
# Registry metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetInfo:
    """Static facts about a dataset that models need in order to size layers."""

    name: str
    channels: int
    native_size: int
    num_classes: int
    class_names: Tuple[str, ...] = ()
    needs_manual_download: bool = False


DATASET_INFO: Dict[str, DatasetInfo] = {
    "mnist": DatasetInfo(
        "mnist", 1, 28, 10, tuple(str(i) for i in range(10))
    ),
    "fashion-mnist": DatasetInfo(
        "fashion-mnist",
        1,
        28,
        10,
        (
            "t-shirt", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle-boot",
        ),
    ),
    "cifar10": DatasetInfo(
        "cifar10",
        3,
        32,
        10,
        (
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ),
    ),
    "celeba64": DatasetInfo("celeba64", 3, 64, 1, ("face",), needs_manual_download=True),
    "flowers102": DatasetInfo("flowers102", 3, 64, 102),
    "shapes": DatasetInfo(
        "shapes", 3, 32, 4, ("square", "circle", "triangle", "cross")
    ),
}

DATASET_NAMES: List[str] = list(DATASET_INFO)


def info(name: str) -> DatasetInfo:
    if name not in DATASET_INFO:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASET_NAMES)}")
    return DATASET_INFO[name]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transform(
    image_size: int,
    channels: int,
    normalize: str = "tanh",
    augment: bool = False,
) -> transforms.Compose:
    """Resize -> optional flip -> tensor -> normalise.

    ``normalize="tanh"`` maps to ``[-1, 1]``, ``normalize="unit"`` keeps ``[0, 1]``.
    """
    ops: List[Callable] = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    if normalize == "tanh":
        ops.append(transforms.Normalize([0.5] * channels, [0.5] * channels))
    elif normalize != "unit":
        raise ValueError(f"normalize must be 'tanh' or 'unit', got {normalize!r}")
    return transforms.Compose(ops)


def denormalize(x: torch.Tensor, normalize: str = "tanh") -> torch.Tensor:
    """Bring a batch back to ``[0, 1]`` for saving/plotting."""
    if normalize == "tanh":
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Shapes: a procedural RGB dataset that needs no network at all
# --------------------------------------------------------------------------- #
class Shapes(Dataset):
    """Coloured geometric shapes on a coloured background, generated procedurally.

    This dataset exists so that every project in this repository can be run and
    smoke-tested without downloading anything, and so the 3-channel code paths
    stay exercisable on machines with no internet access.  It is deliberately
    easy: four shape classes, random colours, positions and sizes.  Use it to
    check that a pipeline runs, then switch to CIFAR-10 or CelebA for results
    that actually say something about the model.
    """

    SHAPES = ("square", "circle", "triangle", "cross")

    def __init__(
        self,
        num_samples: int = 6000,
        image_size: int = 32,
        seed: int = 0,
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.image_size = image_size
        generator = np.random.default_rng(seed)
        self.images = np.zeros((num_samples, image_size, image_size, 3), dtype=np.uint8)
        self.labels = generator.integers(0, len(self.SHAPES), size=num_samples)
        for i in range(num_samples):
            self.images[i] = self._draw(int(self.labels[i]), image_size, generator)

    @staticmethod
    def _draw(label: int, size: int, rng: "np.random.Generator") -> "np.ndarray":
        bg = rng.integers(0, 90, size=3)
        fg = rng.integers(150, 256, size=3)
        canvas = np.tile(bg.astype(np.uint8), (size, size, 1))

        radius = int(rng.integers(size // 5, size // 3))
        cy = int(rng.integers(radius, size - radius))
        cx = int(rng.integers(radius, size - radius))
        yy, xx = np.mgrid[0:size, 0:size]
        dy, dx = yy - cy, xx - cx

        if label == 0:  # square
            mask = (np.abs(dy) <= radius) & (np.abs(dx) <= radius)
        elif label == 1:  # circle
            mask = dy**2 + dx**2 <= radius**2
        elif label == 2:  # triangle
            mask = (dy <= radius) & (dy >= -radius) & (np.abs(dx) <= (dy + radius) / 2)
        else:  # cross
            arm = max(1, radius // 3)
            mask = ((np.abs(dy) <= arm) & (np.abs(dx) <= radius)) | (
                (np.abs(dx) <= arm) & (np.abs(dy) <= radius)
            )
        canvas[mask] = fg.astype(np.uint8)
        return canvas

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        from PIL import Image

        img = Image.fromarray(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[index])


# --------------------------------------------------------------------------- #
# CelebA-64
# --------------------------------------------------------------------------- #
CELEBA_HELP = """\
CelebA could not be downloaded automatically.  The official archive is hosted on
Google Drive, which frequently rejects scripted downloads.

To use CelebA, download 'img_align_celeba.zip' manually (for example from
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html or the Kaggle mirror
'jessicali9530/celeba-dataset') and unzip it so that the images live at:

    <data-root>/celeba/img_align_celeba/000001.jpg
    <data-root>/celeba/img_align_celeba/000002.jpg
    ...

Every project falls back to a smaller dataset, so CelebA is always optional.
"""


class CelebA64(Dataset):
    """Folder-backed CelebA reader: centre-crops the aligned images before resizing.

    torchvision's ``datasets.CelebA`` is avoided on purpose because its Google
    Drive download is unreliable.  Any directory of aligned CelebA jpgs works.
    """

    def __init__(self, root: str | Path, transform: Optional[Callable] = None) -> None:
        self.dir = Path(root) / "celeba" / "img_align_celeba"
        if not self.dir.is_dir():
            raise FileNotFoundError(CELEBA_HELP)
        self.files = sorted(self.dir.glob("*.jpg")) or sorted(self.dir.glob("*.png"))
        if not self.files:
            raise FileNotFoundError(CELEBA_HELP)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        from PIL import Image

        img = Image.open(self.files[index]).convert("RGB")
        # The aligned images are 178x218; the standard crop keeps the face.
        img = transforms.functional.center_crop(img, [178, 178])
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def get_dataset(
    name: str,
    root: str | Path = "data",
    image_size: Optional[int] = None,
    train: bool = True,
    normalize: str = "tanh",
    augment: bool = False,
    download: bool = True,
) -> Dataset:
    """Build one of the registered datasets with a consistent output contract."""
    meta = info(name)
    image_size = image_size or meta.native_size
    tf = build_transform(image_size, meta.channels, normalize=normalize, augment=augment)
    root = str(root)

    if name == "mnist":
        return datasets.MNIST(root, train=train, transform=tf, download=download)
    if name == "fashion-mnist":
        return datasets.FashionMNIST(root, train=train, transform=tf, download=download)
    if name == "cifar10":
        return datasets.CIFAR10(root, train=train, transform=tf, download=download)
    if name == "celeba64":
        full = CelebA64(root, transform=tf)
        # Deterministic 95/5 split so evaluation never sees training images.
        cut = int(len(full) * 0.95)
        indices = list(range(cut)) if train else list(range(cut, len(full)))
        return Subset(full, indices)
    if name == "flowers102":
        split = "train" if train else "test"
        return datasets.Flowers102(root, split=split, transform=tf, download=download)
    if name == "shapes":
        return Shapes(
            num_samples=6000 if train else 1500,
            image_size=image_size,
            seed=0 if train else 1,
            transform=tf,
        )
    raise KeyError(name)


def get_dataloader(
    name: str,
    root: str | Path = "data",
    image_size: Optional[int] = None,
    train: bool = True,
    batch_size: int = 128,
    normalize: str = "tanh",
    augment: bool = False,
    num_workers: int = 2,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    download: bool = True,
) -> DataLoader:
    dataset = get_dataset(
        name,
        root=root,
        image_size=image_size,
        train=train,
        normalize=normalize,
        augment=augment,
        download=download,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        drop_last=train if drop_last is None else drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def download_dataset(name: str, root: str | Path = "data") -> str:
    """Fetch a dataset and report what landed on disk."""
    meta = info(name)
    try:
        train = get_dataset(name, root=root, train=True, download=True)
    except FileNotFoundError as exc:
        return f"{name}: NOT AVAILABLE\n{exc}"
    try:
        test = get_dataset(name, root=root, train=False, download=True)
        n_test = len(test)
    except Exception:  # some splits are optional
        n_test = 0
    return (
        f"{name}: ready ({len(train)} train / {n_test} test images, "
        f"{meta.channels}x{meta.native_size}x{meta.native_size})"
    )


# --------------------------------------------------------------------------- #
# Class-filtered views, used by CycleGAN to build two unpaired domains
# --------------------------------------------------------------------------- #
def class_subset(dataset: Dataset, class_indices: Sequence[int]) -> Subset:
    """Keep only the samples whose label is in ``class_indices``."""
    wanted = set(int(c) for c in class_indices)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        labels = [int(dataset[i][1]) for i in range(len(dataset))]
    else:
        labels = [int(t) for t in targets]
    keep = [i for i, y in enumerate(labels) if y in wanted]
    if not keep:
        raise ValueError(f"no samples found for classes {sorted(wanted)}")
    return Subset(dataset, keep)


class UnpairedDataset(Dataset):
    """Zips two domains with independent (randomly offset) indexing.

    CycleGAN never sees a correspondence between the domains, so sample ``i``
    from A is deliberately paired with an arbitrary sample from B.
    """

    def __init__(self, dataset_a: Dataset, dataset_b: Dataset, seed: int = 0) -> None:
        self.a = dataset_a
        self.b = dataset_b
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return max(len(self.a), len(self.b))

    def __getitem__(self, index: int):
        a = self.a[index % len(self.a)][0]
        j = int(torch.randint(len(self.b), (1,), generator=self.generator).item())
        b = self.b[j][0]
        return a, b


# --------------------------------------------------------------------------- #
# Edges -> photo pairs, used by Pix2Pix so a paired task always works offline
# --------------------------------------------------------------------------- #
class EdgesToPhoto(Dataset):
    """Derives an exactly-paired translation task from any image dataset.

    The input is a Sobel edge map of the photo and the target is the photo
    itself.  This is the same setup as pix2pix's ``edges2shoes``, only the edge
    maps are computed on the fly so no extra download is required.
    """

    def __init__(self, base: Dataset, threshold: float = 0.25) -> None:
        self.base = base
        self.threshold = threshold
        kx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        self.sobel = torch.stack([kx, kx.t()]).unsqueeze(1)  # (2, 1, 3, 3)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        import torch.nn.functional as F

        photo = self.base[index][0]  # (C, H, W) in [-1, 1]
        gray = photo.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)
        grad = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), self.sobel)
        magnitude = grad.pow(2).sum(dim=1, keepdim=True).sqrt().squeeze(0)
        magnitude = magnitude / magnitude.amax().clamp(min=1e-8)
        edges = (magnitude > self.threshold).float()
        edges = edges * 2.0 - 1.0  # match the [-1, 1] convention
        return edges, photo


# --------------------------------------------------------------------------- #
# Facades (the real pix2pix dataset) -- optional download
# --------------------------------------------------------------------------- #
FACADES_URL = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
HORSE2ZEBRA_URL = (
    "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip"
)


class SideBySideDataset(Dataset):
    """Reads the pix2pix layout where input and target are one wide image."""

    def __init__(self, directory: str | Path, image_size: int = 64, input_right: bool = True):
        self.dir = Path(directory)
        self.files = sorted(self.dir.glob("*.jpg")) + sorted(self.dir.glob("*.png"))
        if not self.files:
            raise FileNotFoundError(f"no images under {self.dir}")
        self.image_size = image_size
        self.input_right = input_right
        self.tf = build_transform(image_size, 3, normalize="tanh")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        from PIL import Image

        wide = Image.open(self.files[index]).convert("RGB")
        w, h = wide.size
        left = wide.crop((0, 0, w // 2, h))
        right = wide.crop((w // 2, 0, w, h))
        source, target = (right, left) if self.input_right else (left, right)
        return self.tf(source), self.tf(target)


class ImageFolderFlat(Dataset):
    """A flat directory of images, returned as ``(image, 0)``."""

    def __init__(self, directory: str | Path, image_size: int = 64):
        self.dir = Path(directory)
        self.files = sorted(
            p for p in self.dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.files:
            raise FileNotFoundError(f"no images under {self.dir}")
        self.tf = build_transform(image_size, 3, normalize="tanh")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        from PIL import Image

        return self.tf(Image.open(self.files[index]).convert("RGB")), 0


def download_archive(url: str, root: str | Path, name: str) -> Path:
    """Download and extract a tar/zip archive, skipping work if it already exists."""
    from torchvision.datasets.utils import download_and_extract_archive

    root = Path(root)
    target = root / name
    if target.exists():
        return target
    download_and_extract_archive(url, download_root=str(root), extract_root=str(root))
    return target
