"""Two image collections with no correspondence between them.

    python data.py --task shapes --preview                       # generated, no download
    python data.py --task classes --dataset fashion-mnist        # ~30 MB
    python data.py --task datasets                              # MNIST <-> Fashion-MNIST
    python data.py --task horse2zebra                           # ~110 MB, the original

The word that matters here is **unpaired**. Project 10 (Pix2Pix) is handed matched
``(source, target)`` pairs and can compute a pixel loss against the right answer. This
project is handed two *piles* of images and told only which pile each came from. Nothing
in the data says which image in pile B corresponds to a given image in pile A, because
nothing does.

That changes what a dataset has to provide, and it changes what can be measured. There is
no target to compare an output against, so the loaders here deliberately pair images *at
random* and reshuffle that pairing every epoch - any correspondence the model appears to
find is one it invented.

Four tasks, in increasing order of cost:

``shapes``       generated. Domain A is hollow outlines, domain B is filled colour, drawn
                 from the same shape vocabulary but sampled **independently** - so no
                 outline in A is the outline of any shape in B. This is the default: it
                 needs no download, and because the underlying vocabulary is shared there
                 is a sensible translation to find, which makes success legible in a figure.
``classes``      two class subsets of one labelled dataset, e.g. Fashion-MNIST's sandals
                 and sneakers. Real images, one download, and the two domains genuinely
                 differ in shape rather than only in colour.
``datasets``     two datasets entirely, e.g. MNIST digits and Fashion-MNIST garments. The
                 hardest of the reachable options - the domains share almost no structure,
                 which is exactly when cycle consistency starts fighting the adversarial
                 loss.
``horse2zebra``  the dataset CycleGAN was published on. Needs
                 ``people.eecs.berkeley.edu``; if that host is unreachable the other three
                 tasks demonstrate the same mechanics, and the README says so.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

HORSE2ZEBRA_URL = (
    "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip"
)

TASKS = ("shapes", "classes", "datasets", "horse2zebra")

# Datasets reachable without the Berkeley host, and their shapes.
DATASETS = {
    "mnist": {"channels": 1, "size": 32, "num_classes": 10, "download_mb": 12},
    "fashion-mnist": {"channels": 1, "size": 32, "num_classes": 10, "download_mb": 30},
    "cifar10": {"channels": 3, "size": 32, "num_classes": 10, "download_mb": 170},
}

FASHION_CLASSES = (
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "boot",
)
CIFAR_CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def class_names(dataset: str) -> tuple[str, ...]:
    if dataset == "fashion-mnist":
        return FASHION_CLASSES
    if dataset == "cifar10":
        return CIFAR_CLASSES
    return tuple(str(i) for i in range(10))


class ShapesDomain(Dataset):
    """One of the two generated domains: outlines, or filled colour.

    Both domains draw from the same vocabulary - circle, square, triangle, at random
    positions and sizes - but a domain is seeded independently, so image ``i`` of the
    outline domain has nothing to do with image ``i`` of the filled domain. A model that
    learns outline -> filled here has learned it from the *distributions*, which is the
    claim CycleGAN makes.
    """

    COLOURS = [
        (0.85, 0.20, 0.20), (0.25, 0.70, 0.35), (0.25, 0.45, 0.85),
        (0.93, 0.83, 0.25), (0.60, 0.35, 0.75),
    ]

    def __init__(
        self, n: int, filled: bool, canvas: int = 64, seed: int = 0, max_shapes: int = 3
    ) -> None:
        self.n = n
        self.filled = filled
        self.canvas = canvas
        self.seed = seed
        self.max_shapes = max_shapes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        size = self.canvas
        canvas = np.full((3, size, size), 0.08, dtype=np.float32) if self.filled else np.zeros(
            (3, size, size), dtype=np.float32
        )
        ys, xs = np.mgrid[0:size, 0:size]

        for _ in range(int(rng.integers(1, self.max_shapes + 1))):
            kind = int(rng.integers(0, 3))
            colour = np.array(self.COLOURS[int(rng.integers(0, len(self.COLOURS)))],
                              dtype=np.float32)
            radius = int(rng.integers(size // 8, size // 4))
            cx = int(rng.integers(radius + 1, size - radius - 1))
            cy = int(rng.integers(radius + 1, size - radius - 1))

            if kind == 0:
                distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
                area = distance <= radius
                edge = np.abs(distance - radius) <= 1.2
            elif kind == 1:
                inside = np.maximum(np.abs(xs - cx), np.abs(ys - cy))
                area = inside <= radius
                edge = np.abs(inside - radius) <= 1.2
            else:
                bary = (ys - (cy - radius)) / 2 + 1
                area = (ys >= cy - radius) & (ys <= cy + radius) & (np.abs(xs - cx) <= bary)
                inner = (ys >= cy - radius + 2) & (ys <= cy + radius - 2) & (
                    np.abs(xs - cx) <= bary - 2
                )
                edge = area & ~inner

            if self.filled:
                canvas[:, area] = colour[:, None]
            else:
                canvas[:, edge] = 1.0

        return torch.from_numpy(canvas)


class ImageFolderFlat(Dataset):
    """Every image directly under one directory, no class subfolders."""

    def __init__(self, root: str | Path, image_size: int) -> None:
        self.paths = sorted(
            path for path in Path(root).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not self.paths:
            raise FileNotFoundError(f"no images directly under {root}")
        self.transform = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[index]).convert("RGB"))


class ImagesOnly(Dataset):
    """Drops the label from a torchvision dataset: a domain has no labels to give."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.base[index][0]


class UnpairedDataset(Dataset):
    """Draws one image from each domain, with the pairing deliberately arbitrary.

    ``reshuffle(epoch)`` redraws the permutation, so a given A never sits opposite the
    same B twice. If the pairing were fixed, a determined model could memorise it and the
    result would quietly become a paired problem.
    """

    def __init__(self, set_a: Dataset, set_b: Dataset, seed: int = 0) -> None:
        self.set_a = set_a
        self.set_b = set_b
        self.seed = seed
        self.order: list[int] = []
        self.reshuffle(0)

    def reshuffle(self, epoch: int) -> None:
        generator = torch.Generator().manual_seed(self.seed * 7919 + epoch)
        self.order = torch.randint(
            len(self.set_b), (len(self),), generator=generator
        ).tolist()

    def __len__(self) -> int:
        return max(len(self.set_a), 1)

    def __getitem__(self, index: int):
        return self.set_a[index], self.set_b[self.order[index]]


def dataset_info(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(DATASETS)}")
    return dict(DATASETS[name])


def _torchvision_split(name: str, root: str, image_size: int, train: bool) -> Dataset:
    transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
    ])
    if name == "mnist":
        return datasets.MNIST(root, train=train, download=True, transform=transform)
    if name == "fashion-mnist":
        return datasets.FashionMNIST(root, train=train, download=True, transform=transform)
    return datasets.CIFAR10(root, train=train, download=True, transform=transform)


def _class_subset(base: Dataset, wanted: int) -> Dataset:
    targets = getattr(base, "targets", None)
    if targets is None:
        raise ValueError("the dataset exposes no targets, so a class subset cannot be built")
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    indices = [i for i, label in enumerate(targets) if int(label) == wanted]
    if not indices:
        raise ValueError(f"class {wanted} has no examples in this split")
    return Subset(ImagesOnly(base), indices)


def _limit(dataset: Dataset, size: int) -> Dataset:
    """Evenly spread subset, so a smaller run still spans the whole domain."""
    if not size or size <= 0 or size >= len(dataset):
        return dataset
    stride = len(dataset) / size
    return Subset(dataset, [int(i * stride) for i in range(size)])


def download(task: str, dataset: str = "fashion-mnist", root: str = "data") -> None:
    if task in ("classes",):
        _torchvision_split(dataset, root, 32, True)
        _torchvision_split(dataset, root, 32, False)
    elif task == "datasets":
        for name in ("mnist", "fashion-mnist"):
            _torchvision_split(name, root, 32, True)
            _torchvision_split(name, root, 32, False)
    elif task == "horse2zebra":
        directory = Path(root) / "horse2zebra"
        if not directory.exists():
            from torchvision.datasets.utils import download_and_extract_archive

            print("downloading horse2zebra ...")
            download_and_extract_archive(HORSE2ZEBRA_URL, download_root=str(Path(root)))


def get_domains(
    task: str = "shapes",
    dataset: str = "fashion-mnist",
    domain_a: str = "5",
    domain_b: str = "7",
    root: str = "data",
    image_size: int | None = None,
    train_size: int = 2000,
    val_size: int = 200,
    test_size: int = 500,
    seed: int = 0,
    synthetic: bool = False,
):
    """Return ``(train_a, train_b, val_a, val_b, test_a, test_b, info)``.

    Every split keeps the two domains as separate datasets: they are only brought
    together by ``make_loader``, and then only at random.
    """
    if synthetic:
        info = {"channels": 3, "size": 32, "name_a": "outline", "name_b": "filled",
                "task": "shapes"}
        return (
            ShapesDomain(48, False, 32, seed=1), ShapesDomain(48, True, 32, seed=101),
            ShapesDomain(16, False, 32, seed=2), ShapesDomain(16, True, 32, seed=102),
            ShapesDomain(16, False, 32, seed=3), ShapesDomain(16, True, 32, seed=103),
            info,
        )

    if task == "shapes":
        size = image_size or 32
        info = {"channels": 3, "size": size, "name_a": "outline", "name_b": "filled",
                "task": task}
        # The +100 offset on the B seeds is what makes the domains independent.
        return (
            ShapesDomain(train_size, False, size, seed=1),
            ShapesDomain(train_size, True, size, seed=101),
            ShapesDomain(val_size, False, size, seed=2),
            ShapesDomain(val_size, True, size, seed=102),
            ShapesDomain(test_size, False, size, seed=3),
            ShapesDomain(test_size, True, size, seed=103),
            info,
        )

    if task == "classes":
        meta = dataset_info(dataset)
        size = image_size or meta["size"]
        index_a, index_b = int(domain_a), int(domain_b)
        names = class_names(dataset)
        train_base = _torchvision_split(dataset, root, size, True)
        test_base = _torchvision_split(dataset, root, size, False)
        train_a = _class_subset(train_base, index_a)
        train_b = _class_subset(train_base, index_b)
        test_a = _class_subset(test_base, index_a)
        test_b = _class_subset(test_base, index_b)
        info = {"channels": meta["channels"], "size": size, "task": task,
                "name_a": names[index_a], "name_b": names[index_b]}
        # The validation domains come out of the *training* split, so the test split
        # stays untouched until evaluate.py.
        return (
            _limit(train_a, train_size), _limit(train_b, train_size),
            _limit(test_a, val_size), _limit(test_b, val_size),
            _limit(test_a, test_size), _limit(test_b, test_size),
            info,
        )

    if task == "datasets":
        name_a = domain_a if domain_a in DATASETS else "mnist"
        name_b = domain_b if domain_b in DATASETS else "fashion-mnist"
        meta_a, meta_b = dataset_info(name_a), dataset_info(name_b)
        if meta_a["channels"] != meta_b["channels"]:
            raise ValueError(
                f"{name_a} has {meta_a['channels']} channels and {name_b} has "
                f"{meta_b['channels']}; pick two datasets with the same channel count"
            )
        size = image_size or max(meta_a["size"], meta_b["size"])
        info = {"channels": meta_a["channels"], "size": size, "task": task,
                "name_a": name_a, "name_b": name_b}
        return (
            _limit(ImagesOnly(_torchvision_split(name_a, root, size, True)), train_size),
            _limit(ImagesOnly(_torchvision_split(name_b, root, size, True)), train_size),
            _limit(ImagesOnly(_torchvision_split(name_a, root, size, False)), val_size),
            _limit(ImagesOnly(_torchvision_split(name_b, root, size, False)), val_size),
            _limit(ImagesOnly(_torchvision_split(name_a, root, size, False)), test_size),
            _limit(ImagesOnly(_torchvision_split(name_b, root, size, False)), test_size),
            info,
        )

    if task == "horse2zebra":
        size = image_size or 64
        directory = Path(root) / "horse2zebra"
        if not (directory / "trainA").is_dir():
            raise FileNotFoundError(
                f"{directory / 'trainA'} not found. Run: python data.py --task horse2zebra\n"
                "If that host is unreachable, --task shapes and --task classes show the "
                "same mechanics with no download."
            )
        info = {"channels": 3, "size": size, "task": task, "name_a": "horse",
                "name_b": "zebra"}
        train_a = ImageFolderFlat(directory / "trainA", size)
        train_b = ImageFolderFlat(directory / "trainB", size)
        test_a = ImageFolderFlat(directory / "testA", size)
        test_b = ImageFolderFlat(directory / "testB", size)
        return (
            _limit(train_a, train_size), _limit(train_b, train_size),
            _limit(test_a, val_size), _limit(test_b, val_size),
            _limit(test_a, test_size), _limit(test_b, test_size),
            info,
        )

    raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")


def make_loader(
    set_a: Dataset,
    set_b: Dataset,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 2,
    seed: int = 0,
) -> DataLoader:
    """Loader over ``(image_from_A, image_from_B)`` with an arbitrary pairing."""
    return DataLoader(
        UnpairedDataset(set_a, set_b, seed=seed),
        batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


def single_loader(
    dataset: Dataset, batch_size: int = 16, num_workers: int = 2
) -> DataLoader:
    """One domain on its own, for scoring a direction against real images."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class DomainLabelled(Dataset):
    """Both domains concatenated, labelled 0 for A and 1 for B.

    This exists for the FID feature network and nothing else. Everywhere else in the
    project the domain label is withheld from the model - that is the point of the task -
    but the metric has to measure "does this output look like domain B", and a network
    trained to tell the two domains apart is the most direct feature space for that
    question. It is also the only labelling available for the generated ``shapes`` task,
    which has no object classes.
    """

    def __init__(self, set_a: Dataset, set_b: Dataset) -> None:
        self.set_a = set_a
        self.set_b = set_b

    def __len__(self) -> int:
        return len(self.set_a) + len(self.set_b)

    def __getitem__(self, index: int):
        if index < len(self.set_a):
            return self.set_a[index], 0
        return self.set_b[index - len(self.set_a)], 1


def domain_classifier_loader(
    set_a: Dataset, set_b: Dataset, batch_size: int = 64, num_workers: int = 2, seed: int = 0
) -> DataLoader:
    """Shuffled loader of ``(image, domain)`` for fitting the feature network."""
    return DataLoader(
        DomainLabelled(set_a, set_b),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
    )


def feature_net_cache(root: str, task: str, dataset: str, size: int) -> Path:
    return Path(root) / f"fid_net_{task}_{dataset}_{size}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or inspect the two domains.")
    parser.add_argument("--task", default="shapes", choices=TASKS)
    parser.add_argument("--dataset", default="fashion-mnist", choices=sorted(DATASETS))
    parser.add_argument("--domain-a", default="5", help="class index, or dataset name")
    parser.add_argument("--domain-b", default="7")
    parser.add_argument("--root", default="data")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--preview", action="store_true", help="save outputs/data_preview.png")
    args = parser.parse_args()

    download(args.task, args.dataset, args.root)
    train_a, train_b, val_a, val_b, test_a, test_b, info = get_domains(
        args.task, args.dataset, args.domain_a, args.domain_b, args.root,
        image_size=args.image_size,
    )
    print(f"task    : {args.task}")
    print(f"domain A: {info['name_a']}  {len(train_a)} train / {len(val_a)} val / "
          f"{len(test_a)} test")
    print(f"domain B: {info['name_b']}  {len(train_b)} train / {len(val_b)} val / "
          f"{len(test_b)} test")
    print(f"image   : {info['channels']}x{info['size']}x{info['size']} in [0, 1]")
    print("\nthe two domains are unpaired: nothing links an image in A to one in B")

    if args.preview:
        from utils import plot_image_rows

        loader = make_loader(val_a, val_b, batch_size=8, num_workers=0)
        image_a, image_b = next(iter(loader))
        plot_image_rows(
            [(f"domain A: {info['name_a']}", image_a), (f"domain B: {info['name_b']}", image_b)],
            "outputs/data_preview.png",
            title=f"{args.task}: two unpaired collections (rows do not correspond)",
        )
        print("\npreview -> outputs/data_preview.png")


if __name__ == "__main__":
    main()
