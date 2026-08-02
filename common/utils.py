"""Small helpers shared by every project: seeding, devices, checkpoints, logging."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Reproducibility and devices
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 0) -> None:
    """Seed python, numpy and torch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` / ``mps`` into a real device."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(module: torch.nn.Module) -> int:
    """Number of trainable parameters, handy for the README tables."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def describe_model(name: str, module: torch.nn.Module) -> str:
    return f"{name}: {count_parameters(module) / 1e6:.2f}M trainable parameters"


# --------------------------------------------------------------------------- #
# Filesystem
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_checkpoint(path: str | Path, **objects: Any) -> None:
    """Save a checkpoint. ``nn.Module``/optimizers are stored as state dicts."""
    path = Path(path)
    ensure_dir(path.parent)
    payload: Dict[str, Any] = {}
    for key, value in objects.items():
        if hasattr(value, "state_dict"):
            payload[key] = value.state_dict()
        else:
            payload[key] = value
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------- #
# Metric bookkeeping
# --------------------------------------------------------------------------- #
class AverageMeter:
    """Running mean of a scalar, reset once per epoch."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class History:
    """Collects per-epoch scalars so they can be plotted and dumped to JSON."""

    def __init__(self) -> None:
        self.records: Dict[str, list] = {}

    def add(self, **values: float) -> None:
        for key, value in values.items():
            self.records.setdefault(key, []).append(float(value))

    def save(self, path: str | Path) -> None:
        save_json(self.records, path)

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def __getitem__(self, key: str) -> list:
        return self.records[key]


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start

    @property
    def elapsed(self) -> float:
        return getattr(self, "seconds", time.perf_counter() - self.start)


def progress(iterable: Iterable, desc: str = "", total: Optional[int] = None):
    """Thin tqdm wrapper.

    The bar is suppressed when stderr is not a terminal (CI logs, ``tee``,
    redirected output), where a redrawing progress bar produces megabytes of
    control characters instead of readable logs.  Set ``MODELPEDIA_PROGRESS=1``
    to force it on anyway.
    """
    import os
    import sys

    forced = os.environ.get("MODELPEDIA_PROGRESS") == "1"
    if not forced and not sys.stderr.isatty():
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, leave=False, dynamic_ncols=True)
    except ImportError:  # pragma: no cover - tqdm is in requirements.txt
        return iterable


# --------------------------------------------------------------------------- #
# Argument parsing shared by the training scripts
# --------------------------------------------------------------------------- #
def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags every training script understands."""
    parser.add_argument("--data-root", default="data", help="where datasets are stored")
    parser.add_argument("--out-dir", default=None, help="where checkpoints/samples are written")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny run over a few batches, used to verify the pipeline end to end",
    )
    return parser


def resolve_out_dir(args: argparse.Namespace, default: str) -> Path:
    return ensure_dir(args.out_dir or default)


def maybe_limit(loader, args: argparse.Namespace, quick_batches: int = 4):
    """Yield at most ``quick_batches`` batches when ``--quick`` is set."""
    for i, batch in enumerate(loader):
        if args.quick and i >= quick_batches:
            return
        yield batch


def quick_len(loader, args: argparse.Namespace, quick_batches: int = 4) -> int:
    return min(len(loader), quick_batches) if args.quick else len(loader)
