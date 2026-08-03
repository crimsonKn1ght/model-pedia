"""Metrics and plotting for generative models, including FID, KID and
generative precision/recall - all implemented here.

**About FID in this repository.** The published FID number is computed from the
2048-dimensional pool3 features of a particular ImageNet InceptionV3 checkpoint.
Downloading a 90 MB classifier to score 32x32 digits would be the largest
dependency in the repository by an order of magnitude, so instead a small
classifier is trained on the dataset itself and its penultimate features are used.

The consequence matters and is stated everywhere these numbers appear: **these FID
and KID values are not comparable with published ones.** They are comparable
*within* a project - between checkpoints, between architectures, between epochs -
which is all the comparisons here need. The feature network is trained once per
dataset with a fixed seed and cached, so it is the same ruler for every arm of
every study.

Precision and recall follow Kynkaanniemi et al. 2019: build a k-nearest-neighbour
manifold for each set of features and ask what fraction of one lands inside the
other. They separate the two failure modes FID conflates - precision falls when
samples are unrealistic, recall falls when they are realistic but cover only part
of the data.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB, one value per image."""
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(data_range**2 / mse)


# --------------------------------------------------------------------------
# the feature network FID and KID are measured through
# --------------------------------------------------------------------------


class FeatureNet(nn.Module):
    """Small classifier whose penultimate layer is the FID/KID feature space."""

    def __init__(self, in_channels: int = 1, width: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        channels = [in_channels, width, width * 2, width * 4]
        blocks = []
        for index in range(3):
            blocks += [
                nn.Conv2d(channels[index], channels[index + 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels[index + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = channels[-1]
        self.head = nn.Linear(self.feature_dim, num_classes)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(images)).flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))


def train_feature_net(
    loader,
    in_channels: int,
    num_classes: int,
    device: torch.device,
    epochs: int = 2,
    lr: float = 2e-3,
    seed: int = 1234,
    verbose: bool = False,
) -> FeatureNet:
    """Fit the feature network on real images. Labels are used *only* here."""
    generator_state = torch.get_rng_state()
    torch.manual_seed(seed)  # one fixed ruler, regardless of the caller's seed
    net = FeatureNet(in_channels, num_classes=num_classes).to(device)
    torch.set_rng_state(generator_state)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        meter, correct, total = AverageMeter(), 0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), images.size(0))
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
        if verbose:
            print(f"  feature net epoch {epoch + 1}/{epochs}: "
                  f"loss {meter.avg:.4f}, train accuracy {correct / max(total, 1):.4f}")
    net.eval()
    return net


def random_feature_net(in_channels: int, device: torch.device, seed: int = 1234) -> FeatureNet:
    """Fixed-seed untrained feature network, for datasets that have no labels.

    Random convolutional features are a weaker ruler than trained ones, but they
    are deterministic and free, and they still rank samples consistently - which is
    what a within-project comparison needs. Anything measured through them is
    labelled ``feature_net: random`` in the metrics file.
    """
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    net = FeatureNet(in_channels, num_classes=1).to(device)
    torch.set_rng_state(state)
    net.eval()
    return net


def load_or_train_feature_net(
    cache: str | Path | None,
    loader,
    in_channels: int,
    num_classes: int | None,
    device: torch.device,
    epochs: int = 2,
    verbose: bool = False,
) -> tuple[FeatureNet, str]:
    """Return ``(feature_net, kind)``; ``kind`` is ``classifier`` or ``random``.

    ``num_classes=None`` means the dataset has no labels, so no classifier can be
    fitted and the fixed-seed random network is used instead.
    """
    if num_classes is None:
        return random_feature_net(in_channels, device), "random"

    net = FeatureNet(in_channels, num_classes=num_classes).to(device)
    if cache is not None and Path(cache).exists():
        net.load_state_dict(torch.load(cache, map_location=device, weights_only=True))
        net.eval()
        return net, "classifier"

    net = train_feature_net(
        loader, in_channels, num_classes, device, epochs=epochs, verbose=verbose
    )
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), cache)
    return net, "classifier"


@torch.no_grad()
def features_from_loader(net: FeatureNet, loader, device: torch.device, limit: int | None = None):
    net.eval()
    collected, seen = [], 0
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        collected.append(net.features(images.to(device)).cpu())
        seen += images.size(0)
        if limit is not None and seen >= limit:
            break
    features = torch.cat(collected)
    return features[:limit] if limit is not None else features


@torch.no_grad()
def features_from_sampler(net: FeatureNet, sample_fn, total: int, batch_size: int, device):
    """Features for ``total`` generated images, produced ``batch_size`` at a time."""
    net.eval()
    collected = []
    remaining = total
    while remaining > 0:
        n = min(batch_size, remaining)
        images = sample_fn(n).to(device)
        collected.append(net.features(images).cpu())
        remaining -= n
    return torch.cat(collected)


# --------------------------------------------------------------------------
# distribution distances
# --------------------------------------------------------------------------


def _psd_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """Symmetric PSD square root via eigendecomposition (avoids a scipy dependency)."""
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min(0).sqrt()
    return (vectors * values) @ vectors.transpose(-1, -2)


def frechet_distance(
    mu_real: torch.Tensor,
    sigma_real: torch.Tensor,
    mu_fake: torch.Tensor,
    sigma_fake: torch.Tensor,
) -> float:
    """``|mu1-mu2|^2 + tr(S1) + tr(S2) - 2 tr((S1 S2)^{1/2})``.

    The trace of the matrix square root is computed from the eigenvalues of the
    symmetric product ``S1^{1/2} S2 S1^{1/2}``, which has the same spectrum as
    ``S1 S2`` and is numerically much better behaved.
    """
    difference = (mu_real - mu_fake).double()
    sigma_real = sigma_real.double()
    sigma_fake = sigma_fake.double()

    root = _psd_sqrt(sigma_real)
    middle = root @ sigma_fake @ root
    trace_sqrt = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum()
    value = difference.dot(difference) + sigma_real.trace() + sigma_fake.trace() - 2 * trace_sqrt
    return float(value.clamp_min(0))


def _mean_cov(features: torch.Tensor):
    features = features.double()
    mean = features.mean(dim=0)
    centred = features - mean
    covariance = centred.T @ centred / max(features.size(0) - 1, 1)
    return mean, covariance


def fid_score(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    mu_real, sigma_real = _mean_cov(real_features)
    mu_fake, sigma_fake = _mean_cov(fake_features)
    return frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


def _polynomial_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dimension = x.size(1)
    return (x @ y.T / dimension + 1.0) ** 3


def kid_score(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    subsets: int = 10,
    subset_size: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """Unbiased kernel inception distance: MMD^2 under a cubic polynomial kernel.

    Reported as ``(mean, std)`` over random subsets. Unlike FID it is unbiased for
    finite samples, so it is the more trustworthy of the two when only a few
    thousand images are available - which is always the case here.
    """
    generator = torch.Generator().manual_seed(seed)
    size = min(subset_size, real_features.size(0), fake_features.size(0))
    estimates = []

    for _ in range(subsets):
        real = real_features[torch.randperm(real_features.size(0), generator=generator)[:size]]
        fake = fake_features[torch.randperm(fake_features.size(0), generator=generator)[:size]]
        real, fake = real.double(), fake.double()

        k_rr = _polynomial_kernel(real, real)
        k_ff = _polynomial_kernel(fake, fake)
        k_rf = _polynomial_kernel(real, fake)
        # Exclude the diagonal terms: that is what makes the estimator unbiased.
        term_rr = (k_rr.sum() - k_rr.diag().sum()) / (size * (size - 1))
        term_ff = (k_ff.sum() - k_ff.diag().sum()) / (size * (size - 1))
        estimates.append(float(term_rr + term_ff - 2 * k_rf.mean()))

    return float(np.mean(estimates)), float(np.std(estimates))


def _knn_radii(features: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each point to its k-th nearest neighbour within the same set."""
    distances = torch.cdist(features, features)
    distances.fill_diagonal_(float("inf"))
    k = min(k, distances.size(0) - 1)
    return distances.topk(k, largest=False).values[:, -1]


def precision_recall(
    real_features: torch.Tensor, fake_features: torch.Tensor, k: int = 3
) -> tuple[float, float]:
    """Improved precision and recall for generative models.

    **Precision** is the fraction of generated samples that fall within the k-NN
    manifold of the real features - how often the model produces something
    plausible. **Recall** is the fraction of real samples inside the generated
    manifold - how much of the data distribution the model covers. A model that
    memorises one convincing image scores high precision and near-zero recall;
    FID alone will not tell you which of the two went wrong.
    """
    real = real_features.double()
    fake = fake_features.double()
    real_radii = _knn_radii(real, k)
    fake_radii = _knn_radii(fake, k)

    cross = torch.cdist(fake, real)
    precision = float((cross <= real_radii.unsqueeze(0)).any(dim=1).double().mean())
    recall = float((cross.T <= fake_radii.unsqueeze(0)).any(dim=1).double().mean())
    return precision, recall


def generative_metrics(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    kid_subsets: int = 10,
    kid_subset_size: int = 500,
    k: int = 3,
) -> dict:
    """FID, KID and precision/recall in one call, on cached features."""
    kid_mean, kid_std = kid_score(
        real_features, fake_features, subsets=kid_subsets, subset_size=kid_subset_size
    )
    precision, recall = precision_recall(real_features, fake_features, k=k)
    return {
        "fid": fid_score(real_features, fake_features),
        "kid": kid_mean,
        "kid_std": kid_std,
        "precision": precision,
        "recall": recall,
        "real_samples": int(real_features.size(0)),
        "fake_samples": int(fake_features.size(0)),
    }


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------


def show_image(ax, image: torch.Tensor) -> None:
    array = image.detach().cpu().clamp(0, 1).numpy()
    if array.shape[0] == 1:
        ax.imshow(array[0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(np.transpose(array, (1, 2, 0)))
    ax.axis("off")


def plot_image_grid(
    images: torch.Tensor, path: str | Path, columns: int = 8, title: str | None = None
) -> None:
    n = images.size(0)
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.3 * columns, 1.35 * rows + 0.4),
                             squeeze=False)
    for index in range(rows * columns):
        ax = axes[index // columns][index % columns]
        if index < n:
            show_image(ax, images[index])
        else:
            ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_image_rows(
    rows: list[tuple[str, torch.Tensor]], path: str | Path, title: str | None = None
) -> None:
    n_rows = len(rows)
    n_cols = min(len(images) for _, images in rows)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.35 * n_cols, 1.45 * n_rows + 0.4), squeeze=False
    )
    for row_idx, (label, images) in enumerate(rows):
        for col_idx in range(n_cols):
            show_image(axes[row_idx][col_idx], images[col_idx])
            if col_idx == 0:
                axes[row_idx][col_idx].set_ylabel(label, fontsize=8)
                axes[row_idx][col_idx].axis("on")
                axes[row_idx][col_idx].set_xticks([])
                axes[row_idx][col_idx].set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(history: dict, path: str | Path, keys: tuple[str, ...] = ("loss",)) -> None:
    present = [key for key in keys if f"train_{key}" in history or f"val_{key}" in history]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), squeeze=False)
    for ax, key in zip(axes[0], present):
        for split in ("train", "val"):
            values = history.get(f"{split}_{key}")
            if values:
                ax.plot(range(1, len(values) + 1), values, label=split, marker="o", markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pca_2d(features: torch.Tensor) -> torch.Tensor:
    centred = features - features.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(centred, q=min(8, *centred.shape))
    return centred @ components[:, :2]


def plot_latent_scatter(
    latents: torch.Tensor,
    labels: torch.Tensor,
    path: str | Path,
    class_names: list[str] | None = None,
    title: str | None = None,
    max_points: int = 3000,
) -> None:
    """Latent codes in 2D, coloured by class. PCA first if the latent is wider than 2."""
    n = min(max_points, latents.size(0))
    points = latents[:n]
    points = (points if points.size(1) == 2 else pca_2d(points)).numpy()
    targets = labels[:n].numpy()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for cls in np.unique(targets):
        mask = targets == cls
        name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        ax.scatter(points[mask, 0], points[mask, 1], s=4, alpha=0.5, label=name)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=2)
    ax.set_title(title or "latent space")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bars(
    labels: list[str],
    series: dict[str, list[float]],
    path: str | Path,
    ylabel: str = "score",
    title: str | None = None,
) -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 3, 4))
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(bars, fmt="%.3g", fontsize=7, padding=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
