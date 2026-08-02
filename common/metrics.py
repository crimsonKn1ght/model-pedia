"""Evaluation metrics for the generative projects.

The headline numbers are FID and KID.  Published FID uses an InceptionV3
network trained on ImageNet; downloading those weights is a 100 MB dependency
that also makes little sense for 28x28 MNIST digits.  This repository therefore
computes the same statistics in a **domain-specific feature space**: a small
convolutional network is trained once per dataset (a couple of epochs, seconds
on a CPU) and cached under ``outputs/feature-extractors``.

Consequences, stated plainly:

* the scores are directly comparable **between projects in this repository**,
  because every project reuses the same cached extractor for a given dataset;
* they are **not** comparable to FID values quoted in papers;
* lower is better for FID and KID, exactly as usual.

Pass ``--feature-extractor inception`` to any evaluation script to switch to
torchvision's ImageNet InceptionV3 instead, if the weights are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import data as data_mod
from common.utils import DEFAULT_FEATURE_CACHE, ensure_dir, get_device, progress, set_seed


# --------------------------------------------------------------------------- #
# Feature extractor
# --------------------------------------------------------------------------- #
class SmallConvFeatures(nn.Module):
    """A compact CNN whose penultimate activations act as the feature space."""

    # 64 dimensions is a deliberate choice: FID fits a full covariance matrix,
    # which needs comfortably more samples than dimensions to be stable. These
    # projects evaluate on 1-2k samples, and per-class breakdowns on a few
    # hundred, so a narrow feature space keeps those estimates well conditioned.
    def __init__(self, channels: int = 1, num_classes: int = 10, feature_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.project = nn.Linear(128, feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim
        self.channels = channels

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            x = x.mean(dim=1, keepdim=True) if self.channels == 1 else x.repeat(1, 3, 1, 1)
        return self.project(self.body(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(F.relu(self.features(x)))


class InceptionFeatures(nn.Module):
    """torchvision InceptionV3 pool3 features, for comparison with the literature."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = nn.Identity()
        net.eval()
        self.net = net
        self.feature_dim = 2048
        self.channels = 3

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # torchvision expects ImageNet normalisation on [0, 1] inputs.
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return self.net((x - mean) / std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def _rotation_batch(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a 4-way rotation-prediction task, used for unlabelled datasets."""
    outs, labels = [], []
    for k in range(4):
        outs.append(torch.rot90(x, k, dims=(2, 3)))
        labels.append(torch.full((x.shape[0],), k, dtype=torch.long))
    return torch.cat(outs), torch.cat(labels)


def train_feature_extractor(
    dataset_name: str,
    root: str = "data",
    image_size: int = 32,
    device: Optional[torch.device] = None,
    epochs: int = 2,
    batch_size: int = 256,
    max_batches: int = 120,
    num_workers: int = 2,
) -> SmallConvFeatures:
    """Train (and return) the small CNN whose features define the metric space."""
    device = device or get_device()
    meta = data_mod.info(dataset_name)
    set_seed(1234)

    supervised = meta.num_classes > 1
    num_classes = meta.num_classes if supervised else 4
    model = SmallConvFeatures(meta.channels, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    # Trained on [0, 1] inputs, which is the range every ``features_*`` helper
    # hands over after denormalising. Keeping both sides in one range matters:
    # otherwise the metric is measured in a space the extractor never saw.
    loader = data_mod.get_dataloader(
        dataset_name,
        root=root,
        image_size=image_size,
        train=True,
        batch_size=batch_size if supervised else max(batch_size // 4, 32),
        normalize="unit",
        num_workers=num_workers,
    )
    model.train()
    for _ in range(epochs):
        for i, (x, y) in enumerate(progress(loader, desc=f"feature extractor [{dataset_name}]")):
            if i >= max_batches:
                break
            if supervised:
                x, y = x.to(device), y.to(device)
            else:
                x, y = _rotation_batch(x)
                x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def get_feature_extractor(
    dataset_name: str,
    root: str = "data",
    image_size: int = 32,
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None,
    kind: str = "small-cnn",
    num_workers: int = 2,
) -> nn.Module:
    """Load the cached extractor for ``dataset_name``, training it on first use."""
    device = device or get_device()
    if kind == "inception":
        return InceptionFeatures().to(device).eval()
    if kind != "small-cnn":
        raise ValueError(f"unknown feature extractor {kind!r}")

    cache = ensure_dir(cache_dir or DEFAULT_FEATURE_CACHE) / f"{dataset_name}-{image_size}.pt"
    meta = data_mod.info(dataset_name)
    num_classes = meta.num_classes if meta.num_classes > 1 else 4
    if cache.exists():
        model = SmallConvFeatures(meta.channels, num_classes).to(device)
        model.load_state_dict(torch.load(cache, map_location=device))
        return model.eval()

    model = train_feature_extractor(
        dataset_name, root=root, image_size=image_size, device=device, num_workers=num_workers
    )
    torch.save(model.state_dict(), cache)
    return model.eval()


# --------------------------------------------------------------------------- #
# Feature collection
# --------------------------------------------------------------------------- #
@torch.no_grad()
def features_from_loader(
    extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 2048,
    normalize: str = "tanh",
) -> torch.Tensor:
    """Feature matrix for real images coming out of a dataloader."""
    extractor.eval()
    chunks, seen = [], 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = data_mod.denormalize(x, normalize).to(device)
        chunks.append(extractor.features(x).cpu())
        seen += x.shape[0]
        if seen >= max_samples:
            break
    return torch.cat(chunks)[:max_samples]


@torch.no_grad()
def features_from_sampler(
    extractor: nn.Module,
    sampler: Callable[[int], torch.Tensor],
    device: torch.device,
    num_samples: int = 2048,
    batch_size: int = 256,
    normalize: str = "tanh",
) -> torch.Tensor:
    """Feature matrix for generated images produced by ``sampler(n)``."""
    extractor.eval()
    chunks, seen = [], 0
    while seen < num_samples:
        n = min(batch_size, num_samples - seen)
        x = sampler(n)
        x = data_mod.denormalize(x.to(device), normalize)
        chunks.append(extractor.features(x).cpu())
        seen += n
    return torch.cat(chunks)[:num_samples]


# --------------------------------------------------------------------------- #
# FID / KID
# --------------------------------------------------------------------------- #
def _mean_cov(features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    f = features.detach().cpu().double().numpy()
    return f.mean(axis=0), np.cov(f, rowvar=False)


def fid_from_features(real: torch.Tensor, fake: torch.Tensor) -> float:
    """Frechet distance between two Gaussians fitted to the feature sets."""
    from scipy import linalg

    mu1, sigma1 = _mean_cov(real)
    mu2, sigma2 = _mean_cov(fake)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # Singular product; nudge the diagonals and retry (standard practice).
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def kid_from_features(
    real: torch.Tensor,
    fake: torch.Tensor,
    num_subsets: int = 50,
    subset_size: int = 256,
    degree: int = 3,
) -> Tuple[float, float]:
    """Unbiased MMD^2 with a polynomial kernel, averaged over random subsets.

    Returns ``(mean, std)``.  Unlike FID this is unbiased for small sample
    counts, which matters at the sample sizes these lightweight projects use.
    """
    r = real.detach().cpu().double()
    f = fake.detach().cpu().double()
    d = r.shape[1]
    subset_size = int(min(subset_size, r.shape[0], f.shape[0]))
    if subset_size < 8:
        return float("nan"), float("nan")

    values = []
    generator = torch.Generator().manual_seed(0)
    for _ in range(num_subsets):
        x = r[torch.randperm(r.shape[0], generator=generator)[:subset_size]]
        y = f[torch.randperm(f.shape[0], generator=generator)[:subset_size]]
        kxx = (x @ x.t() / d + 1.0) ** degree
        kyy = (y @ y.t() / d + 1.0) ** degree
        kxy = (x @ y.t() / d + 1.0) ** degree
        m = subset_size
        term_xx = (kxx.sum() - kxx.diag().sum()) / (m * (m - 1))
        term_yy = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
        values.append(float(term_xx + term_yy - 2.0 * kxy.mean()))
    return float(np.mean(values)), float(np.std(values))


# --------------------------------------------------------------------------- #
# Improved precision / recall (Kynkaanniemi et al., 2019)
# --------------------------------------------------------------------------- #
def _knn_radii(features: torch.Tensor, k: int) -> torch.Tensor:
    dist = torch.cdist(features, features)
    # The k-th neighbour excluding the point itself.
    return dist.kthvalue(k + 1, dim=1).values


def precision_recall(
    real: torch.Tensor,
    fake: torch.Tensor,
    k: int = 3,
    max_samples: int = 1024,
) -> Tuple[float, float]:
    """Fraction of fakes inside the real manifold, and vice versa."""
    r = real.detach().cpu().float()[:max_samples]
    f = fake.detach().cpu().float()[:max_samples]
    if min(r.shape[0], f.shape[0]) <= k + 1:
        return float("nan"), float("nan")

    radii_r = _knn_radii(r, k)
    radii_f = _knn_radii(f, k)
    cross = torch.cdist(f, r)  # (n_fake, n_real)

    precision = float((cross <= radii_r.unsqueeze(0)).any(dim=1).float().mean())
    recall = float((cross.t() <= radii_f.unsqueeze(0)).any(dim=1).float().mean())
    return precision, recall


def standardize_pair(
    real: torch.Tensor, fake: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Whiten both feature sets using the *real* set's per-dimension statistics.

    The learned feature space has an arbitrary scale, and the polynomial kernel
    behind KID is cubic in that scale, so raw features produce values in the
    thousands that are hard to read.  Applying one fixed affine map -- derived
    from the reference distribution only, never from the generated one -- puts
    every dataset on a comparable footing without changing which of two models
    scores better.
    """
    mean = real.mean(dim=0, keepdim=True)
    std = real.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (real - mean) / std, (fake - mean) / std


def generative_metrics(
    real: torch.Tensor,
    fake: torch.Tensor,
    with_precision_recall: bool = True,
    standardize: bool = True,
) -> Dict[str, float]:
    """FID + KID (+ precision/recall) in one call."""
    if standardize:
        real, fake = standardize_pair(real, fake)
    kid_mean, kid_std = kid_from_features(real, fake)
    out = {
        "fid": fid_from_features(real, fake),
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "num_real": float(real.shape[0]),
        "num_fake": float(fake.shape[0]),
    }
    if with_precision_recall:
        p, r = precision_recall(real, fake)
        out["precision"] = p
        out["recall"] = r
    return out


# --------------------------------------------------------------------------- #
# Reconstruction / paired-image metrics
# --------------------------------------------------------------------------- #
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    """Global SSIM with an 11x11 Gaussian window."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    channels = pred.shape[1]

    coords = torch.arange(11, dtype=torch.float32, device=pred.device) - 5.0
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.t() @ g).expand(channels, 1, 11, 11).contiguous()

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, window, padding=5, groups=channels)

    mu_p, mu_t = blur(pred), blur(target)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t
    sigma_p = blur(pred * pred) - mu_p2
    sigma_t = blur(target * target) - mu_t2
    sigma_pt = blur(pred * target) - mu_pt

    numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p2 + mu_t2 + c1) * (sigma_p + sigma_t + c2)
    return float((numerator / denominator).mean())


def lpips_proxy(extractor: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> float:
    """Perceptual distance in the same feature space the FID uses.

    This stands in for LPIPS, which would require yet another pretrained
    network.  It is a cosine distance between features, so 0 means identical.
    """
    with torch.no_grad():
        fp = F.normalize(extractor.features(pred), dim=1)
        ft = F.normalize(extractor.features(target), dim=1)
    return float((1.0 - (fp * ft).sum(dim=1)).mean())


# --------------------------------------------------------------------------- #
# Classifier accuracy on generated images (used by the conditional GAN)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def classifier_accuracy(
    classifier: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    normalize: str = "tanh",
) -> float:
    classifier.eval()
    x = data_mod.denormalize(images.to(device), normalize)
    pred = classifier(x).argmax(dim=1).cpu()
    return float((pred == labels.cpu()).float().mean())


def format_metrics(metrics: Dict[str, float], indent: str = "  ") -> str:
    lines = []
    for key in sorted(metrics):
        value = metrics[key]
        lines.append(f"{indent}{key:<22} {value:.4f}" if isinstance(value, float) else f"{indent}{key:<22} {value}")
    return "\n".join(lines)
