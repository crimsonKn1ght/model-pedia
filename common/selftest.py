"""Sanity checks for the metric code, runnable with ``python -m common.selftest``.

These are not unit tests of implementation details. Each one pins a property the
metric must have for the numbers in the project READMEs to mean anything, using
inputs whose correct answer is known in advance:

* two halves of the real test set must score well on everything;
* a single repeated sample must show the mode-collapse signature -- high
  precision, near-zero recall;
* an invertible transform must round-trip exactly.

The mode-collapse check earned its place: a conditional GAN in this repository
reported precision 0.70 with recall exactly 0.00, and this is what established
that the metric was right and the model was wrong, rather than the reverse.
"""

from __future__ import annotations

import sys

import torch

from common import data as data_mod
from common import metrics as metrics_mod
from common import utils

PASS = "  ok  "
FAIL = " FAIL "


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        marker = PASS if condition else FAIL
        print(f"[{marker}] {name}{('  -- ' + detail) if detail else ''}")
        if not condition:
            self.failures += 1


def main() -> int:
    utils.set_seed(0)
    device = utils.get_device()
    checker = Checker()

    print("collecting features from the MNIST test split")
    loader = data_mod.get_dataloader(
        "mnist",
        root=str(utils.DEFAULT_DATA_ROOT),
        image_size=32,
        train=False,
        batch_size=128,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    extractor = metrics_mod.get_feature_extractor(
        "mnist", root=str(utils.DEFAULT_DATA_ROOT), image_size=32, device=device, num_workers=0
    )
    features = metrics_mod.features_from_loader(extractor, loader, device, max_samples=4096)
    a, b = features[:2048], features[2048:4096]

    # -- FID of a distribution against itself must be near zero ------------- #
    fid_self = metrics_mod.fid_from_features(*metrics_mod.standardize_pair(a, a))
    checker.check("FID(x, x) is ~0", abs(fid_self) < 1e-3, f"{fid_self:.6f}")

    # -- two halves of real data must look alike ---------------------------- #
    fid_split = metrics_mod.fid_from_features(*metrics_mod.standardize_pair(a, b))
    checker.check("FID between two real halves is small", fid_split < 2.0, f"{fid_split:.4f}")

    p, r = metrics_mod.precision_recall(*metrics_mod.standardize_pair(a, b))
    checker.check("real-vs-real precision > 0.8", p > 0.8, f"{p:.3f}")
    checker.check("real-vs-real recall > 0.8", r > 0.8, f"{r:.3f}")

    # -- the mode-collapse signature ---------------------------------------- #
    collapsed = a[:1].repeat(a.shape[0], 1) + torch.randn_like(a) * 0.01
    p_c, r_c = metrics_mod.precision_recall(*metrics_mod.standardize_pair(a, collapsed))
    checker.check("collapse keeps precision high", p_c > 0.9, f"{p_c:.3f}")
    checker.check("collapse drives recall to ~0", r_c < 0.05, f"{r_c:.3f}")

    # -- FID must be monotone in how wrong the samples are ------------------ #
    mild = b + torch.randn_like(b) * 0.5
    severe = b + torch.randn_like(b) * 3.0
    fid_mild = metrics_mod.fid_from_features(*metrics_mod.standardize_pair(a, mild))
    fid_severe = metrics_mod.fid_from_features(*metrics_mod.standardize_pair(a, severe))
    checker.check(
        "FID grows with corruption",
        fid_split < fid_mild < fid_severe,
        f"{fid_split:.2f} < {fid_mild:.2f} < {fid_severe:.2f}",
    )

    # -- KID must be near zero for identical distributions ------------------ #
    kid_self, _ = metrics_mod.kid_from_features(*metrics_mod.standardize_pair(a, a))
    checker.check("KID(x, x) is ~0", abs(kid_self) < 1e-2, f"{kid_self:.6f}")

    # -- SSIM and PSNR on identical images ---------------------------------- #
    images = next(iter(loader))[0]
    checker.check("SSIM(x, x) is 1", abs(metrics_mod.ssim(images, images) - 1.0) < 1e-4)
    checker.check("PSNR(x, x) is infinite", metrics_mod.psnr(images, images) == float("inf"))

    noisy = (images + torch.randn_like(images) * 0.3).clamp(-1, 1)
    checker.check("SSIM drops on noise", metrics_mod.ssim(noisy, images) < 0.9)

    print()
    if checker.failures:
        print(f"{checker.failures} check(s) FAILED")
        return 1
    print("all metric checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
