"""Check every hand-written metric in this repository against known answers.

    python scripts/metric_selftest.py
    python scripts/metric_selftest.py --groups classification detection

Almost every number these projects report comes from a metric implemented here
rather than pulled from a library: confusion matrices and F1, PSNR and SSIM, IoU
and Dice, non-maximum suppression and mean average precision, BLEU, METEOR and
CIDEr-D, FID, KID, generative precision/recall, and bits per dimension. That is
deliberate - they are short enough to be worth reading - but it means a mistake in
any of them silently corrupts every result downstream, and a smoke test that only
checks shapes will not notice.

So each check below feeds in an input whose correct answer is known in advance,
either analytically or by construction, and asserts the metric returns it. Nothing
is downloaded and nothing is trained; the whole file runs in seconds on a CPU.

These are property checks, not unit tests of internals. The properties are the ones
the READMEs lean on when they interpret a number - that FID between two unit
Gaussians a distance ``d`` apart is exactly ``d^2``, that mode collapse shows up as
high precision with near-zero recall, that mean IoU and pixel accuracy disagree in a
specific computable way on an unbalanced mask, that a flow's inverse really inverts
its forward pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def load(project: str, module: str):
    """Import ``<project>/<module>.py`` under a unique name.

    Every project keeps its own ``utils.py`` and ``model.py``, so they have to be
    loaded by path and registered under distinct names - a plain ``import utils``
    would bind whichever directory happened to be on ``sys.path`` first.
    """
    path = REPO_ROOT / project / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"{project}__{module}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError(f"cannot load {path}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


class Checker:
    """Collects pass/fail lines so one failure does not hide the rest."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        suffix = f"   {detail}" if detail else ""
        if condition:
            self.passes += 1
            print(f"  [ ok ] {name}{suffix}")
        else:
            self.failures.append(name)
            print(f"  [FAIL] {name}{suffix}")

    def close(self, name: str, value: float, expected: float, tol: float) -> None:
        """Assert ``value ~= expected``, printing both either way."""
        self.check(
            f"{name} == {expected:.6g}",
            abs(value - expected) <= tol,
            f"got {value:.6g}",
        )


# ---------------------------------------------------------------------------
# classification: confusion matrix, precision/recall/F1
# ---------------------------------------------------------------------------
def check_classification(checker: Checker, seed: int = 0) -> None:
    utils = load("01_classic_classifier", "utils")

    # Three classes, and a deliberately lopsided set of mistakes so that accuracy
    # and macro F1 have to come apart. Class 2 is never predicted correctly.
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    cm = utils.confusion_matrix(y_true, y_pred, 3)

    checker.check("confusion matrix rows sum to the true support",
                  cm.sum(axis=1).tolist() == [4, 4, 2], f"got {cm.sum(axis=1).tolist()}")
    checker.check("confusion matrix total equals the sample count",
                  int(cm.sum()) == 10, f"got {int(cm.sum())}")

    metrics = utils.per_class_metrics(cm)
    # 3 of class 0 and 4 of class 1 are right, nothing else: 7 of 10.
    checker.close("accuracy", metrics["accuracy"], 0.7, 1e-12)
    # Class 0: tp 3, predicted 4 -> precision 3/4; actual 4 -> recall 3/4; F1 3/4.
    checker.close("class 0 precision", float(metrics["precision"][0]), 0.75, 1e-12)
    checker.close("class 0 recall", float(metrics["recall"][0]), 0.75, 1e-12)
    # Class 1: tp 4, predicted 6 -> precision 2/3; actual 4 -> recall 1; F1 0.8.
    checker.close("class 1 precision", float(metrics["precision"][1]), 2 / 3, 1e-12)
    checker.close("class 1 recall", float(metrics["recall"][1]), 1.0, 1e-12)
    checker.close("class 1 F1", float(metrics["f1"][1]), 0.8, 1e-12)
    # Class 2 is never predicted: everything about it is zero, and it must not be
    # a division-by-zero NaN, because macro F1 averages over it.
    checker.close("class 2 F1 (never predicted)", float(metrics["f1"][2]), 0.0, 1e-12)
    checker.close("macro F1", metrics["macro_f1"], (0.75 + 0.8 + 0.0) / 3, 1e-12)
    checker.check("macro F1 is below accuracy when a class is missed",
                  metrics["macro_f1"] < metrics["accuracy"],
                  f"{metrics['macro_f1']:.4f} < {metrics['accuracy']:.4f}")

    # Perfect prediction: every derived number pinned at 1.
    perfect = utils.per_class_metrics(utils.confusion_matrix(y_true, y_true, 3))
    checker.close("perfect accuracy", perfect["accuracy"], 1.0, 1e-12)
    checker.close("perfect macro F1", perfect["macro_f1"], 1.0, 1e-12)


# ---------------------------------------------------------------------------
# reconstruction: PSNR and SSIM
# ---------------------------------------------------------------------------
def check_reconstruction(checker: Checker, seed: int = 0) -> None:
    utils = load("07_vector_quantized_ae", "utils")
    torch.manual_seed(seed)
    images = torch.rand(4, 1, 32, 32)

    # Both metrics return one value per image, so a batch is a batch of scores.
    checker.check("PSNR returns one value per image",
                  tuple(utils.psnr(images, images).shape) == (4,),
                  f"got {tuple(utils.psnr(images, images).shape)}")

    # An identical pair has zero MSE, which the implementation clamps to 1e-12 so
    # the result stays finite and JSON-serialisable. That puts the ceiling at
    # 10 log10(1 / 1e-12) = 120 dB exactly, rather than at infinity.
    checker.close("PSNR(x, x) sits at the 120 dB clamp ceiling",
                  float(utils.psnr(images, images)[0]), 120.0, 1e-6)
    checker.close("SSIM(x, x) == 1", float(utils.ssim(images, images).mean()), 1.0, 1e-4)

    # PSNR has a closed form: 10 log10(range^2 / mse). A constant offset makes the
    # MSE exact, so the metric can be checked against arithmetic rather than a
    # previous run of itself. Compared per image, since averaging dB values and
    # taking the dB of an averaged MSE are not the same number.
    offset = (images + 0.1).clamp(0, 1)
    mse = (offset - images).pow(2).flatten(1).mean(dim=1)
    expected = 10 * torch.log10(1.0 / mse)
    error = float((utils.psnr(offset, images) - expected).abs().max())
    checker.check("PSNR matches 10 log10(1/mse) per image", error < 1e-4,
                  f"max err {error:.2e}")

    # Both metrics must be monotone in corruption, or no reconstruction table in
    # the repository means anything.
    mild = (images + torch.randn_like(images) * 0.05).clamp(0, 1)
    severe = (images + torch.randn_like(images) * 0.30).clamp(0, 1)
    psnr_mild = float(utils.psnr(mild, images).mean())
    psnr_severe = float(utils.psnr(severe, images).mean())
    checker.check("PSNR falls as noise grows", psnr_mild > psnr_severe,
                  f"{psnr_mild:.2f} dB > {psnr_severe:.2f} dB")
    ssim_mild = float(utils.ssim(mild, images).mean())
    ssim_severe = float(utils.ssim(severe, images).mean())
    checker.check("SSIM falls as noise grows", 1.0 > ssim_mild > ssim_severe,
                  f"{ssim_mild:.4f} > {ssim_severe:.4f}")

    # SSIM is bounded above by 1; a blur must not be allowed to beat the original.
    blurred = torch.nn.functional.avg_pool2d(images, 3, 1, 1)
    checker.check("SSIM(blur, x) < 1", float(utils.ssim(blurred, images).mean()) < 1.0)


# ---------------------------------------------------------------------------
# VQ codebook health
# ---------------------------------------------------------------------------
def check_codebook(checker: Checker, seed: int = 0) -> None:
    utils = load("07_vector_quantized_ae", "utils")

    # Perplexity is exp of the histogram entropy, so uniform usage of K codes must
    # give exactly K, and total collapse exactly 1. Those two endpoints are what
    # make the number readable in project 07's tables.
    uniform = torch.arange(128).repeat(10)
    stats = utils.codebook_usage(uniform, 128)
    checker.close("uniform usage -> perplexity == K", stats["perplexity"], 128.0, 1e-6)
    checker.check("uniform usage -> all codes used", stats["codes_used"] == 128,
                  f"got {stats['codes_used']}")
    checker.close("uniform usage fraction", stats["usage_fraction"], 1.0, 1e-12)

    collapsed = torch.zeros(1000, dtype=torch.long)
    stats = utils.codebook_usage(collapsed, 128)
    checker.close("total collapse -> perplexity == 1", stats["perplexity"], 1.0, 1e-6)
    checker.check("total collapse -> one code used", stats["codes_used"] == 1,
                  f"got {stats['codes_used']}")

    # Half the codebook used uniformly must read as exactly half, which is the
    # case the README's "a 128-entry codebook running at perplexity 49" relies on.
    half = torch.arange(64).repeat(10)
    stats = utils.codebook_usage(half, 128)
    checker.close("half the codebook -> perplexity == K/2", stats["perplexity"], 64.0, 1e-6)
    checker.close("half the codebook -> usage 0.5", stats["usage_fraction"], 0.5, 1e-12)


# ---------------------------------------------------------------------------
# generative metrics: FID, KID, precision/recall
# ---------------------------------------------------------------------------
def check_generative(checker: Checker, seed: int = 0) -> None:
    utils = load("06_variational_autoencoder", "utils")
    torch.manual_seed(seed)

    # -- the analytic case -------------------------------------------------- #
    # For two Gaussians with identity covariance whose means are a distance d
    # apart, FID reduces to exactly d^2. Feeding the moments in directly (rather
    # than sampling them) removes estimator noise, so this is an exact check of
    # the Frechet formula and the matrix square root behind it.
    dimension = 16
    identity = torch.eye(dimension)
    mu_zero = torch.zeros(dimension)
    for distance in (0.0, 1.0, 3.0):
        mu_shift = torch.zeros(dimension)
        mu_shift[0] = distance
        value = utils.frechet_distance(mu_zero, identity, mu_shift, identity)
        checker.close(f"FID between unit Gaussians d={distance:g} apart",
                      value, distance**2, 1e-8)

    # Same means, different scales: FID = tr(S1) + tr(S2) - 2 tr((S1 S2)^1/2),
    # which for s*I and t*I is dimension * (sqrt(s) - sqrt(t))^2.
    value = utils.frechet_distance(mu_zero, identity * 4.0, mu_zero, identity * 1.0)
    checker.close("FID between scaled Gaussians", value, dimension * (2.0 - 1.0) ** 2, 1e-8)

    # -- sampled features --------------------------------------------------- #
    features_a = torch.randn(2000, 32)
    features_b = torch.randn(2000, 32)

    checker.close("FID(x, x) == 0", utils.fid_score(features_a, features_a), 0.0, 1e-6)

    # FID is biased at finite sample counts: two independent draws from the *same*
    # distribution do not score 0. KID's estimator is unbiased and does. This is
    # the reason project 06's README calls KID the more trustworthy of the two, so
    # it is worth pinning rather than trusting.
    fid_same = utils.fid_score(features_a, features_b)
    kid_same, _ = utils.kid_score(features_a, features_b, subsets=5, subset_size=400)
    checker.check("FID is biased above 0 for identical distributions",
                  fid_same > 0.05, f"FID {fid_same:.4f}")
    checker.check("KID is unbiased for identical distributions",
                  abs(kid_same) < 0.02, f"KID {kid_same:+.5f}")

    # Both must grow with the real discrepancy.
    shifted = features_b + 1.0
    far = features_b + 4.0
    fid_shift = utils.fid_score(features_a, shifted)
    fid_far = utils.fid_score(features_a, far)
    checker.check("FID grows with distributional shift",
                  fid_same < fid_shift < fid_far,
                  f"{fid_same:.3f} < {fid_shift:.3f} < {fid_far:.3f}")
    kid_shift, _ = utils.kid_score(features_a, shifted, subsets=5, subset_size=400)
    checker.check("KID grows with distributional shift",
                  kid_same < kid_shift, f"{kid_same:+.5f} < {kid_shift:+.5f}")

    # -- precision and recall separate the two failure modes ---------------- #
    # Two independent draws from the same distribution. The absolute level depends
    # on the feature dimension and on k - a k=3 manifold estimate in 32 dimensions
    # is conservative and lands near 0.7, not 1.0 - so the property worth pinning
    # is that the two numbers are high *and symmetric*: neither failure mode is
    # present, and that is what distinguishes this from the two cases below.
    precision, recall = utils.precision_recall(features_a[:500], features_b[:500])
    checker.check("real vs real: precision high", precision > 0.6, f"{precision:.3f}")
    checker.check("real vs real: recall high", recall > 0.6, f"{recall:.3f}")
    checker.check("real vs real: precision and recall agree",
                  abs(precision - recall) < 0.1, f"|{precision:.3f} - {recall:.3f}|")

    # Mode collapse: one point repeated. Every sample is plausible (it sits inside
    # the real manifold) and the samples cover almost none of the data.
    collapsed = features_a[:1].repeat(500, 1) + torch.randn(500, 32) * 0.01
    precision, recall = utils.precision_recall(features_a[:500], collapsed)
    checker.check("mode collapse: precision stays high", precision > 0.9, f"{precision:.3f}")
    checker.check("mode collapse: recall near zero", recall < 0.05, f"{recall:.3f}")

    # The opposite failure: samples far too spread out. Now coverage is fine and
    # plausibility is not, which is the asymmetry FID alone cannot report.
    scattered = torch.randn(500, 32) * 4.0
    precision, recall = utils.precision_recall(features_a[:500], scattered)
    checker.check("over-dispersed samples: precision collapses",
                  precision < 0.2, f"{precision:.3f}")

    # -- the PSD square root itself ----------------------------------------- #
    torch.manual_seed(seed + 1)
    factor = torch.randn(12, 12)
    psd = (factor @ factor.T).double() + torch.eye(12).double() * 1e-3
    root = utils._psd_sqrt(psd)
    error = float((root @ root - psd).abs().max())
    checker.check("PSD square root: sqrt(S) @ sqrt(S) == S", error < 1e-8, f"max err {error:.2e}")


# ---------------------------------------------------------------------------
# segmentation: IoU, Dice, pixel accuracy
# ---------------------------------------------------------------------------
def check_segmentation(checker: Checker, seed: int = 0) -> None:
    utils = load("18_semantic_segmentation", "utils")

    # Perfect prediction first.
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = 1
    matrix = utils.ConfusionMatrix(2)
    matrix.update(target, target)
    checker.close("perfect mean IoU", matrix.mean_iou(), 1.0, 1e-9)
    checker.close("perfect pixel accuracy", matrix.pixel_accuracy(), 1.0, 1e-9)
    checker.close("perfect mean Dice", matrix.mean_dice(), 1.0, 1e-9)

    # The case project 18 is about. 100 pixels: 90 of class 0, 10 of class 1, and
    # the model predicts class 0 everywhere - the majority-class baseline.
    #   pixel accuracy = 90/100                       = 0.90
    #   IoU class 0    = 90 / (90 + 10 + 0)           = 0.90
    #   IoU class 1    = 0  / (0 + 0 + 10)            = 0.00
    #   mean IoU                                      = 0.45
    #   Dice class 0   = 2*90 / (2*90 + 10 + 0)       = 180/190
    target = torch.cat([torch.zeros(90, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    prediction = torch.zeros(100, dtype=torch.long)
    matrix = utils.ConfusionMatrix(2)
    matrix.update(prediction, target)
    # Tolerances here are 1e-7 rather than exact: the counts are integers but the
    # ratios are computed in float32, where 0.9 is not representable.
    checker.close("majority baseline: pixel accuracy", matrix.pixel_accuracy(), 0.90, 1e-7)
    checker.close("majority baseline: IoU of the majority class",
                  float(matrix.iou()[0]), 0.90, 1e-7)
    checker.close("majority baseline: IoU of the ignored class",
                  float(matrix.iou()[1]), 0.0, 1e-9)
    checker.close("majority baseline: mean IoU", matrix.mean_iou(), 0.45, 1e-7)
    checker.close("majority baseline: Dice of the majority class",
                  float(matrix.dice()[0]), 180 / 190, 1e-7)
    # The point of reporting both: one model, two defensible-looking numbers, and
    # a factor of two between them. A project that quoted only pixel accuracy
    # would present a model that has never once predicted class 1 as 90% correct.
    checker.check("pixel accuracy flatters the majority baseline over mean IoU",
                  matrix.pixel_accuracy() - matrix.mean_iou() > 0.4,
                  f"accuracy {matrix.pixel_accuracy():.2f} vs mIoU {matrix.mean_iou():.2f}")

    # Dice is never below IoU, for any confusion matrix.
    torch.manual_seed(seed)
    matrix = utils.ConfusionMatrix(4)
    matrix.update(torch.randint(0, 4, (2000,)), torch.randint(0, 4, (2000,)))
    checker.check("Dice >= IoU class by class",
                  bool((matrix.dice() >= matrix.iou() - 1e-9).all()))

    # A class absent from the ground truth must be excluded rather than scored 0,
    # or every mean IoU in the project would be dragged down by unused labels.
    matrix = utils.ConfusionMatrix(5)
    matrix.update(target, target)
    checker.close("classes absent from the target are excluded from mean IoU",
                  matrix.mean_iou(), 1.0, 1e-9)

    # Accumulation must be additive: two updates of the same batch equal one
    # update of it repeated, since the matrix is summed over batches.
    single = utils.ConfusionMatrix(2)
    single.update(prediction, target)
    doubled = utils.ConfusionMatrix(2)
    doubled.update(prediction, target)
    doubled.update(prediction, target)
    checker.close("accumulating twice leaves the ratios unchanged",
                  doubled.mean_iou(), single.mean_iou(), 1e-9)


# ---------------------------------------------------------------------------
# detection: IoU, NMS, average precision
# ---------------------------------------------------------------------------
def check_detection(checker: Checker, seed: int = 0) -> None:
    utils = load("19_object_detection", "utils")

    box = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    checker.close("IoU(box, box) == 1", float(utils.box_iou(box, box)[0, 0]), 1.0, 1e-7)

    # Half-overlap: intersection 2, areas 4 and 4, union 6 -> 1/3 exactly.
    other = torch.tensor([[1.0, 0.0, 3.0, 2.0]])
    checker.close("IoU of half-overlapping boxes", float(utils.box_iou(box, other)[0, 0]),
                  1 / 3, 1e-7)
    # Disjoint, and touching-but-not-overlapping, must both be 0.
    checker.close("IoU of disjoint boxes",
                  float(utils.box_iou(box, torch.tensor([[5.0, 5.0, 6.0, 6.0]]))[0, 0]),
                  0.0, 1e-9)
    checker.close("IoU of edge-touching boxes",
                  float(utils.box_iou(box, torch.tensor([[2.0, 0.0, 4.0, 2.0]]))[0, 0]),
                  0.0, 1e-9)
    # Containment: the small box inside the big one -> area ratio.
    checker.close("IoU of a box inside another",
                  float(utils.box_iou(torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
                                      torch.tensor([[1.0, 1.0, 3.0, 3.0]]))[0, 0]),
                  4 / 16, 1e-7)

    # CIoU: zero for identical boxes, and positive with a gradient even when the
    # boxes do not overlap at all - which is the whole reason it replaces IoU loss.
    loss_same = float(utils.complete_iou_loss(box, box)[0])
    checker.close("CIoU loss of a box against itself == 0", loss_same, 0.0, 1e-6)
    disjoint = torch.tensor([[10.0, 10.0, 12.0, 12.0]])
    loss_disjoint = float(utils.complete_iou_loss(box, disjoint)[0])
    checker.check("CIoU loss is finite and large for disjoint boxes",
                  1.0 < loss_disjoint < 10.0, f"{loss_disjoint:.4f}")
    nearer = torch.tensor([[3.0, 3.0, 5.0, 5.0]])
    checker.check("CIoU loss falls as a disjoint box approaches",
                  float(utils.complete_iou_loss(box, nearer)[0]) < loss_disjoint)

    # -- NMS ---------------------------------------------------------------- #
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],   # score 0.9, kept
        [1.0, 1.0, 11.0, 11.0],   # score 0.8, suppressed by the first
        [50.0, 50.0, 60.0, 60.0],  # score 0.7, far away, kept
    ])
    scores = torch.tensor([0.9, 0.8, 0.7])
    kept = utils.nms(boxes, scores, iou_threshold=0.5)
    checker.check("NMS suppresses the overlapping duplicate",
                  kept.tolist() == [0, 2], f"kept {kept.tolist()}")
    # Raising the threshold above the pair's IoU must keep all three.
    kept_all = utils.nms(boxes, scores, iou_threshold=0.95)
    checker.check("NMS keeps everything at a permissive threshold",
                  sorted(kept_all.tolist()) == [0, 1, 2], f"kept {kept_all.tolist()}")
    checker.check("NMS returns the highest score first",
                  int(utils.nms(boxes, scores, 0.5)[0]) == 0)
    checker.check("NMS on an empty input returns an empty index",
                  utils.nms(torch.zeros(0, 4), torch.zeros(0)).numel() == 0)

    # Class-wise NMS must not suppress across classes: two objects of different
    # classes are allowed to overlap completely.
    labels = torch.tensor([0, 1, 0])
    kept = utils.class_wise_nms(boxes, scores, labels, iou_threshold=0.5)
    checker.check("class-wise NMS keeps overlapping boxes of different classes",
                  sorted(kept.tolist()) == [0, 1, 2], f"kept {kept.tolist()}")

    # -- average precision -------------------------------------------------- #
    # Everything matched, two ground truths: AP is exactly 1.
    ap, _, _ = utils.average_precision(torch.tensor([0.9, 0.8]),
                                       torch.tensor([True, True]), 2)
    checker.close("AP with every prediction correct", ap, 1.0, 1e-9)
    # Nothing matched: AP is exactly 0.
    ap, _, _ = utils.average_precision(torch.tensor([0.9, 0.8]),
                                       torch.tensor([False, False]), 2)
    checker.close("AP with no prediction correct", ap, 0.0, 1e-9)
    # One of two found. The precision envelope is 1 up to recall 0.5 and 0 above
    # it, so the 101-point rule samples 1.0 at 51 grid points: AP = 51/101.
    ap, _, _ = utils.average_precision(torch.tensor([0.9, 0.8]),
                                       torch.tensor([True, False]), 2)
    checker.close("AP with half the objects found", ap, 51 / 101, 1e-6)
    # Ranking matters: the same two predictions with the hit ranked last scores
    # worse, because precision at the recall it reaches is halved.
    ap_bad, _, _ = utils.average_precision(torch.tensor([0.9, 0.8]),
                                          torch.tensor([False, True]), 2)
    checker.check("AP punishes ranking a false positive above a true one",
                  ap_bad < ap, f"{ap_bad:.4f} < {ap:.4f}")
    # No ground truth at all is undefined, not zero - it has to be excluded from
    # the mean rather than counted as a failure.
    ap, _, _ = utils.average_precision(torch.tensor([0.9]), torch.tensor([False]), 0)
    checker.check("AP with no ground truth is NaN, not 0", math.isnan(ap), f"got {ap}")


# ---------------------------------------------------------------------------
# captioning: BLEU, METEOR, CIDEr-D
# ---------------------------------------------------------------------------
def check_captioning(checker: Checker, seed: int = 0) -> None:
    utils = load("20_image_captioning", "utils")

    sentence = "a red square sits on a blue background".split()

    # A candidate identical to its reference: every n-gram precision is 1 and the
    # brevity penalty is 1, so BLEU at every order is exactly 1.
    scores = utils.corpus_bleu([sentence], [[sentence]])
    checker.close("BLEU-1 of an exact match", scores["bleu_1"], 1.0, 1e-9)
    checker.close("BLEU-4 of an exact match", scores["bleu_4"], 1.0, 1e-9)
    checker.close("brevity penalty of an equal-length match", scores["brevity_penalty"], 1.0, 1e-9)
    checker.close("length ratio of an exact match", scores["length_ratio"], 1.0, 1e-9)

    # No shared vocabulary: every precision is 0, and the log floor keeps BLEU
    # finite and tiny rather than raising.
    unrelated = "completely different words entirely".split()
    scores = utils.corpus_bleu([unrelated], [[sentence]])
    checker.check("BLEU of a disjoint candidate is ~0", scores["bleu_4"] < 1e-4,
                  f"{scores['bleu_4']:.3e}")

    # Brevity penalty has a closed form, exp(1 - r/c), when the candidate is
    # shorter than its reference.
    truncated = sentence[:3]
    scores = utils.corpus_bleu([truncated], [[sentence]])
    checker.close("brevity penalty == exp(1 - r/c)", scores["brevity_penalty"],
                  math.exp(1 - len(sentence) / len(truncated)), 1e-9)
    checker.check("a longer candidate is not penalised",
                  utils.corpus_bleu([sentence + ["extra"]], [[sentence]])["brevity_penalty"] == 1.0)

    # Word order must matter at orders above 1: reversing the sentence keeps every
    # unigram and destroys every 4-gram, so BLEU-1 is untouched and BLEU-4 collapses.
    # That is the property that makes reporting both orders worthwhile.
    reversed_words = list(reversed(sentence))
    scores = utils.corpus_bleu([reversed_words], [[sentence]])
    checker.close("BLEU-1 is order-insensitive", scores["bleu_1"], 1.0, 1e-9)
    checker.check("BLEU-4 collapses when the order is reversed", scores["bleu_4"] < 0.2,
                  f"{scores['bleu_4']:.4f}")

    # Multiple references: a candidate matching any one of them scores as an exact
    # match, because n-gram counts are clipped to the best reference.
    other = "a blue circle on a red background".split()
    scores = utils.corpus_bleu([other], [[sentence, other]])
    checker.close("BLEU against the best of several references", scores["bleu_4"], 1.0, 1e-9)

    # -- METEOR ------------------------------------------------------------- #
    # An exact match gives F-mean 1, reduced only by the fragmentation penalty,
    # which for a single chunk of m words is 0.5 * (1/m)^3 - small but not zero.
    matches = len(sentence)
    expected = 1.0 - 0.5 * (1 / matches) ** 3
    checker.close("METEOR of an exact match", utils.corpus_meteor([sentence], [[sentence]]),
                  expected, 1e-9)
    checker.close("METEOR with no shared words",
                  utils.corpus_meteor([unrelated], [[sentence]]), 0.0, 1e-9)
    checker.check("METEOR penalises a reordering it aligns in many chunks",
                  utils.corpus_meteor([reversed_words], [[sentence]])
                  < utils.corpus_meteor([sentence], [[sentence]]))

    # -- CIDEr-D ------------------------------------------------------------ #
    # CIDEr needs a corpus: its weighting is TF-IDF over the whole split, so a
    # single-document call is degenerate (every IDF is log(1/1) = 0). Build a few
    # documents with distinct content and check the ordering it must produce.
    references = [
        [sentence],
        [other],
        ["a green triangle above a yellow line".split()],
        ["two small circles beside a large square".split()],
    ]
    exact = [refs[0] for refs in references]
    checker.check("CIDEr-D rewards exact matches across a corpus",
                  utils.corpus_cider(exact, references) > 5.0,
                  f"{utils.corpus_cider(exact, references):.3f}")

    wrong = [unrelated for _ in references]
    checker.close("CIDEr-D of unrelated captions", utils.corpus_cider(wrong, references),
                  0.0, 1e-9)

    # The "-D" in CIDEr-D is the repetition clipping. A degenerate caption that
    # repeats one high-value word must not out-score the real thing.
    repeated = [["a"] * 8 for _ in references]
    checker.check("CIDEr-D clipping stops a repeated word from scoring",
                  utils.corpus_cider(repeated, references)
                  < utils.corpus_cider(exact, references),
                  f"{utils.corpus_cider(repeated, references):.3f} < "
                  f"{utils.corpus_cider(exact, references):.3f}")

    # -- the diversity guard ------------------------------------------------ #
    # Emitting one safe caption for every image is the classic failure, and it can
    # post a respectable BLEU. distinct_captions is what makes it visible.
    metrics = utils.caption_metrics(exact, references)
    checker.close("distinct_captions of four different captions",
                  metrics["distinct_captions"], 1.0, 1e-9)
    metrics = utils.caption_metrics([sentence] * 4, references)
    checker.close("distinct_captions when one caption is repeated",
                  metrics["distinct_captions"], 0.25, 1e-9)


# ---------------------------------------------------------------------------
# normalizing flow: invertibility and the bits-per-dimension convention
# ---------------------------------------------------------------------------
def check_flow(checker: Checker, seed: int = 0) -> None:
    model_module = load("11_normalizing_flow", "model")
    torch.manual_seed(seed)

    # Deliberately small: this checks the algebra, not the model.
    flow = model_module.RealNVP(in_channels=1, image_size=16, hidden=16, couplings_per_stage=2)
    flow.eval()
    images = torch.rand(4, 1, 16, 16)

    with torch.no_grad():
        latent, log_det = flow(images, dequantise=False)
        recovered = flow.inverse(latent)
    error = float((recovered - images).abs().max())
    # A flow whose inverse does not invert its forward pass reports a likelihood
    # for a density it is not actually modelling, and nothing downstream catches
    # it - the samples still look like something.
    checker.check("RealNVP inverse recovers its input", error < 1e-5,
                  f"max err {error:.2e}")

    checker.check("the log-determinant is finite and per-image",
                  log_det.shape == (4,) and bool(torch.isfinite(log_det).all()),
                  f"shape {tuple(log_det.shape)}")

    # bits/dim must be exactly -log p(x) / (D ln 2) + 8. The +8 is the discrete
    # correction (log2 of 256 levels); it is what makes project 11's number
    # comparable with published flow results, so the convention is worth pinning.
    with torch.no_grad():
        log_prob = flow.log_prob(images, dequantise=False)
        bpd = flow.bits_per_dimension(images, dequantise=False)
    expected = -log_prob / (flow.dimension * math.log(2)) + 8.0
    checker.check("bits/dim follows the +8 discrete convention",
                  float((bpd - expected).abs().max()) < 1e-5,
                  f"max err {float((bpd - expected).abs().max()):.2e}")

    # The squeeze operation is a pure reshape and must round-trip exactly, not
    # approximately - it carries no parameters and no log-determinant.
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    squeezed = model_module.squeeze(x)
    checker.check("squeeze quadruples the channels and halves the resolution",
                  tuple(squeezed.shape) == (2, 12, 4, 4), f"got {tuple(squeezed.shape)}")
    checker.check("unsqueeze(squeeze(x)) == x",
                  bool(torch.equal(model_module.unsqueeze(squeezed), x)))


GROUPS = {
    "classification": check_classification,
    "reconstruction": check_reconstruction,
    "codebook": check_codebook,
    "generative": check_generative,
    "segmentation": check_segmentation,
    "detection": check_detection,
    "captioning": check_captioning,
    "flow": check_flow,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the hand-written metrics.")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Each group re-seeds from this, so --seed genuinely varies the random inputs
    # rather than only the first group's. Every threshold below is chosen to hold
    # across seeds; if one starts failing intermittently, the threshold is wrong.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checker = Checker()
    for name in args.groups:
        print(f"\n{name}")
        GROUPS[name](checker, args.seed)

    total = checker.passes + len(checker.failures)
    print()
    if checker.failures:
        print(f"{len(checker.failures)} of {total} checks FAILED:")
        for name in checker.failures:
            print(f"  - {name}")
        return 1
    print(f"all {total} metric checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
