"""Boxes, non-maximum suppression, and mean average precision - written out.

The interesting content of a detection project is not the network, it is this
file. A detector emits a few hundred overlapping guesses per image and the metric
has to turn that into one number, which takes three steps that are all easy to get
subtly wrong:

1. **IoU**, on the corner representation, with the degenerate cases handled.
2. **NMS**, which decides that two boxes are the same object. Everything that
   survives is a separate prediction, so a wrong threshold here shows up as either
   duplicate detections or missed neighbours.
3. **Average precision**, which needs the predictions sorted by confidence
   *across the whole split*, greedy one-to-one matching against ground truth, and
   integration under the precision envelope. mAP@0.5 asks whether the object was
   found; mAP@0.5:0.95 averages over ten IoU thresholds and mostly asks how well
   the box fits.

``torchvision.ops`` has `nms` and `box_iou`; they are re-implemented here because
they are short and because a detection metric you have not read is a detection
metric you cannot debug.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

COCO_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
BOX_COLOURS = ["#e6550d", "#3182bd", "#31a354", "#756bb1", "#d6616b", "#e7ba52"]


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


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of ``xyxy`` boxes, ``(N, M)``."""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.size(0), boxes_b.size(0)))

    top_left = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    overlap = (bottom_right - top_left).clamp_min(0)
    intersection = overlap[..., 0] * overlap[..., 1]

    union = box_area(boxes_a)[:, None] + box_area(boxes_b)[None, :] - intersection
    return intersection / union.clamp_min(1e-9)


def complete_iou_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``1 - CIoU`` per box pair, both ``xyxy``.

    Plain L1 on the corners treats a 2-pixel error the same on a 10-pixel box as
    on a 100-pixel one, and IoU alone has no gradient once two boxes stop
    overlapping. Complete IoU adds a centre-distance term normalised by the
    enclosing box (so there is a gradient when IoU is 0) and an aspect-ratio term,
    which is why it converges faster than either.
    """
    top_left = torch.max(prediction[:, :2], target[:, :2])
    bottom_right = torch.min(prediction[:, 2:], target[:, 2:])
    overlap = (bottom_right - top_left).clamp_min(0)
    intersection = overlap[:, 0] * overlap[:, 1]
    union = box_area(prediction) + box_area(target) - intersection
    iou = intersection / union.clamp_min(1e-9)

    pred_c = xyxy_to_cxcywh(prediction)
    target_c = xyxy_to_cxcywh(target)
    centre_distance = (pred_c[:, :2] - target_c[:, :2]).pow(2).sum(dim=1)

    enclose_tl = torch.min(prediction[:, :2], target[:, :2])
    enclose_br = torch.max(prediction[:, 2:], target[:, 2:])
    enclose = (enclose_br - enclose_tl).clamp_min(0)
    diagonal = enclose.pow(2).sum(dim=1).clamp_min(1e-9)

    pred_ratio = torch.atan(pred_c[:, 2] / pred_c[:, 3].clamp_min(1e-6))
    target_ratio = torch.atan(target_c[:, 2] / target_c[:, 3].clamp_min(1e-6))
    v = (4 / torch.pi**2) * (pred_ratio - target_ratio).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp_min(1e-9)

    return 1 - iou + centre_distance / diagonal + alpha * v


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """Greedy non-maximum suppression; returns kept indices, highest score first."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        best = order[0]
        keep.append(best)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[best].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep)


def class_wise_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """NMS applied per class - two different objects may legitimately overlap."""
    keep = []
    for cls in labels.unique():
        selector = (labels == cls).nonzero(as_tuple=True)[0]
        kept = nms(boxes[selector], scores[selector], iou_threshold)
        keep.append(selector[kept])
    if not keep:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    merged = torch.cat(keep)
    return merged[scores[merged].argsort(descending=True)]


def average_precision(
    scores: torch.Tensor, matched: torch.Tensor, num_ground_truth: int
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """AP by 101-point interpolation of the precision envelope (the COCO rule).

    ``matched`` is a boolean per prediction, already decided by the IoU matching.
    Returns ``(ap, precision_envelope, recall)`` so the PR curve can be plotted.
    """
    if num_ground_truth == 0:
        return float("nan"), torch.zeros(0), torch.zeros(0)
    if scores.numel() == 0:
        return 0.0, torch.zeros(0), torch.zeros(0)

    order = scores.argsort(descending=True)
    matched = matched[order].float()
    true_positive = matched.cumsum(0)
    false_positive = (1 - matched).cumsum(0)

    recall = true_positive / num_ground_truth
    precision = true_positive / (true_positive + false_positive).clamp_min(1e-9)
    # Envelope: precision is replaced by the best precision at any higher recall,
    # which removes the sawtooth and is what both VOC and COCO integrate under.
    envelope = precision.flip(0).cummax(0).values.flip(0)

    grid = torch.linspace(0, 1, 101)
    index = torch.searchsorted(recall.contiguous(), grid)
    sampled = torch.where(
        index < envelope.numel(), envelope[index.clamp(max=envelope.numel() - 1)], torch.zeros(1)
    )
    return float(sampled.mean()), envelope, recall


def evaluate_detections(
    predictions: list[dict],
    targets: list[dict],
    num_classes: int,
    thresholds: tuple[float, ...] = COCO_THRESHOLDS,
) -> dict:
    """mAP over a whole split.

    ``predictions[i]`` and ``targets[i]`` describe image ``i`` with keys ``boxes``
    (``xyxy``), ``labels`` and - for predictions - ``scores``.

    The matching is the standard greedy rule: within a class, predictions from the
    entire split are ranked by confidence, and each is matched to the highest-IoU
    ground-truth box in its own image that is still free. A second prediction on an
    already-matched object is a false positive, which is exactly what makes NMS
    worth doing.
    """
    per_class: dict[int, dict] = {}
    curves: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for cls in range(num_classes):
        num_gt = sum(int((target["labels"] == cls).sum()) for target in targets)
        entries = []  # (score, image index, box)
        for image_index, prediction in enumerate(predictions):
            selector = (prediction["labels"] == cls).nonzero(as_tuple=True)[0]
            for i in selector.tolist():
                entries.append((float(prediction["scores"][i]), image_index, prediction["boxes"][i]))
        if num_gt == 0 and not entries:
            continue

        aps = {}
        for threshold in thresholds:
            claimed = {
                image_index: torch.zeros(int((target["labels"] == cls).sum()), dtype=torch.bool)
                for image_index, target in enumerate(targets)
            }
            gt_boxes = {
                image_index: target["boxes"][target["labels"] == cls]
                for image_index, target in enumerate(targets)
            }

            scores, matched = [], []
            for score, image_index, box in sorted(entries, key=lambda e: -e[0]):
                scores.append(score)
                boxes = gt_boxes[image_index]
                if boxes.numel() == 0:
                    matched.append(False)
                    continue
                ious = box_iou(box.unsqueeze(0), boxes)[0]
                ious = torch.where(claimed[image_index], torch.full_like(ious, -1.0), ious)
                best = int(ious.argmax())
                if float(ious[best]) >= threshold:
                    claimed[image_index][best] = True
                    matched.append(True)
                else:
                    matched.append(False)

            ap, envelope, recall = average_precision(
                torch.tensor(scores), torch.tensor(matched), num_gt
            )
            aps[threshold] = ap
            if threshold == 0.5:
                curves[cls] = (envelope, recall)

        per_class[cls] = {"num_ground_truth": num_gt, "ap": aps}

    def mean_over(threshold_subset) -> float:
        values = [
            ap
            for entry in per_class.values()
            for threshold, ap in entry["ap"].items()
            if threshold in threshold_subset and not np.isnan(ap)
        ]
        return float(np.mean(values)) if values else float("nan")

    return {
        "map_50": mean_over({0.5}),
        "map_75": mean_over({0.75}) if 0.75 in thresholds else float("nan"),
        "map_50_95": mean_over(set(thresholds)),
        "per_class": {
            cls: {
                "ap_50": entry["ap"].get(0.5, float("nan")),
                "ap_50_95": float(
                    np.mean([v for v in entry["ap"].values() if not np.isnan(v)])
                )
                if entry["ap"]
                else float("nan"),
                "num_ground_truth": entry["num_ground_truth"],
            }
            for cls, entry in per_class.items()
        },
        "curves": curves,
        "thresholds": list(thresholds),
    }


def draw_boxes(
    ax,
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor | None = None,
    class_names: list[str] | None = None,
) -> None:
    """Draw one image with its boxes; grayscale and RGB both handled."""
    array = image.detach().cpu().clamp(0, 1).numpy()
    if array.shape[0] == 1:
        ax.imshow(array[0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(np.transpose(array, (1, 2, 0)))
    ax.set_xticks([])
    ax.set_yticks([])

    for index in range(boxes.size(0)):
        x1, y1, x2, y2 = boxes[index].tolist()
        cls = int(labels[index])
        colour = BOX_COLOURS[cls % len(BOX_COLOURS)]
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour, linewidth=1.4
            )
        )
        name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        text = name if scores is None else f"{name} {float(scores[index]):.2f}"
        ax.text(
            x1, max(y1 - 1, 0), text, fontsize=6, color="white",
            bbox={"facecolor": colour, "pad": 0.6, "edgecolor": "none"},
        )


def plot_detections(
    images: torch.Tensor,
    predictions: list[dict],
    targets: list[dict],
    class_names: list[str] | None,
    path: str | Path,
    title: str | None = None,
) -> None:
    """Two rows: ground truth on top, predictions below, same images."""
    n = min(images.size(0), len(predictions), len(targets))
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 4.0), squeeze=False)
    for column in range(n):
        draw_boxes(
            axes[0][column], images[column], targets[column]["boxes"],
            targets[column]["labels"], None, class_names,
        )
        draw_boxes(
            axes[1][column], images[column], predictions[column]["boxes"],
            predictions[column]["labels"], predictions[column]["scores"], class_names,
        )
    axes[0][0].set_ylabel("truth", fontsize=8)
    axes[1][0].set_ylabel("predicted", fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(
    curves: dict[int, tuple[torch.Tensor, torch.Tensor]],
    per_class: dict,
    class_names: list[str] | None,
    path: str | Path,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    drawn = 0
    for cls, (precision, recall) in sorted(curves.items()):
        if recall.numel() == 0:
            continue
        name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        ap = per_class.get(cls, {}).get("ap_50", float("nan"))
        ax.plot(recall.numpy(), precision.numpy(), label=f"{name} (AP {ap:.2f})", linewidth=1.2)
        drawn += 1

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    if drawn:
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.text(0.5, 0.5, "no detections above threshold", ha="center", va="center")
    ax.set_title(title or "precision-recall at IoU 0.5")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(history: dict, path: str | Path, keys: tuple[str, ...] = ("loss", "map_50")) -> None:
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


def plot_bars(
    labels: list[str],
    series: dict[str, list[float]],
    path: str | Path,
    ylabel: str = "mAP",
    title: str | None = None,
) -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)

    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 3, 4))
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)

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
