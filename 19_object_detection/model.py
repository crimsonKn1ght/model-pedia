"""A single-scale anchor-free detector, its target assignment, and its loss.

    python model.py

This is the smallest thing that is still honestly a detector. A convolutional
backbone reduces the image by ``stride``, and one 1x1 convolution turns every cell
of the resulting grid into a prediction of five plus ``num_classes`` numbers:

    channel 0        objectness logit - is the centre of an object in this cell?
    channels 1..4    box: centre offset inside the cell, and log size
    channels 5..     class logits

Three design choices are worth naming, because they are the ones that separate
this from a classifier with extra outputs:

**Anchor-free.** No prior box shapes. A cell predicts a size directly as
``stride * exp(t)``, so an untrained network starts out guessing boxes exactly one
cell across. Anchors were how detectors handled scale before this worked; skipping
them removes a whole set of hyperparameters and costs surprisingly little.

**Centre assignment.** A ground-truth box belongs to the single cell containing its
centre. That makes the target unambiguous and the loss sparse - two or three
positive cells out of a few hundred - which is why the objectness term needs
separate weighting for the positives and the empty cells.

**One box per cell.** The grid resolution therefore *is* the limit on how many
objects can be found, and two objects whose centres land in the same cell cannot
both be predicted. ``compare.py --study stride`` measures exactly that.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import class_wise_nms, complete_iou_loss, cxcywh_to_xyxy


class ConvBlock(nn.Module):
    """conv 3x3 -> BN -> SiLU, optionally halving the resolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyDetector(nn.Module):
    """Backbone plus a single decoupled head on one feature map."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        base_channels: int = 32,
        stride: int = 8,
    ) -> None:
        super().__init__()
        if stride not in (2, 4, 8, 16):
            raise ValueError(f"stride must be 2, 4, 8 or 16, got {stride}")
        self.stride = stride
        self.num_classes = num_classes

        stages = [ConvBlock(in_channels, base_channels)]
        channels = base_channels
        downsamples = int(torch.log2(torch.tensor(float(stride))))
        for _ in range(downsamples):
            # Width doubles per stage but is capped, so changing the stride changes
            # the grid resolution without also changing the model size by 16x -
            # otherwise the stride study would be measuring capacity instead.
            out_channels = min(channels * 2, base_channels * 4)
            stages.append(ConvBlock(channels, out_channels, stride=2))
            channels = out_channels
            stages.append(ConvBlock(channels, channels))
        self.backbone = nn.Sequential(*stages)

        # A separate 3x3 before the prediction convolution: classification and
        # localisation want different features, and giving them one shared layer of
        # their own is the cheap version of a decoupled head.
        self.neck = ConvBlock(channels, channels)
        self.head = nn.Conv2d(channels, 5 + num_classes, 1)
        # Start with objectness strongly negative: almost every cell is empty, and
        # without this the first few hundred steps are spent unlearning optimism.
        nn.init.constant_(self.head.bias[0], -4.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.neck(self.backbone(images)))

    def decode_boxes(self, prediction: torch.Tensor) -> torch.Tensor:
        """Raw head output -> ``xyxy`` boxes in pixels, ``(N, H, W, 4)``."""
        _, _, grid_h, grid_w = prediction.shape
        device = prediction.device
        rows = torch.arange(grid_h, device=device).view(1, grid_h, 1)
        cols = torch.arange(grid_w, device=device).view(1, 1, grid_w)

        centre_x = (cols + prediction[:, 1].sigmoid()) * self.stride
        centre_y = (rows + prediction[:, 2].sigmoid()) * self.stride
        width = self.stride * prediction[:, 3].clamp(-4, 4).exp()
        height = self.stride * prediction[:, 4].clamp(-4, 4).exp()

        boxes = torch.stack([centre_x, centre_y, width, height], dim=-1)
        return cxcywh_to_xyxy(boxes)

    @torch.no_grad()
    def detect(
        self,
        prediction: torch.Tensor,
        image_size: int,
        conf_threshold: float = 0.05,
        nms_iou: float = 0.5,
        max_detections: int = 100,
    ) -> list[dict]:
        """Grid predictions -> a list of per-image ``{boxes, scores, labels}``.

        The confidence of a detection is ``P(object) * P(class)``, the product of
        the two things the head predicts separately. Only the best class per cell
        is kept, since the cell can only carry one box anyway.
        """
        boxes = self.decode_boxes(prediction).clamp(0, image_size)
        objectness = prediction[:, 0].sigmoid()
        class_probability, class_index = prediction[:, 5:].softmax(dim=1).max(dim=1)
        scores = objectness * class_probability

        results = []
        for image in range(prediction.size(0)):
            flat_boxes = boxes[image].reshape(-1, 4)
            flat_scores = scores[image].reshape(-1)
            flat_labels = class_index[image].reshape(-1)

            keep = flat_scores > conf_threshold
            flat_boxes, flat_scores, flat_labels = (
                flat_boxes[keep], flat_scores[keep], flat_labels[keep]
            )
            if flat_boxes.numel():
                kept = class_wise_nms(flat_boxes, flat_scores, flat_labels, nms_iou)[:max_detections]
                flat_boxes, flat_scores, flat_labels = (
                    flat_boxes[kept], flat_scores[kept], flat_labels[kept]
                )
            results.append(
                {"boxes": flat_boxes.cpu(), "scores": flat_scores.cpu(), "labels": flat_labels.cpu()}
            )
        return results


def build_targets(
    targets: list[dict],
    grid_h: int,
    grid_w: int,
    stride: int,
    device: torch.device,
) -> dict:
    """Turn per-image box lists into dense grid targets.

    Boxes are assigned largest-first, so when two centres fall in the same cell the
    *smaller* object wins it - small objects are the ones the detector struggles
    with, and handing the cell to the large one loses the harder example.
    """
    n = len(targets)
    objectness = torch.zeros(n, grid_h, grid_w, device=device)
    classes = torch.full((n, grid_h, grid_w), -1, dtype=torch.long, device=device)
    boxes = torch.zeros(n, grid_h, grid_w, 4, device=device)

    for index, target in enumerate(targets):
        gt_boxes = target["boxes"].to(device)
        gt_labels = target["labels"].to(device)
        if gt_boxes.numel() == 0:
            continue

        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        for order in areas.argsort(descending=True).tolist():
            box = gt_boxes[order]
            centre_x = (box[0] + box[2]) / 2
            centre_y = (box[1] + box[3]) / 2
            col = int(centre_x.item() // stride)
            row = int(centre_y.item() // stride)
            col = min(max(col, 0), grid_w - 1)
            row = min(max(row, 0), grid_h - 1)

            objectness[index, row, col] = 1.0
            classes[index, row, col] = gt_labels[order]
            boxes[index, row, col] = box

    return {"objectness": objectness, "classes": classes, "boxes": boxes}


def detection_loss(
    model: TinyDetector,
    prediction: torch.Tensor,
    targets: list[dict],
    box_loss: str = "ciou",
    box_weight: float = 5.0,
    class_weight: float = 1.0,
    noobj_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Objectness + classification + box regression.

    The objectness term is split into positives and negatives and each is averaged
    separately. With two or three positives among several hundred cells, a plain
    mean over all cells would let the empty ones drown out the objects entirely.
    """
    grid_h, grid_w = prediction.shape[-2:]
    target = build_targets(targets, grid_h, grid_w, model.stride, prediction.device)
    positive = target["objectness"] > 0

    objectness_logits = prediction[:, 0]
    bce = F.binary_cross_entropy_with_logits(
        objectness_logits, target["objectness"], reduction="none"
    )
    loss_obj = bce[positive].mean() if positive.any() else objectness_logits.sum() * 0
    loss_noobj = bce[~positive].mean() if (~positive).any() else objectness_logits.sum() * 0

    if positive.any():
        class_logits = prediction[:, 5:].permute(0, 2, 3, 1)[positive]
        loss_cls = F.cross_entropy(class_logits, target["classes"][positive])

        predicted_boxes = model.decode_boxes(prediction)[positive]
        target_boxes = target["boxes"][positive]
        if box_loss == "ciou":
            loss_box = complete_iou_loss(predicted_boxes, target_boxes).mean()
        else:
            # L1 on the corners, normalised by the image scale so the two options
            # are on a comparable footing.
            loss_box = (predicted_boxes - target_boxes).abs().mean() / model.stride
    else:
        loss_cls = prediction.sum() * 0
        loss_box = prediction.sum() * 0

    total = loss_obj + noobj_weight * loss_noobj + class_weight * loss_cls + box_weight * loss_box
    parts = {
        "loss": float(total.detach()),
        "obj": float(loss_obj.detach()),
        "noobj": float(loss_noobj.detach()),
        "cls": float(loss_cls.detach()),
        "box": float(loss_box.detach()),
        "positives": int(positive.sum()),
    }
    return total, parts


def build_detector(
    in_channels: int = 1,
    num_classes: int = 10,
    base_channels: int = 32,
    stride: int = 8,
) -> TinyDetector:
    return TinyDetector(in_channels, num_classes, base_channels, stride)


if __name__ == "__main__":
    image_size = 96
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 34.0, 40.0], [50.0, 55.0, 78.0, 88.0]]),
            "labels": torch.tensor([3, 7]),
        },
        {"boxes": torch.tensor([[20.0, 20.0, 44.0, 52.0]]), "labels": torch.tensor([1])},
    ]
    images = torch.rand(2, 1, image_size, image_size)

    for stride in (4, 8, 16):
        model = build_detector(1, 10, 32, stride)
        prediction = model(images)
        loss, parts = detection_loss(model, prediction, targets)
        detections = model.detect(prediction, image_size, conf_threshold=0.0)
        params = sum(p.numel() for p in model.parameters())
        grid = prediction.shape[-1]
        print(
            f"stride {stride:2d}: grid {grid:2d}x{grid:2d} = {grid * grid:3d} cells, "
            f"{parts['positives']} positive  loss {loss.item():.3f}  "
            f"detections {len(detections[0]['boxes'])}  params {params:,}"
        )

    # A perfectly fitted box must give zero CIoU loss, so the loss can reach 0.
    model = build_detector(1, 10, 32, 8)
    prediction = model(images)
    print("\nloss parts at init:", {k: round(v, 3) for k, v in
                                   detection_loss(model, prediction, targets)[1].items()})
