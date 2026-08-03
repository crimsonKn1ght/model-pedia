# 19 - Object detection (what, and where, and how many)

A classifier answers one question per image. A detector answers a variable number of
questions per image, each with a box attached, and nobody tells it how many there are.
That single difference is responsible for almost everything unusual in this project:
the ragged batches, the assignment rule, the non-maximum suppression, and a metric
that takes a hundred lines to compute.

The model is a single-scale anchor-free detector - a convolutional backbone, then one
1x1 convolution that turns every cell of the resulting grid into an objectness score,
a class distribution and a box. It is the smallest thing that is still honestly a
detector, and it is small enough to read in one sitting.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset digits --preview       # 12 MB download, builds scenes from MNIST
python train.py --dataset digits --epochs 10     # about ten minutes on CPU
python evaluate.py --checkpoint outputs/digits_stride8/best.pt --nms-sweep 0.3 0.5 0.7 0.9
```

`digits` composes scenes from MNIST: one to three digits, randomly scaled, pasted on a
textured canvas, with the box read off the digit's own non-zero pixels. Two reasons it
is the default. The boxes are **exact**, so nothing in the mAP you measure is
annotation noise - which matters when the metric is the thing being taught. And it is
small enough to reach a real mAP in minutes, which is what makes the metric
implementation worth trusting.

```bash
python data.py --dataset voc2007                 # ~880 MB download, 20 real classes
python train.py --dataset voc2007 --epochs 40 --batch-size 16
```

Pascal VOC is the real benchmark and the code path is there, but be honest about it:
a single-scale detector this size, trained for minutes on a CPU, will produce a low
mAP on VOC. It wants a GPU and an order of magnitude more epochs.

## The metric is the project

`utils.py` implements IoU, NMS and mean average precision from scratch. That is
deliberate - `torchvision.ops` has the first two, and a detection metric you have not
read is a detection metric you cannot debug. The pieces:

1. **IoU** on corner boxes, degenerate cases handled.
2. **NMS**, per class, greedy. This is what decides that two boxes are the same
   object. Everything surviving it counts as a separate prediction.
3. **Average precision**: rank every prediction of a class by confidence *across the
   whole split*, match each greedily to the highest-IoU unclaimed ground-truth box in
   its own image, then integrate under the precision envelope by 101-point
   interpolation - the COCO rule.

Two headline numbers come out, and the gap between them is the interesting part:

- **mAP@0.5** - overlap by half counts. Mostly asks *did you find it*.
- **mAP@0.5:0.95** - averaged over ten thresholds up to 0.95. Mostly asks *how well
  does the box fit*. Always lower, often much lower.

The implementation is checked against cases whose answers are known by hand: a perfect
detector scores 1.0 at every threshold; boxes shifted by 6 pixels on a 20-pixel object
score 0, because their IoU is 0.32; finding one of two objects gives AP 0.505 under
101-point interpolation, not 0.5.

## Measured results

`digits`, 4000 training scenes at 96x96, stride 8 (a 12x12 grid), 10 epochs on four CPU
cores (500 s). Validation mAP@0.5 climbed 0.429, 0.649, 0.845, 0.887, 0.888, 0.927,
0.926, 0.933, 0.939, 0.941. Scored on 1000 test scenes containing 2006 objects, with
3190 detections kept above confidence 0.05:

| Metric | Value |
|---|---|
| mAP@0.5 | **0.9275** |
| mAP@0.75 | 0.8332 |
| mAP@0.5:0.95 | 0.6975 |

Per-class AP@0.5 runs from 0.896 (digit 1) to 0.947 (digits 2 and 6); per-class
AP@0.5:0.95 from 0.612 to 0.783. Digit 1 is worst at both, which makes sense - a thin
vertical stroke has the least distinctive box of the ten.

The 0.93 / 0.70 gap between the two mAPs is the thing to take away. The detector finds
almost everything (93% at the loose threshold) and its boxes are only roughly right, so
two thirds of that score evaporates as the IoU requirement tightens. Reporting only
mAP@0.5 would have hidden it entirely.

### Non-maximum suppression, measured

Same weights, four suppression thresholds:

| NMS IoU | Detections kept | mAP@0.5 | mAP@0.75 |
|---|---|---|---|
| 0.30 | 2885 | 0.9245 | 0.8317 |
| 0.50 | 3190 | **0.9275** | 0.8332 |
| 0.70 | 4052 | 0.9184 | **0.8446** |
| 0.90 | 5507 | 0.8677 | 0.8090 |

Not one parameter changed between those rows. mAP@0.5 peaks at 0.5 and mAP@0.75 at 0.7,
which is worth a moment: a looser threshold keeps more near-duplicate boxes, and some of
those duplicates are *tighter* than the one that would have suppressed them, so the
strict-IoU metric improves while the loose one gets worse. At 0.9 the duplicates
overwhelm both. Post-processing is a real part of the reported accuracy.

## What the grid stride costs you

This detector predicts **one box per cell**, so the stride is a hard limit on how close
two object centres can be and still both be found. Not a soft degradation - a
structural ceiling.

```bash
python compare.py --study stride    # 16 / 8 / 4
python compare.py --study crowd     # 1 / 3 / 5 objects, fixed grid
```

The `crowd` study attacks the same limit from the other side: hold the grid fixed and
put more objects in each image. mAP falls, and the fall is not the network getting
worse at recognition - it is the assignment rule running out of cells. Assignment is
largest-box-first in `build_targets`, so when two centres collide the *smaller* object
wins the cell; small objects are the harder examples and handing the cell to the large
one would throw them away.

## Non-maximum suppression is part of the accuracy

```bash
python evaluate.py --checkpoint ... --nms-sweep 0.3 0.5 0.7 0.9
```

The same weights, re-scored under different suppression thresholds. Too strict and
neighbouring objects suppress each other, costing recall; too loose and duplicates
survive as false positives. Nothing about the model changes between those rows, which
makes it a clean demonstration that a meaningful chunk of a detector's reported
accuracy lives in its post-processing rather than its parameters.

Note also that `--conf-threshold` defaults to a low 0.05. mAP rewards ranked
low-confidence guesses rather than punishing them - raising the threshold to make the
pictures look tidy will *lower* your mAP.

## Reading the loss

The printed loss is a sum of three things measured in different units, so it is a poor
progress signal on its own; that is why checkpoints are selected on validation
mAP@0.5. The breakdown separates `obj` (cells containing an object) from `noobj`
(everything else), because with two or three positives among 144 cells a single
averaged objectness term would be dominated by the empty ones.

Reading a real run: over ten epochs on `digits` the objectness terms start small and
stay small (`noobj` 0.150 to 0.061, `obj` 0.104 to 0.027) - "nothing here" is right
almost everywhere and the model works that out quickly - while the classification term
does nearly all the moving (1.778 to 0.197) and the box term improves slowly and
steadily (0.332 to 0.184). Recognising the digit is the hard part; deciding that a cell
is empty is not.

The head's objectness bias is initialised to -4.0 for the same reason: without it, the
first few hundred steps are spent unlearning optimism.

## Files

| File | What it does |
|---|---|
| `data.py` | The MNIST-derived `digits` scenes, Pascal VOC 2007, and the ragged-batch collate |
| `model.py` | Backbone, decoupled head, box decoding, centre assignment, and the loss |
| `train.py` | Training with per-epoch validation mAP, checkpoint selection on it |
| `evaluate.py` | Test mAP@0.5 and @0.5:0.95, per-class AP, PR curves, NMS sweep |
| `compare.py` | `stride`, `box_loss` and `crowd` studies |
| `utils.py` | IoU, CIoU loss, NMS, average precision, mAP, box drawing |

## Outputs

In `outputs/<dataset>_stride<n>/`:

- `best.pt`, `history.json`, `curves.png`, `detections_val.png`
- `metrics.json` - both mAPs, per-class AP, and the NMS sweep if requested
- `pr_curves.png` - precision against recall at IoU 0.5, one curve per class
- `detections.png` - ground truth above, predictions with scores below
- `nms_sweep.png` - the same weights at several suppression thresholds

## Knobs worth turning

- `--box-loss l1` against the default `ciou`. L1 on the corners treats a 2-pixel error
  the same on a 10-pixel box as on a 100-pixel one, and it shows up in mAP@0.75 long
  before mAP@0.5.
- `--stride 4` - four times the cells. Better on crowded images, slower, and more
  empty cells to learn to ignore.
- `--max-objects 6` - past the point where a 12x12 grid can represent the scene.
- `--noobj-weight 0.1` - what happens when the empty cells stop being punished.
