# 18 - Semantic segmentation (a label for every pixel)

Classification asks *what is in this image*. Segmentation asks it once per pixel,
which changes the architecture and changes the metric, and the metric is the part
most worth your attention.

The tension in the architecture is that recognising *what* something is wants a large
receptive field, and the cheap way to get one is to downsample - but knowing *where
its edges are* wants full resolution, which downsampling destroys. A U-Net answers
this by doing both: downsample for context, upsample back, and pass full-resolution
features across skip connections so the decoder can put the edges back.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset shapes --preview            # generated, no download
python train.py --dataset shapes --epochs 8           # about eight minutes on CPU
python evaluate.py --checkpoint outputs/shapes_unet/best.pt
```

For the real dataset:

```bash
python data.py --dataset oxford-pets                  # ~810 MB download
python train.py --dataset oxford-pets --epochs 10 --image-size 96
```

`shapes` is generated: coloured circles, rectangles and triangles on a textured
background, with pixel-exact masks. It exists so that a full run costs minutes and no
part of the number you get back is annotation noise. `oxford-pets` is the real one,
and its three-class trimap (background / pet / border) is a better teacher precisely
because the border class is hard.

## Why mean IoU and not pixel accuracy

Both datasets are dominated by background - 80% of pixels in `shapes`. So:

| Metric | What "predict background everywhere" scores |
|---|---|
| Pixel accuracy | **0.803** - looks like a working model |
| Mean IoU | **0.201** - correctly says it segmented nothing |

Every table in this project prints that baseline, computed from the ground-truth
masks of the split being evaluated. Pixel accuracy rewards getting the easy majority
right; mean IoU gives a two-pixel-wide class the same weight as the background. When
the two disagree, mIoU is the one telling the truth - which is why checkpoints are
selected on validation mIoU, not on loss and not on accuracy.

Dice is reported alongside. It carries the same information with the intersection
weighted double, is standard in medical imaging, and is less brutal than IoU on small
structures.

## Measured results

`shapes`, 4000 training images at 64x64, U-Net for 8 epochs on four CPU cores
(503 s), scored on the 1000-image test split:

| Class | IoU | Dice | Pixel share |
|---|---|---|---|
| background | 0.9999 | 1.0000 | 80.3% |
| circle | 0.9840 | 0.9920 | 6.3% |
| rectangle | 0.9886 | 0.9943 | 8.5% |
| triangle | 0.9827 | 0.9913 | 4.9% |
| **mean** | **0.9888** | **0.9944** | |

Pixel accuracy 0.9986, against the majority-class baseline's 0.8032 accuracy and
0.2008 mIoU.

### What the skip connections are worth

```bash
python compare.py --study arch
```

Six epochs, everything else held fixed:

| Arm | Parameters | Train s | mIoU | Dice | Pixel acc |
|---|---|---|---|---|---|
| `dilated` | 66,020 | 270.9 | 0.9847 | 0.9922 | 0.9980 |
| `no skips` | 421,188 | 456.9 | 0.9744 | 0.9870 | 0.9961 |
| `unet` | 467,268 | 333.7 | 0.9849 | 0.9924 | 0.9980 |

Per-class IoU, same runs:

| Arm | background | circle | rectangle | triangle |
|---|---|---|---|---|
| `dilated` | 0.9999 | 0.9757 | 0.9849 | 0.9782 |
| `no skips` | 0.9982 | 0.9584 | 0.9779 | 0.9631 |
| `unet` | 0.9999 | 0.9790 | 0.9842 | 0.9766 |

Cutting the skips costs 0.010 mIoU but only 0.002 pixel accuracy - the loss is
concentrated in the thin classes (circle drops 0.021, triangle 0.014) and the
background barely notices. That is the whole argument for the architecture, and it is
also a small lesson in metric choice: the ablation is four times more visible in mIoU
than in accuracy.

The `dilated` control is the other way to keep resolution - never downsample, and grow
the receptive field with dilation instead. It matches the U-Net here with a seventh of
the parameters, because these shapes need very little context to identify. On Oxford
Pets, where telling a pet from a background needs to look at more than a few pixels,
the balance shifts.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `arch` | U-Net vs no skips vs full-resolution dilated | Skips buy detail; the gap lands on the thin classes |
| `loss` | Cross entropy vs Dice vs both | Cross entropy is a per-pixel average, so a 5% class gets 5% of the gradient; Dice normalises per class |
| `capacity` | Base width 8 / 16 / 32 | How much of the score is capacity rather than architecture |

## Files

| File | What it does |
|---|---|
| `data.py` | Oxford-IIIT Pet trimaps and the generated `shapes` set; joint image/mask transforms |
| `model.py` | U-Net, the no-skip control, and a full-resolution dilated control |
| `train.py` | Training with CE / Dice / both, checkpoint selection on validation mIoU |
| `evaluate.py` | Test mIoU, per-class IoU and Dice, confusion matrix, overlays |
| `compare.py` | `arch`, `loss` and `capacity` studies |
| `utils.py` | Confusion-matrix metrics, Dice loss, mask colouring and overlays |

## Outputs

In `outputs/<dataset>_<model>/`:

- `best.pt`, `history.json`, `curves.png`, `predictions_val.png`
- `metrics.json` - mIoU, per-class IoU and Dice, pixel accuracy, and the baseline
- `confusion.png` - row-normalised, so it says *what* the errors are
- `examples.png` - image / truth / prediction / overlay

## Knobs worth turning

- `--loss ce+dice` on `oxford-pets`. The border class is where it earns its keep.
- `evaluate.py --image-size 128` on a model trained at 64. Both architectures are
  fully convolutional, so this runs - and the drop tells you how much of what the
  model learned was tied to the scale it was trained at.
- `--model dilated` and watch the epoch time. Full resolution is not free once the
  images are bigger than 64 pixels.
