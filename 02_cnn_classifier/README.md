# 02 - CNN classifier (ResNet on natural images)

Natural images are harder than MNIST: the same class appears at different
scales, poses, colours and backgrounds. This project trains ResNets written
from scratch on CIFAR-10 / CIFAR-100 / SVHN and isolates what makes them work.

The one idea to take away is the **identity shortcut**. `plain20` is byte-for-byte
`resnet20` with the skip connections deleted; `compare.py --study residual`
trains both under identical conditions and the gap is the shortcut alone. Add
depth (`plain32` vs `resnet32`) and the plain net gets *worse* while the
residual net keeps improving - the observation ResNet was invented for.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset cifar10                # download (~170 MB)
python train.py --arch resnet20 --epochs 12
python evaluate.py --checkpoint outputs/cifar10_resnet20/best.pt
```

Controlled comparisons:

```bash
python compare.py --study residual   # skip connections on/off, at two depths
python compare.py --study depth      # 20 vs 32 layers
python compare.py --study augment    # random crop + flip on/off
```

### Keeping it short

The defaults train on a 15 000-image subset for 12 epochs, which is roughly
seven minutes on four CPU cores. For the numbers people quote in papers:

```bash
python train.py --arch resnet20 --train-subset 0 --epochs 30   # full split
```

That is about 45 minutes on four CPU cores and a couple of minutes on a GPU.

## Architectures

| Name | Layers | Params | Notes |
|---|---|---|---|
| `resnet20` | 20 | 0.27 M | The CIFAR ResNet from the paper; the default |
| `resnet32` | 32 | 0.47 M | Same design, deeper |
| `plain20` | 20 | 0.27 M | `resnet20` with the shortcuts removed - the control |
| `plain32` | 32 | 0.46 M | The deeper control; expect it to degrade |
| `resnet18` | 18 | 11.2 M | ImageNet widths, 3x3 stem for 32x32 inputs |

`resnet18` is the model named in most tutorials, but at 32x32 it is 40x the
compute of `resnet20` for a couple of points of accuracy. It is here so you can
measure that trade-off rather than take it on faith.

## Files

| File | What it does |
|---|---|
| `data.py` | Downloads CIFAR-10/100 or SVHN, crop/flip augmentation, validation split |
| `model.py` | `BasicBlock` and a generic `ResNet`; the `residual` flag builds the control net |
| `train.py` | SGD with Nesterov momentum and a one-cycle LR schedule |
| `evaluate.py` | Top-1 / top-5, per-class accuracy, most-confused pairs, prediction grid |
| `compare.py` | Runs one ablation study and prints a summary table |
| `utils.py` | Seeding, device selection, metrics, plotting |

## Outputs

In `outputs/<dataset>_<arch>/` (or `outputs/<study>/<arm>/` for comparisons):

- `best.pt`, `history.json`, `curves.png`
- `metrics.json` - top-1, top-5, per-class accuracy, most-confused pairs
- `confusion_matrix.png`, `predictions.png`

## Typical numbers

Indicative CIFAR-10 top-1 accuracy; expect variation with seed and hardware.

| Setup | resnet20 | plain20 |
|---|---|---|
| 15 k images, 12 epochs (default) | ~0.74 | ~0.71 |
| full split, 30 epochs | ~0.90 | ~0.86 |

The most-confused pairs are the interesting output: cat/dog and
automobile/truck dominate the errors, which is a sensible failure mode rather
than noise.
