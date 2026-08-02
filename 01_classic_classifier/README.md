# 01 - Classic classifier (MLP, then LeNet)

The starting point of supervised deep learning: map a small grayscale image to
one of ten classes. Two models are trained through exactly the same loop, so
the difference between them is purely architectural.

- **MLP** - flattens the image, so it has no idea which pixels are neighbours.
- **LeNet-5** - convolutions share weights across the image, so an edge detector
  learned in one place works everywhere.

LeNet wins with roughly a quarter of the parameters. That is the lesson: the
right inductive bias beats raw capacity.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset mnist                 # download (~12 MB)
python train.py --model lenet --dataset mnist --epochs 5
python evaluate.py --checkpoint outputs/mnist_lenet/best.pt
```

Or train both and get one table:

```bash
python compare.py --dataset fashion-mnist --epochs 5
```

Both models train in a few minutes on a CPU. Fashion-MNIST is the more
interesting dataset - MNIST is close to saturated, and the confusion matrix on
Fashion-MNIST shows a real, interpretable failure mode (shirt vs coat vs
pullover).

## Files

| File | What it does |
|---|---|
| `data.py` | Downloads MNIST / Fashion-MNIST, builds the transforms, carves a validation split out of the train split |
| `model.py` | `MLP` and `LeNet5`, both taking `1x28x28` and returning logits |
| `train.py` | Training loop, cosine LR schedule, saves the best-validation checkpoint |
| `evaluate.py` | Test-set accuracy, macro-F1, per-class table, confusion matrix, grid of mistakes |
| `compare.py` | Trains every model on the same data and prints a summary table |
| `utils.py` | Seeding, device selection, confusion matrix and F1 by hand, plotting |

## Outputs

Everything lands in `outputs/<dataset>_<model>/`:

- `best.pt` - checkpoint with the highest validation accuracy
- `history.json`, `curves.png` - loss and accuracy per epoch
- `metrics.json` - test accuracy, macro-F1, per-class precision/recall/F1
- `confusion_matrix.png` - row-normalised heatmap
- `mistakes.png` - test images the model gets wrong, true label vs prediction

## Typical numbers

Indicative test accuracy after 5 epochs on CPU; expect small variation with
seed and hardware.

| Dataset | MLP (~0.24 M params) | LeNet-5 (~0.06 M params) |
|---|---|---|
| MNIST | ~0.977 | ~0.990 |
| Fashion-MNIST | ~0.884 | ~0.905 |

## Knobs worth turning

- `--epochs` - more epochs mostly help the MLP catch up on MNIST, not on Fashion-MNIST.
- `--augment` - mild rotation/translation jitter; helps LeNet, hurts the MLP,
  which is exactly what you would predict from weight sharing.
- `--train-subset 5000` - the CNN degrades far more gracefully with less data.
- `--model mlp --epochs 20` - capacity alone does not close the gap.
