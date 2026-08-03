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
python compare.py --dataset fashion-mnist --epochs 12
```

Each model trains in about two minutes per five epochs on four CPU cores.

Fashion-MNIST is the more interesting dataset, for two reasons. Its confusion
matrix shows a real, interpretable failure mode - shirt is confused with
T-shirt, coat and pullover, and nothing else - where MNIST is close to
saturated. And it is where LeNet's advantage has to be earned: at five epochs
the two models are level, because this LeNet is small enough to still be
underfitting. Give it twelve and it pulls ahead. On MNIST the gap is obvious
after one epoch. Same two architectures, different lesson.

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

## Measured numbers

Test accuracy from `compare.py`, seed 0, four CPU cores. Expect small variation
with seed and hardware.

| Dataset | Epochs | MLP (235 k params) | LeNet-5 (62 k params) | Gap |
|---|---|---|---|---|
| MNIST | 5 | 0.9788 | **0.9900** | +1.1 pts |
| Fashion-MNIST | 5 | 0.8829 | 0.8849 | +0.2 pts |
| Fashion-MNIST | 12 | 0.8928 | **0.8998** | +0.7 pts |

LeNet has 3.8x fewer parameters in every row. The Fashion-MNIST rows are the
honest version of the story: the architectural advantage is real but it is not
free, and a short run will not show it.

Per-class F1 for LeNet on Fashion-MNIST after 5 epochs makes the failure mode
concrete - trouser 0.98, bag 0.97, ankle boot 0.96, but shirt only 0.69. Nearly
all of the remaining error is one class the model genuinely cannot separate
from its neighbours.

## Knobs worth turning

- `--epochs` - on Fashion-MNIST this is what separates the two models; five
  epochs is not enough to distinguish them.
- `--augment` - mild rotation/translation jitter; helps LeNet, hurts the MLP,
  which is exactly what you would predict from weight sharing.
- `--train-subset 5000` - the CNN degrades far more gracefully with less data.
- `--model mlp --epochs 20` - capacity alone does not close the gap.
