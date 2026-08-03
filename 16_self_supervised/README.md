# 16 - Self-supervised learning (labels are not the only supervision)

Train an image encoder with no labels at all, then measure how much it learned by
freezing it and asking a linear layer to classify on top. The labels are downloaded,
and then deliberately not used until evaluation time.

Both methods here take the same two augmented views of one image and try to make
their embeddings agree. They differ in what stops that from collapsing to a
constant:

- **SimCLR** pushes the two views of an image together and every other image in the
  batch apart. The other images are the negatives, which is why it wants a big batch.
- **BYOL** has no negatives. One view goes through the network plus a predictor, the
  other through an exponential-moving-average copy of the same network that receives
  no gradient. The asymmetry alone prevents collapse, which is a genuinely surprising
  result.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset cifar10 --preview     # download (~170 MB), look at the views
python train.py --method simclr                 # six epochs, ~10 min on four CPU cores
python evaluate.py --checkpoint outputs/cifar10_simclr/best.pt
```

## The two things this project insists on

**1. Always report the random-init control.** A randomly initialised convolutional
network is a decent feature extractor - random filters still compute edge-like
responses - so a linear probe on one scores far above chance. `evaluate.py` therefore
probes the pretrained encoder *and* an untrained encoder of the identical
architecture, in the same run, and prints the gap. Only the gap is attributable to
pretraining.

**2. The loss is not the metric.** SimCLR's loss falls when the model gets better at
telling batch-mates apart, and BYOL's can be driven to zero by a representation that
has collapsed. So `train.py` runs a **k-NN probe** on the validation split every
epoch and selects the checkpoint on that. Watching the loss and the k-NN curve
disagree is the point of `curves.png`.

## The augmentation *is* the supervision

This is the idea that makes self-supervised learning work, and it is easiest to see
by taking it away:

```bash
python compare.py --study augment
```

| Arm | What the model can use to match the views |
|---|---|
| `no augment` | Both views identical - the task is trivial and teaches nothing |
| `crop only` | Must relate a part of the image to another part |
| `crop+colour` | Colour jitter and grayscale close off the "average colour" shortcut |

Whatever the augmentations destroy is what the representation learns to ignore. With
colour jitter removed, matching views by their colour histogram is enough, and the
model takes that shortcut rather than learning about content.

## Measured results

MNIST, 10 000 unlabelled training images, SimCLR for 6 epochs on four CPU cores
(487 s), then probed on the 10 000-image test split:

```bash
python train.py --dataset mnist                # six epochs is the default
python evaluate.py --checkpoint outputs/mnist_simclr/best.pt
```

| Probe | Pretrained | Random init | Gain |
|---|---|---|---|
| 20-NN | 0.9389 | 0.5719 | +0.3670 |
| Linear, 1% of labels | 0.6597 | 0.2640 | +0.3957 |
| Linear, 10% of labels | 0.8637 | 0.4993 | +0.3644 |
| Linear, 100% of labels | 0.9450 | 0.7735 | +0.1715 |

The k-NN monitor rose from 0.4917 (untrained) to 0.8970 over the six epochs.

Two things to read off that table. First, the untrained encoder is not at chance -
it reaches 0.77 with all the labels, which is exactly why the control column has to
be there. Second, the gain is **widest where labels are scarcest**: at 1% of the
labels pretraining is worth +0.40, at 100% it is worth +0.17. Label efficiency is
the practical reason to pretrain, and it is invisible if you only ever report the
full-label number.

CIFAR-10 is the more interesting dataset and the default. Its images are larger and in
colour, so an epoch costs roughly a third more than an MNIST one, and it takes more
epochs to separate from its control - natural images give the augmentations far more to
work with, which is the point, and also the cost.

## Files

| File | What it does |
|---|---|
| `data.py` | Datasets, the two-view augmentation pipeline, `--preview` to see the pairs |
| `model.py` | Small ResNet encoder, projection head, SimCLR and BYOL objectives |
| `train.py` | Label-free pretraining with a per-epoch k-NN monitor |
| `evaluate.py` | k-NN and linear probes on frozen features, against a random-init control |
| `compare.py` | `method`, `augment` and `width` studies |
| `utils.py` | Seeding, feature extraction, both probes, PCA scatter, plotting |

## Outputs

In `outputs/<dataset>_<method>/`:

- `best.pt`, `history.json`, `curves.png` - loss falling next to k-NN accuracy rising
- `views.png` - the augmented pairs the model is asked to match
- `metrics.json`, `probe_accuracy.png` - both probes, both encoders
- `features_pca.png` - test features projected to 2D, pretrained vs random

## Knobs worth turning

- `--epochs 0` checkpoints the untrained encoder. Evaluate it to see the control on
  its own.
- `--method byol` - no negatives, and it works. Try `--batch-size 32` on both
  methods: SimCLR degrades noticeably, BYOL much less, because BYOL never needed the
  batch to supply negatives.
- `--dataset stl10` is the dataset this family of methods was designed around, at the
  cost of a 2.6 GB download.
- `--label-fractions 0.001 0.01 0.1 1.0` in `evaluate.py` pushes the
  label-efficiency table further into the regime where pretraining wins most.
