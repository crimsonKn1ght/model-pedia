# 04 - Conditional GAN (cGAN and ACGAN)

Class-controlled generation: ask for a "7" and get a 7.

## The idea

A DCGAN produces *some* sample from the data distribution. A conditional GAN
models `p(x | y)` instead, so the label becomes an input you control.

Both variants here condition the generator the same way -- a learned class
embedding is concatenated to the noise vector. They differ in how the
**discriminator** learns about the label, and that difference is the lesson.

**cGAN** (Mirza & Osindero, 2014) gives the discriminator the label too, as
extra constant-valued input channels. It answers one question: *is this a real
image of this class?* A real image paired with the wrong label counts as fake,
which is what forces the generator to respect the conditioning.

**ACGAN** (Odena et al., 2017) hides the label from the discriminator's input and
adds a second head that must **classify** the image. The discriminator answers
two questions -- *is this real?* and *which class is it?* -- and the
classification loss is applied to real and generated images alike. That extra
signal pushes the generator towards class-distinctive samples.

ACGAN tends to train more stably at the cost of a known failure mode: the
generator can satisfy the classifier by producing an easily-classified prototype
per class, reducing within-class diversity. The per-class FID in `evaluate.py`
is there to catch exactly that.

## Run it

```bash
python download_data.py
python train.py                    # ACGAN on MNIST
python evaluate.py
```

Useful variations:

```bash
python train.py --mode cgan                 # the concatenation-style cGAN
python train.py --dataset cifar10 --epochs 20 --base-channels 64
python train.py --aux-weight 0.1            # weaken ACGAN's classification loss
```

## Files

| File | What it holds |
|---|---|
| `model.py` | `ConditionalGenerator`, `ProjectionDiscriminator` (both modes) |
| `download_data.py` | dataset fetcher |
| `train.py` | adversarial loop with the mode-dependent losses |
| `evaluate.py` | classifier accuracy, per-class FID, class grid |

## Test-set evaluation

Overall FID answers neither question that matters for a conditional model, so
`evaluate.py` adds two that do.

**1. Is the conditioning obeyed?** An independent classifier -- the same small
CNN the FID feature space is built from, trained only on real training images --
labels the generated samples. Agreement with the requested class is
`classifier_accuracy`. A model with a good FID and a `classifier_accuracy` near
chance is generating fine images and ignoring the label completely.

**2. Is any class neglected?** FID is computed per class, against real test
images of that class only. `fid_per_class_worst` catches the model that posts a
respectable average while producing garbage for one or two classes.

Output in `outputs/evaluation/`:

* `metrics.json` -- overall FID/KID/precision/recall, `classifier_accuracy`,
  per-class FID with mean and worst;
* `class-grid.png` -- one row per class, ten samples wide;
* `fid-per-class.png`.

## What to look at

* The class grid is the fastest diagnostic. Read down the rows: if row 3 does
  not contain 3s, conditioning failed regardless of what FID says.
* Compare `--mode cgan` against `--mode acgan` at equal epochs. ACGAN usually
  reaches higher `classifier_accuracy` sooner; check whether it also shows less
  variety within each row.
* Push `--aux-weight` up to 5 and watch `classifier_accuracy` rise while the
  per-class FID gets worse. That trade-off is ACGAN's characteristic weakness.

## A note on FID and KID in this repository

Computed in a small per-dataset feature space, not ImageNet InceptionV3. See
`../01-vae/README.md`. Note that `classifier_accuracy` uses that same network,
so it is only available with `--feature-extractor small-cnn`.
