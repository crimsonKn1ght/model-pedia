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

ACGAN is often described as the more stable of the two. That is not what the
measurements in this project show: at equal budget its known failure mode --
the generator satisfying the classifier with one easily-classified prototype per
class -- dominates completely, and intra-class diversity collapses to nothing.
The per-class FID and the precision/recall split in `evaluate.py` exist to catch
exactly that, and here they do.

## Run it

```bash
python download_data.py
python train.py                    # cGAN on MNIST
python evaluate.py
```

Useful variations:

```bash
python train.py --mode acgan                # the auxiliary-classifier variant
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

## Reference run, and why cGAN is the default

MNIST, 8 epochs, `base_channels=32`, CPU only (about 16 minutes each). All three
rows use the identical generator, discriminator trunk, budget and metric space,
so the differences are attributable to the conditioning scheme alone:

| Variant | classifier accuracy | FID | worst-class FID | precision | recall |
|---|---|---|---|---|---|
| **cGAN** (default) | **0.976** | **0.795** | **15.2** | 0.864 | **0.769** |
| ACGAN, `aux_weight=1.0` | 0.623 | 12.8 | 308 | 0.701 | 0.000 |
| ACGAN, `aux_weight=0.2` | 0.306 | 16.6 | 493 | 0.460 | 0.000 |

The cGAN is not marginally better, it is better by an order of magnitude, and
its FID of 0.795 is the best generative score in this repository -- ahead of the
unconditional DCGAN's 2.94 on the same data. That is not a surprise once stated:
the label tells the generator what to draw and gives the discriminator a sharper
question to ask, so conditioning makes the problem easier, not harder.

**Every ACGAN configuration tried here collapses to recall 0.00.** Lowering the
auxiliary weight from 1.0 to 0.2 did not rescue it -- it made the conditioning
worse without restoring diversity. The auxiliary classification objective is
what drives the collapse, and weakening it weakens the conditioning before it
weakens the collapse.

## The ACGAN feedback loop, and why the default differs from the paper

The original ACGAN trains the discriminator's classifier head on **both** real
and generated images. Doing that here collapses the model completely, and the
mechanism is worth understanding because it is not obvious from the loss:

1. the generator drifts towards whichever image is most easily classified as
   class `k`, because that minimises its auxiliary loss;
2. the discriminator's classifier is then trained on *those* images, labelled
   `k`, so it becomes ever more confident about that exact prototype;
3. which makes the prototype an even better answer for the generator.

Nothing in the objective opposes this. The adversarial term should, but a single
sharp realistic digit satisfies it well enough. The end state is one image per
class, identical for every `z` -- the noise input is ignored entirely.

Measured here on MNIST, 8 epochs, with `--aux-on-fake`:

| Metric | Value | Reading |
|---|---|---|
| precision | 0.76 | each sample looks real |
| **recall** | **0.00** | **no diversity whatsoever** |
| classifier accuracy | 0.20 | and the prototypes are mostly the wrong class |
| worst-class FID | 1176 | some classes are far off |

The class grid makes it unmistakable: every column within a row is the same
image. Note that overall FID was 18.4 -- bad but not obviously catastrophic. The
precision/recall split and the per-class breakdown are what expose it, which is
the argument for reporting them.

**The default here trains the classifier head on real images only**, which
breaks the loop by keeping the classifier grounded in data the generator cannot
influence. Pass `--aux-on-fake` to reproduce the paper's formulation and watch
the collapse happen:

```bash
python train.py --aux-on-fake --out-dir outputs/aux-on-fake
python evaluate.py --checkpoint outputs/aux-on-fake/conditional-gan.pt --out-dir outputs/aux-on-fake
```

## What to look at

* The class grid is the fastest diagnostic, and it has two axes. Read *down* the
  rows to check conditioning: if row 3 does not contain 3s, the label is being
  ignored regardless of what FID says. Then read *across* a row to check
  diversity: if the samples are identical, the model has collapsed and only
  recall will tell you.
* Run `--mode acgan` and compare. The measured outcome here contradicts the
  usual framing -- ACGAN does not merely trade diversity for conditioning
  accuracy, it loses on both at this budget.
* `recall` is the number to watch during any conditional GAN experiment. It went
  to exactly 0.00 in every ACGAN configuration tried here while FID stayed in a
  range that looks unremarkable. FID alone would not have caught it.

## A note on FID and KID in this repository

Computed in a small per-dataset feature space, not ImageNet InceptionV3. See
`../01-vae/README.md`. Note that `classifier_accuracy` uses that same network,
so it is only available with `--feature-extractor small-cnn`.
