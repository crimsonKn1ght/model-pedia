# 09 - Conditional GAN (generation you can steer, and therefore grade)

An unconditional GAN samples *something* from the data distribution. A conditional one
samples something of a **requested class**. That is more useful, and - the reason this
project exists as a separate row - it is much easier to evaluate: generate an image for a
known label, ask an independent classifier what it sees, and count agreement.

```bash
pip install -r ../requirements.txt

python data.py --dataset fashion-mnist              # ~30 MB
python train.py --dataset fashion-mnist --mode acgan
python evaluate.py --checkpoint outputs/fashion-mnist_acgan/best.pt
```

## Two ways to condition a discriminator

The generator's side is the same in both: embed the label, concatenate it to the latent
vector, carry on. The difference is what the discriminator does with the label.

**cGAN** gives it the label too, as extra input channels holding a broadcast class
embedding. The discriminator judges *pairs*: is this a real image **of this class**? A
perfect horse handed over with the label "car" is fake. Direct, and it makes the
discriminator's job harder.

**ACGAN** keeps the real/fake head unconditional and adds a second head that classifies the
image. The generator is rewarded when that classifier agrees with the label it was given.
Cheaper, and it has a known failure mode - see below.

## The trap: accuracy and diversity pull apart

A generator can score near-perfect class accuracy by producing one over-typical example per
class. Accuracy would call that a triumph. So `evaluate.py` always reports **recall**
beside it, and `compare.py --study aux_weight` makes the trade visible: turning up the
auxiliary classifier weight buys class accuracy and, past a point, costs recall.

This is the same lesson as project 08's mode collapse, arriving through a different door -
and it is why the checkpoint is selected on FID rather than on class accuracy.

## FID by class

An overall FID hides which classes a model is bad at, and conditional models are usually
uneven - the classes with the most distinctive silhouette come out first. Per-class FID is
computed against that class's real test images only, so the comparison is like for like, and
`per_class.png` puts quality and obedience on the same axes.

`latent_sweep.png` is the qualitative counterpart: one noise vector conditioned on every
label in turn. What stays constant across the row is what the model has decided "style"
means.

## The classifier is independent

Class accuracy is measured with the same small network the FID features come from, trained
on real data before the GAN starts and never updated. Using the discriminator's own
auxiliary head would be circular - it is part of what is being trained.

As in every generative project here, FID and KID go through that small classifier rather
than InceptionV3, so **the values are comparable within this repository and not with
published numbers.**

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `mode` | cGAN vs ACGAN | ACGAN usually reaches higher class accuracy; check its recall |
| `aux_weight` | 0.2 / 1.0 / 5.0 | Accuracy up, recall down - the trade this project exists to show |
| `latent` | 16 vs 64 dimensions | A small latent caps within-class variety |

## Files

| File | What it does |
|---|---|
| `data.py` | Labelled datasets; unlabelled ones are rejected with an explanation |
| `model.py` | Conditional generator, both discriminator variants, both loss forms |
| `train.py` | Adversarial training with per-epoch FID *and* class accuracy |
| `evaluate.py` | Overall and per-class FID, class accuracy, class grids, latent sweep |
| `compare.py` | `mode`, `aux_weight` and `latent` studies |
| `utils.py` | FID, KID, generative precision/recall, and the feature network |

## Knobs worth turning

- `--aux-weight 10` and read `recall`, not `class accuracy`.
- `--mode cgan` on CIFAR-10, where the classes are visually closer and the conditioning has
  more work to do.
- Compare per-class FID with per-class accuracy in `metrics.json`. They do not rank the
  classes the same way, which says the two are measuring different things.
