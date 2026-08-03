# 08 - DCGAN

Unconditional image generation with the convolutional GAN architecture that made
adversarial training reproducible.

## The idea

Two networks compete. The generator `G` maps noise `z ~ N(0, I)` to an image;
the discriminator `D` scores how real an image looks. They play

```
min_G max_D  E_x[ log D(x) ]  +  E_z[ log(1 - D(G(z))) ]
```

In practice the generator is trained on the **non-saturating** loss -- maximise
`log D(G(z))` rather than minimise `log(1 - D(G(z)))`. The two have the same
fixed point, but the original form's gradient vanishes precisely when the
generator is bad, which is exactly when it needs signal most.

Unlike a VAE there is no reconstruction term and no likelihood anywhere. The
generator never sees a real image -- it only ever learns from the discriminator's
opinion. That is why GAN samples are sharp (nothing averages over pixels) and
also why they can miss entire modes without any loss value complaining.

DCGAN's contribution was a set of architectural rules that turned this from a
research curiosity into something that trains:

* strided convolutions instead of pooling; transposed convolutions to upsample;
* batch normalisation everywhere except the generator's output layer and the
  discriminator's input layer;
* ReLU in `G`, LeakyReLU(0.2) in `D`;
* `tanh` output, so images live in `[-1, 1]`;
* Adam, `lr = 2e-4`, `beta1 = 0.5`.

## Run it

```bash
python data.py
python train.py                        # MNIST, ~9 min on 4 CPU cores
python evaluate.py
```

Useful variations:

```bash
python train.py --dataset fashion-mnist --epochs 12
python train.py --dataset cifar10 --base-channels 64 --epochs 20
python train.py --dataset celeba64 --image-size 64
python train.py --label-smoothing 0     # remove the one-sided smoothing
```

## Files

| File | What it holds |
|---|---|
| `model.py` | `Generator`, `Discriminator`, DCGAN weight initialisation |
| `data.py` | dataset fetcher |
| `train.py` | the alternating adversarial loop |
| `evaluate.py` | FID/KID, precision-recall, sample grid, latent interpolation |

## Test-set evaluation

`evaluate.py` writes `outputs/evaluation/`:

* `metrics.json` -- FID, KID, and **precision/recall reported separately**;
* `samples.png`, `real-vs-generated.png`;
* `interpolation.png` -- straight-line walks between latent codes.

Precision and recall are the useful pair, because a single FID hides *why* a GAN
is bad:

| precision | recall | diagnosis |
|---|---|---|
| high | low | sharp samples, **mode collapse** -- it makes a few things well |
| low | high | broad coverage, poor quality |
| high | high | what you want |

## Reference run

MNIST, 8 epochs, `base_channels=32`, generator 0.37M / discriminator 0.17M
parameters, CPU only (4 cores, about 12 minutes end to end):

| Metric | Value |
|---|---|
| FID (see note) | 2.94 |
| KID | 0.132 +/- 0.049 |
| precision / recall | 0.70 / 0.87 |

Worth comparing against project 06's VAE on the same dataset, same feature
space, similar budget: **FID 11.6 for the VAE against 2.94 here**, with recall
rising from 0.74 to 0.87. Put the two sample grids side by side and the reason
is obvious -- the VAE's outputs are recognisable but soft, the GAN's have clean
strokes and sharp edges.

That gap is structural, not a matter of tuning. The VAE optimises a
reconstruction likelihood, and under uncertainty the likelihood-maximising
answer is to average over plausible outcomes, which looks like blur. The GAN has
no such term: the discriminator penalises blur directly, because blurry images
are easy to identify as fake. The GAN pays for this elsewhere -- no likelihood,
no encoder, and a training process that can collapse without warning.

Watch the printed `D(x)` and `D(G(z))` above: they converge towards each other
through epoch 4 and then begin separating again as the discriminator pulls
ahead. That drift is normal, but if `D(G(z))` keeps falling towards zero the
generator is running out of gradient.

## Reading the training log

The printed `D(x)` and `D(G(z))` are the discriminator's average confidence on
real and fake batches. Healthy training keeps them in tension, with `D(G(z))`
drifting up towards 0.5 as the generator improves. Two failure signatures:

* **`loss_D` collapses to ~0, `D(x)` -> 1, `D(G(z))` -> 0.** The discriminator
  has won outright; the generator's gradients dry up. Lower `D`'s learning rate
  or raise `--label-smoothing`.
* **Sample grids stop changing between epochs, `loss_G` flat.** Mode collapse.
  The recall number in `evaluate.py` will confirm it.

Interpolation is worth looking at too: smooth, semantically continuous
transitions mean the generator learned a structured mapping. Abrupt jumps
between a few fixed images mean it memorised a handful of modes.

## A note on FID and KID in this repository

Computed in a small per-dataset feature space, not ImageNet InceptionV3.
Comparable within this repository, not to published numbers. See
`../06_vae/README.md`, or pass `--feature-extractor inception`.
