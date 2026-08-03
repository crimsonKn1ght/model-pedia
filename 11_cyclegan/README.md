# 11 - CycleGAN

Unpaired image-to-image translation: two piles of images, no correspondence
between them.

## The idea

Pix2Pix needs matched pairs. For most interesting translations they do not
exist -- nobody has photographs of the same horse as a zebra. CycleGAN drops the
requirement entirely.

Without pairs the adversarial loss is hopelessly under-constrained: *any*
mapping whose outputs land in domain B satisfies the discriminator, including
one that ignores its input completely. **Cycle consistency** supplies the
missing constraint. Train a second generator running the other way and require

```
F(G(a)) ~= a        and        G(F(b)) ~= b
```

A translation must retain enough information to be reversible, which forces it
to preserve content while changing style. That one idea is the whole method.

Two details from the paper are kept because they matter in practice:

* **Least-squares GAN loss** -- squared error against 1/0 rather than binary
  cross entropy. Markedly more stable, and the reason `loss_D` here sits near
  0.5 rather than near 0.7.
* **Identity loss** -- `G(b) ~= b`. Feeding a generator an image already in its
  target domain should change nothing. Without it, generators tend to shift
  colours they had no reason to touch.

An **image pool** shows the discriminator a mix of the generator's current
output and its recent history, which stops the two networks chasing each other
around a cycle of one-step-behind responses.

## The domains

Three ways to build the two unpaired collections:

| `--task` | What it does |
|---|---|
| `classes` (default) | two class subsets of one labelled dataset, e.g. CIFAR-10 horse (7) vs deer (4) |
| `datasets` | two different datasets, e.g. MNIST digits vs Fashion-MNIST garments |
| `horse2zebra` | the original CycleGAN dataset, via `python data.py --horse2zebra` |

`horse2zebra` needs outbound access to `people.eecs.berkeley.edu`. The other two
need nothing extra and demonstrate identical mechanics.

## Run it

```bash
python data.py
python train.py                                    # CIFAR-10 horse <-> deer
python evaluate.py
```

Useful variations:

```bash
python train.py --domain-a 7 --domain-b 9          # horse <-> truck
python train.py --task datasets --domain-a mnist --domain-b fashion-mnist
python train.py --task horse2zebra --image-size 64 --epochs 30
python train.py --lambda-cycle 0 --out-dir outputs/no-cycle    # the key ablation
```

## Files

| File | What it holds |
|---|---|
| `model.py` | `ResnetGenerator`, `PatchDiscriminator`, `ImagePool` |
| `domains.py` | builds the two unpaired domains |
| `data.py` | dataset fetcher, optional horse2zebra download |
| `train.py` | four losses across two generators and two discriminators |
| `evaluate.py` | domain-transfer FID plus cycle-reconstruction error |

## Test-set evaluation

There is no ground-truth translation for unpaired data, so the evaluation splits
into two questions that must both be answered:

**Did the output land in the target domain?** FID/KID of `G(A)` against real
domain-B test images, and the reverse direction.

**Was the content preserved?** Cycle-reconstruction error `F(G(a))` vs `a`,
reported as L1, PSNR and SSIM.

Neither alone is sufficient. A generator that discards its input and emits a
generic domain-B image scores a fine FID; the cycle error exposes it. A
generator that learns the identity function has a perfect cycle error and a
terrible FID.

`outputs/evaluation/` contains `metrics.json` and `translations.png`, the latter
showing real, translated and cycled images for both directions.

## Reference run

MNIST digits against Fashion-MNIST garments as the two unpaired domains, 8000
images each, 4 epochs, CPU only (about 10 minutes end to end):

| Metric | MNIST -> Fashion | Fashion -> MNIST |
|---|---|---|
| FID in the target domain | 13.6 | 6.6 |
| KID | 0.602 | 0.181 |
| cycle L1 | 0.056 | 0.144 |
| cycle PSNR | 22.0 dB | 17.8 dB |
| cycle SSIM | 0.894 | 0.678 |

Both questions get a satisfactory answer, which is what makes the run
meaningful. FID in the low tens says the outputs genuinely land in the target
domain, and cycle SSIM of 0.89 in the digit direction says the content survives
the round trip -- the model is translating, not discarding its input and drawing
a generic sample.

The asymmetry between the directions is the interesting part. Translating *into*
MNIST is easier (FID 6.6) because digits are a narrower, more constrained
distribution than clothing. But the cycle back through the garment domain is
harder (SSIM 0.678 against 0.894), because compressing a textured garment into a
digit-like image discards information that cannot be recovered. Cycle
consistency constrains the mapping; it does not make it lossless.

## What to look at

* Run the `--lambda-cycle 0` ablation. The adversarial losses keep falling while
  the translations stop having anything to do with their inputs -- the clearest
  possible demonstration of why the cycle term exists.
* Instance normalisation, not batch normalisation, throughout the generator.
  Style transfer works per image; batch statistics would leak information
  between samples in a batch.
* Unpaired translation is genuinely harder than paired. Expect the class-subset
  task to change colour and texture convincingly while struggling with shape --
  that limitation is real and well documented for CycleGAN, not an artefact of
  the small budget here.

## A note on FID and KID in this repository

Computed in a small per-dataset feature space, not ImageNet InceptionV3. See
`../06_vae/README.md`.
