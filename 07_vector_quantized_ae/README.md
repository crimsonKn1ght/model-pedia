# 07 - Vector-quantized autoencoder (a discrete description of an image)

A VAE (project 06) compresses an image into a continuous vector. A VQ-VAE compresses
it into a **grid of integers**: the encoder produces a feature map, every position is
snapped to its nearest entry in a learned codebook, and the decoder works from the
snapped version. A 32x32 greyscale image - 1024 numbers - becomes an 8x8 grid of 64
integers.

That sounds like a worse autoencoder, and as an autoencoder it is. Its importance is
what the discreteness enables: once an image is a short sequence of integers,
generating images becomes a **sequence-modelling problem**, and everything the field
knows about modelling sequences applies. That is why this architecture ended up under
so much later work.

## Two stages, and why

```bash
pip install -r ../requirements.txt

python data.py --dataset fashion-mnist                                   # ~30 MB
python train.py --dataset fashion-mnist                                  # stage 1, ~4 min
python train_prior.py --checkpoint outputs/fashion-mnist_k128/best.pt    # stage 2, ~4 min
python evaluate.py --checkpoint outputs/fashion-mnist_k128/best.pt
```

**Stage one** learns the encoder, the codebook and the decoder. At the end of it the
model can compress and reconstruct, and it **cannot generate at all** - the codes are
integers with no distribution attached, so there is nothing to sample.

**Stage two** fixes that with a second model: a small causal Transformer over the 64
code positions in raster order, trained on the codes the frozen encoder produces.
Sampling an image is then: sample 64 integers, look them up, decode. Running
`evaluate.py` without a `prior.pt` says so rather than silently reporting nothing.

This split is the shape of the idea, not an implementation shortcut. The original
paper used a PixelCNN for stage two; a small Transformer is the same job with fewer
lines.

## Three problems the code solves

**argmin has no gradient.** The straight-through estimator sets
`z_q = z_e + (z_q - z_e).detach()`: the forward pass sees the quantised tensor, the
backward pass pretends quantisation was the identity. It is a biased estimator and it
works. `model.py` asserts the encoder does receive gradient through it.

**The codebook needs training too.** Either a loss term pulls entries towards the
encoder outputs assigned to them, or - the default here, and the more stable - an
exponential moving average of those assignments does it with no gradient at all,
which is k-means in disguise. Laplace smoothing keeps an unused entry from dividing by
zero. `compare.py --study codebook_rule` measures the difference.

**The encoder and codebook can drift apart.** The commitment term pulls the encoder
towards the code it chose. `--commitment` sets how hard.

## Codebook collapse, counted

The number to watch is **perplexity**: `exp` of the entropy of the code-usage
histogram. It equals the codebook size when every entry is used equally, and 1 when
the model has collapsed onto a single code.

This matters because a 512-entry codebook running at perplexity 40 *is* a 40-entry
codebook - it just cost 512 entries of memory - and no reconstruction metric will tell
you that has happened. `compare.py --study codebook` grows the codebook and prints
perplexity beside PSNR, so you can see where the extra entries stop being used.
`codebook_usage.png` shows the whole histogram.

## What is measured

| Group | Numbers | Notes |
|---|---|---|
| Reconstruction | PSNR, SSIM, MSE, compression ratio | SSIM and PSNR disagree usefully here: PSNR forgives the blur a small codebook produces, SSIM does not |
| Codebook health | entries used, usage fraction, perplexity | The diagnostic no reconstruction metric can replace |
| Prior | nats per code, next to the `log(K)` uniform baseline | A real likelihood over the latent - no bound, unlike the VAE's ELBO - but over *codes*, so not comparable with bits-per-dimension on pixels |
| Samples | FID, KID, precision, recall | Only with a prior. Measured through a small classifier trained on the dataset, **not** InceptionV3, so comparable within this repository only |

The FID caveat is the same one as project 06 and is stated at the top of `utils.py`,
recorded in `metrics.json` as `feature_net`, and repeated in every table.

## Measured results

Fashion-MNIST, 20 000 training images at 32x32, 128 codes on an 8x8 grid (16x compression),
EMA codebook. Stage one ran 10 epochs, stage two 15 epochs; scored on the 10 000-image test
split.

| Quantity | Value |
|---|---|
| compression | 16x (1024 pixels -> 64 integers) |
| reconstruction | **25.50 dB PSNR**, SSIM 0.8507, MSE 0.00322 |
| codebook used | **70 / 128 entries (54.7%)** |
| perplexity | **49.0** |
| prior | **1.516 nats/code** against a 4.852 uniform baseline |
| FID (prior samples) | 2.991 |
| precision / recall | 0.785 / 0.513 |

The codebook numbers are the interesting part. A nominally 128-entry codebook is running at
perplexity 49, and 58 of its entries were never used at all - so this is effectively a
49-entry codebook that cost 128 entries of memory. Nothing in the reconstruction metrics
would have told you that; only the perplexity column does.

The prior is worth 3.34 nats per position over assuming the codes are uniform, which is what
makes sampling work at all - and its samples reach FID 2.99 against the VAE's 14.33 on the
same dataset and the same feature network, with better precision and recall. Discrete codes
plus a sequence model beats a continuous latent with a Gaussian prior, which is the whole
argument for the architecture.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `codebook` | 32 / 128 / 512 entries | PSNR improves sub-linearly; perplexity shows why |
| `codebook_rule` | EMA vs a loss term | EMA is steadier and usually keeps more of the codebook alive |
| `commitment` | 0.1 / 0.25 / 1.0 | Too low and the encoder drifts from its codes; too high and it cannot move |

Add `--with-prior` to any of them to train a prior per arm and score samples too, at
roughly double the runtime.

## Files

| File | What it does |
|---|---|
| `data.py` | MNIST, Fashion-MNIST, CIFAR-10, CelebA and any image folder, in `[0, 1]` |
| `model.py` | Vector quantiser (straight-through, EMA), the VQ-VAE, and the code prior |
| `train.py` | Stage one; selects on validation PSNR and prints codebook perplexity |
| `train_prior.py` | Stage two; extracts codes once, then trains the prior on them |
| `evaluate.py` | Reconstruction, codebook health, and sample metrics when a prior exists |
| `compare.py` | `codebook`, `codebook_rule` and `commitment` studies |
| `utils.py` | PSNR, SSIM, codebook usage, FID, KID and generative precision/recall |

## Outputs

In `outputs/<dataset>_k<codes>/`:

- `best.pt`, `history.json`, `curves.png` - loss, PSNR and perplexity per epoch
- `reconstructions_val.png`, `examples.png` - input, the code grid drawn as an image, reconstruction
- `codebook_usage.png` - the usage histogram
- `prior.pt`, `prior_curves.png`, `prior_samples.png` - stage two
- `samples.png` - prior samples at the chosen temperature
- `metrics.json` - everything above, including `perplexity` and the sample metrics

## Knobs worth turning

- `--num-codes 512` then read the perplexity. This is the fastest way to see codebook
  collapse happen.
- `--no-ema` to train the codebook by gradient instead. Watch how many entries stay
  alive.
- `evaluate.py --temperature 0.8` sharpens the prior. Lower temperature buys precision
  and costs recall, and the two numbers show it directly.
- Compare the reconstruction PSNR here with project 04's continuous autoencoder at a
  similar compression ratio. Quantisation costs something real; what it buys is stage
  two.
