# 02 - Vector-Quantized Autoencoder (VQ-VAE)

Discrete image representation, and generation from an autoregressive prior over
the discrete codes.

## The idea

A VAE has a continuous latent. A VQ-VAE keeps a **codebook** of `K` embedding
vectors and snaps each spatial position of the encoder output to its nearest
entry. An image's latent representation becomes a small grid of integers -- for
a 32x32 input downsampled by 4, an 8x8 grid of symbols drawn from `K`.

Two problems follow, and their solutions are the interesting part:

**`argmin` has no gradient.** The straight-through estimator copies the
decoder's incoming gradient directly onto the encoder's output, pretending the
quantisation step was the identity. Two auxiliary terms keep the two sides from
drifting apart: a codebook loss pulling embeddings towards the encoder outputs
assigned to them, and a commitment loss pulling encoder outputs towards their
chosen embedding. The EMA variant used here (`--decay 0.99`) replaces the
codebook loss with an exponential moving average update, which trains more
stably.

**A discrete latent has no prior you can sample from.** `N(0, I)` is meaningless
over index grids. So a second model -- a **PixelCNN** -- is trained to model
`p(indices)` autoregressively. Sampling an index grid from it and decoding gives
a new image. A VQ-VAE alone is a compressor; VQ-VAE plus prior is a generative
model. This two-stage split is the ancestor of the modern token-based image
generators.

## Run it

```bash
python download_data.py           # CIFAR-10 by default
python train.py                   # stage 1 (VQ-VAE) then stage 2 (PixelCNN)
python evaluate.py
```

Useful variations:

```bash
python train.py --dataset fashion-mnist --epochs 4 --prior-epochs 3
python train.py --dataset celeba64 --image-size 64
python train.py --num-embeddings 64      # a smaller codebook, watch perplexity
python train.py --decay 0                # plain codebook loss instead of EMA
python train.py --prior-epochs 0         # compressor only, skip generation
```

## Files

| File | What it holds |
|---|---|
| `model.py` | `VectorQuantizer`, `Encoder`, `Decoder`, `VQVAE`, `PixelCNNPrior`, `MaskedConv2d` |
| `download_data.py` | dataset fetcher |
| `train.py` | stage 1 then stage 2 |
| `evaluate.py` | reconstruction metrics, codebook health, sample FID/KID |

## Test-set evaluation

`evaluate.py` writes `outputs/evaluation/`:

* `metrics.json` -- reconstruction MSE/PSNR/SSIM, **codebook perplexity**, how
  many of the `K` codes were used at all, the compression ratio, and FID/KID of
  images sampled from the PixelCNN prior;
* `fid_reconstructions` -- FID of *reconstructions* rather than samples. This
  isolates the autoencoder from the prior: it is the floor that generation
  cannot beat;
* `reconstructions.png`, `prior-samples.png`, `codebook-usage.png`.

**Codebook collapse** is the failure to watch for. If perplexity sits far below
`K` and only a handful of codes are ever selected, the model has thrown away
most of its capacity -- the reconstructions may still look acceptable while the
representation is nearly useless. The usage bar chart makes this obvious.

## What to look at

* Perplexity should climb during training and settle at a decent fraction of the
  codebook size. Shrinking `--num-embeddings` to 64 and watching what happens to
  both perplexity and reconstruction error is the most instructive single
  experiment here.
* The gap between `fid_reconstructions` and `fid` tells you which stage is the
  bottleneck. A large gap means the PixelCNN prior is the weak link and deserves
  more epochs; a small gap means the autoencoder is.
* PixelCNN sampling is a raster-scan loop -- one forward pass per latent
  position. On an 8x8 grid that is 64 sequential passes, which is why
  `evaluate.py` defaults to fewer samples than the other projects.

## A note on FID and KID in this repository

These scores are computed in a small per-dataset feature space trained on first
use, not with ImageNet InceptionV3. They compare models *within this
repository* and are not comparable to published FID values. See
`../01-vae/README.md` for the full explanation, or pass
`--feature-extractor inception`.

## Reference run

Fashion-MNIST, 20000 training images, 3 VQ-VAE epochs then 2 PixelCNN epochs,
`K=256`, 8x8 latent grid, CPU only (about 8 minutes end to end):

| Metric | Value |
|---|---|
| test reconstruction MSE | 0.0127 |
| test PSNR | 25.0 dB |
| test SSIM | 0.888 |
| **codebook perplexity** | **182.9 / 256** |
| **codes used** | **256 / 256 (100%)** |
| compression ratio | 16x |
| FID of prior samples | 10.1 |
| **FID of reconstructions** | **1.90** |
| precision / recall | 0.80 / 0.62 |

Two readings matter here.

**The codebook is fully alive.** All 256 entries are used and perplexity sits at
183, meaning roughly 183 codes are effectively active. Without dead-code
restarts the same configuration starts near a perplexity of 4 and climbs only
slowly. This is the single largest quality lever in the project.

**Reconstruction FID is 1.90; sample FID is 10.1.** That five-fold gap localises
the weakness precisely. The autoencoder is excellent -- a 16x compression that
loses almost nothing perceptually. The PixelCNN prior is what limits generation,
and more `--prior-epochs` is where extra budget should go. Without the
reconstruction-only number you would not know which half to blame.

## Codebook collapse and dead-code restarts

Early training is where the codebook is most fragile. A plain EMA codebook only
updates entries that win at least one input, so a code that stops winning
receives no update, never moves, and cannot come back. On Fashion-MNIST with
`K = 256`, a plain EMA run starts with a perplexity around 4 -- roughly four
effective codes out of 256 -- and climbs only slowly from there (about 39 after
four epochs, on a run measured here). Most of the codebook sits unused for a
long time, which is capacity paid for and not used.

**Dead-code restarts** address it directly: any code whose EMA cluster size
falls below `--restart-threshold` is reseeded to a random encoder output from
the current batch, putting it back somewhere the data actually is. Restarts are
on by default; `--restart-threshold 0` disables them.

Run it both ways and watch the `perplexity` column:

```bash
python train.py --dataset fashion-mnist --epochs 3 --restart-threshold 0
python train.py --dataset fashion-mnist --epochs 3
```

Then compare `codebook-usage.png` and the `codebook_used` figure in
`metrics.json`. The reconstruction error usually differs far less than the
perplexity does, which is the point worth absorbing: a badly collapsed codebook
can look fine on reconstruction while wasting most of its representational
capacity, and only the usage statistics reveal it.
