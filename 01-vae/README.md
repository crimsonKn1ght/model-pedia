# 01 - Variational Autoencoder (Conv-VAE / beta-VAE)

Reconstruct and generate images with a latent-variable model trained by
maximising the evidence lower bound.

## The idea

A VAE assumes every image `x` was produced from an unobserved code `z`. The
decoder models `p(x|z)`, the encoder approximates the intractable posterior
with a Gaussian `q(z|x) = N(mu, diag(sigma^2))`, and training maximises

```
ELBO = E_q(z|x)[ log p(x|z) ]  -  beta * KL( q(z|x) || N(0, I) )
```

The first term wants faithful reconstructions. The second pulls the posterior
towards the prior, which is what makes the latent space *smooth*: because every
training image is encoded to a blob rather than a point, nearby codes must
decode to plausible images. That is why you can sample `z ~ N(0, I)` afterwards
and get a digit rather than noise.

`beta = 1` is the original VAE. `beta > 1` (the beta-VAE) over-weights the KL
term, forcing the model to use fewer, more independent latent dimensions. It
trades reconstruction sharpness for disentanglement -- the latent traversal
figure is where you can see the difference.

The reparameterisation trick, `z = mu + sigma * eps` with `eps ~ N(0, I)`, is
what makes this trainable: it moves the randomness into an input so gradients
can flow through the sampling step.

## Run it

```bash
python download_data.py                 # MNIST + Fashion-MNIST
python train.py                         # ~4 min for 5 epochs on 4 CPU cores
python evaluate.py
```

Useful variations:

```bash
python train.py --dataset fashion-mnist --epochs 8
python train.py --beta 4 --out-dir outputs/beta4     # the beta-VAE
python evaluate.py --checkpoint outputs/beta4/vae.pt --out-dir outputs/beta4
python train.py --dataset celeba64 --image-size 64 --likelihood gaussian
python train.py --quick                 # 4 batches, just checks the plumbing
```

`--likelihood bernoulli` (the default) treats pixels as Bernoulli probabilities
and uses binary cross entropy -- correct for MNIST-like data. Use
`--likelihood gaussian` (mean squared error) for natural images.

## Files

| File | What it holds |
|---|---|
| `model.py` | `Encoder`, `Decoder`, `VAE`, and the ELBO in `vae_loss` |
| `download_data.py` | dataset fetcher |
| `train.py` | training loop, per-epoch sample and reconstruction grids |
| `evaluate.py` | test ELBO, latent map, latent traversal, FID/KID |

## Test-set evaluation

`evaluate.py` writes `outputs/evaluation/`:

* `metrics.json` -- test negative ELBO split into its reconstruction and KL
  terms, bits/dim, and FID/KID/precision/recall of prior samples;
* `reconstructions.png` -- test images above their reconstructions;
* `prior-samples.png` -- 64 images decoded from `z ~ N(0, I)`;
* `latent-space.png` -- posterior means projected to 2-D and coloured by class;
* `latent-traversal.png` -- one row per latent dimension, swept from -3 to +3
  while the others stay at zero.

## Reference run

MNIST, 5 epochs, `latent_dim=16`, `beta=1`, 0.43M parameters, CPU only
(4 cores, about 4 minutes end to end):

| Metric | Value |
|---|---|
| test negative ELBO | 146.6 nats/image |
| reconstruction term | 123.9 |
| KL term | 22.7 |
| bits/dim | 0.207 |
| FID (see note) | 11.6 |
| KID | 0.534 +/- 0.075 |
| precision / recall | 0.68 / 0.74 |

Samples are clearly digits after five epochs, and the latent scatter already
separates several classes. Longer training and a larger `latent_dim` sharpen
reconstructions but also make prior samples blurrier unless `beta` is retuned --
that tension *is* the VAE.

## A note on FID and KID in this repository

Published FID uses an ImageNet-trained InceptionV3. That is a poor fit for
32x32 greyscale digits and a heavy dependency for a lightweight project, so
these projects compute the same statistics in a **small CNN feature space
trained once per dataset** and cached under `outputs/feature-extractors/`. The
scores are comparable *between projects here*, and are **not** comparable to FID
numbers quoted in papers. Pass `--feature-extractor inception` to switch to
torchvision's InceptionV3 if you want literature-comparable values.

## What to look at

* The KL term should settle at a positive value. If it collapses to ~0 the
  decoder is ignoring the latents (posterior collapse) and samples become one
  blurry average image.
* Compare `--beta 1` against `--beta 4`: the ELBO gets worse, reconstructions
  blur, and the traversal rows become noticeably more single-purpose.
