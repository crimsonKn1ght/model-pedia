# 07 - RealNVP (normalizing flow)

Exact-likelihood density modelling. The only project here whose headline metric
is directly comparable to published numbers.

## The idea

A VAE optimises a *bound* on the likelihood. A GAN never touches it. A
normalizing flow is built entirely from invertible transforms, so the
change-of-variables formula gives the exact log-density:

```
log p(x) = log p(z) + log | det (dz / dx) |        with  z = f(x)
```

The obstacle is that a general Jacobian determinant costs `O(D^3)`. RealNVP's
**affine coupling layer** sidesteps it: split the input, leave one half
untouched, and use that half to predict a scale and shift for the other.

```
z_a = x_a
z_b = x_b * exp(s(x_a)) + t(x_a)
```

The Jacobian is triangular, so its determinant is just `sum(s(x_a))` -- linear to
compute. And the layer inverts trivially no matter how complicated `s` and `t`
are, because computing them only ever needs the untouched half. Alternating
which half is fixed lets every dimension get transformed.

Two masking patterns alternate, as in the paper:

* **checkerboard**, capturing local spatial structure;
* **channel-wise**, applied after a squeeze that trades resolution for channels,
  capturing longer-range structure.

The price of exactness is expressiveness per parameter: flows must preserve
dimensionality throughout, so they need many more parameters than a VAE or GAN
to reach comparable sample quality. What you get in return is a real likelihood.

## Data handling

Two preprocessing steps are load-bearing, not incidental:

* **Uniform dequantisation.** Pixel values are discrete; a continuous density
  would put infinite mass on them. Adding uniform noise turns the discrete
  distribution into a continuous one whose likelihood is a valid bound on the
  discrete-data likelihood.
* **Logit transform.** Data lives in `[0, 1]`; an unconstrained Gaussian latent
  fits a bounded interval badly. `logit(alpha + (1 - 2*alpha) * x)` moves it to
  the whole real line.

## Run it

```bash
python download_data.py
python train.py                    # MNIST
python evaluate.py
```

Useful variations:

```bash
python train.py --dataset cifar10 --epochs 12 --hidden 96
python train.py --num-scales 3 --couplings-per-scale 4     # a deeper flow
```

## Files

| File | What it holds |
|---|---|
| `model.py` | `AffineCoupling`, masking, `squeeze`/`unsqueeze`, `RealNVP` |
| `download_data.py` | dataset fetcher |
| `train.py` | maximises the exact log-likelihood |
| `evaluate.py` | test bits/dim, invertibility check, temperature sweep |

## Test-set evaluation

`outputs/evaluation/metrics.json` reports:

* **`test_bits_per_dim`** -- the headline. Negative log-likelihood in bits per
  subpixel. Lower is better, no reference network is involved, and the number
  means the same thing here as in any flow paper. A uniform distribution over
  8-bit pixels scores exactly 8.0, so anything below that is real modelling.
* **`max_roundtrip_error`** -- encode test images, decode them again, measure the
  worst error. This should sit at floating-point noise. If it does not, the flow
  is not invertible and every likelihood it reports is meaningless.
* FID/KID, for comparison with the other projects here.

Figures: `samples.png` and `temperature-sweep.png`.

## Reference run

MNIST, 2 epochs, 2 scales, 3 couplings per scale, 0.18M parameters, CPU only
(4 cores, about 15 minutes end to end):

| Metric | Value |
|---|---|
| train bits/dim, epoch 1 | 2.328 |
| train bits/dim, epoch 2 | 1.749 |
| **test bits/dim** | **1.713** |
| max round-trip error | 0.0 |
| FID (see note) | 22.8 |
| KID | 0.863 +/- 0.073 |
| precision / recall | 0.51 / 0.44 |

Two things in that table are worth sitting with.

**The invertibility check returns exactly 0.0.** Not "small" -- zero to
floating-point precision. That is the guarantee a flow gives you and neither the
VAE nor the GAN can: the transform is invertible by construction, so the
likelihood it reports is exact rather than a bound or an estimate.

**It has the worst FID of the three models measured here**, on the same dataset
and the same feature space:

| Model | test bits/dim | FID | precision / recall |
|---|---|---|---|
| VAE (01) | 0.207 (ELBO bound) | 11.6 | 0.68 / 0.74 |
| DCGAN (03) | not available | **2.94** | 0.70 / 0.87 |
| RealNVP (07) | **1.713** (exact) | 22.8 | 0.51 / 0.44 |

The flow is the only one of the three that can tell you the exact likelihood of
a test image, and it produces the least convincing samples. Look at
`temperature-sweep.png`: after two epochs the outputs are stroke-like blobs with
digit statistics rather than digits.

This is not a bug and not merely undertraining, though more epochs would help.
Bits/dim is dominated by getting the high-frequency pixel noise distribution
right, because that is where most of the entropy in an image lives. A model can
win on likelihood by modelling texture statistics well while never learning that
a digit needs to be one connected stroke. **Likelihood and perceptual quality
are different objectives**, and this is the clearest place in the repository to
see them come apart.

Note also the cost: a flow must preserve dimensionality at every layer, so its
0.18M parameters buy far less than the same budget spent on a GAN.

## What to look at

* The temperature sweep. Flows sample `z ~ N(0, T^2 I)`; `T < 1` concentrates
  near the distribution's mode. At `T = 0.5` samples are clean and repetitive;
  at `T = 1.0` they are diverse and noisier. Glow uses exactly this knob for its
  headline figures, and it is a fair reminder that published flow samples are
  usually not drawn at `T = 1`.
* Bits/dim falls fast at first and then crawls. Flow likelihoods are dominated
  by high-frequency pixel noise, which is why a model can score well on bits/dim
  while producing unimpressive samples -- likelihood and perceptual quality are
  genuinely different objectives, and this project is the clearest place in the
  repository to see that.
* `--num-scales 3` roughly doubles depth and cost. Watch how much bits/dim
  actually improves for it.

## A note on FID and KID in this repository

The bits/dim figure is standard and comparable to the literature. The FID/KID
here are not -- they use a small per-dataset feature space. See
`../01-vae/README.md`.
