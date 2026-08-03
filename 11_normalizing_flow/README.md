# 11 - Normalizing flow (the exact likelihood)

Project 06's VAE reports a *bound* on the likelihood. Project 08's GAN has no likelihood at
all. This project reports the exact number, and the price is a constraint on the
architecture: **every layer must be invertible with a tractable Jacobian determinant.**

The change of variables formula is the whole idea. If `z = f(x)` is invertible,

```
log p(x) = log p_z(f(x)) + log |det df/dx|
```

Choose `p_z` to be a unit Gaussian and computing `log p(x)` is just a forward pass plus a
correction term. Sampling is the same network run backwards.

```bash
pip install -r ../requirements.txt

python data.py --dataset fashion-mnist        # ~30 MB
python train.py --dataset fashion-mnist        # 15 epochs, ~4 min on four CPU cores
python evaluate.py --checkpoint outputs/fashion-mnist_h64/best.pt
```

## The affine coupling layer

Split the input in two. Pass the first half through unchanged, and use it to predict a scale
and a shift for the second:

```
y_a = x_a
y_b = x_b * exp(s(x_a)) + t(x_a)
```

Inverting is trivial - subtract and divide, using `x_a = y_a` which you already have - and
the Jacobian is triangular, so its log-determinant is just `sum(s(x_a))`. The networks `s`
and `t` can be arbitrarily complicated, because they are **never inverted**. That is the
trick, and everything else in `model.py` is bookkeeping around it:

- masks alternate between checkerboard and channel-wise, so every dimension gets both
  transformed and used as a condition;
- `squeeze` trades resolution for channels, which is what lets the channel masks see
  spatial neighbours;
- the last convolution of each coupling network is zero-initialised, so an untrained flow is
  exactly the identity and training starts from a sane place.

`python model.py` asserts the two properties the likelihood depends on: the network inverts
to within 1e-7, and `squeeze`/`unsqueeze` are exact inverses.

## Bits per dimension, and why 8.0 is the line

This is the one number in the generative half of this repository that **is** comparable with
published results - unlike the FID values, which go through a feature network of this
repository's own making.

The convention: the density is over `[0, 1]` while the data has 256 grey levels, so
converting to bits per dimension of the discrete data adds `log2(256) = 8`. A uniform
distribution over those levels scores exactly **8.0 bits/dim**, so that is the number to
beat - anything above it means the model is worse than assuming nothing.

## Dequantisation is not optional

Pixels are discrete, and a continuous density can place unbounded mass on a finite set of
points. A flow fitted directly to quantised pixels reports a bits-per-dimension that
improves without limit and describes nothing.

The fix is to add uniform noise within each quantisation bin, which turns the problem back
into a continuous one and makes the result a proper bound on the discrete likelihood.
`compare.py --study dequantisation` runs both arms and evaluates them identically, so the
difference in the table is entirely a difference in what they were trained on. Run it once
to see what an impossible number looks like.

The logit transform in the same function is the other half of the preprocessing: pixels live
in a box and a Gaussian does not, so mapping to logit space saves the flow from spending
capacity on squashing.

## Likelihood is not perceptual quality

`evaluate.py` also reports FID, KID and precision/recall, so the flow can be compared with
projects 08 and 12 on the same dataset. The expected result is that the flow wins on
likelihood and loses visibly on samples. That is not a bug in either metric - they are
different objectives, and this is the cleanest place in the repository to watch them
disagree.

`temperature.png` shows the standard mitigation: sampling from a narrower prior gives
cleaner and less varied images, trading recall for precision without retraining.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `capacity` | Coupling width 32 / 64 / 128 | Bits/dim improves with diminishing returns |
| `depth` | 2 / 3 / 5 couplings per stage | More couplings, more expressive; watch the inversion error stay small |
| `dequantisation` | With and without | The warning, not a tuning exercise |

## Files

| File | What it does |
|---|---|
| `data.py` | The same datasets as the other generative projects, in `[0, 1]` |
| `model.py` | Coupling layers, masks, squeeze, preprocessing, exact log-likelihood |
| `train.py` | Maximises likelihood directly; the loss *is* the metric here |
| `evaluate.py` | Exact bits/dim, the invertibility check, samples, temperature sweep |
| `compare.py` | `capacity`, `depth` and `dequantisation` studies |
| `utils.py` | FID, KID and precision/recall, shared with the other generative projects |

## Knobs worth turning

- `--couplings-per-stage 6` and watch bits/dim improve while sample quality does not keep up.
- `evaluate.py --temperature 0.7` for the cleaner-but-less-varied trade.
- Put the bits/dim number next to project 06's `bits_per_dimension_bound` on the same
  dataset. The gap between a bound and the exact value is the thing you cannot see from
  inside a VAE.
