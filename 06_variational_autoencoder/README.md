# 06 - Variational autoencoder (a latent space you can sample from)

Project 04 built an autoencoder: an encoder that maps each image to a *point*, and a
decoder that maps it back. It compresses well and its latent space is full of holes -
decode a random code and you get noise, because nothing ever asked the space between
training points to mean anything.

A VAE asks. The encoder outputs a **distribution** - a mean and a log-variance - a
sample is drawn from it, and the decoder must reconstruct from that sample. Because
the encoder has to place a whole Gaussian where each image lives, and because a KL
term pulls every one of those Gaussians towards a shared unit Gaussian prior, the
latent fills in. Interpolations stay on the manifold, and sampling from the prior
produces images.

The objective is a bound on the log likelihood, the ELBO:

```
log p(x)  >=  E[ log p(x|z) ]  -  KL( q(z|x) || p(z) )
              reconstruction        stay near the prior
```

Those two terms fight, and `--beta` sets the exchange rate. At `beta=1` it is a VAE;
raise it and you have a beta-VAE.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset fashion-mnist --preview     # download (~30 MB)
python train.py --dataset fashion-mnist              # 12 epochs, ~4 min on four CPU cores
python evaluate.py --checkpoint outputs/fashion-mnist_beta1_z16/best.pt
```

## Posterior collapse, counted

The most useful number this project prints is not the ELBO. It is `active`: the
number of latent dimensions whose KL exceeds 0.01 nats.

A dimension with near-zero KL has posterior equal to prior. The encoder puts nothing
in it and the decoder cannot read it - it has been switched off. This is **posterior
collapse**, and it is the failure mode that makes VAE results hard to interpret,
because a model with 3 of 16 dimensions alive can still post a respectable ELBO.

Two things follow, and both are visible in `compare.py`:

- raising `beta` does not degrade the latent smoothly, it switches dimensions off one
  at a time;
- adding latent capacity beyond what the data needs costs nothing, because the KL
  term prunes the surplus by itself.

`kl_per_dimension.png` shows the whole distribution, and `traversals.png` varies the
most informative dimensions one at a time - which is where a beta-VAE's
disentanglement claim either shows up or does not.

## What the numbers mean, and do not mean

**-ELBO is an upper bound on the negative log likelihood, not the likelihood.** It is
loose by exactly the KL between the approximate posterior and the true one, and
nothing in this project can tell you by how much. `bits_per_dimension_bound` is
reported in the same units project 11 (normalizing flows) reports its *exact*
likelihood, which is the honest comparison and the reason flows exist.

**FID here is not published FID.** Computing the standard metric means downloading a
90 MB ImageNet InceptionV3, which would be the largest dependency in the repository
by an order of magnitude. Instead a small classifier is trained on the dataset itself
(2 epochs, cached in `data/`) and its penultimate features are used. The values are
therefore only comparable *within* this repository - which is what every comparison
here needs, and it is the same ruler for every arm of every study. `utils.py` says so
at the top, `metrics.json` records `feature_net`, and every table repeats it.

**KID is the more trustworthy of the two** at these sample counts. FID is biased for
finite samples: two thousand samples from *the same* distribution score around 2
rather than 0 in this feature space, while KID's unbiased estimator sits at
approximately zero. The implementation is checked against that, and against the
analytic case where two unit Gaussians a distance `d` apart must give FID exactly
`d^2`.

**Precision and recall separate the two failure modes FID conflates.** Precision is
the fraction of samples inside the real data's k-NN manifold - are they plausible.
Recall is the fraction of real images inside the samples' manifold - do they cover
the data. A model that memorises one convincing image scores precision 1.0 and recall
0.0; the implementation is checked against exactly that case.

## Measured results

Fashion-MNIST, 20 000 training images at 32x32, latent 16, `beta=1`, Bernoulli likelihood,
12 epochs on four CPU cores (286 s), scored on the 10 000-image test split:

| Quantity | Value |
|---|---|
| -ELBO | **318.38 nats** (reconstruction 301.16 + KL 17.22) |
| bits/dim (bound) | 0.4486 |
| active latent units | **16 / 16** |
| reconstruction PSNR | 22.15 dB |
| FID (prior samples) | 14.331 |
| KID | +1.479 +- 0.104 |
| precision / recall | 0.490 / 0.403 |

Two things to read off that. At `beta=1` **nothing has collapsed** - all sixteen dimensions
carry more than 0.01 nats, so the KL total of 17.22 is spread across the latent rather than
concentrated in three surviving dimensions. Raise beta and that changes, which is what the
`beta` study is for.

And the recall of 0.40 is the number to carry forward. Put it next to project 12's diffusion
model on the same dataset: a VAE covers less of the distribution than diffusion does, and its
samples are visibly blurrier, which is the practical reason diffusion displaced it for image
synthesis. The bits/dim bound of 0.449 is also worth putting next to project 11's *exact*
number on the same data - a bound and a likelihood are not the same claim.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `beta` | 0.5 / 1 / 4 / 8 | Higher beta: lower KL, fewer active units, blurrier reconstructions, tidier latent |
| `latent` | 2 / 8 / 32 dimensions | Two is visibly too few; thirty-two is not better than eight, it just prunes itself |
| `likelihood` | Bernoulli vs unit-variance Gaussian | Bernoulli trains better on greyscale even though the data is not binary |

Every arm is scored on **-ELBO with beta set back to 1**, so the numbers compare even
though the training objectives did not.

## Files

| File | What it does |
|---|---|
| `data.py` | MNIST, Fashion-MNIST, CIFAR-10, CelebA and any image folder, all in `[0, 1]` |
| `model.py` | Conv VAE, the reparameterisation trick, prior sampling, interpolation, traversals |
| `train.py` | ELBO training with a beta-free validation number and the active-unit counter |
| `evaluate.py` | Test ELBO and bits/dim, reconstruction PSNR, FID/KID/precision/recall, latent figures |
| `compare.py` | `beta`, `latent` and `likelihood` studies |
| `utils.py` | FID, KID, generative precision/recall and the feature network they run through |

## Outputs

In `outputs/<dataset>_beta<b>_z<d>/`:

- `best.pt`, `history.json`, `curves.png`, `reconstructions_val.png`
- `samples.png` - decoded from the prior, the test a plain autoencoder fails
- `latent_space.png` - posterior means coloured by class
- `interpolation.png` - straight lines between two posterior means
- `traversals.png` - one latent dimension at a time, most informative first
- `kl_per_dimension.png` - the active-unit picture
- `metrics.json` - ELBO, bits/dim bound, active units, and the sample metrics

## Knobs worth turning

- `--beta 0` turns the VAE into a plain autoencoder with a noisy encoder. Sample from
  the prior afterwards and compare with project 04: this is the clearest way to see
  what the KL term is for.
- `--latent-dim 2` makes `latent_space.png` exact rather than a PCA projection, which
  is worth seeing once even though two dimensions cost real reconstruction quality.
- `--dataset cifar10` is much harder: colour photographs under a per-pixel
  independent likelihood produce the famous blur, and the FID gap against the
  diffusion projects (12-14) on the same dataset is the reason those exist.
