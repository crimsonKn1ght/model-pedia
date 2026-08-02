# model-pedia

Eleven self-contained projects covering the main families of generative image
models, plus a Vision Transformer for classification. Each one is small enough
to train on a laptop CPU in minutes, and each is built around the one idea that
makes its model family work.

Every project follows the same shape:

```
NN-name/
  README.md          the idea, what to run, what the numbers mean
  model.py           the architecture and its loss
  download_data.py   fetch the datasets
  train.py           training loop, writes checkpoints and per-epoch figures
  evaluate.py        held-out evaluation, writes metrics.json and figures
```

## The projects

| # | Project | Model | Task | Default dataset |
|---|---|---|---|---|
| [01](01-vae/) | Variational autoencoder | Conv-VAE, beta-VAE | reconstruct and generate | MNIST |
| [02](02-vq-vae/) | Vector-quantized AE | VQ-VAE + PixelCNN prior | discrete representation, generation | CIFAR-10 |
| [03](03-dcgan/) | GAN | DCGAN | unconditional generation | MNIST |
| [04](04-conditional-gan/) | Conditional GAN | cGAN, ACGAN | class-controlled generation | MNIST |
| [05](05-pix2pix/) | Image-to-image, paired | Pix2Pix | edges -> photo, facades | CIFAR-10 |
| [06](06-cyclegan/) | Image-to-image, unpaired | CycleGAN | domain translation | CIFAR-10 |
| [07](07-realnvp/) | Normalizing flow | RealNVP | exact-likelihood density modelling | MNIST |
| [08](08-ddpm/) | Diffusion | DDPM with a U-Net | generation and denoising | MNIST |
| [09](09-ddim/) | Faster diffusion sampling | DDIM | speed/quality against DDPM | reuses 08 |
| [10](10-latent-diffusion/) | Latent diffusion | autoencoder + diffusion | efficient generation | Fashion-MNIST |
| [11](11-vit/) | Vision Transformer | ViT-Tiny vs ResNet-18 | classification at matched compute | CIFAR-10 |

Projects 09 and 10 import the U-Net and diffusion process from project 08 rather
than copying them, so the sampler comparison and the latent-space variant always
run against the same architecture.

## Getting started

```bash
pip install -r requirements.txt

cd 01-vae
python download_data.py
python train.py
python evaluate.py
```

Every script takes `--help`. Three flags are shared by all of them:

| Flag | Purpose |
|---|---|
| `--quick` | run a handful of batches; verifies the pipeline in seconds |
| `--device` | `auto` (default), `cpu`, `cuda`, `mps` |
| `--data-root` | dataset location, shared across projects by default |

Datasets land in `data/` at the repository root and are shared, so MNIST is
downloaded once no matter how many projects use it. Checkpoints and figures go
to each project's own `outputs/`. Both directories are git-ignored.

### On a CPU-only machine

Install the CPU wheels, which are far smaller than the CUDA build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Defaults throughout are tuned for a 4-core CPU: most projects finish in 5-15
minutes and produce results that show what the model actually does. They are
deliberately *not* tuned for state-of-the-art numbers. On a GPU, raise
`--epochs` / `--max-steps` and `--base-channels`, and switch to the larger
datasets suggested in each project's README.

## Datasets

| Name | Size | Notes |
|---|---|---|
| `mnist`, `fashion-mnist` | 28x28 greyscale | downloaded automatically |
| `cifar10` | 32x32 colour | downloaded automatically |
| `flowers102` | variable, resized | downloaded automatically |
| `celeba64` | 64x64 faces | **manual download**, see below |
| `shapes` | 32x32 colour | generated procedurally, needs no network at all |

`celeba64` is the one exception: the official archive lives on Google Drive,
which routinely rejects scripted downloads. Fetch `img_align_celeba.zip` by hand
and unzip it so images sit at `data/celeba/img_align_celeba/*.jpg`. Every
project that offers CelebA also works on a smaller dataset, so it is always
optional.

`shapes` exists so any pipeline can be run and smoke-tested with no network
access at all. It is deliberately easy -- use it to check that something runs,
then switch to real data for results that mean anything.

## How the metrics work

Two of the reported numbers are standard and comparable to published work:
**bits per dimension** (project 07) and **classification accuracy** (project 11).

**FID and KID here are not.** Published FID uses an ImageNet-trained
InceptionV3, which is a poor fit for 32x32 greyscale digits and a heavy
dependency for a lightweight project. These projects instead compute the same
statistics in a **small per-dataset feature space**: a compact CNN is trained
once per dataset on first use (seconds on a CPU) and cached under
`outputs/feature-extractors/`. So:

* the scores are directly comparable **between projects in this repository**,
  because every project reuses the same cached extractor for a given dataset;
* they are **not** comparable to FID values quoted in papers;
* lower is still better, as usual.

Pass `--feature-extractor inception` to any `evaluate.py` for literature-
comparable values, if the torchvision weights are available to you.

Beyond FID, each project reports the metrics that actually diagnose its own
model: precision/recall for GANs, codebook perplexity for VQ-VAE, cycle error
for CycleGAN, an invertibility check for RealNVP, per-class FID and classifier
agreement for the conditional GAN, and the stage-1 floor for latent diffusion. A
single headline number rarely says anything useful on its own, and each README
explains what its numbers are for.

## Repository layout

```
common/            shared code: dataset registry, figures, metrics
  data.py          datasets, transforms, paired/unpaired wrappers
  metrics.py       FID, KID, precision/recall, SSIM, PSNR, feature extractors
  viz.py           sample grids, curves, latent plots
  utils.py         seeding, devices, checkpoints, argument parsing
01-vae/ ... 11-vit/
data/              downloaded datasets (git-ignored)
outputs/           cached feature extractors (git-ignored)
```

Projects import `common` through a one-line `bootstrap.py` that puts the
repository root on `sys.path`, so scripts run correctly from inside their own
folder.

## Suggested order

The projects build on each other, and reading them in order is the intended
path:

1. **01 VAE** -- latent variables, the ELBO, why sampling from a prior works.
2. **02 VQ-VAE** -- what changes when the latent goes discrete, and why that
   needs a second model to sample from.
3. **03 DCGAN** -- adversarial training instead of likelihood.
4. **04 cGAN/ACGAN** -- adding control, and how to check it is real.
5. **05 Pix2Pix** and **06 CycleGAN** -- conditioning on an image; what cycle
   consistency replaces when pairs are unavailable.
6. **07 RealNVP** -- the exact-likelihood alternative, and what it costs.
7. **08 DDPM** -> **09 DDIM** -> **10 latent diffusion** -- the modern line, and
   the two efficiency ideas that made it practical.
8. **11 ViT** -- a change of subject: inductive bias versus expressiveness,
   measured rather than argued.
