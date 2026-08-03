# model-pedia

Small, self-contained reference implementations of the models you meet on the
way into deep learning. Each one is a folder you can read in a sitting and run
in a few minutes on a laptop CPU.

Every project answers the same three questions with runnable code: **what is
the model**, **what does it train on**, and **how do you know it worked**. The
last one gets as much attention as the first two - each project ends in a
test-set number next to the baseline that number should be compared against,
and a figure that shows the idea rather than just the score.

## Projects

### Part I - supervised learning and representations

| # | Project | Model | Dataset | What it shows |
|---|---|---|---|---|
| 01 | [Classic classifier](01_classic_classifier/) | MLP, LeNet-5 | MNIST, Fashion-MNIST | Weight sharing beats raw capacity: LeNet wins with a quarter of the parameters |
| 02 | [CNN classifier](02_cnn_classifier/) | ResNet-20/32/18, plain-net control | CIFAR-10/100, SVHN | What the identity shortcut is actually worth, measured against the same net without it |
| 03 | [Transfer learning](03_transfer_learning/) | Pretrained ResNet-18, MobileNetV3, EfficientNet-B0 | Flowers-102, Oxford-IIIT Pets | Frozen linear probe vs full fine-tuning, with a random-init control |
| 04 | [Autoencoder](04_autoencoder/) | Convolutional AE | Fashion-MNIST, CIFAR-10 | A bottleneck turns compression into representation learning |
| 05 | [Denoising autoencoder](05_denoising_autoencoder/) | U-Net, no-skip control | CIFAR-10, any image folder | A learned image prior, and where it stops generalising |

### Part II - generative models

| # | Project | Model | Dataset | What it shows |
|---|---|---|---|---|
| 06 | [Variational autoencoder](06_vae/) | Conv-VAE, beta-VAE | MNIST, Fashion-MNIST, CelebA | The ELBO, and why a KL term makes a latent space you can sample from |
| 07 | [Vector-quantized AE](07_vq_vae/) | VQ-VAE + PixelCNN prior | CIFAR-10, CelebA | A discrete latent, and the second model you then need in order to sample |
| 08 | [GAN](08_dcgan/) | DCGAN | MNIST, Fashion-MNIST, CIFAR-10 | Adversarial training instead of likelihood, and why the samples are sharper |
| 09 | [Conditional GAN](09_conditional_gan/) | cGAN, ACGAN control | MNIST, CIFAR-10 | Control over what gets generated, and how to check the label is obeyed |
| 10 | [Image-to-image, paired](10_pix2pix/) | Pix2Pix (U-Net + PatchGAN) | edges-to-photo, Facades | Skip connections and a patch discriminator; what the L1 term is holding up |
| 11 | [Image-to-image, unpaired](11_cyclegan/) | CycleGAN | CIFAR-10 classes, horse-to-zebra | Cycle consistency as the constraint that replaces paired data |
| 12 | [Normalizing flow](12_realnvp/) | RealNVP | MNIST, CIFAR-10 | An exact likelihood, and the price paid for it in sample quality |
| 13 | [Diffusion](13_ddpm/) | DDPM with a U-Net | MNIST, CIFAR-10, CelebA | Generation as learned denoising, with the simplest loss in the repository |
| 14 | [Faster diffusion sampling](14_ddim/) | DDIM | reuses project 13 | 15x fewer network evaluations at the same quality, no retraining |
| 15 | [Latent diffusion](15_latent_diffusion/) | Autoencoder + diffusion | Fashion-MNIST, CelebA | Diffusing in a compressed space: better samples for half the compute |
| 16 | [Vision Transformer](16_vit/) | ViT-Tiny, ResNet-18 baseline | Fashion-MNIST, CIFAR-10, Flowers-102 | Inductive bias vs expressiveness, compared at matched compute |

Projects 14 and 15 import the U-Net and diffusion process from project 13 rather
than copying them, so the sampler comparison and the latent-space variant always
run against the same architecture.

## Quickstart

```bash
pip install -r requirements.txt

cd 01_classic_classifier
python data.py --dataset fashion-mnist     # download
python train.py --model lenet --epochs 5   # train
python evaluate.py --checkpoint outputs/fashion-mnist_lenet/best.pt
```

Every project follows the same layout, so once you have read one you can
navigate the rest:

| File | Role |
|---|---|
| `data.py` | Downloads the dataset and builds the splits. Runnable on its own |
| `model.py` | The architecture, and nothing else. Runnable on its own to print shapes and parameter counts |
| `train.py` | Training loop; writes a checkpoint, `history.json` and a curves figure |
| `evaluate.py` | Loads a checkpoint, scores the **test** split, writes `metrics.json` and figures |
| `compare.py` / `sweep.py` | A controlled study: one thing changes, everything else is held fixed |
| `utils.py` (part I) / `common/` (part II) | Seeding, device selection, metrics, plotting |

Shared conventions:

- `--device auto` picks CUDA, then MPS, then CPU. Every script takes `--seed`.
- The validation split drives checkpoint selection; the test split is touched
  only by `evaluate.py`.
- Results land in `<project>/outputs/`, git-ignored. Part I keeps datasets in
  `<project>/data/`; part II shares one `data/` at the repository root, so MNIST
  is downloaded once no matter how many projects use it.
- Part I's scripts take `--smoke-test`; part II's take `--quick`. Both run the
  whole pipeline on a handful of batches so you can check it works before
  committing to a download or a training run.
- Part II adds `--train-subset N` for a predictable runtime on a small machine.

### Why part II has a shared `common/`

Part I gives every project its own `utils.py`, which is the right call when the
helpers are seeding, plotting and an accuracy function. Part II's evaluation is
heavier - FID, KID, precision/recall, SSIM and a cached per-dataset feature
extractor come to several hundred lines - and copying that into eleven folders
would guarantee they drift apart. It lives in `common/` instead, and the
per-project file contract is otherwise identical.

## Runtimes

Defaults are tuned so that each project finishes in a few minutes on four CPU
cores, while still showing the effect it is about. Where that meant training on
a subset, the README says so and gives the command for the full run.

| Project | Default run | Download |
|---|---|---|
| 01 Classic classifier | ~2 min | 12-30 MB |
| 02 CNN classifier | ~7 min (15k-image subset) | 170 MB |
| 03 Transfer learning | ~2 min frozen, ~6 min fine-tune | 345 MB + backbone weights |
| 04 Autoencoder | ~3 min | 30 MB |
| 05 Denoising autoencoder | ~5 min | 170 MB |
| 06 Variational autoencoder | ~4 min | 12 MB |
| 07 Vector-quantized AE | ~8 min | 30-170 MB |
| 08 GAN | ~9 min | 12 MB |
| 09 Conditional GAN | ~16 min | 12 MB |
| 10 Image-to-image, paired | ~8 min | 30 MB |
| 11 Image-to-image, unpaired | ~10 min | 42 MB |
| 12 Normalizing flow | ~15 min | 12 MB |
| 13 Diffusion | ~13 min | 12 MB |
| 14 Faster diffusion sampling | ~10 min, no training | reuses 13 |
| 15 Latent diffusion | ~7 min | 30 MB |
| 16 Vision Transformer | ~7 min ResNet, ~17 min ViT | 30 MB |

A GPU is not required anywhere. If you have one, raise `--epochs` (or
`--max-steps` for the diffusion projects) and drop `--train-subset` for numbers
comparable with the literature. The diffusion projects are the ones that most
repay a GPU: project 13 is visibly undertrained at its CPU default, and says so.

## Checking the repository

```bash
python scripts/smoke_test.py    # part I, on synthetic data, no downloads
./verify.sh                     # part II, every project in --quick mode
python -m common.selftest       # part II, the metric checks alone
```

`verify.sh` runs the metric self-test first, and deliberately so. It pins the
properties the reported numbers depend on: FID of a distribution against itself
is zero, FID grows with corruption, and a collapsed generator shows high
precision with near-zero recall. That last check is not hypothetical - a
conditional GAN here reported precision 0.70 with recall exactly 0.00, and these
checks are what established the metric was right and the model was wrong.

## How the generative metrics work

Part II reports two numbers that are standard and comparable to published work:
**bits per dimension** (project 12) and **classification accuracy** (project 16).

**FID and KID here are not.** Published FID uses an ImageNet-trained
InceptionV3, which is a poor fit for 32x32 greyscale digits and a heavy
dependency for a repository whose whole premise is a short requirements file.
These projects compute the same statistics in a **small per-dataset feature
space**: a compact CNN trained once per dataset on first use, cached under
`outputs/feature-extractors/`. So:

* the scores are directly comparable **between projects here**, because every
  project reuses the same cached extractor for a given dataset;
* they are **not** comparable to FID values quoted in papers;
* lower is still better, as usual.

Pass `--feature-extractor inception` to any part II `evaluate.py` for
literature-comparable values, if the torchvision weights are available to you.

Beyond FID, each project reports the metric that actually diagnoses its own
model: precision/recall for the GANs, codebook perplexity for VQ-VAE, cycle
error for CycleGAN, an invertibility check for RealNVP, per-class FID and
classifier agreement for the conditional GAN, and the stage-1 floor for latent
diffusion. A single headline number rarely says anything useful on its own.

## Measured results

Every part II project was trained and evaluated on four CPU cores with no GPU.
FID uses each dataset's own cached feature space, so numbers are comparable
down a group and not across groups or to published work.

**MNIST**, comparable budgets:

| Project | FID | precision / recall | Also |
|---|---|---|---|
| 09 Conditional GAN | **0.80** | 0.86 / 0.77 | 97.6% conditioning accuracy |
| 08 DCGAN | 2.94 | 0.70 / 0.87 | |
| 06 VAE | 11.6 | 0.68 / 0.74 | 0.207 bits/dim (ELBO bound) |
| 12 RealNVP | 22.8 | 0.51 / 0.44 | **1.713 bits/dim, exact** |
| 13 DDPM | 46.6 | 0.27 / 0.29 | 300 network evals per image |
| 14 DDIM | 178 at 20 evals | | **15x fewer evals, flat FID** |

**Fashion-MNIST**, 20000-image subsets:

| Project | Headline | Note |
|---|---|---|
| 10 Pix2Pix | FID 0.53, SSIM 0.765 | paired translation is the easiest task here |
| 07 VQ-VAE | recon FID 1.90, sample FID 10.1 | 256/256 codes used, 16x compression |
| 11 CycleGAN | FID 6.6 / 13.6 | cycle SSIM 0.89 / 0.68 |
| 15 Latent diffusion | FID 37.8 | beats pixel-space DDPM at half the training time |
| 16 ViT vs ResNet-18 | 0.826 vs **0.920** | the CNN wins on accuracy *and* FLOPs |

Three of these are worth reading twice. **RealNVP has the best likelihood and
the worst samples**, the cleanest demonstration here that the two objectives
differ. **The ViT loses to the ResNet while spending 25 percent more compute**,
which is why that comparison carries a FLOPs column. And **latent diffusion
beats pixel-space diffusion on both quality and cost**, which is the whole
reason the method exists.

These come from short CPU runs and are not tuned. Project 13 in particular is
undertrained at this budget and its README says so; raising `--max-steps` on a
GPU changes that picture substantially.

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

`shapes` exists so any part II pipeline can be run with no network access at
all. It is deliberately easy - use it to check that something runs, then switch
to real data for results that mean anything.

## Roadmap

Still to come, roughly in order: self-supervised learning, masked image
modelling, semantic segmentation, object detection and image captioning.

## Requirements

Python 3.9+, PyTorch 2.0+, torchvision, numpy, scipy, matplotlib, tqdm, Pillow.
Nothing else - the metrics that would normally justify a heavier dependency
(confusion matrices, F1, PSNR, SSIM, FID, KID) are implemented in the projects
themselves, and are short enough to be worth reading once.
