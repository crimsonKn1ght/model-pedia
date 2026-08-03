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

| # | Project | Model | Dataset | What it shows |
|---|---|---|---|---|
| 01 | [Classic classifier](01_classic_classifier/) | MLP, LeNet-5 | MNIST, Fashion-MNIST | Weight sharing beats raw capacity: LeNet wins with a quarter of the parameters |
| 02 | [CNN classifier](02_cnn_classifier/) | ResNet-20/32/18, plain-net control | CIFAR-10/100, SVHN | What the identity shortcut is actually worth, measured against the same net without it |
| 03 | [Transfer learning](03_transfer_learning/) | Pretrained ResNet-18, MobileNetV3, EfficientNet-B0 | Flowers-102, Oxford-IIIT Pets | Frozen linear probe vs full fine-tuning, with a random-init control |
| 04 | [Autoencoder](04_autoencoder/) | Convolutional AE | Fashion-MNIST, CIFAR-10 | A bottleneck turns compression into representation learning |
| 05 | [Denoising autoencoder](05_denoising_autoencoder/) | U-Net, no-skip control | CIFAR-10, any image folder | A learned image prior, and where it stops generalising |
| 06 | [Variational autoencoder](06_variational_autoencoder/) | Conv VAE, beta-VAE | Fashion-MNIST, CIFAR-10, CelebA | A latent space you can sample from, and posterior collapse counted |
| 07 | [Vector-quantized AE](07_vector_quantized_ae/) | VQ-VAE + autoregressive code prior | Fashion-MNIST, CIFAR-10 | A discrete description of an image, and why generation needs a second stage |
| 08 | [GAN](08_gan/) | DCGAN | Fashion-MNIST, CIFAR-10, CelebA | Learning with no loss you can evaluate; mode collapse measured, not guessed |
| 09 | [Conditional GAN](09_conditional_gan/) | cGAN, ACGAN | Fashion-MNIST, CIFAR-10 | Steerable generation, and the accuracy/diversity trade it exposes |
| 10 | [Image-to-image GAN](10_image_to_image_gan/) | Pix2Pix (U-Net + PatchGAN) | Facades, generated pairs | Two losses that disagree, and why you need both |
| 11 | [Normalizing flow](11_normalizing_flow/) | RealNVP | Fashion-MNIST, CIFAR-10 | The exact likelihood, and what dequantisation is for |
| 12 | [Diffusion](12_diffusion/) | DDPM with a U-Net | Fashion-MNIST, CIFAR-10 | Many small easy problems instead of one hard one |
| 13 | [Faster diffusion inference](13_faster_diffusion/) | DDIM | Project 12's trained DDPM | Quality against sampling cost, as a curve rather than a number |
| 14 | [Latent diffusion](14_latent_diffusion/) | Autoencoder + diffusion in its latent | Fashion-MNIST, CIFAR-10 | Denoise a smaller tensor, and measure the ceiling that sets |
| 15 | [Vision Transformer](15_vision_transformer/) | ViT-Tiny, size-matched ResNet | CIFAR-10/100, Flowers-102 | What the convolution was giving you for free |
| 16 | [Self-supervised learning](16_self_supervised/) | SimCLR, BYOL | CIFAR-10, STL-10, MNIST | Labels are not the only supervision - and the augmentation *is* the supervision |
| 17 | [Masked image modelling](17_masked_image_modeling/) | MAE with a ViT encoder | CIFAR-10, any `ImageFolder` | Hide 75% of the image; reconstruction is the objective, the encoder is the result |
| 18 | [Semantic segmentation](18_semantic_segmentation/) | U-Net, no-skip and dilated controls | Oxford-IIIT Pets, generated shapes | A label per pixel, and why mean IoU and pixel accuracy disagree |
| 19 | [Object detection](19_object_detection/) | Anchor-free single-scale detector | MNIST-derived scenes, Pascal VOC 2007 | Variable-length output: assignment, NMS and mAP written out |
| 20 | [Image captioning](20_image_captioning/) | CNN / frozen-ImageNet encoder + Transformer decoder | Flickr8k, generated shapes | A sequence out, and three metrics that disagree about it |

The numbering follows the reference table this repository works through. Projects 13 and 14
build on project 12 rather than standing alone - DDIM re-samples an already-trained DDPM, and
latent diffusion is a composition of an autoencoder with one - so they import its model and
metrics instead of copying them. Every other project is self-contained.

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
| `train.py` | Training loop; writes a checkpoint, `history.json` and `curves.png` |
| `evaluate.py` | Loads a checkpoint, scores the **test** split, writes `metrics.json` and figures |
| `compare.py` / `sweep.py` | A controlled study: one thing changes, everything else is held fixed |
| `utils.py` | Seeding, device selection, metrics, plotting |

Shared conventions:

- `--device auto` picks CUDA, then MPS, then CPU. Every script takes `--seed`.
- The validation split drives checkpoint selection; the test split is touched
  only by `evaluate.py`.
- Datasets land in `<project>/data/`, results in `<project>/outputs/`. Both are
  git-ignored.
- `train.py`, `evaluate.py` and the study scripts take `--smoke-test`, which
  runs the whole pipeline on random tensors so you can check it works before
  committing to a download.

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
| 06 Variational autoencoder | ~5 min | 30 MB |
| 07 Vector-quantized AE | ~4 min stage one, ~4 min the code prior | 30 MB |
| 08 GAN | ~10 min (FID every epoch) | 30 MB |
| 09 Conditional GAN | ~11 min | 30 MB |
| 10 Image-to-image GAN | ~8 min | none (generated pairs), 30 MB (facades) |
| 11 Normalizing flow | ~5 min | 30 MB |
| 12 Diffusion | ~12 min (sampling dominates) | 30 MB |
| 13 Faster diffusion inference | ~4 min, no training | reuses project 12 |
| 14 Latent diffusion | ~4 min stage one, ~8 min stage two | 30 MB |
| 15 Vision Transformer | ~8 min per architecture | 170 MB |
| 16 Self-supervised learning | ~11 min (CIFAR-10, 10k subset); ~8 min on MNIST | 12-170 MB |
| 17 Masked image modelling | ~2 min pretrain, ~5 min for the fine-tune comparison | 12-170 MB |
| 18 Semantic segmentation | ~8 min | none (`shapes`), 810 MB (Oxford Pets) |
| 19 Object detection | ~8 min train, ~2 min evaluate | 12 MB (`digits`), 880 MB (VOC 2007) |
| 20 Image captioning | ~9 min | none (`shapes`), 1.1 GB (Flickr8k) |

Projects 18, 19 and 20 default to a small generated or MNIST-derived dataset so that a
first run costs minutes and no download. Each also supports the real dataset the task is
normally taught on - Oxford-IIIT Pets, Pascal VOC 2007, Flickr8k - and the project README
gives the command and says what it costs.

A GPU is not required anywhere. If you have one, raise `--epochs` and drop
`--train-subset` for numbers comparable with the literature.

## Checking the repository

```bash
python scripts/smoke_test.py       # metrics, then every project's pipeline
python scripts/metric_selftest.py  # just the metrics, ~4 seconds
```

Two things get checked, and the distinction matters.

**`scripts/metric_selftest.py`** checks that the metrics are *correct*. Nearly
every number in this repository comes from a metric implemented here rather than
imported - confusion matrices and F1, PSNR and SSIM, IoU and Dice, NMS and mean
average precision, BLEU, METEOR and CIDEr-D, FID, KID, generative
precision/recall, bits per dimension - so a mistake in one of them quietly
corrupts every result downstream. Each of the 96 checks feeds in an input whose
answer is known in advance and asserts the metric returns it: FID between two unit
Gaussians a distance `d` apart must be exactly `d^2`; a codebook used uniformly
must have perplexity exactly equal to its size; the majority-class baseline on a
90/10 mask must give pixel accuracy 0.90 and mean IoU exactly 0.45; AP with one of
two objects found must be exactly 51/101 under the COCO rule; RealNVP's inverse
must invert its forward pass. Nothing is downloaded and nothing is trained.

**`scripts/smoke_test.py`** checks that the *pipelines run*. It executes
`train.py`, `evaluate.py` and each project's study script on synthetic tensors,
verifying imports, shapes, checkpoint round-trips and figure writing for all
twenty projects, then cleans up after itself. It runs the metric self-test first,
since it otherwise only proves a number was produced, not that it was right.

## A note on FID

Projects 06-14 report FID, KID and generative precision/recall. The published FID number comes
from a specific ImageNet InceptionV3 checkpoint; downloading a 90 MB classifier to score 32x32
images would be the largest dependency here by an order of magnitude, so instead a small
classifier is trained on the dataset itself and cached. **Those values are not comparable with
published FID.** They are comparable within this repository - the same ruler for every arm of
every study - which is what the comparisons need. Every project's `metrics.json` records which
feature network was used, and `utils.py` explains it at the top.

The *implementation* is a separate question from the feature space, and it is checked:
`scripts/metric_selftest.py` pins FID against the analytic Gaussian cases where the answer is
known in closed form, confirms KID's estimator is unbiased where FID's is not, and reproduces
the mode-collapse signature (high precision, near-zero recall) on constructed inputs.

Project 11's bits-per-dimension is the exception to the comparability caveat: it is exact and
uses the standard convention, so it *is* comparable with published numbers.

## Roadmap

All twenty rows of the reference table are implemented.

## Requirements

Python 3.9+, PyTorch 2.0+, torchvision, numpy, matplotlib, tqdm, Pillow. Nothing else -
the metrics that would normally justify a heavier dependency are implemented in the
projects themselves, and are short enough to be worth reading once: confusion matrices,
F1, PSNR and SSIM; IoU and Dice; non-maximum suppression and mean average precision;
BLEU, METEOR and CIDEr-D. Writing them by hand is only defensible if they are also
checked, which is what `scripts/metric_selftest.py` is for - including the matrix square
root behind FID, which is where scipy would otherwise be the one dependency needed.
