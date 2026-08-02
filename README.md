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

A GPU is not required anywhere. If you have one, raise `--epochs` and drop
`--train-subset` for numbers comparable with the literature.

## Checking the repository

```bash
python scripts/smoke_test.py
```

Runs `train.py`, `evaluate.py` and each project's study script on synthetic
data, with no downloads. It verifies imports, shapes, checkpoint round-trips
and figure writing in about two minutes, and cleans up after itself.

## Roadmap

The projects above are the first five rows of a longer reference table. Still to
come, roughly in order: variational autoencoders, VQ-VAE, GANs (DCGAN,
conditional, image-to-image), normalizing flows, diffusion (DDPM, DDIM, latent),
vision transformers, self-supervised learning, masked image modelling,
segmentation, detection and captioning.

## Requirements

Python 3.9+, PyTorch 2.0+, torchvision, numpy, matplotlib, tqdm. Nothing else -
the metrics that would normally justify a heavier dependency (confusion
matrices, F1, PSNR, SSIM) are implemented in the projects themselves, and are
short enough to be worth reading once.
