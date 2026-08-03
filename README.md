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
| 16 | [Self-supervised learning](16_self_supervised/) | SimCLR, BYOL | CIFAR-10, STL-10, MNIST | Labels are not the only supervision - and the augmentation *is* the supervision |
| 17 | [Masked image modelling](17_masked_image_modeling/) | MAE with a ViT encoder | CIFAR-10, any `ImageFolder` | Hide 75% of the image; reconstruction is the objective, the encoder is the result |
| 18 | [Semantic segmentation](18_semantic_segmentation/) | U-Net, no-skip and dilated controls | Oxford-IIIT Pets, generated shapes | A label per pixel, and why mean IoU and pixel accuracy disagree |
| 19 | [Object detection](19_object_detection/) | Anchor-free single-scale detector | MNIST-derived scenes, Pascal VOC 2007 | Variable-length output: assignment, NMS and mAP written out |
| 20 | [Image captioning](20_image_captioning/) | CNN / frozen-ImageNet encoder + Transformer decoder | Flickr8k, generated shapes | A sequence out, and three metrics that disagree about it |

The numbering follows the reference table this repository works through, so the gap
between 05 and 16 is rows still to come rather than anything missing.

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
python scripts/smoke_test.py
```

Runs `train.py`, `evaluate.py` and each project's study script on synthetic
data, with no downloads and no pretrained weights. It verifies imports, shapes,
checkpoint round-trips and figure writing for all ten projects in about three
minutes, and cleans up after itself.

## Roadmap

The reference table runs to twenty rows; rows 01-05 and 16-20 are done. Still to come,
roughly in order: variational autoencoders, VQ-VAE, GANs (DCGAN, conditional,
image-to-image), normalizing flows, diffusion (DDPM, DDIM, latent) and vision
transformers as a supervised classifier in its own right.

## Requirements

Python 3.9+, PyTorch 2.0+, torchvision, numpy, matplotlib, tqdm, Pillow. Nothing else -
the metrics that would normally justify a heavier dependency are implemented in the
projects themselves, and are short enough to be worth reading once: confusion matrices,
F1, PSNR and SSIM; IoU and Dice; non-maximum suppression and mean average precision;
BLEU, METEOR and CIDEr-D.
