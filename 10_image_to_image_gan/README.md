# 10 - Image-to-image GAN (Pix2Pix: two losses that disagree)

Paired translation has an obvious loss - L1 against the target - and using only that loss
produces blurry output. The reason is the same one that blurs a VAE: when several outputs are
plausible, the L1-optimal answer is a compromise between them, and a compromise between two
sharp images is a soft one.

Pix2Pix adds a discriminator, so the generator is scored not only on being close to the target
but on being *the kind of image a target could be*. The combination is the method, and this
project is arranged around measuring it.

```bash
pip install -r ../requirements.txt

python data.py --dataset shapes --preview     # generated, no download
python train.py --dataset shapes               # 20 epochs
python evaluate.py --checkpoint outputs/shapes_l1100/best.pt
python compare.py --study l1_weight
```

`shapes` is generated: an outline drawing in, the filled coloured version out. It is the
default because the targets are *exact*, so L1 and SSIM mean what they say and the failure
mode - right shape, wrong colour - is legible in the figure. The outline carries shape and
position but no colour, so the model has to invent the colour, which is precisely where the
two losses disagree.

`--dataset facades` is the dataset Pix2Pix was published on: architectural label maps paired
with photographs. It is also a better illustration of the underlying problem, because a label
map does not determine a photograph - there are many right answers, and L1 punishes all but
one of them.

## The ablation the method is built on

```bash
python compare.py --study l1_weight    # 0 / 10 / 100 / 1000
```

| Arm | Result |
|---|---|
| GAN only (`--l1-weight 0`) | Sharp, and free to produce something that does not match the input |
| L1-dominated (`1000`) | Correct on average, soft |
| Both (`100`) | What the paper ships |

`compare.py` prints the best-PSNR and best-FID arms at the bottom of the table. When those
disagree - and they do - neither metric alone is enough to pick a model. Reading only PSNR
picks the blurriest entry; reading only FID picks one that ignores its input.

## Two architectural choices

**A U-Net generator, not an encoder-decoder.** Input and output are aligned pixel by pixel, so
most of what the decoder needs is already sitting at the matching encoder resolution. The skips
carry it across. `compare.py --study skips` builds the control.

**A PatchGAN discriminator.** Instead of one verdict per image it outputs a grid of verdicts,
each covering a patch, which makes it a texture critic - enough receptive field to judge local
realism, not enough to police global layout, which is the input's job to determine.
`--patch-layers` sets that receptive field; `python model.py` prints the verdict grid and
parameter count for each setting, and a 1-layer PatchGAN has 7 thousand parameters against a
5-layer one's 7 million.

The discriminator sees the input **and** the output, concatenated. Without that it could only
judge whether an image looks real, not whether it answers the question asked.

## What is measured, and why both kinds

| Family | Metrics | Bias |
|---|---|---|
| Pixel | L1, PSNR, SSIM | Rewards blur - the pixel-optimal answer is an average |
| Distribution | FID, KID, precision/recall | Notices the blur PSNR forgives |

LPIPS would be the third metric and is deliberately absent: it needs pretrained ImageNet
weights, a heavier dependency than anything else in this repository. SSIM plus FID covers the
same ground with code you can read.

One caveat specific to this project: the targets carry no class labels, so the FID feature
network falls back to fixed-seed **random** features rather than a trained classifier. That is
recorded as `feature_net` in `metrics.json`, and it makes these FID values comparable within
this project only.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `l1_weight` | 0 / 10 / 100 / 1000 | The central trade; PSNR and FID rank the arms differently |
| `discriminator` | PatchGAN 1 / 3 vs whole image | A small receptive field is enough, and much cheaper |
| `skips` | U-Net vs no skips | Aligned tasks lean on the skips heavily |

## Files

| File | What it does |
|---|---|
| `data.py` | The generated `shapes` pairs, the facades download, and the joint transform |
| `model.py` | U-Net generator (skips optional), PatchGAN discriminator, both loss terms |
| `train.py` | Adversarial plus L1, selecting on validation SSIM |
| `evaluate.py` | Pixel metrics and distribution metrics side by side, with figures |
| `compare.py` | `l1_weight`, `discriminator` and `skips` studies |
| `utils.py` | PSNR, SSIM, FID, KID and precision/recall |

## Knobs worth turning

- `--l1-weight 0` and look at `examples.png`. The output will be sharp and often the wrong
  colour, which is the clearest possible picture of what the L1 term is for.
- `--no-skips` on `shapes`. The task is pixel-aligned, so this hurts more than it did in
  project 05.
- `--patch-layers 1`. A 7-thousand-parameter discriminator is often enough, which says
  something about how little of the image the adversarial signal needs to see.
- `--dataset facades --image-size 128` for the real thing, and note that L1 gets *worse* as
  the outputs get more plausible - there are many right answers and it can only reward one.
