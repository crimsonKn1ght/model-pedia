# 05 - Pix2Pix

Paired image-to-image translation with a U-Net generator and a PatchGAN
discriminator.

## The idea

Given matched pairs `(source, target)`, learn the mapping between them. Two
design choices carry the method.

**The U-Net generator.** Translation preserves structure -- an edge in the input
belongs in the same place in the output. Skip connections wire every encoder
stage directly to the matching decoder stage so spatial detail travels sideways
instead of being squeezed through the bottleneck and rebuilt. Remove the skips
and the outputs go mushy immediately.

**The PatchGAN discriminator.** Instead of one verdict per image, it outputs a
grid of verdicts, each covering a local patch. This aims the adversarial loss at
*texture* -- exactly what an L1 loss cannot model, because L1's optimal answer
under uncertainty is to blur. The full objective splits the labour:

```
L = L_cGAN(G, D)  +  lambda_L1 * || target - G(source) ||_1
```

with `lambda_L1 = 100`. L1 gets the low frequencies right; the adversarial term
sharpens. The discriminator always sees the **pair**, never the output alone,
which is what makes it judge correspondence rather than mere realism.

## The datasets

The default task is **edges -> photo**, derived on the fly: the target is a real
photo, the input is its Sobel edge map. The pairing is exact, it needs no extra
download, and it is the same setup as the paper's `edges2shoes`.

`--task facades` uses the original pix2pix facades dataset, fetched by
`python download_data.py --facades`. That needs outbound access to
`efrosgans.eecs.berkeley.edu`; if the host is unreachable the edges task
demonstrates identical mechanics.

## Run it

```bash
python download_data.py                  # CIFAR-10, for the edges task
python train.py
python evaluate.py
```

Useful variations:

```bash
python train.py --dataset fashion-mnist --epochs 5
python train.py --dataset celeba64 --image-size 64
python download_data.py --facades && python train.py --task facades --image-size 64
python train.py --lambda-l1 0 --out-dir outputs/no-l1     # the key ablation
```

## Files

| File | What it holds |
|---|---|
| `model.py` | `UNetGenerator`, `PatchDiscriminator`, `DownBlock`/`UpBlock` |
| `data_pairs.py` | builds the paired loaders for either task |
| `download_data.py` | dataset fetcher, optional facades download |
| `train.py` | adversarial loop with the L1 term |
| `evaluate.py` | paired fidelity metrics plus FID/KID |

## Test-set evaluation

Paired data means the ground truth is known, so fidelity can be measured
directly rather than inferred. `evaluate.py` writes `outputs/evaluation/`:

* `metrics.json` -- **L1, PSNR, SSIM** against the true target, an LPIPS-style
  perceptual distance, and FID/KID against the target distribution;
* `translations.png` -- input, generated and target in three rows.

Both families matter. A model can post a good FID by producing realistic images
that are the *wrong* translation of their input; only the per-pair metrics catch
that. Conversely a model can win on L1 by outputting a blur that no per-pair
metric penalises enough -- FID catches that one.

## Reference run

Edges-to-photo derived from Fashion-MNIST, 20000 training pairs, 4 epochs,
`lambda_l1=100`, CPU only (about 8 minutes end to end):

| Metric | Value |
|---|---|
| test L1 | 0.109 |
| test PSNR | 20.1 dB |
| test SSIM | 0.765 |
| FID | 0.53 |
| precision / recall | 0.97 / 0.90 |

This is the strongest set of numbers in the repository, and the reason is
structural rather than flattering: paired translation is by far the easiest task
here. The model is handed an edge map that already fixes the shape and position
of the output, so it only has to fill in interior intensity. Compare against the
unconditional DCGAN, which must invent the whole image from noise and reaches
FID 2.94.

The lesson is about reading FID in context. A low number here means much less
than the same number would for unconditional generation, which is exactly why
the per-pair metrics are reported alongside: PSNR of 20.1 dB and SSIM of 0.765
say the outputs are close to the true target, not merely plausible.

## What to look at

* Run the `--lambda-l1 0` ablation. The adversarial losses keep improving while
  the outputs become confident, plausible texture that has stopped tracking the
  input. PSNR and SSIM collapse; FID may barely move.
* The dropout in the first two decoder blocks stays active at test time. It is
  Pix2Pix's only source of output diversity -- the model is otherwise
  deterministic, which is a real limitation of the method, not a bug here.
* At `image_size 32` the PatchGAN output is a 6x6 grid; at 64 it is 14x14. The
  receptive field of each patch verdict is what sets the scale of texture the
  adversarial loss can police.

## A note on FID and KID in this repository

Computed in a small per-dataset feature space, not ImageNet InceptionV3, and
`test_lpips_proxy` is a cosine distance in that same space rather than real
LPIPS. See `../01-vae/README.md`.
