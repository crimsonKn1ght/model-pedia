# 17 - Masked image modelling (fill in the blanks)

Hide three quarters of an image and train a model to reconstruct the missing pixels.
No labels. Then throw the reconstruction machinery away and keep the encoder, which
is the only part that was ever the point.

This is BERT's trick applied to pixels, and the interesting thing is that it does not
transfer directly. Language is discrete and information-dense, so masking 15% of the
words is already hard. Images are continuous and enormously redundant - a masked
patch is usually guessable from its neighbours - so at a low mask ratio the model
learns interpolation and nothing else. MAE's answer is to mask **most** of the image,
which is what makes the task require some understanding of what is in it.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset cifar10                       # download (~170 MB)
python train.py --dataset cifar10 --epochs 8            # a few minutes on CPU
python evaluate.py --checkpoint outputs/cifar10_mae/best.pt
```

`evaluate.py` fine-tunes two classifiers, which is the slow part. Skip it with
`--tasks recon probe` while you are iterating.

## Three ideas, and where they are in the code

**1. Mask most of the image.** `--mask-ratio 0.75` by default. `compare.py --study
mask_ratio` shows what happens either side of it.

**2. The encoder never sees the mask.** Dropped patches are *removed from the
sequence*, not replaced by a mask token (`ViTEncoder.random_masking` in `model.py`).
Two consequences: the encoder runs on a quarter of the tokens, so pretraining is
cheap; and it is never asked to process a mask token, which will not exist at
fine-tuning time.

**3. The decoder is small and disposable.** It re-inserts one shared mask token at the
right positions - only the position embedding tells it *which* patch it is being asked
about - and predicts raw pixels. Here it is 111k parameters against the encoder's
800k, and after pretraining it is deleted.

## Reconstruction is the objective, not the result

The project reports three things, in increasing order of how much they tell you:

| Measurement | What it says | What it does not say |
|---|---|---|
| Masked-patch PSNR | The decoder can fill in plausible pixels | Nothing about content; local smoothness scores well |
| Linear probe | Whether the features are linearly separable | Almost nothing about MAE - measured below, it comes out *negative* against the random-init control |
| Fine-tuning | What the pretrained weights are actually worth | - |

That ordering is the main lesson of the project. A model can reconstruct nicely and
probe badly, and MAE deliberately trades the second for the third: its features are
not arranged for a linear boundary, they are arranged to be a good starting point.
Compare with project 16, where a contrastive objective optimises almost directly for
linear separability and probes much better at equal cost.

Every classification number is paired with **the identical ViT trained from random
initialisation under the identical budget**. Without that control, "the fine-tuned MAE
reaches X%" is not a claim about pretraining.

## Measured results

MNIST, 10 000 training images at 28x28 with 4x4 patches (49 patches, 12 of them visible
at 75% masking). 8 pretraining epochs on four CPU cores took 101 s; the downstream
comparison is 3 fine-tuning epochs per arm.

```bash
python train.py --dataset mnist --epochs 8
python evaluate.py --checkpoint outputs/mnist_mae/best.pt
```

Reconstruction on the 10 000-image test split, swept over test-time mask ratio:

| Mask ratio | Objective | Masked-patch PSNR |
|---|---|---|
| 0.25 | 0.3827 | 14.20 dB |
| 0.50 | 0.3444 | 14.32 dB |
| 0.75 (trained at) | 0.3321 | 14.30 dB |
| 0.90 | 0.3338 | 14.21 dB |

Classification, each read-out run twice - once from the pretrained encoder, once from
the identical ViT at random initialisation, same schedule, same seed:

| Read-out | Pretrained | Random init | Gain |
|---|---|---|---|
| Linear probe | 0.5748 | 0.6417 | **-0.0669** |
| Fine-tune | 0.8522 | 0.4850 | **+0.3672** |

That pair of rows is the most useful result in this project, and the negative one is not
a bug. MAE features are genuinely poor under a linear probe - here *worse than random
initialisation* - because nothing in the objective asks them to be linearly separable;
predicting pixels rewards keeping local detail, not arranging classes into half-spaces.
Fine-tune the same weights and they are worth +0.37 accuracy over training from scratch
under an identical budget.

The practical lesson is that a linear probe is the wrong instrument for this family of
methods. Had this project reported only the probe - the default choice, and the right
one for the contrastive methods in project 16 - it would have concluded that MAE
pretraining is worthless, from a correctly computed number.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `mask_ratio` | 0.25 / 0.5 / 0.75 / 0.9 | Reconstruction degrades as it rises; representation quality does not, until very late |
| `decoder` | Decoder depth 1 / 2 / 4 | Changes reconstruction visibly, downstream accuracy barely - the decoder is disposable |
| `target` | Raw pixels vs per-patch normalised | Normalising removes the easy part of the problem (a patch's mean brightness) |

## A note on the reconstruction figures

With per-patch normalisation on (the default, and what the paper does), the decoder
predicts *normalised* pixels, so its output is not directly viewable. `reconstruct()`
puts each patch's original mean and standard deviation back to make a picture. Those
are ground-truth statistics the model never saw, so read the figures as a check on
structure, not on absolute colour. The reference implementation's demo does the same
thing; `--raw-pixel-loss` avoids the issue at a small cost in feature quality.

## Files

| File | What it does |
|---|---|
| `data.py` | Datasets in `[0, 1]`, weak pretraining augmentation, `--dataset folder` for Tiny ImageNet or your own |
| `model.py` | ViT encoder, MAE decoder, and the classifier that reuses the encoder |
| `train.py` | Pretraining with fixed-seed validation masks, masked-patch PSNR |
| `evaluate.py` | Mask-ratio sweep, linear probe and fine-tune, each against a random-init control |
| `compare.py` | `mask_ratio`, `decoder` and `target` studies |
| `utils.py` | Masked PSNR, cached-feature linear probe, plotting |

## Outputs

In `outputs/<dataset>_mae/`:

- `best.pt`, `history.json`, `curves.png`, `reconstruction_val.png`
- `metrics.json` - the mask-ratio sweep and both classification read-outs
- `mask_ratio_sweep.png`, `examples.png` - original / masked / reconstructed
- `downstream_accuracy.png` - pretrained vs from scratch

## Knobs worth turning

- `--mask-ratio 0.9` - the pictures become guesses and the features stay useful.
- `--dataset folder --data-root path/to/tiny-imagenet-200` runs any `ImageFolder`
  layout (`train/<class>/*`, `val/<class>/*`). Tiny ImageNet ships its validation set
  flat with a separate annotation file, so it needs rearranging into per-class folders
  first.
- `--decoder-depth 1 --decoder-dim 32` - how little decoder can you get away with.
- `--epochs 0` checkpoints the untrained model, so `evaluate.py` can report the
  control on its own.
