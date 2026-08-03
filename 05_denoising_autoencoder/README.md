# 05 - Denoising autoencoder (a learned image prior)

Take a clean image, add Gaussian noise, ask the network for the clean one back.
The target is never the input, so the model cannot cheat by copying - it has to
know something about what images look like. That knowledge is an **image
prior**, and it is learned entirely from the data.

The point is easiest to see in the architecture ablation. A bottleneck
autoencoder is the wrong shape for this job: squeezing the image through a
narrow code destroys the very detail you are trying to recover, and the output
comes back clean *and* blurred. A U-Net keeps the encoder-decoder shape but
adds skip connections, so full-resolution detail bypasses the bottleneck while
the coarse levels decide what is signal and what is noise.

```bash
python compare.py --study skips     # identical net, skip connections on/off
```

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset cifar10                # download (~170 MB)
python train.py --epochs 6                      # about five minutes on CPU
python evaluate.py --checkpoint outputs/cifar10_unet/best.pt
```

## Two things this project insists on

**1. Always report the do-nothing baseline.** A denoiser that returns its input
unchanged still scores a respectable PSNR. So every table here prints the
noisy input's PSNR/SSIM next to the model's, and the gain between them. A
number like "31 dB" is meaningless until you know the input was 24 dB.

**2. Train blind.** The noise level is drawn fresh from `--sigma-range` for
every batch, so the model is never told how noisy its input is - which is the
only situation you ever face in practice. Noise is added in the training loop
rather than baked into the dataset, so each epoch sees a new corruption of the
same image and the network cannot memorise one.

`evaluate.py` then sweeps sigma **past** the trained range and marks the
out-of-range rows with `*`. Watching the gain collapse there is the most
useful single output of the project: it is a clean, quantitative picture of a
model failing outside its training distribution.

## Running a classical benchmark set

BSD68, Set12 and Kodak do not ship with torchvision, so point the evaluator at
a folder instead:

```bash
python evaluate.py --checkpoint outputs/cifar10_unet/best.pt --test-dir path/to/BSD68
```

The U-Net is fully convolutional, so a model trained on 32x32 CIFAR patches
runs unchanged on 481x321 photographs. Images are cropped to a multiple of 4
and processed one at a time. Expect the numbers to drop: CIFAR is a poor proxy
for natural-image statistics, and seeing exactly how much it costs is a
worthwhile experiment in itself.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `skips` | U-Net vs the same net with skips removed | The largest gap of the three; without skips the output is visibly softer |
| `target` | Predicting the clean image vs predicting the noise | Residual prediction is an easier target - the noise is near zero almost everywhere |
| `loss` | L1 vs L2 | L2 optimises PSNR directly; L1 is less willing to hedge with a blur and usually looks better |

## Files

| File | What it does |
|---|---|
| `data.py` | Datasets, the Gaussian noise model, and `ImageDirectory` for benchmark folders |
| `model.py` | Three-level U-Net; `skips=False` builds the control, `predict_residual` switches the target |
| `train.py` | Blind training with per-batch noise levels, fixed-seed validation noise |
| `evaluate.py` | Sweeps noise levels against the do-nothing baseline, writes examples and curves |
| `compare.py` | Runs one ablation and prints a summary table |
| `utils.py` | Seeding, PSNR and SSIM from scratch, image-grid plotting |

## Outputs

In `outputs/<dataset>_<model>/`:

- `best.pt`, `history.json`, `curves.png`, `denoising_val.png`
- `metrics.json` - per-sigma PSNR/SSIM for both the noisy input and the output
- `sigma_sweep.png` - the two curves, with the trained noise range shaded
- `examples.png` - clean / noisy / denoised rows at several noise levels

## Typical numbers

Indicative CIFAR-10 test results after 6 epochs on a 10 000-image subset,
trained blind over sigma in [0.05, 0.25].

| Sigma | Noisy PSNR | Denoised PSNR | Gain |
|---|---|---|---|
| 0.05 | ~26.3 dB | ~31 dB | +5 dB |
| 0.15 | ~17.2 dB | ~25 dB | +8 dB |
| 0.25 | ~13.4 dB | ~22 dB | +9 dB |
| 0.35 * | ~11.2 dB | ~19 dB | +8 dB |
| 0.50 * | ~9.2 dB | ~15 dB | +6 dB |

`*` outside the trained range. The gain peaks inside the training range and
falls away outside it - more noise is a harder problem, and a problem the model
was never shown.

## Knobs worth turning

- `--sigma-range 0.15 0.15` - train at a single level, then evaluate across the
  sweep. It wins slightly at 0.15 and falls apart everywhere else, which is the
  argument for blind training in one experiment.
- `--base-channels 16` - halves the cost; shows how much of the gain is capacity.
- `--dataset fashion-mnist` - much easier, because the images are nearly binary
  and the prior is close to trivial.
