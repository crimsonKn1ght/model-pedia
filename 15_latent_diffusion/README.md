# 15 - Latent Diffusion

Run the diffusion process in a compressed space instead of on pixels.

## The idea

Pixel-space diffusion spends most of its capacity on detail a cheap autoencoder
could have reproduced, and every one of its hundreds of sampling steps pays the
full resolution cost. Latent diffusion (Rombach et al., 2022 -- the basis of
Stable Diffusion) splits the problem:

**Stage 1, perceptual compression.** An autoencoder maps a `C x H x W` image to
a `d x H/f x W/f` latent and back. Trained once on reconstruction, then
**frozen**.

**Stage 2, semantic generation.** A DDPM is trained *in that latent space*. At
`f = 4` the U-Net operates on 1/16th of the spatial positions, so training and
sampling both get dramatically cheaper.

Generation is: sample a latent with diffusion, decode it once. The expensive
iterative part happens at low resolution; the single expensive upsampling
happens once at the end.

The division of labour is the point. Stage 1 handles high-frequency detail --
the part that is perceptually necessary but semantically boring. Stage 2 handles
composition and content. Diffusion is a poor use of compute for the former and
an excellent one for the latter.

## Two details that decide whether it works

**The KL weight is deliberately tiny** (`1e-6`). Stage 1 is *not* a VAE you
sample from -- it is a codec. A VAE-strength KL term would blur exactly the
detail stage 2 depends on. But some regularisation is needed, or the latent
scale drifts arbitrarily.

**The latent scaler.** Diffusion's noise schedule is defined against a
unit-variance signal, and stage 1 makes no such promise. A single scalar is
estimated from the training set and applied in both directions. Stable Diffusion
does exactly this with its `0.18215` constant; get it wrong and stage 2 simply
does not converge.

## Run it

```bash
python data.py
python train.py                       # Fashion-MNIST: stage 1, then stage 2
python evaluate.py
```

Useful variations:

```bash
python train.py --dataset celeba64 --image-size 64
python train.py --downsample-factor 2         # milder compression, better floor
python train.py --downsample-factor 8         # aggressive; watch the floor rise
python train.py --latent-channels 8           # a wider latent
```

## Files

| File | What it holds |
|---|---|
| `model.py` | `AutoencoderKL` (stage 1), `LatentScaler` |
| `data.py` | dataset fetcher |
| `train.py` | stage 1 then stage 2; imports the U-Net from project 13 |
| `evaluate.py` | sample quality, the stage-1 floor, latent-vs-pixel cost |

The diffusion U-Net is imported from `../13_ddpm/model.py` unchanged. The
architecture does not care whether its input is pixels or latents -- which is
precisely the observation latent diffusion is built on.

## Test-set evaluation

`outputs/evaluation/metrics.json`:

* FID/KID of decoded samples, and `seconds_per_image`;
* **`fid_autoencoder_floor`** -- the FID of stage-1 *reconstructions*. This is a
  floor latent diffusion can never beat, because every sample it produces has
  passed through the decoder. The gap between it and the sample FID tells you
  which stage is the bottleneck: a small gap means stage 1 is the limit and more
  diffusion training will not help;
* **`step_speedup_vs_pixel`** -- both U-Nets are timed on this machine, so the
  cost claim is measured rather than asserted;
* `compression_ratio`, and the wall-clock minutes each stage took.

Figures: `samples.png`, `stage1-reconstructions.png`, `cost-per-step.png`.

## Reference run

Fashion-MNIST, 20000 training images, `f=4` compression, 2 autoencoder epochs
then 1500 diffusion steps, CPU only:

| Metric | Value |
|---|---|
| stage 1 training | 3.3 min |
| stage 2 training | 3.8 min |
| autoencoder MSE | 0.0098 |
| **FID of samples** | **37.8** |
| **FID of the stage-1 floor** | **1.11** |
| precision / recall | 0.81 / 0.06 |
| latent vs pixel elements | 256 vs 1024 |
| **cost per denoising step** | **7.2x cheaper** |

**The efficiency claim holds.** One denoising step costs 0.6 ms in latent space
against 4.6 ms in pixel space, a 7.2x saving from a 4x reduction in values. And
the comparison that matters: pixel-space DDPM in project 13 reached FID 46.6
after 13.3 minutes of training, while this reaches 37.8 in 7.1 minutes across
both stages. Better samples, roughly half the compute. That is the entire
argument for latent diffusion, reproduced at laptop scale.

Note the measured speedup is below the naive 4x-implies-4x arithmetic in one
direction and above it in another -- attention and fixed per-call overhead do
not scale with resolution, so the relationship is never the simple ratio.

**The floor tells you where the remaining budget should go.** Stage 1
reconstructs at FID 1.11, so the autoencoder is very nearly lossless in the
metric space. Samples score 37.8. The entire gap is stage 2, meaning more
diffusion steps -- not a better autoencoder -- is what would improve this.

**The honest caveat is recall at 0.06.** Precision is high, so individual
samples look right, but diversity is poor after only 1500 diffusion steps. This
is the same undertraining that limits project 13, and it shows up here in the
metric built to detect it rather than in FID.

## What to look at

* **The floor is the whole story at small scale.** Sweep `--downsample-factor`
  from 2 to 8 and watch `fid_autoencoder_floor` climb as compression gets more
  aggressive while `step_speedup_vs_pixel` improves. That trade-off is the
  central design decision of the method, and it is fully visible in three runs.
* Look at `stage1-reconstructions.png` before judging the samples. If the
  autoencoder cannot reconstruct the data, nothing downstream can fix it.
* The measured step speedup is smaller than the raw element-count ratio, because
  attention and the fixed per-call overhead do not scale with resolution. Real
  speedups are always below the naive arithmetic.

## A note on FID and KID in this repository

Computed in a small per-dataset feature space, not ImageNet InceptionV3. See
`../06_vae/README.md`.
