# 13 - DDPM (denoising diffusion)

Image generation by learning to reverse a fixed noising process.

## The idea

Diffusion splits generation into two processes.

**Forward -- fixed, nothing learned.** Gaussian noise is added over `T` steps
until only noise remains. The algebraic fact that makes this practical is that
any step is reachable in closed form:

```
x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * eps
```

so training never simulates the chain. Pick a random `t`, jump straight there.

**Reverse -- learned.** A U-Net `eps_theta(x_t, t)` predicts the noise that was
added. Sampling starts from pure noise and walks back down the chain.

The training objective is the mean squared error between true and predicted
noise:

```
L = E_{x_0, t, eps} || eps - eps_theta(x_t, t) ||^2
```

That is all of it. No adversarial game to balance, no posterior to approximate,
no likelihood bound to wrestle with -- which is a large part of why diffusion
displaced GANs. The cost is paid at sampling time: `T` sequential network
evaluations per image. Project 14 (DDIM) attacks exactly that.

The U-Net is conditioned on the timestep through a sinusoidal embedding added
inside every residual block, because one network must handle every noise level
and needs to know which one it is looking at.

## Run it

```bash
python data.py
python train.py                    # MNIST, ~10 min on 4 CPU cores
python evaluate.py
```

Useful variations:

```bash
python train.py --dataset fashion-mnist --max-steps 4000
python train.py --dataset cifar10 --max-steps 8000 --base-channels 64
python train.py --schedule linear        # the original DDPM schedule
python train.py --timesteps 1000         # more steps, slower sampling
```

Training is budgeted in **optimizer steps**, not epochs. Sample quality tracks
step count far more closely than epoch count, and a step budget makes the
runtime predictable on any machine.

## Two things that matter more than they look

**The cosine schedule** (default) destroys information more gradually than the
original linear one. At 32x32 the linear schedule turns images to noise too
early, wasting a large fraction of the timesteps on inputs that carry no signal.

**The EMA weights.** An exponential moving average of the parameters is
maintained alongside the live model, and that is what gets sampled. For
diffusion this is not a nicety -- EMA weights produce visibly cleaner samples at
a given step count, which matters enormously on a small budget. Compare with
`python evaluate.py --weights raw`.

## Files

| File | What it holds |
|---|---|
| `model.py` | `UNet`, `SinusoidalTimeEmbedding`, `GaussianDiffusion`, the schedules |
| `data.py` | dataset fetcher |
| `train.py` | the noise-prediction loop with EMA |
| `evaluate.py` | FID/KID, trajectory figures, sampling throughput |

Projects 14 and 15 import `UNet` and `GaussianDiffusion` from this file directly
rather than copying them, so the three diffusion projects can never drift apart.

## Test-set evaluation

`outputs/evaluation/`:

* `metrics.json` -- FID/KID, plus `seconds_per_image` and
  `network_evaluations_per_image`. Those last two are the baseline that project
  09 improves on, so they are recorded here deliberately.
* `forward-process.png` -- real images being destroyed by the schedule. This is
  the process the model learns to invert, and it costs nothing to compute.
* `denoising-trajectory.png` -- the reverse process, from pure noise on the left
  to a finished image on the right. The single clearest picture of what
  diffusion does.
* `samples.png`.

## Reference run

MNIST, 2500 optimizer steps, `base_channels=32`, 300 timesteps, cosine
schedule, 0.84M parameters, CPU only (4 cores, 13.3 minutes to train):

| Metric | Value |
|---|---|
| final noise MSE | 0.0324 |
| FID (see note) | 46.6 |
| KID | 3.67 +/- 0.20 |
| precision / recall | 0.27 / 0.29 |
| network evaluations per image | 300 |

**This is the weakest generative result in the repository, and honestly so.**
The DCGAN in project 08 reaches FID 2.94 on the same dataset in a comparable
wall-clock budget. Diffusion is by far the most compute-hungry model here, and
2500 steps on four CPU cores does not do it justice -- the samples are
digit-shaped but rough, and precision/recall near 0.28 says both quality and
coverage are limited.

That is a statement about the budget, not the method. Diffusion overtakes GANs
at scale, which is why it won; it simply does not get there in thirteen minutes
on a laptop CPU. `--max-steps 20000` on a GPU is a different picture. The
architecture, schedule and objective here are the real ones, so scaling the step
count is the only change needed.

The `noise MSE` figure is the honest progress signal: the target is unit
Gaussian noise, so a network predicting zero scores exactly 1.0. Reaching 0.032
means the model explains most of the noise, and it was still improving when the
budget ran out.

## What to look at

* The denoising trajectory typically shows coarse layout settling early and
  detail arriving late. That ordering is not accidental: high noise levels
  destroy high frequencies first, so the reverse process necessarily recovers
  low frequencies first.
* `noise MSE` hovering near 1.0 at the start is correct, not a bug -- the target
  is unit Gaussian noise, so an untrained network predicting zero scores exactly
  1.0. Progress below 1.0 is real progress.
* Sampling is the slow part. 300 timesteps means 300 forward passes per image.
  Note `seconds_per_image` here, then run project 14.

## A note on FID and KID in this repository

Computed in a small per-dataset feature space, not ImageNet InceptionV3. See
`../06_vae/README.md`.
