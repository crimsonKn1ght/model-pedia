# 12 - Diffusion (DDPM: many small easy problems)

A GAN learns to jump from noise to an image in one step, and fights a discriminator to do
it. A diffusion model refuses the jump. It defines a process that destroys an image by adding
a little Gaussian noise at a time, and learns to undo **one step** of that process. Sampling
is then a walk back down the whole trajectory.

The result is a model with no adversarial game, no mode collapse, and a training objective as
simple as mean squared error - at the cost of needing hundreds of network evaluations per
sample. Project 13 attacks that cost.

```bash
pip install -r ../requirements.txt

python data.py --dataset fashion-mnist            # ~30 MB
python train.py --dataset fashion-mnist            # 20 epochs, FID every 4
python evaluate.py --checkpoint outputs/fashion-mnist_cosine_t200/best.pt
```

## The two processes

**Forward** (no learning): add noise repeatedly. Because each step is Gaussian, the
composition of `t` steps is Gaussian too, and there is a closed form - `q_sample` - that
jumps straight to any timestep. That is what makes training cheap: a step picks a random
`t`, jumps there in one shot, and asks the network to undo it.

**Reverse** (all the learning): one network, told which timestep it is looking at, predicts
the noise that was added. `python model.py` checks the closed form against the definition.

## Two choices worth naming

**Predict the noise, not the image.** Algebraically equivalent - either determines the other
- and they train very differently, because the noise target has the same scale at every
timestep while the clean image is nearly free at `t=1` and impossible at `t=T`.
`compare.py --study prediction` measures it. This is a good example of a reparameterisation
that changes nothing mathematically and everything practically.

**The schedule matters.** A linear beta schedule destroys most of the signal in the first
third of the trajectory, so much of the network's capacity goes to timesteps with nothing
left to learn from. `python model.py` prints the signal-to-noise ratio along both schedules;
cosine is the default for that reason.

## The loss is real and still not the metric

Unlike a GAN, this loss means something - mean squared error on predicted noise. But it
averages over timesteps, so a model that is excellent at `t=5` and useless at `t=180` can
post the same loss as one that is mediocre everywhere, and only the second makes good
samples. FID is computed every `--fid-every` epochs and selects the checkpoint.
`compare.py` prints validation loss beside FID precisely so you can see how little the loss
separates arms that FID separates clearly.

## What sampling costs

`evaluate.py` reports wall-clock milliseconds per image and network evaluations per image. A
GAN needs one forward pass; this needs `T`. That single number is what the entire
fast-sampling literature exists to attack, and project 13 measures the first and most useful
attack on it.

`trajectory.png` shows the same sample at several points on the way back from noise.
Structure appears early and detail late, which is the practical reason step-skipping works.

## Where diffusion beats the GAN

Compare `recall` here with project 08 on the same dataset. Diffusion usually wins by a wide
margin, because it has no discriminator to satisfy and therefore no incentive to abandon the
difficult parts of the distribution. FID and KID go through the same feature network in both
projects, so those comparisons are valid - **but not against published FID**, since the
feature network is a small classifier trained here rather than InceptionV3.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `prediction` | Predict noise vs predict `x0` | Equivalent on paper, not in practice |
| `schedule` | Cosine vs linear betas | Linear wastes timesteps at low `T` |
| `steps` | 100 / 200 / 400 timesteps | The cost/quality dial; sampling time scales linearly |
| `attention` | With and without the mid-block attention | Cheap at the lowest resolution, and usually worth it |

## Files

| File | What it does |
|---|---|
| `data.py` | The shared generative datasets, in `[0, 1]` |
| `model.py` | Time-conditioned U-Net, both schedules, both parameterisations, the samplers |
| `train.py` | Noise-prediction training with periodic FID |
| `evaluate.py` | FID/KID/precision/recall, the reverse trajectory, and the cost of sampling |
| `compare.py` | `prediction`, `schedule`, `steps` and `attention` studies |
| `utils.py` | FID, KID and generative precision/recall |

Projects 13 and 14 import this project's `model.py`, `data.py` and `utils.py` rather than
copying them.

## Knobs worth turning

- `--steps 50` for a fast, visibly worse model - and then run project 13 on the 200-step one
  to get the speed without the quality loss.
- `--prediction x0` to see a mathematically equivalent objective train badly.
- `--dataset cifar10` and compare FID with project 06's VAE on the same dataset. The gap is
  why diffusion displaced VAEs for image synthesis.
