# 13 - Faster diffusion inference (DDIM)

Project 12's DDPM makes good samples and takes 200 network evaluations to make one. A
GAN takes one. That gap is the single biggest practical objection to diffusion models,
and this project measures the first and most useful answer to it.

**Nothing is trained here.** DDIM is a different way to *sample* from a model that has
already been trained as a DDPM. This folder therefore reuses project 12's U-Net,
diffusion process, datasets and metric code rather than copying them - see `_paths.py` -
which is also what the reference table means when it lists this row's dataset as "same
trained DDPM". Every other project in the repository is self-contained; this one and
project 14 are deliberately not.

## Run it

```bash
pip install -r ../requirements.txt

cd ../12_diffusion && python train.py --dataset fashion-mnist   # if you have not already
cd ../13_faster_diffusion
python evaluate.py --checkpoint ../12_diffusion/outputs/fashion-mnist_cosine_t200/best.pt
```

## The two ideas

**The update can jump.** Given `x_t` and the network's noise prediction, you can
estimate the clean image directly, then re-noise that estimate to *any* earlier
timestep:

```
x_prev = sqrt(a_prev) * x0_hat + sqrt(1 - a_prev - sigma^2) * eps_hat + sigma * z
```

Nothing requires `prev` to be `t - 1`. Take 20 timesteps out of the trained 200 and the
same weights still work.

**The noise can be removed.** `sigma` above is free. DDPM corresponds to one particular
choice; `--eta 0` sets it to zero, and sampling becomes **deterministic** - one latent
maps to exactly one image. That is what makes `interpolation.png` meaningful: with a
stochastic sampler, two nearby latents give unrelated images, so there is nothing to
interpolate. `sampler.py` asserts both properties when run directly.

## What is measured

`evaluate.py` samples the same weights at several step counts and scores each run with
the FID, KID and precision/recall used everywhere else in this repository, plus
wall-clock milliseconds per image. The full-length DDPM ancestral sampler is included as
the reference row, so the table answers two questions:

- how much does skipping steps cost?
- does DDIM at full length differ from DDPM at full length? It does, because `eta=0`
  removes the sampling noise, and the two are not the same sampler.

The output is a curve rather than a number - `quality_vs_steps.png` plots FID against
network evaluations and against wall-clock - because the honest answer to "how fast can
you make it" is a trade-off, not a value.

`same_latent.png` is the figure to look at next: one latent decoded at 5, 10, 20 and 50
steps. With `eta=0` these are the same function evaluated at different accuracies, so
the images agree on layout and differ in polish. That is the mechanism, made visible.

## Which eta, at which budget

```bash
python compare.py --checkpoint ../12_diffusion/outputs/fashion-mnist_cosine_t200/best.pt
```

`eta` values do not rank the same way at every step count. Injected noise costs little
when there are many steps left to absorb it and hurts when there are few, so the best
`eta` depends on the budget - which is precisely why `compare.py` sweeps both axes
instead of fixing one. A comparison run at a single step count would report whichever
answer that budget happened to favour and present it as a general result.

## Files

| File | What it does |
|---|---|
| `_paths.py` | Puts project 12 on the import path, and explains why this project is not self-contained |
| `sampler.py` | The DDIM sampler; run it directly to see the determinism check |
| `train.py` | Delegates to project 12's training loop, so the folder has the standard entry points |
| `evaluate.py` | Sweeps step counts, scores each, prices the trade against full DDPM |
| `compare.py` | Sweeps `eta` against step budget |

There is no `model.py`, `data.py` or `utils.py` here by design - importing project 12's
copies is what keeps the two from drifting apart.

## Outputs

In `outputs/<dataset>_eta<eta>/`:

- `metrics.json` - every row of the sweep: FID, KID, precision, recall, ms/image
- `quality_vs_steps.png` - FID against evaluations, and against wall-clock
- `same_latent.png` - one latent at several step counts
- `samples.png`, and `interpolation.png` with `--interpolate`

## Knobs worth turning

- `--steps 2 5 10 20 50 100 200` to find where the curve turns. The interesting region is
  the knee, and it is lower than most people guess.
- `--eta 1.0` recovers noise injection. Compare its 5-step row with `eta=0`'s.
- `--interpolate` with `--eta 0`, then again with `--eta 1`. The second one produces
  eight unrelated images, which is the clearest possible demonstration of what
  determinism buys.
- Point `--checkpoint` at a project 12 model trained with `--schedule linear`. Fast
  samplers and noise schedules interact, and a schedule that wastes timesteps has more
  of them to skip.
