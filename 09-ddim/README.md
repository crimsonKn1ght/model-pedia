# 09 - DDIM (faster diffusion sampling)

The same trained DDPM, sampled in a fraction of the steps.

## The idea

A DDPM needs one network evaluation per timestep -- 300 here, 1000 in the
original paper. That is the method's main practical drawback.

DDIM (Song et al., 2020) starts from an observation about the DDPM training
objective: it only ever depends on the **marginals** `q(x_t | x_0)`, never on the
particular Markov chain connecting them. So the sampling chain can be swapped
for a different one with the same marginals -- including a non-Markovian,
**deterministic** one, defined over an arbitrary subsequence of timesteps.

The update:

```
x_{t-1} = sqrt(a_{t-1}) * x0_hat
        + sqrt(1 - a_{t-1} - sigma^2) * eps_hat
        + sigma * z

sigma_t = eta * sqrt((1 - a_{t-1}) / (1 - a_t)) * sqrt(1 - a_t / a_{t-1})
```

Two consequences:

* **`eta = 0` removes the noise term.** Sampling becomes deterministic, so one
  latent maps to exactly one image -- and latents can be interpolated the way GAN
  latents are, which a stochastic DDPM cannot do.
* **Steps can be skipped.** Because the marginals hold at any spacing, 20-50
  steps often match what the DDPM needed hundreds for.

**No retraining is involved.** `eta = 1` recovers DDPM ancestral sampling on the
chosen subsequence, so the two methods are endpoints of one family.

## Run it

```bash
python download_data.py
python train.py            # reuses project 08's checkpoint; trains one only if absent
python evaluate.py
```

`train.py` deliberately does not train anything when `../08-ddpm/outputs/ddpm.pt`
exists -- DDIM adds no parameters and no objective, and the comparison in
`evaluate.py` is only meaningful against the *same* trained network. Use
`--force-train` to train a fresh one here anyway.

Useful variations:

```bash
python evaluate.py --steps 5 10 20 50 100 200
python evaluate.py --eta 1.0        # stochastic; should approach DDPM quality
python evaluate.py --eta 0.5        # halfway between the two
```

## Files

| File | What it holds |
|---|---|
| `model.py` | `DDIMSampler`, including the timestep subsequence and slerp interpolation |
| `ddpm_bridge.py` | imports the U-Net and diffusion process from project 08 |
| `download_data.py` | dataset fetcher (for the reference metrics) |
| `train.py` | reuses or, if necessary, trains the DDPM |
| `evaluate.py` | the quality-versus-steps sweep |

## Test-set evaluation

This is the experiment the project exists for. The same network is sampled with
full DDPM ancestral sampling and with DDIM at several step counts. For each
setting `evaluate.py` records FID, KID, seconds per image and network
evaluations.

`outputs/evaluation/`:

* `metrics.json` -- one entry per setting, plus `max_speedup`;
* `fid-vs-steps.png` -- the headline curve;
* `speed-vs-steps.png`;
* `samples-by-step-count.png` -- the same comparison by eye;
* `latent-interpolation.png` -- only possible because `eta = 0` is
  deterministic.

## What to look at

* **The shape of the FID curve.** The claim DDIM makes is not "fewer steps are
  free" but "quality is flat over a wide range and degrades only at very few
  steps". Find the knee -- the point where cutting further starts costing real
  quality. That is the number worth remembering.
* **Speed scales linearly with steps; quality does not.** That asymmetry is the
  entire practical result.
* **Try `--eta 1.0`.** It should track DDPM quality more closely at matched step
  counts while losing the deterministic-latent property. The trade-off between
  the two is real.
* The interpolation figure only makes sense at `eta = 0`. With a stochastic
  chain, feeding the same latent twice gives two different images, so there is
  no latent space to interpolate in.

## A note on FID and KID in this repository

Computed in a small per-dataset feature space, not ImageNet InceptionV3. Since
every setting here uses the identical feature space and the same trained
network, the *relative* comparison across step counts is exactly what it looks
like. See `../01-vae/README.md`.
