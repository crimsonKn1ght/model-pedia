# 08 - GAN (learning without a loss you can evaluate)

Every model before this one had a loss you could compute on a single example: a
reconstruction error, a likelihood, a bound. A GAN has none. There is no target to compare
the output against - only a second network whose job is to tell generated images from real
ones, and a generator whose job is to stop it.

That is what makes GANs produce sharp images: nothing in the objective rewards hedging. It
is also what makes them hard, and this project is organised around the practical
consequence.

## The loss curves are not a progress signal

A GAN's losses measure who is currently winning a game whose equilibrium is both networks
being confused. A falling generator loss can mean better samples or a collapsing
discriminator, and **the number cannot tell you which.** So `train.py` computes FID every
epoch against a fixed set of real features and selects the checkpoint on that. Watch
`curves.png`: the losses wander, the FID descends.

```bash
pip install -r ../requirements.txt

python data.py --dataset fashion-mnist            # ~30 MB
python train.py --dataset fashion-mnist            # 20 epochs, FID every epoch
python evaluate.py --checkpoint outputs/fashion-mnist_bce/best.pt
```

Two diagnostics print beside the losses, and they are what you read when a run goes wrong:

- **`D(real)` and `D(fake)`** - the discriminator's mean probabilities. Both near 0.5 is a
  healthy game. `D(real)` at 1.0 with `D(fake)` at 0.0 means the discriminator has won and
  the generator has no gradient left to follow.
- **`recall`** in `evaluate.py` - the mode-collapse detector, below.

## Mode collapse is low recall at high precision

FID is one number and cannot distinguish "the samples look wrong" from "the samples look
right and there are only three of them". Precision and recall can:

| | high precision | low precision |
|---|---|---|
| **high recall** | working | noisy but varied |
| **low recall** | **mode collapse** | not training |

Precision is the fraction of samples inside the real data's k-NN manifold; recall is the
fraction of real images inside the samples' manifold. The implementation is checked against
the degenerate case - a "generator" that emits one point scores precision 1.0 and recall 0.0.

## The memorisation check

A generator that memorised its training set would score beautifully on everything above.
`evaluate.py` therefore finds, for each sample, the closest training image in feature space,
and compares that distance with the distance between two *different* real images. A ratio
well below 1 means copying. `nearest_neighbours.png` puts the pairs side by side so you can
judge rather than trust the number.

`interpolation.png` is the other check: a generator that learned structure interpolates
smoothly between latents; one that memorised jumps between training images.

## About FID here

FID and KID are measured through a small classifier trained on the dataset itself, not
InceptionV3 - downloading a 90 MB ImageNet model would be the largest dependency in this
repository by an order of magnitude. **These values are therefore not comparable with
published FID.** They are comparable within this repository, using the same cached feature
network for every arm of every study, which is what the comparisons need. `utils.py` says so
at the top and `metrics.json` records it as `feature_net`.

KID is the more trustworthy of the two at these sample counts: it is unbiased for finite
samples, while FID is not.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `loss` | BCE vs hinge | Hinge is often steadier; neither dominates at this scale |
| `smoothing` | One-sided label smoothing 0 vs 0.1 | A discriminator that cannot be certain keeps producing gradient |
| `balance` | 1 vs 2 discriminator steps per generator step | Theory says train D to optimality; practice says a D that wins too clearly stalls the run |
| `latent` | 16 / 64 / 128 latent dimensions | Too small caps diversity; too large is mostly wasted |

## Files

| File | What it does |
|---|---|
| `data.py` | MNIST, Fashion-MNIST, CIFAR-10, CelebA and any image folder, in `[0, 1]` |
| `model.py` | DCGAN generator and discriminator, the paper's initialisation, both losses |
| `train.py` | Adversarial training with per-epoch FID and the D-probability diagnostics |
| `evaluate.py` | FID, KID, precision/recall, interpolation, and the memorisation check |
| `compare.py` | `loss`, `smoothing`, `balance` and `latent` studies |
| `utils.py` | FID, KID, generative precision/recall, and the feature network they run through |

## Knobs worth turning

- `--label-smoothing 0.1` then read `D(real)`. It should stop pinning to 1.0.
- `--d-steps 3` to watch the failure the theory does not predict.
- Compare the `recall` here with project 12's diffusion model on the same dataset. Diffusion
  usually wins on recall by a wide margin, because it has no discriminator to satisfy and so
  no incentive to abandon the difficult parts of the distribution.
