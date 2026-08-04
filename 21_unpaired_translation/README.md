# 21 - Unpaired image-to-image translation (learning a mapping with no pairs)

Project 10 trained Pix2Pix on **pairs**: for every input it had the correct output, so it
could compute a pixel loss against the right answer. Almost no real translation problem
comes with pairs. Nobody has photographs of the same horse as a zebra, or the same street
in summer and winter from the identical angle.

CycleGAN drops the requirement. It is given two *collections* - a pile of images from
domain A, a pile from domain B - and told nothing about which image corresponds to which,
because nothing does.

## Why the obvious objective fails

Remove the pairs and the adversarial loss is all that is left: train `G: A -> B` so a
discriminator cannot tell `G(a)` from a real domain-B image. That objective is
**satisfied by throwing the input away**. A generator that ignores `a` entirely and emits
the same convincing domain-B image every time fools the discriminator perfectly. Nothing
in the loss says the output should have anything to do with the input.

**Cycle consistency** is the constraint that fixes it. Train a second generator in the
opposite direction and require the round trip to return where it started:

```
F(G(a)) ~= a          G(F(b)) ~= b
```

A mapping that discards its input cannot be inverted, so it cannot satisfy this. That is
the entire idea, and `compare.py --study cycle` is the experiment that shows it: with the
cycle term removed, the adversarial losses keep improving while the translations stop
corresponding to their inputs at all.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --task shapes --preview     # generated, no download
python train.py --task shapes              # 10 epochs, ~8 min on four CPU cores
python evaluate.py --checkpoint outputs/shapes_cycle10/best.pt
python compare.py --study cycle            # the ablation that matters
```

Four tasks, in increasing order of cost:

| Task | Domains | Download |
|---|---|---|
| `shapes` (default) | hollow outlines vs filled colour, generated independently | none |
| `classes` | two classes of one dataset, e.g. Fashion-MNIST sandal vs sneaker | 12-170 MB |
| `datasets` | two datasets, e.g. MNIST digits vs Fashion-MNIST garments | 42 MB |
| `horse2zebra` | the dataset the paper used | ~110 MB, host often blocked |

`shapes` is the default because it needs no download and because the two domains share a
shape vocabulary, so there *is* a sensible translation to find and you can see in the
figure whether it was found. The domains are seeded independently, so no outline in A is
the outline of any shape in B - the pairing genuinely does not exist. `horse2zebra` needs
`people.eecs.berkeley.edu`; if that host is unreachable the other three show the same
mechanics.

## Measuring a problem with no right answer

This is the part worth reading, because it is where unpaired translation differs from
everything else in this repository. There is no target image, so there is no
reconstruction error to report. Two things can be measured, and **each one alone can be
gamed**:

| Question | Metric | How to cheat it |
|---|---|---|
| Did the output reach the target domain? | FID/KID of `G(A)` against real B | Ignore the input and emit one convincing B image |
| Did the content survive? | cycle SSIM/PSNR/L1 of `F(G(a))` against `a` | Learn the identity map and translate nothing |

So both are always reported together, each against the baseline that makes it readable:

- **The identity mapping is the FID baseline.** Copy the input unchanged and its "FID
  against domain B" is just the distance between the two domains. That is the number to
  beat: score above it and the translation moved the images *away* from the target.
- **The identity mapping is also the cycle-SSIM ceiling**, at exactly 1.000, which is why
  cycle SSIM is a guard rather than a score. Its job is to catch the first failure, not to
  be maximised.

Checkpoint selection uses mean FID across both directions, never the cycle error - because
selecting on cycle error would reward doing nothing.

## Measured results

Generated `shapes`, 2000 images per domain at 32x32, `lambda_cycle=10`,
`lambda_identity=0.5`, two 609k-parameter generators and two 95k-parameter
discriminators, 10 epochs in 492 s on four CPU cores. Checkpoint selected at epoch 9 on
mean validation FID; scored on the 500-image-per-domain test split, which `train.py` never
touches.

| Direction | FID | identity baseline | KID | precision | recall | cycle SSIM |
|---|---|---|---|---|---|---|
| outline -> filled | **0.531** | 240.94 | +0.043 | 0.626 | 0.252 | **0.901** |
| filled -> outline | **1.553** | 240.94 | +0.251 | 0.934 | 0.338 | 0.686 |

Three things are worth reading off that.

**The translation works, and the baseline is what says so.** The identity mapping scores
FID 240.94 - that is simply how far apart the two domains are, and it is identical in both
directions because it is a property of the data, not the model. Against that, 0.531 means
the outlines were moved essentially all the way into the filled domain. Without the
baseline column, "FID 0.531" would be a number with no scale attached.

**Both metrics had to be reported.** Cycle SSIM of 0.901 rules out the failure FID cannot
see: the model is not ignoring its input, because the round trip still recovers the
original. A model that had collapsed to one output would show the same FID and a cycle
SSIM near zero.

**The two directions are not symmetric, and they cannot be.** `filled -> outline` scores
worse on both counts, and the cycle number explains why: going from filled colour to a
hollow outline *destroys* the colour, so coming back the generator has to invent it and
cannot match what was there. Cycle consistency is only fully satisfiable when the
translation is close to information-preserving, and this asymmetry is that limit showing up
in a measurement rather than in an argument.

**The recall of 0.25-0.34 is the honest weak point, and `translations.png` shows what it
means.** The translations are individually plausible - precision 0.63 and 0.93 - but they
cover only part of the target domain's variety, and the figure says why: the model learned
*fill the interior of the shape*, which is most of the task, but fills it with a mottled
multi-coloured texture rather than one flat colour from the five-colour palette the real
domain uses. Geometry and position transfer cleanly; the colour statistics do not. The
round trip back through `filled -> outline -> filled` also frequently returns a different
colour than it started with, which is the information-destroying asymmetry above made
visible.

None of that is contradicted by FID 0.531 - it is what a low recall against a high
precision looks like when you go and look. Ten epochs on 2000 images is a demonstration
rather than a converged model, and the recall column is the one to watch when raising
`--epochs`, since it is the number the mottling shows up in.

**Validation FID by epoch**, showing the selection working: 24.9, 8.2, 4.4, 3.0, 2.1, 1.4,
1.26, 1.10, **1.00**, 1.57. Epoch 10 was worse than epoch 9, which is why the checkpoint
is not simply the last one.

## What the FID number here is, and is not

Same caveat as projects 06-14, with one addition specific to this project. FID is measured
through a small classifier trained on the data rather than ImageNet InceptionV3, so
**these values are not comparable with published FID** - only within this repository.

The addition: the classifier here is trained to tell **domain A from domain B**, a
two-class problem, rather than to recognise objects. That is the most direct feature space
for the question "does this output look like domain B", and for `shapes` it is the only
labelling available. But it is worth knowing that the feature space is deliberately tuned
to the one axis being measured, and that on `shapes` the two domains are trivially
separable - the classifier reaches 100% train accuracy - so FID here is a sharp
measurement along the domain axis and says little about anything else.
`metrics.json` records `feature_net` for every run.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `cycle` | `lambda_cycle` 0 / 1 / 10 | The important one. At 0 the FID column stays respectable and the cycle column collapses - FID cannot see the failure |
| `identity` | `lambda_identity` 0 / 0.5 | Smaller effect: without it the generators shift colours they had no reason to touch |
| `pool` | image buffer 0 / 50 | Without the buffer the generator/discriminator pair oscillates, visible as a noisier loss curve |

## Files

| File | What it does |
|---|---|
| `data.py` | The four unpaired tasks, the random re-pairing each epoch, and the domain-labelled loader the FID network is fitted on |
| `model.py` | ResNet generators, PatchGAN discriminators, the image buffer, and the four-part objective |
| `train.py` | Trains all four networks; selects on mean FID, logs cycle quality beside it |
| `evaluate.py` | Test-split FID/KID/precision/recall and cycle quality, both directions, both against the identity baseline |
| `compare.py` | `cycle`, `identity` and `pool` studies |
| `utils.py` | PSNR, SSIM, FID, KID, generative precision/recall and the feature network |

## Outputs

In `outputs/<task>_cycle<lambda>/`:

- `best.pt`, `history.json`, `train_summary.json`
- `curves.png` - adversarial, cycle and discriminator losses with validation FID
- `translations_val.png` - both directions and the round trip, refreshed every epoch
- `translations.png` - the same on the test split, six rows
- `fid_vs_baseline.png` - each direction against the identity mapping
- `metrics.json` - every number above, including `feature_net`

## Knobs worth turning

- `--lambda-cycle 0` and look at `translations.png`. This is the fastest way to see why the
  method needs the cycle term, and why FID alone would not have told you.
- `--lambda-cycle 100` over-constrains it the other way: the round trip becomes near
  perfect because the generators have learned to barely change anything.
- `--task datasets --domain-a mnist --domain-b fashion-mnist` is much harder than `shapes`:
  the domains share almost no structure, so cycle consistency and the adversarial loss pull
  against each other rather than agreeing.
- Compare `translations.png` with project 10's on the paired task. Pix2Pix has the easier
  problem and a U-Net with skip connections to exploit it; the ResNet generator here has no
  skips on purpose, because the two domains need not align pixel for pixel.
