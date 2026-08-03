# 14 - Latent diffusion (do the expensive part on a smaller tensor)

Diffusion cost scales with the number of values being denoised, and the expensive part of an
image is high-frequency detail that carries almost no semantic information. So: let a cheap
autoencoder handle the detail once, and run the expensive iterative model on the small
tensor that is left.

That is latent diffusion, and it is a **composition** rather than a new model. This project
therefore imports project 12's U-Net, diffusion process, datasets and metrics rather than
copying them - see `_paths.py`. The one genuinely new piece is the first stage in
`autoencoder.py`. Project 13 is arranged the same way and for the same reason.

## Two stages

```bash
pip install -r ../requirements.txt

python train_autoencoder.py --dataset fashion-mnist                        # stage 1
python train.py --autoencoder outputs/fashion-mnist_z4x2/autoencoder.pt     # stage 2
python evaluate.py --checkpoint outputs/fashion-mnist_z4x2/best.pt \
  --pixel-checkpoint ../12_diffusion/outputs/fashion-mnist_cosine_t200/best.pt
```

**Stage one** trains a small autoencoder with a *spatial* latent. Project 06's VAE compresses
to a single vector, which is the wrong shape - a U-Net needs height and width to convolve
over. Here a 32x32 image becomes a 4x8x8 tensor: 256 values instead of 1024 for greyscale,
or instead of 3072 for colour.

**Stage two** is project 12's training loop with one substitution: encode the batch first,
and denoise the latent instead of the image. That single change is the method.

## The ceiling

Everything stage two can achieve is bounded by how well stage one reconstructs, because every
sample goes out through the same decoder. `evaluate.py` measures that bound directly - real
test images encoded and decoded with no diffusion at all - and reports it as the
**reconstruction ceiling**.

This is the number that makes latent-diffusion results interpretable:

- FID close to the ceiling means the **first stage** is the limit. Training the diffusion
  model longer will not help.
- FID far above the ceiling means there is headroom left in stage two.

Those two situations look identical if you only read FID, and they call for opposite actions.
`ceiling.png` shows real images, the same images through the latent, and fresh samples, in
three rows.

## The latent scale

Stage one measures the standard deviation of its latent over the training set and stores it in
the checkpoint; stage two divides by it. A diffusion process assumes its input has roughly
unit variance, and a latent whose scale had drifted would break the noise schedule *quietly* -
the loss would still fall and the samples would be wrong. The light KL term in stage one
exists for the same reason, not to build a prior worth sampling from.

## The cost table

Pass `--pixel-checkpoint` and `evaluate.py` prints milliseconds per image and values denoised
for both models side by side, plus the speedup. That comparison is the entire reason latent
diffusion exists and the one thing a FID table alone will not show you.

Note where the saving comes from: the diffusion model runs once per timestep, and the decoder
runs once in total. Compressing by 4x makes each of `T` steps cheaper and adds one decode.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `downsample` | 2x vs 4x compression | More compression, cheaper diffusion, lower ceiling |
| `channels` | 2 / 4 / 8 latent channels | Wider latents raise the ceiling and cost more to denoise |

Each arm trains **both** stages, because the two interact - a more aggressive first stage
makes diffusion cheaper and lowers what it is working under. Every row prints the ceiling
next to the FID and the headroom between them.

## Files

| File | What it does |
|---|---|
| `_paths.py` | Imports project 12's U-Net, data and metrics; explains why this project is not self-contained |
| `autoencoder.py` | The first stage: a spatial, lightly KL-regularised autoencoder |
| `train_autoencoder.py` | Stage one, and it measures the latent scale stage two needs |
| `train.py` | Stage two: diffusion inside the frozen latent |
| `evaluate.py` | FID/KID, the reconstruction ceiling, and the cost table against pixel space |
| `compare.py` | `downsample` and `channels` studies |

## Knobs worth turning

- `--levels 3` on 32x32 images gives a 4x4 latent. Watch the ceiling collapse - this is the
  fastest way to see stage one become the bottleneck.
- `--latent-channels 1` for the same lesson from the other direction.
- Train stage one for twice as long and rerun stage two unchanged. If the FID improves, the
  ceiling was the limit all along.
