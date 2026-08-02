# 04 - Autoencoder (compression as representation learning)

No labels here. The model is handed an image and asked to reproduce it, with
one obstacle in the way: everything has to pass through a `latent_dim`-number
bottleneck. A 32x32 grayscale image is 1024 numbers; a 32-dimensional code is
32x compression. The only way through is to throw away what is predictable and
keep what is not, which is a working definition of a representation.

Remove the bottleneck and the model learns the identity function and teaches
you nothing. The bottleneck *is* the model.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset fashion-mnist               # download (~30 MB)
python train.py --latent-dim 32 --epochs 10          # about two minutes on CPU
python evaluate.py --checkpoint outputs/fashion-mnist_latent32/best.pt
```

Then the study that makes the point:

```bash
python sweep.py --latent-dims 2 8 16 32 64 128       # about ten minutes
```

`--dataset cifar10` is the harder version: colour and texture do not survive a
small code nearly as gracefully as clothing silhouettes do.

## What evaluate.py reports

Reconstruction quality, three ways, because they disagree in useful places:

- **MSE** - what the loss optimises.
- **PSNR** (dB) - log-scaled MSE, comparable across datasets.
- **SSIM** - compares local means, variances and covariance, so it punishes the
  blur that PSNR happily forgives. An autoencoder that has learned to hedge
  will show decent PSNR and poor SSIM.

And then the question that actually matters - is the code *useful*? A
1-nearest-neighbour classifier is run twice, once on the 32-number latent codes
and once on the 1024-number raw pixels. If the latent does about as well with
32x fewer numbers, the encoder kept the structure and dropped the redundancy.
Nothing in the training objective asked for this; it falls out of the
bottleneck.

## Figures

| File | What to look at |
|---|---|
| `reconstructions.png` | Original, reconstruction, and rescaled absolute error. The error concentrates on edges and texture - the high-frequency detail is what a bottleneck spends its budget on last |
| `latent_pca.png` | First two principal components of the code, coloured by class. Classes cluster without ever having been shown a label |
| `interpolation.png` | Decoding a straight line between two codes |
| `curves.png` | Loss, PSNR and SSIM per epoch |
| `sweep.png`, `bottleneck_comparison.png` | From `sweep.py`: quality against bottleneck size, and the same images decoded through every bottleneck |

About `interpolation.png`: a plain autoencoder is **not** a generative model.
Nothing in the objective forces the space between two codes to decode to
anything sensible, so the midpoints tend to be a soft blend rather than a
plausible new garment. That failure is worth seeing directly - closing it is
exactly what a variational autoencoder adds.

## Files

| File | What it does |
|---|---|
| `data.py` | Downloads Fashion-MNIST / MNIST / CIFAR-10, resizes to 32x32, keeps pixels in `[0, 1]` |
| `model.py` | `ConvAutoencoder`: three stride-2 convolutions down, a linear bottleneck, and the mirror back up |
| `train.py` | Reconstruction training, tracks validation PSNR and SSIM |
| `evaluate.py` | Test metrics, latent-vs-pixel 1-NN, PCA plot, interpolation, error maps |
| `sweep.py` | Trains one model per bottleneck size and plots the trade-off |
| `utils.py` | Seeding, PSNR and SSIM implemented from scratch, image-grid plotting |

The SSIM here matches `skimage.metrics.structural_similarity` (with
`gaussian_weights=True, sigma=1.5, use_sample_covariance=False`) to six decimal
places, so the numbers are comparable with anything else you read.

## Typical numbers

Indicative Fashion-MNIST test results, 10 epochs on a 20 000-image subset.

| Latent | Compression | PSNR | SSIM | Latent 1-NN | Pixel 1-NN |
|---|---|---|---|---|---|
| 2 | 512x | ~14 dB | ~0.45 | ~0.45 | ~0.78 |
| 8 | 128x | ~18 dB | ~0.68 | ~0.72 | ~0.78 |
| 32 | 32x | ~21 dB | ~0.82 | ~0.79 | ~0.78 |
| 128 | 8x | ~25 dB | ~0.92 | ~0.80 | ~0.78 |

The interesting row is 32: the code matches raw pixels for nearest-neighbour
classification while being 32 times smaller.

## Knobs worth turning

- `--loss bce` - Fashion-MNIST and MNIST pixels are nearly binary, and treating
  them as Bernoulli probabilities gives visibly sharper reconstructions.
- `--base-channels 16` - a narrower encoder; shows that the bottleneck, not the
  parameter count, is the binding constraint.
- `--dataset cifar10 --latent-dim 128` - the same 8x compression on colour
  images is a much harder problem.
