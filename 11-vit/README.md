# 11 - Vision Transformer (ViT-Tiny) vs ResNet-18

Image classification, and an honest comparison at matched compute.

## The idea

A ViT discards nearly every assumption a CNN builds in. The image is cut into
fixed patches, each patch is linearly projected into a token, a learned position
embedding is added, and a standard transformer encoder does the rest. No
locality prior, no spatial weight sharing, no pooling hierarchy. A class token
aggregates information and a linear head reads the answer off it.

That is both the appeal and the problem. Self-attention lets any patch attend to
any other from the very first layer, which is strictly more expressive than a
convolution's local window. But convolutions get **translation equivariance**
for free, and a ViT has to learn it from data. On ImageNet-scale datasets the
extra expressiveness wins. On CIFAR-10 from scratch it usually does not.

**The interesting result in this project is often that the ResNet wins.**
Understanding why is worth more than a leaderboard number: it is the clearest
practical demonstration that inductive bias is worth data, and that
architectural expressiveness is not free.

Two things make small-data ViT training survivable, and both are on by default:

* **Stochastic depth** -- randomly skip residual branches during training, with
  the drop rate increasing linearly with depth;
* **Mixup and augmentation** -- ViTs overfit small datasets aggressively, and
  these are among the few regularisers that reliably help.

Note the patch size: `4`, not the original paper's `16`. On a 32x32 image, 16x16
patches would leave four tokens, which is far too coarse to say anything.

## Run it

```bash
python download_data.py
python train.py                        # ViT on CIFAR-10
python train.py --model resnet18       # the baseline
python evaluate.py                     # compares both
```

Useful variations:

```bash
python train.py --dataset fashion-mnist --epochs 10
python train.py --dataset flowers102 --image-size 64
python train.py --mixup 0 --no-augment          # watch the ViT overfit
python train.py --depth 4 --dim 128             # a smaller ViT
```

## Files

| File | What it holds |
|---|---|
| `model.py` | `VisionTransformer`, `PatchEmbedding`, `TransformerBlock`, `DropPath`, `ResNet18`, `mixup` |
| `download_data.py` | dataset fetcher |
| `train.py` | AdamW, cosine schedule with warmup, mixup |
| `evaluate.py` | accuracy, FLOPs, latency, per-class breakdown, attention maps |

## Test-set evaluation

An accuracy comparison alone proves nothing, so `evaluate.py` reports the
context needed to interpret it:

* **top-1 and top-5 accuracy** on the test split;
* **parameter count, measured MFLOPs per image, and wall-clock latency**. Two
  models are only comparable at matched compute -- a ViT that loses while using
  three times the FLOPs has lost twice;
* **per-class accuracy and the worst class**, since a good average can hide one
  class the model never learned;
* **`vit-attention.png`** -- attention rollout from the class token back to the
  patches, showing what the prediction actually drew on.

Everything lands in `outputs/comparison/`.

## Reference run

Fashion-MNIST, 20000 training images, 6 epochs each, CPU only. Both models were
sized to land near the same parameter count so the comparison is about
architecture rather than capacity:

| Model | accuracy | top-5 | params | MFLOPs/image | ms/image | train time |
|---|---|---|---|---|---|---|
| ViT-Tiny | 0.826 | 0.995 | 2.69M | 172.7 | 0.98 | 17.2 min |
| **ResNet-18** | **0.920** | **0.998** | 2.80M | **138.7** | 0.92 | **7.1 min** |

**The ResNet wins on every axis at once.** Higher accuracy, fewer FLOPs, faster
inference, and less than half the training time. This is the expected outcome at
this scale and it is worth stating without softening: at 20000 small greyscale
images, a convolution's built-in translation equivariance is worth more than
self-attention's flexibility, and the ViT has to spend both parameters and data
learning a prior the CNN gets for free.

The ViT is not broken -- 82.6% with 99.5% top-5 is real learning, and its
training curve was still improving when the budget ran out while the ResNet's
had largely flattened. That asymmetry is the honest caveat: a short budget
flatters the CNN, and the ranking would narrow with more epochs and more data.
It would not obviously reverse at this dataset size, which is the point.

If you take one thing from this project, make it that **architecture comparisons
without a compute column are close to meaningless**. Quoting only the accuracies
would suggest a tuning gap; the FLOPs column shows the ViT losing while spending
25 percent more compute per image.

## What to look at

* **Match the compute before drawing conclusions.** `--resnet-base` and
  `--dim`/`--depth` let you tune the two towards equal MFLOPs. The comparison is
  only meaningful once they are close.
* **Run the ViT with `--mixup 0 --no-augment`.** Training accuracy climbs while
  test accuracy stalls or falls -- textbook overfitting, and much more dramatic
  than the same ablation on the ResNet. That difference *is* the inductive-bias
  argument, made empirically.
* **Attention rollout is suggestive, not an explanation.** It averages heads and
  ignores the value pathway. Useful for intuition, not for claims about what the
  model "looks at".
* Longer training changes the ranking. The ViT's curve is typically still
  climbing when the ResNet's has flattened, so a short budget flatters the CNN.
  Worth noting before generalising from a 12-epoch run.
