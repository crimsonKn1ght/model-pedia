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
