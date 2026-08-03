# 15 - Vision Transformer (what the convolution was giving you for free)

A convolution has two assumptions built in: nearby pixels belong together, and a feature that
matters in one place matters everywhere. Those assumptions are true of images, and they are
why a small CNN learns from a small dataset.

A Vision Transformer throws both away. It cuts the image into patches, embeds each one, and
runs plain self-attention - every patch may look at every other patch from the first layer,
and nothing tells it which patches are adjacent except a position embedding it has to learn.
Strictly more general, strictly harder to learn, and that is the whole story of ViT on small
data.

This project is built to show that honestly rather than to pick a winner.

```bash
pip install -r ../requirements.txt

python data.py --dataset cifar10 --augment strong --preview   # ~170 MB
python train.py --arch vit --dataset cifar10
python train.py --arch resnet --dataset cifar10
python evaluate.py --checkpoint outputs/cifar10_vit/best.pt
python compare.py --study arch
```

## The comparison is controlled twice

**Matched parameters.** The default ViT (dim 192, depth 6) has 2.69M parameters; the ResNet
control at `width=64` has 2.78M - within 3%.

**Matched wall-clock.** Parameter count is only half a control. Attention costs
O(tokens squared) and convolution costs O(pixels x channels), so two models of equal size can
differ by a factor in what they cost to run. `measure_throughput` times a real
forward-backward step, and `compare.py` reports images per second and accuracy per 1000
seconds of training beside the accuracy. A comparison at matched size that quietly spends
different compute is not a comparison.

On CIFAR-10 at this scale the ResNet is expected to win. The interesting question is by how
much, and what closes it.

## Augmentation is the answer, and it is measurable

```bash
python compare.py --study augment    # none / basic / strong
```

The transformer is the model that has to *learn* locality and translation equivariance, so it
is the model that gains from having more views of the data. `--augment` has three settings
and the study prices each one. This is the practical form of "ViT needs more data": you either
bring more data, or you manufacture more views of what you have.

## The attention maps

`attention.png` is what a ViT gives you that a CNN does not. Attention in one layer says where
that layer looked; composing the layers - with the residual path added in as an identity, then
row-normalised - gives what the class token at the top depends on in the *input* patches. That
is Abnar and Zuidema's rollout, implemented in `model.attention_rollout`.

It is worth looking at even when the ViT is losing on accuracy: a model attending to the
object is failing differently from one attending to the background.

## Two properties checked in code

`python model.py` verifies both:

- **without position embeddings the ViT is permutation-equivariant** - zero the table, shuffle
  the patches, and the logits do not change. This is the assumption a convolution does not
  need to be told.
- **with them, patch order matters.** The position embedding is doing real work.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `arch` | ViT vs ResNet at matched parameters and time | The ResNet wins at this scale; read both columns |
| `augment` | none / basic / strong | The transformer gains more than the ResNet would |
| `patch` | 2 / 4 / 8 pixels | Halving the patch quadruples the tokens and the attention cost, for a much smaller accuracy gain |
| `depth` | 4 / 6 / 8 blocks | Where depth stops paying on 20 000 images |

## Files

| File | What it does |
|---|---|
| `data.py` | CIFAR-10/100, Flowers-102, MNIST and Fashion-MNIST, with three augmentation levels |
| `model.py` | ViT with returnable attention and rollout, plus the size-matched ResNet control |
| `train.py` | One script for both architectures, same schedule, and it times the model |
| `evaluate.py` | Test accuracy, per-class accuracy, confusion matrix, attention maps |
| `compare.py` | `arch`, `augment`, `patch` and `depth` studies |
| `utils.py` | Accuracy and confusion, throughput measurement, attention overlays |

## Knobs worth turning

- `--patch-size 2` on CIFAR-10: 256 tokens instead of 64. Read the img/s column before the
  accuracy.
- `--train-subset 0` to use the full 50 000 images. The gap to the ResNet should narrow - the
  transformer is the one that was data-starved.
- Compare with project 17's MAE-pretrained ViT on the same dataset. That is the other way to
  close the gap, and it is the one that made ViTs practical at this scale.
- `--dataset flowers102` has 102 classes and about 1000 training images. Expect the ViT to do
  badly, and expect that to be informative.
