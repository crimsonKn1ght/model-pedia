# 03 - Transfer learning (frozen vs fine-tuned)

Flowers-102 gives you **ten training images per class** for 102 classes. Trained
from scratch that is hopeless. Starting from an ImageNet backbone it is a
twenty-minute job, because the features a network learns on one dataset -
edges, textures, petal-shaped parts - are not specific to that dataset.

The project runs both ways of reusing those features:

- **frozen** (linear probe) - the backbone is a fixed feature extractor. The
  features are computed **once**, cached, and a linear head is trained on the
  cached vectors. Training the head takes seconds.
- **finetune** - the backbone keeps learning, at a learning rate 10x smaller
  than the head's, so the pretrained weights are adjusted rather than erased.

Fine-tuning usually wins by several points; the frozen probe gets most of the
way there for a fraction of the compute. Which trade-off is right depends on
how much data and time you have, and running both is how you find out.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset flowers102          # download (~345 MB)
python train.py --mode both --epochs 10      # trains both arms, prints the table
python evaluate.py --checkpoint outputs/flowers102_resnet18_frozen/best.pt
python evaluate.py --checkpoint outputs/flowers102_resnet18_finetune/best.pt
```

On four CPU cores the frozen arm takes about two minutes (nearly all of it
caching features) and the fine-tune arm about six. `--dataset pets` swaps in
Oxford-IIIT Pets (37 breeds, ~800 MB).

Note that the ImageNet weights themselves are downloaded by torchvision from
`download.pytorch.org` the first time a backbone is built.

## The control arm

If you doubt that the pretrained weights are doing the work:

```bash
python train.py --mode finetune --no-pretrained --epochs 10
```

Same architecture, same schedule, random initialisation. On Flowers-102 it
lands near single-digit accuracy. The gap between that run and the default one
*is* the transferred knowledge.

## Backbones

| Name | Feature dim | Params | Relative CPU cost |
|---|---|---|---|
| `mobilenet_v3_small` | 1024 | 1.6 M | fastest; use it if you are impatient |
| `resnet18` | 512 | 11.2 M | the default, and the usual reference point |
| `efficientnet_b0` | 1280 | 4.1 M | fewer params than ResNet-18, slower per image |
| `resnet50` | 2048 | 23.7 M | strongest features, ~3x the fine-tuning cost |

## Files

| File | What it does |
|---|---|
| `data.py` | Downloads Flowers-102 / Pets, ImageNet-style preprocessing, augmentation |
| `model.py` | Strips the ImageNet head off a torchvision backbone, bolts on a new one |
| `train.py` | Both modes: cached-feature linear probe, and fine-tuning with two LRs |
| `evaluate.py` | Top-1 / top-5, per-class accuracy, confusion matrix, prediction grid |
| `utils.py` | Seeding, device selection, metrics, plotting |

## Outputs

In `outputs/<dataset>_<backbone>_<mode>/`: `best.pt`, `history.json`,
`curves.png`, `metrics.json`, `confusion_matrix.png`, `predictions.png`.

## Typical numbers

Indicative Flowers-102 test accuracy with `resnet18` at 224 px, 10 epochs.

| Arm | Updated params | Top-1 |
|---|---|---|
| frozen linear probe | ~52 k | ~0.85 |
| fine-tuned | ~11.2 M | ~0.92 |
| fine-tuned, `--no-pretrained` | ~11.2 M | ~0.08 |

## Details worth knowing

- The frozen arm does not use random augmentation. Cached features cannot
  follow a random crop, and recomputing them every epoch would throw away the
  entire speed advantage. `train.py` enforces this rather than letting the two
  quietly disagree.
- Fine-tuning uses one-cycle with separate peaks for backbone and head. A
  single large learning rate across both is the most common way to make
  fine-tuning perform *worse* than a frozen probe.
- Checkpoints store the whole network, so `evaluate.py` never re-downloads
  ImageNet weights.
