# 20 - Image captioning (a grid in, a sequence out)

Every project before this one produced a fixed-shape answer: a label, a mask, a set of
boxes. This one produces a *sentence*, which means the output is variable-length, its
tokens depend on each other, and there is no single correct answer to compare against.
All three of those change the machinery.

The model is the standard arrangement: an encoder turns the image into a short sequence
of feature vectors - one per spatial location, not one per image - and a causal
Transformer decoder generates words while cross-attending into it. Pooling the image to
a single vector also works and is how the first captioning models did it, but then the
decoder has nothing to point at, and "the dog on the left" becomes unreachable.

## Run it

```bash
pip install -r ../requirements.txt

python data.py --dataset shapes --preview        # generated, no download
python train.py --dataset shapes                 # ten epochs, ~9 min on four CPU cores
python evaluate.py --checkpoint outputs/shapes_cnn/best.pt --beam-sizes 1 3 5
```

For real photographs:

```bash
python data.py --dataset flickr8k                                 # ~1.1 GB download
python train.py --dataset flickr8k --encoder resnet18 --epochs 10
```

`flickr8k` is 8091 photographs with five human captions each and the authors' own
splits, fetched from a public mirror of the original release. Use it with
`--encoder resnet18`: the encoder is frozen ImageNet features, so only the decoder
trains and the run stays measured in minutes rather than hours.

`shapes` is generated - one or two coloured shapes per image, four templated captions
each. It is the default because a captioner trained for ten minutes on Flickr8k
produces word salad, and word salad tells you nothing about whether your BLEU
implementation is correct. On `shapes` the model learns to say true sentences, the
metrics move for reasons you can read off the picture, and the failure modes are
legible: getting the shape right and the colour wrong is a visible, gradable error.

## Three metrics, because they catch different failures

All three are implemented in `utils.py` rather than pulled from a package.

| Metric | What it measures | What it forgives |
|---|---|---|
| **BLEU-n** | Clipped n-gram precision, corpus-level, with a brevity penalty | Saying little. It is precision-only, so a short safe caption does well |
| **METEOR** | Unigram precision *and* recall, recall-weighted, with a word-order penalty | Less than BLEU - leaving things out costs something |
| **CIDEr-D** | TF-IDF-weighted n-gram similarity over the reference corpus | Almost nothing generic. Agreeing on *"a"* earns nothing; agreeing on the word that identifies this image earns a lot |

The METEOR here matches exact words only. The real metric also matches stems, synonyms
and paraphrases, so these numbers come out lower than published ones - which is why the
key is `meteor_exact` rather than `meteor`.

BLEU is computed **corpus-level**: counts are summed over the whole split and divided
once, as the definition intends. Averaging per-sentence BLEU is a different and much
noisier number, and mixing the two up is the most common way captioning results stop
being comparable.

### The two diagnostics that matter more than the scores

`mean_length` and `distinct_captions` - the fraction of test images that got a caption
no other image got. The classic failure of a small captioner is to learn one plausible
sentence and emit it for everything; that can post a respectable BLEU-1 with
`distinct_captions` near zero, and no amount of staring at BLEU will tell you. It is
worth seeing this happen: a single under-trained epoch on `shapes` produces exactly
that, the same caption for every image, with BLEU-1 around 0.3.

## Measured results

`shapes`, 2000 training images (8000 image-caption pairs) at 64x64, CNN encoder trained
from scratch, 10 epochs on four CPU cores (448 s). Scored on the 500-image test split
against all four references:

| Decoding | BLEU-1 | BLEU-4 | METEOR (exact) | CIDEr-D | Mean length | Distinct |
|---|---|---|---|---|---|---|
| greedy | 0.9328 | 0.7945 | 0.9278 | 3.198 | 5.9 | 0.524 |
| beam 3 | 0.9821 | **0.9173** | 0.9497 | **4.554** | 12.2 | 0.746 |
| beam 5 | 0.9821 | 0.9183 | 0.9504 | 4.556 | 12.2 | 0.746 |

Three things worth reading carefully.

**Beam search is worth 0.12 BLEU-4 here**, far more than the point or two it usually
buys - and the reason is visible in the length column. Greedy settles on the short
template (*"a small yellow triangle"*, 5.9 tokens); beam search finds the long one
(*"the image shows a small yellow triangle on the left"*, 12.2 tokens) because with the
0.7 length penalty its total log-probability comes out higher. The longer template
contains more of the reference n-grams, so every metric improves. That is a real gain,
but it is a gain from *template selection*, not from better recognition - which is the
kind of thing a score alone will not tell you and `examples.png` will.

**Beam 5 is indistinguishable from beam 3** (+0.001 BLEU-4 for another full decode of
the split). Widening the beam has sharply diminishing returns, and this is what that
looks like when measured instead of assumed.

**The loss stopped moving and the metric did not.** Over the last five epochs validation
cross entropy sat between 0.950 and 0.968 with no trend, while validation BLEU-4 went
0.729, 0.772, 0.787, 0.799, 0.804. Had the checkpoint been selected on loss, epoch 6
would have won and roughly 0.03 BLEU-4 would have been left on the table. This is the
concrete reason `train.py` generates captions every epoch instead of trusting the loss.

The generated captions are accurate on shape, colour and size, and least reliable on
*position* and on the second object of a two-object scene - which is exactly what you
would expect from a 4x4 grid of encoder tokens and 8000 training pairs.

## Teacher forcing, and why the loss is not the metric

Training shows the decoder the true previous words and asks for the next one. That is
fast and stable, and it is not what the model is judged on - at test time it conditions
on its own output and errors compound. So `train.py` **generates** captions for the
validation split every epoch and selects the checkpoint on BLEU-4, not on loss.

Label smoothing makes the same point from the other direction:

```bash
python compare.py --study smoothing
```

Turning it off *improves* the validation cross entropy and usually makes the captions
worse. An image has several correct captions, so a loss demanding all the probability
on one particular next word is optimising for something untrue. This is the cleanest
example in the repository of the loss and the metric disagreeing, with the metric being
right.

## Ablations

| Study | Compares | What to expect |
|---|---|---|
| `smoothing` | Label smoothing 0.0 vs 0.1 | Better loss, worse captions, without smoothing |
| `capacity` | Decoder depth 1 / 3 / 5 | Where extra depth stops paying, on a small vocabulary |
| `encoder` | From-scratch CNN vs frozen ResNet-18 | On `shapes` the CNN wins - ImageNet knows nothing about coloured triangles. On `flickr8k` it is not close |

## Files

| File | What it does |
|---|---|
| `data.py` | Flickr8k download and parsing, the generated `shapes` set, vocabulary, both collate functions |
| `model.py` | CNN and frozen-ImageNet encoders, Transformer decoder, greedy and beam search |
| `train.py` | Teacher-forced training, per-epoch generation, checkpoint selection on BLEU-4 |
| `evaluate.py` | Test BLEU / METEOR / CIDEr-D at several beam widths, plus the diagnostics |
| `compare.py` | `smoothing`, `capacity` and `encoder` studies |
| `utils.py` | BLEU, METEOR and CIDEr-D from scratch, caption figures |

## Outputs

In `outputs/<dataset>_<encoder>/`:

- `best.pt` (weights **and** the vocabulary - the ids are meaningless without it)
- `history.json`, `curves.png` - loss next to BLEU-4 and CIDEr-D
- `examples_val.png`, `examples.png` - images with the generated caption and a reference
- `metrics.json` - every metric at every beam width, plus ten sample captions
- `decoding.png` - the same weights under greedy and beam search

## Knobs worth turning

- `--beam-sizes 1 3 5 10` in `evaluate.py`. Beam search is worth real points and it is
  not free; this prices it. Watch `mean_length` rise with beam width, and note that
  `--length-penalty` exists in `model.generate` for exactly that reason.
- `--label-smoothing 0.0` and read `examples.png`, not the loss.
- `--encoder mobilenet_v3_small` on `flickr8k` - a third of ResNet-18's cost, and the
  captions barely notice.
- `--max-len 16` - a shorter cap makes BLEU look better and the captions worse, which is
  a useful thing to see happen deliberately.
