"""Captioning metrics, written out: BLEU, METEOR and CIDEr-D.

A caption has no single right answer, so every metric here compares a generated
sentence against *several* references and tries to turn "how close is this" into a
number. They disagree with each other, and knowing how is most of the skill:

* **BLEU-n** - clipped n-gram precision with a brevity penalty. Precision-only, so
  it rewards saying little and saying it safely. Corpus-level, as the definition
  intends: summing counts over the whole split and dividing once, not averaging
  per-sentence BLEU, which is a different (and much noisier) number.
* **METEOR** - unigram precision *and* recall, weighted towards recall, with a
  penalty for word order. The variant here matches on exact words only; the real
  metric also matches stems, synonyms and paraphrases, so these numbers come out
  lower than published ones. It is labelled ``meteor_exact`` for that reason.
* **CIDEr-D** - the one built for captioning. n-grams are weighted by TF-IDF over
  the reference corpus, so agreeing with the references on *"a"* earns nothing and
  agreeing on *"snowboarder"* earns a lot. Candidate counts are clipped and a
  Gaussian length penalty is applied, which is the "-D" part.

A caption model that scores well on BLEU and badly on CIDEr is usually producing
fluent generic sentences. Reading ``examples.png`` next to the table is not
optional.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    candidates: list[list[str]],
    references: list[list[list[str]]],
    max_n: int = 4,
) -> dict[str, float]:
    """Corpus-level BLEU-1..``max_n`` with the standard brevity penalty.

    ``candidates[i]`` is a token list; ``references[i]`` is a list of token lists.
    Counts are accumulated over the whole corpus before the division, which is what
    makes this comparable to reported BLEU.
    """
    clipped = [0] * (max_n + 1)
    total = [0] * (max_n + 1)
    candidate_length = 0
    reference_length = 0

    for candidate, refs in zip(candidates, references):
        candidate_length += len(candidate)
        # The brevity penalty uses the reference length closest to the candidate,
        # ties going to the shorter one.
        reference_length += min(
            (abs(len(ref) - len(candidate)), len(ref)) for ref in refs
        )[1] if refs else 0

        for n in range(1, max_n + 1):
            candidate_counts = ngrams(candidate, n)
            if not candidate_counts:
                continue
            maximum = Counter()
            for ref in refs:
                reference_counts = ngrams(ref, n)
                for gram, count in reference_counts.items():
                    maximum[gram] = max(maximum[gram], count)
            clipped[n] += sum(min(count, maximum[gram]) for gram, count in candidate_counts.items())
            total[n] += sum(candidate_counts.values())

    penalty = 1.0
    if candidate_length < reference_length and candidate_length > 0:
        penalty = math.exp(1 - reference_length / candidate_length)

    scores = {}
    log_precisions = []
    for n in range(1, max_n + 1):
        precision = clipped[n] / total[n] if total[n] else 0.0
        # A zero at any order makes the geometric mean zero; the usual practical
        # fix is a small floor so BLEU-4 is not 0 for a short evaluation split.
        log_precisions.append(math.log(precision) if precision > 0 else math.log(1e-9))
        scores[f"bleu_{n}"] = penalty * math.exp(sum(log_precisions) / n)
    scores["brevity_penalty"] = penalty
    scores["length_ratio"] = candidate_length / max(reference_length, 1)
    return scores


def _meteor_single(candidate: list[str], reference: list[str]) -> float:
    """Exact-match METEOR against one reference."""
    if not candidate or not reference:
        return 0.0

    # Greedy alignment on exact word matches, keeping the reference positions so
    # the chunk count (and therefore the fragmentation penalty) is well defined.
    remaining = list(range(len(reference)))
    alignment = []
    for position, word in enumerate(candidate):
        for index in remaining:
            if reference[index] == word:
                alignment.append((position, index))
                remaining.remove(index)
                break

    matches = len(alignment)
    if matches == 0:
        return 0.0

    precision = matches / len(candidate)
    recall = matches / len(reference)
    f_mean = 10 * precision * recall / (recall + 9 * precision)

    # Chunks: runs that are contiguous in both sentences. One chunk means perfect
    # word order, one chunk per word means the words are right and the order is not.
    chunks = 1
    for i in range(1, len(alignment)):
        if not (
            alignment[i][0] == alignment[i - 1][0] + 1
            and alignment[i][1] == alignment[i - 1][1] + 1
        ):
            chunks += 1
    fragmentation = 0.5 * (chunks / matches) ** 3
    return f_mean * (1 - fragmentation)


def corpus_meteor(
    candidates: list[list[str]], references: list[list[list[str]]]
) -> float:
    """Mean over sentences of the best exact-match METEOR across references."""
    scores = [
        max((_meteor_single(candidate, ref) for ref in refs), default=0.0)
        for candidate, refs in zip(candidates, references)
    ]
    return float(np.mean(scores)) if scores else 0.0


def _tfidf(counts: Counter, document_frequency: Counter, num_documents: int) -> dict:
    total = sum(counts.values())
    if total == 0:
        return {}
    vector = {}
    for gram, count in counts.items():
        idf = math.log(max(num_documents, 1) / max(document_frequency.get(gram, 0), 1.0))
        vector[gram] = (count / total) * idf
    return vector


def corpus_cider(
    candidates: list[list[str]],
    references: list[list[list[str]]],
    max_n: int = 4,
    sigma: float = 6.0,
) -> float:
    """CIDEr-D over the split, using the split's own references as the corpus.

    Document frequency is counted per image, so an n-gram appearing in all five
    references of one image counts once. That is what the original implementation
    does, and it matters: without it common phrasings inside one image's references
    would look rare.
    """
    num_documents = len(references)
    document_frequency = [Counter() for _ in range(max_n + 1)]
    for refs in references:
        for n in range(1, max_n + 1):
            present = set()
            for ref in refs:
                present |= set(ngrams(ref, n))
            for gram in present:
                document_frequency[n][gram] += 1

    per_order = [[] for _ in range(max_n + 1)]
    for candidate, refs in zip(candidates, references):
        for n in range(1, max_n + 1):
            candidate_counts = ngrams(candidate, n)
            candidate_vector = _tfidf(candidate_counts, document_frequency[n], num_documents)
            candidate_norm = math.sqrt(sum(v * v for v in candidate_vector.values()))

            similarities = []
            for ref in refs:
                reference_counts = ngrams(ref, n)
                reference_vector = _tfidf(reference_counts, document_frequency[n], num_documents)
                reference_norm = math.sqrt(sum(v * v for v in reference_vector.values()))
                if candidate_norm == 0 or reference_norm == 0:
                    similarities.append(0.0)
                    continue
                # The "-D" clipping: a repeated n-gram cannot earn more than the
                # reference uses it, which is what stops "a a a a a" from scoring.
                dot = sum(
                    min(value, reference_vector.get(gram, 0.0)) * reference_vector.get(gram, 0.0)
                    for gram, value in candidate_vector.items()
                )
                length_penalty = math.exp(-((len(candidate) - len(ref)) ** 2) / (2 * sigma**2))
                similarities.append(length_penalty * dot / (candidate_norm * reference_norm))
            per_order[n].append(float(np.mean(similarities)) if similarities else 0.0)

    means = [float(np.mean(per_order[n])) if per_order[n] else 0.0 for n in range(1, max_n + 1)]
    return 10.0 * float(np.mean(means))


def caption_metrics(
    candidates: list[list[str]], references: list[list[list[str]]]
) -> dict[str, float]:
    """Every metric in one call, on tokenised captions."""
    scores = corpus_bleu(candidates, references)
    scores["meteor_exact"] = corpus_meteor(candidates, references)
    scores["cider_d"] = corpus_cider(candidates, references)
    scores["mean_length"] = float(np.mean([len(c) for c in candidates])) if candidates else 0.0
    unique = {" ".join(candidate) for candidate in candidates}
    # Repeating one safe sentence is the classic failure mode; this makes it visible.
    scores["distinct_captions"] = len(unique) / max(len(candidates), 1)
    return scores


def plot_caption_examples(
    images: torch.Tensor,
    generated: list[str],
    references: list[str],
    path: str | Path,
    columns: int = 4,
    title: str | None = None,
) -> None:
    """Image grid with the generated caption and one reference underneath each."""
    n = min(images.size(0), len(generated), len(references))
    rows = (n + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 4.7 * rows), squeeze=False)

    for index in range(rows * columns):
        ax = axes[index // columns][index % columns]
        ax.axis("off")
        if index >= n:
            continue
        array = images[index].detach().cpu().clamp(0, 1).numpy()
        if array.shape[0] == 1:
            ax.imshow(array[0], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(np.transpose(array, (1, 2, 0)))
        ax.set_title(
            f"model: {_wrap(generated[index])}\nref:   {_wrap(references[index])}",
            fontsize=7,
            loc="left",
        )

    if title:
        fig.suptitle(title)
    # The captions live in the axes titles, which need real vertical room.
    fig.tight_layout(h_pad=3.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _wrap(text: str, width: int = 34) -> str:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\n       ".join(lines) if lines else ""


def plot_curves(history: dict, path: str | Path, keys: tuple[str, ...] = ("loss", "bleu_4")) -> None:
    present = [key for key in keys if f"train_{key}" in history or f"val_{key}" in history]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), squeeze=False)
    for ax, key in zip(axes[0], present):
        for split in ("train", "val"):
            values = history.get(f"{split}_{key}")
            if values:
                ax.plot(range(1, len(values) + 1), values, label=split, marker="o", markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bars(
    labels: list[str],
    series: dict[str, list[float]],
    path: str | Path,
    ylabel: str = "score",
    title: str | None = None,
) -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)

    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 3, 4))
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)

    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
