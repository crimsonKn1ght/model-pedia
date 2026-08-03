"""Encoder-decoder captioning: a CNN or ViT-style image encoder, and a Transformer
decoder that attends over its output.

    python model.py

The shape of the problem is that an image is a grid and a caption is a sequence, so
the model has to be two things joined by attention:

**Encoder** turns the image into a short sequence of feature vectors - one per
spatial location, not one per image. Pooling to a single vector also works and is
how the first captioning models did it, but then the decoder has nothing to point
at, and "the dog on the left" becomes unreachable. ``--encoder cnn`` trains a small
convolutional encoder from scratch; ``resnet18`` and ``mobilenet_v3_small`` use
frozen ImageNet features, which is what makes real photographs tractable in minutes
on a CPU.

**Decoder** is a causal Transformer over the caption tokens with a cross-attention
layer into the encoder output. Each block does three things: look at the words so
far (masked, so it cannot read ahead), look at the image, then think. Training is
teacher-forced - the decoder is given the true previous words - which is why the
loss is cheap and why generation is the expensive part.

Generation offers greedy and beam search. Greedy takes the best word at every step
and is what most quick implementations do; beam search keeps several partial
sentences alive and usually gains a point or two of BLEU, which ``evaluate.py``
measures rather than assumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ENCODERS = {
    "cnn": "small convolutional encoder, trained from scratch",
    "resnet18": "frozen ImageNet ResNet-18 features",
    "mobilenet_v3_small": "frozen ImageNet MobileNetV3-Small features",
}


class CNNEncoder(nn.Module):
    """Four stride-2 blocks; output is a ``/16`` grid flattened into tokens."""

    def __init__(self, in_channels: int = 3, width: int = 32) -> None:
        super().__init__()
        channels = [in_channels, width, width * 2, width * 4, width * 8]
        blocks = []
        for index in range(4):
            blocks += [
                nn.Conv2d(channels[index], channels[index + 1], 3, 2, 1, bias=False),
                nn.BatchNorm2d(channels[index + 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[index + 1], channels[index + 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels[index + 1]),
                nn.ReLU(inplace=True),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.blocks(images)
        return features.flatten(2).transpose(1, 2)  # (N, HW, C)


class FrozenBackbone(nn.Module):
    """ImageNet backbone with its classifier removed, kept in eval mode.

    Normalisation happens here rather than in the data pipeline, so the rest of the
    project can keep images in ``[0, 1]`` and the two encoder families stay
    interchangeable on the command line.
    """

    def __init__(self, name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        from torchvision import models

        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.body = nn.Sequential(*list(backbone.children())[:-2])
            self.out_channels = 512
        elif name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.body = models.mobilenet_v3_small(weights=weights).features
            self.out_channels = 576
        else:
            raise ValueError(f"unknown backbone {name!r}")

        for parameter in self.body.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Frozen means frozen: batch-norm statistics must not drift either.
        super().train(mode)
        self.body.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            normalised = (images - self.mean) / self.std
            features = self.body(normalised)
        return features.flatten(2).transpose(1, 2)


class DecoderBlock(nn.Module):
    """Masked self-attention, cross-attention into the image, then an MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, causal_mask: torch.Tensor):
        h = self.norm_self(x)
        x = x + self.dropout(
            self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0]
        )
        h = self.norm_cross(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, need_weights=False)[0])
        return x + self.dropout(self.mlp(self.norm_mlp(x)))


class TransformerCaptionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        depth: int = 3,
        heads: int = 4,
        max_len: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.position_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList([DecoderBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)
        nn.init.trunc_normal_(self.position_embed, std=0.02)

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        if length > self.max_len:
            raise ValueError(f"sequence of {length} exceeds max_len {self.max_len}")

        x = self.token_embed(tokens) + self.position_embed[:, :length]
        # True means "not allowed to attend": position i may not see anything after i.
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, memory, causal)
        return self.output(self.norm(x))


class CaptionModel(nn.Module):
    """Encoder plus decoder, with the projection that makes their widths agree."""

    def __init__(
        self,
        encoder: nn.Module,
        vocab_size: int,
        bos_index: int,
        eos_index: int,
        dim: int = 256,
        depth: int = 3,
        heads: int = 4,
        max_len: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.project = nn.Linear(encoder.out_channels, dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.decoder = TransformerCaptionDecoder(vocab_size, dim, depth, heads, max_len, dropout)
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.max_len = max_len

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.memory_norm(self.project(self.encoder(images)))

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher forcing: ``tokens`` are the inputs, the caller shifts the targets."""
        return self.decoder(tokens, self.encode(images))

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        beam_size: int = 1,
        max_len: int | None = None,
        length_penalty: float = 0.7,
    ) -> list[list[int]]:
        """Return token ids per image, without ``<bos>``/``<eos>``."""
        self.eval()
        max_len = min(max_len or self.max_len, self.max_len)
        memory = self.encode(images)
        if beam_size <= 1:
            return self._greedy(memory, max_len)
        return [
            self._beam(memory[index : index + 1], beam_size, max_len, length_penalty)
            for index in range(memory.size(0))
        ]

    def _greedy(self, memory: torch.Tensor, max_len: int) -> list[list[int]]:
        """One forward per step for the whole batch - the cheap decoder."""
        n = memory.size(0)
        tokens = torch.full((n, 1), self.bos_index, dtype=torch.long, device=memory.device)
        done = torch.zeros(n, dtype=torch.bool, device=memory.device)

        for _ in range(max_len - 1):
            logits = self.decoder(tokens, memory)[:, -1]
            next_token = logits.argmax(dim=-1)
            # Once a sentence has ended, keep feeding <eos> so the batch stays square.
            next_token = torch.where(done, torch.full_like(next_token, self.eos_index), next_token)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            done |= next_token == self.eos_index
            if bool(done.all()):
                break

        return [self._strip(row.tolist()) for row in tokens]

    def _beam(
        self, memory: torch.Tensor, beam_size: int, max_len: int, length_penalty: float
    ) -> list[int]:
        """Beam search for one image, with the beams batched together."""
        device = memory.device
        memory = memory.expand(beam_size, -1, -1)
        sequences = torch.full((beam_size, 1), self.bos_index, dtype=torch.long, device=device)
        # Only the first beam is alive at step 0, or all beams would expand the same
        # prefix and the search would collapse to greedy with duplicates.
        scores = torch.full((beam_size,), -torch.inf, device=device)
        scores[0] = 0.0
        finished: list[tuple[float, list[int]]] = []

        for _ in range(max_len - 1):
            logits = self.decoder(sequences, memory)[:, -1]
            log_probabilities = F.log_softmax(logits, dim=-1) + scores.unsqueeze(1)
            vocab = log_probabilities.size(1)

            top_scores, top_indices = log_probabilities.view(-1).topk(beam_size)
            beam_index = torch.div(top_indices, vocab, rounding_mode="floor")
            token_index = top_indices % vocab
            sequences = torch.cat([sequences[beam_index], token_index.unsqueeze(1)], dim=1)
            scores = top_scores.clone()

            for slot in range(beam_size):
                if int(token_index[slot]) == self.eos_index:
                    tokens = self._strip(sequences[slot].tolist())
                    # Normalise by length, or the search always prefers short captions.
                    normaliser = max(len(tokens), 1) ** length_penalty
                    finished.append((float(scores[slot]) / normaliser, tokens))
                    scores[slot] = -torch.inf

            if bool(torch.isinf(scores).all()):
                break

        if finished:
            return max(finished, key=lambda item: item[0])[1]
        return self._strip(sequences[int(scores.argmax())].tolist())

    def _strip(self, ids: list[int]) -> list[int]:
        out = []
        for token in ids[1:]:  # drop <bos>
            if token == self.eos_index:
                break
            out.append(int(token))
        return out


def build_encoder(name: str, in_channels: int = 3, width: int = 32, pretrained: bool = True):
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}, expected one of {sorted(ENCODERS)}")
    if name == "cnn":
        return CNNEncoder(in_channels, width)
    try:
        return FrozenBackbone(name, pretrained=pretrained)
    except Exception as error:  # noqa: BLE001 - the cause is almost always the network
        raise RuntimeError(
            f"could not build {name} with pretrained weights ({error}).\n"
            "torchvision fetches them from download.pytorch.org on first use; check the "
            "connection, or use --encoder cnn to train an encoder from scratch."
        ) from error


def build_model(
    vocab_size: int,
    bos_index: int,
    eos_index: int,
    encoder_name: str = "cnn",
    in_channels: int = 3,
    encoder_width: int = 32,
    pretrained: bool = True,
    dim: int = 256,
    depth: int = 3,
    heads: int = 4,
    max_len: int = 24,
    dropout: float = 0.1,
) -> CaptionModel:
    encoder = build_encoder(encoder_name, in_channels, encoder_width, pretrained)
    return CaptionModel(
        encoder, vocab_size, bos_index, eos_index, dim, depth, heads, max_len, dropout
    )


if __name__ == "__main__":
    vocab_size, bos, eos, max_len = 40, 1, 2, 12
    images = torch.rand(3, 3, 64, 64)
    tokens = torch.randint(4, vocab_size, (3, 8))
    tokens[:, 0] = bos

    model = build_model(vocab_size, bos, eos, "cnn", max_len=max_len)
    logits = model(images, tokens)
    memory = model.encode(images)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"cnn encoder : memory {tuple(memory.shape)} (tokens x width)")
    print(f"              logits {tuple(logits.shape)}  trainable {trainable:,}")

    greedy = model.generate(images, beam_size=1, max_len=max_len)
    beam = model.generate(images, beam_size=3, max_len=max_len)
    print(f"greedy      : lengths {[len(ids) for ids in greedy]}")
    print(f"beam 3      : lengths {[len(ids) for ids in beam]}")

    # The causal mask must actually hide the future: changing a later input token
    # cannot be allowed to change an earlier position's logits.
    model.eval()
    altered = tokens.clone()
    altered[:, -1] = (altered[:, -1] + 1) % vocab_size
    with torch.no_grad():
        difference = (model(images, tokens)[:, :-1] - model(images, altered)[:, :-1]).abs().max()
    print(f"causal mask : max change at earlier positions {difference.item():.2e}")
