"""Vision Transformer (ViT-Tiny), and a ResNet-18-style CNN to compare against.

A ViT throws away almost every assumption a CNN builds in.  An image is cut into
fixed patches, each patch is linearly projected into a token, a learned position
embedding is added, and a standard transformer encoder does the rest.  There is
no locality prior, no weight sharing across space, no pooling hierarchy.

That is the whole point and also the whole problem.  Self-attention lets any
patch talk to any other from the first layer, which is more expressive than a
convolution's local window.  But convolutions encode *translation equivariance*
for free, and a ViT has to learn it from data.  On ImageNet-scale datasets the
extra expressiveness wins; on CIFAR-10 from scratch, it usually does not -- the
CNN's built-in prior is worth more than the transformer's flexibility.

Running both at matched compute, which ``evaluate.py`` does, is the point of the
project. The interesting result is often that the ResNet wins, and understanding
why is more valuable than a leaderboard number.

Two tricks make small-data ViT training survivable and are included here:

* **stochastic depth** -- randomly skip residual branches during training;
* **strong augmentation** (RandAugment-style flips/crops, plus mixup in the
  training script), because ViTs overfit small datasets aggressively.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Vision Transformer
# --------------------------------------------------------------------------- #
class PatchEmbedding(nn.Module):
    """Cut the image into patches and linearly project each one to a token.

    Implemented as a strided convolution, which is exactly a per-patch linear
    map with no overlap.
    """

    def __init__(self, image_size: int, patch_size: int, channels: int, dim: int):
        super().__init__()
        if image_size % patch_size:
            raise ValueError(f"patch_size {patch_size} must divide image_size {image_size}")
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(channels, dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, dim)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch for a sample.

    Regularises deep transformers far more effectively than dropout alone, and
    costs nothing at inference.
    """

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: norm -> attention -> add, norm -> MLP -> add."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class VisionTransformer(nn.Module):
    """ViT-Tiny: 12 layers wide 192, 3 heads -- scaled down for small images.

    The default ``patch_size=4`` suits 32x32 inputs: a 16x16 grid of tokens.
    The original ViT's patch size of 16 would leave a 32x32 image with four
    tokens, which is far too coarse.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        channels: int = 3,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, channels, dim)
        num_patches = self.patch_embed.num_patches

        # The class token aggregates information for the final prediction; the
        # position embedding is learned because patch order carries no meaning
        # to a permutation-invariant attention layer.
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Stochastic depth increases linearly with depth, as in the paper.
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout, rates[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        cls = self.class_token.expand(tokens.shape[0], -1, -1)
        h = self.dropout(torch.cat([cls, tokens], dim=1) + self.position_embedding)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h)[:, 0])  # classify from the class token

    @torch.no_grad()
    def attention_rollout(self, x: torch.Tensor) -> torch.Tensor:
        """Attention from the class token to each patch, averaged over heads.

        A crude but useful saliency map: it shows which patches the final
        prediction actually drew on.
        """
        tokens = self.patch_embed(x)
        cls = self.class_token.expand(tokens.shape[0], -1, -1)
        h = torch.cat([cls, tokens], dim=1) + self.position_embedding

        rollout = None
        for block in self.blocks:
            normed = block.norm1(h)
            b, n, d = normed.shape
            attn_module = block.attn
            qkv = (
                attn_module.qkv(normed)
                .reshape(b, n, 3, attn_module.heads, attn_module.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, _ = qkv.unbind(0)
            attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(attn_module.head_dim), dim=-1)
            attn = attn.mean(dim=1)  # average the heads
            # Account for the residual stream, then propagate through layers.
            attn = attn + torch.eye(n, device=attn.device).unsqueeze(0)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            rollout = attn if rollout is None else attn @ rollout
            h = block(h)

        grid = int(math.sqrt(tokens.shape[1]))
        return rollout[:, 0, 1:].reshape(-1, 1, grid, grid)


# --------------------------------------------------------------------------- #
# ResNet-18 baseline
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.shortcut(x))


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18: 3x3 stem, no max-pool, four stages of two blocks.

    The standard ImageNet stem (7x7 stride 2 plus max-pool) throws away far too
    much resolution on a 32x32 input, so it is replaced here -- this is the
    usual CIFAR adaptation.
    """

    def __init__(self, channels: int = 3, num_classes: int = 10, base: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, base, 3, 1, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True)
        )
        widths = [base, base * 2, base * 4, base * 8]
        strides = [1, 2, 2, 2]
        layers = []
        in_ch = base
        for width, stride in zip(widths, strides):
            layers.append(BasicBlock(in_ch, width, stride))
            layers.append(BasicBlock(width, width, 1))
            in_ch = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(h, 1).flatten(1))


def build_model(name: str, image_size: int, channels: int, num_classes: int, **kwargs) -> nn.Module:
    if name == "vit":
        return VisionTransformer(
            image_size=image_size, channels=channels, num_classes=num_classes, **kwargs
        )
    if name == "resnet18":
        return ResNet18(channels=channels, num_classes=num_classes, base=kwargs.get("base", 64))
    raise ValueError(f"unknown model {name!r}")


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Blend pairs of images and their labels.

    ViTs lack the CNN's inductive bias and overfit small datasets quickly;
    mixup is one of the few regularisers that reliably helps.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    index = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam
