"""ViT-Tiny, and a ResNet of comparable size to compare it against.

    python model.py

A convolution has two assumptions built into it: nearby pixels belong together, and a
feature that matters in one place matters everywhere. Those assumptions are correct for
images, and they are why a small CNN learns from a small dataset.

A Vision Transformer throws both away. It cuts the image into patches, embeds each one,
and runs plain self-attention - every patch may look at every other patch from the first
layer, and nothing tells it which patches are adjacent except a position embedding it
has to learn. That is strictly more general and strictly harder to learn, which is the
whole story of ViT on small datasets: it loses to a ResNet of the same size on CIFAR-10
unless you either pretrain it or work much harder on augmentation.

This project is built to show that honestly rather than to pick a winner. Both models
are here, the comparison is at matched parameter count *and* matched wall-clock, and
``compare.py --study augment`` measures how much of the gap augmentation closes.

The attention maps are the compensation. ``attention_rollout`` multiplies the attention
matrices through the depth of the network to show which patches the class token actually
depends on - a form of interpretability a CNN does not hand you for free.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHITECTURES = ("vit", "resnet")


# --------------------------------------------------------------------------
# Vision Transformer
# --------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention that can return its attention matrix."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.project = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = attention.softmax(dim=-1)
        out = (self.dropout(attention) @ v).transpose(1, 2).reshape(batch, tokens, dim)
        out = self.project(out)
        return (out, attention) if return_attention else (out, None)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        out, attention = self.attn(self.norm1(x), return_attention)
        x = x + self.dropout(out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, attention


class VisionTransformer(nn.Module):
    """ViT-Tiny scaled for small images: patch embed, class token, learned positions."""

    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError(f"image size {image_size} must divide by patch {patch_size}")
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid**2

        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        # Learned, not sinusoidal: with a class token in position 0 and only a few
        # hundred patches, a learned table is standard and costs almost nothing.
        self.position = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(x.size(0), -1, -1)
        return self.dropout(torch.cat([cls, x], dim=1) + self.position)

    def forward(self, images: torch.Tensor, return_attention: bool = False):
        x = self.tokens(images)
        attentions = []
        for block in self.blocks:
            x, attention = block(x, return_attention)
            if return_attention:
                attentions.append(attention)
        logits = self.head(self.norm(x)[:, 0])
        return (logits, attentions) if return_attention else logits

    @torch.no_grad()
    def attention_rollout(self, images: torch.Tensor) -> torch.Tensor:
        """Which patches the class token depends on, ``(N, grid, grid)`` in ``[0, 1]``.

        Attention in one layer says where that layer looked. Composing the layers -
        with the residual path added in as an identity, then row-normalised - gives
        what the class token at the top depends on in the *input* patches. This is
        Abnar and Zuidema's rollout, and it is the standard first look inside a ViT.
        """
        _, attentions = self(images, return_attention=True)
        rollout = None
        for attention in attentions:
            head_average = attention.mean(dim=1)  # average the heads
            identity = torch.eye(head_average.size(-1), device=head_average.device)
            augmented = head_average + identity  # the residual connection
            augmented = augmented / augmented.sum(dim=-1, keepdim=True)
            rollout = augmented if rollout is None else augmented @ rollout

        # Row 0 is the class token; drop its self-attention and reshape to the grid.
        maps = rollout[:, 0, 1:].reshape(-1, self.grid, self.grid)
        flat = maps.flatten(1)
        low = flat.min(dim=1).values.view(-1, 1, 1)
        high = flat.max(dim=1).values.view(-1, 1, 1)
        return (maps - low) / (high - low).clamp_min(1e-8)


# --------------------------------------------------------------------------
# the convolutional control
# --------------------------------------------------------------------------


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class SmallResNet(nn.Module):
    """CIFAR-style ResNet.

    ``width=64`` is chosen so the parameter count lands within 3% of the default ViT
    (2.78M against 2.69M). Matching parameters is the weaker of the two controls, so
    ``compare.py`` reports wall-clock as well - two models of equal size can differ by
    a factor in what they cost to train.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        width: int = 64,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        stages, channels = [], width
        for stage, stride in enumerate((1, 2, 2)):
            out_channels = width * 2**stage
            for block in range(blocks_per_stage):
                stages.append(BasicBlock(channels, out_channels, stride if block == 0 else 1))
                channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.stages(self.stem(images))).flatten(1))


def build_model(
    architecture: str = "vit",
    num_classes: int = 10,
    image_size: int = 32,
    in_channels: int = 3,
    patch_size: int = 4,
    dim: int = 192,
    depth: int = 6,
    heads: int = 3,
    dropout: float = 0.1,
    width: int = 64,
) -> nn.Module:
    if architecture == "vit":
        return VisionTransformer(
            num_classes, image_size, patch_size, in_channels, dim, depth, heads, 4.0, dropout
        )
    if architecture == "resnet":
        return SmallResNet(num_classes, in_channels, width)
    raise ValueError(f"unknown architecture {architecture!r}, expected {ARCHITECTURES}")


if __name__ == "__main__":
    images = torch.rand(4, 3, 32, 32)
    for architecture in ARCHITECTURES:
        model = build_model(architecture, 10, 32, 3)
        logits = model(images)
        print(f"{architecture:7s}: logits {tuple(logits.shape)}  "
              f"params {sum(p.numel() for p in model.parameters()):,}")

    vit = build_model("vit", 10, 32, 3, patch_size=4)
    print(f"\nViT patches: {vit.grid}x{vit.grid} = {vit.num_patches} plus a class token")
    rollout = vit.attention_rollout(images)
    print(f"attention rollout: {tuple(rollout.shape)} in "
          f"[{rollout.min():.2f}, {rollout.max():.2f}]")

    # A ViT is permutation-equivariant without its position embedding: zero it out and
    # shuffling the patches must stop changing the answer.
    vit.eval()
    with torch.no_grad():
        baseline = vit(images)
        vit.position.zero_()
        plain = vit(images)
        shuffled = vit.tokens(images)
        order = torch.randperm(shuffled.size(1) - 1) + 1
        shuffled = torch.cat([shuffled[:, :1], shuffled[:, order]], dim=1)
        x = shuffled
        for block in vit.blocks:
            x, _ = block(x)
        shuffled_logits = vit.head(vit.norm(x)[:, 0])
    print(f"without position embeddings, patch order stops mattering: "
          f"{bool(torch.allclose(plain, shuffled_logits, atol=1e-4))}")
    print(f"with them, it does matter: {not bool(torch.allclose(baseline, plain, atol=1e-4))}")
