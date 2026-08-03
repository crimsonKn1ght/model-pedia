"""A small ViT, a masked autoencoder built on it, and a classifier that reuses
the same encoder.

    python model.py

The masked autoencoder (MAE) recipe is three ideas, and each one is a line or two
of code here:

1. **Mask most of the image.** 75% of patches are dropped. At that ratio the task
   cannot be solved by copying a neighbouring pixel, so the model has to learn
   something about how images are put together.
2. **The encoder never sees the mask.** Dropped patches are *removed from the
   sequence*, not replaced by a token. So the encoder runs on a quarter of the
   patches, which is why pretraining is cheap - and, more importantly, it is
   never asked to process a mask token that will not exist at fine-tuning time.
3. **The decoder is small and disposable.** It re-inserts a shared mask token at
   the right positions and predicts raw pixels. After pretraining it is thrown
   away; only the encoder is transferred.

``ViTEncoder`` is shared by ``MAE`` and ``ViTClassifier`` under the same attribute
name, so transferring the pretrained weights is a plain ``load_state_dict``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sincos_pos_embed(dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Fixed 2D sine-cosine position embedding, ``(grid_h * grid_w, dim)``.

    Fixed rather than learned: with a few thousand pretraining images a learned
    table mostly memorises, and MAE reports no benefit from learning it.
    """
    if dim % 4 != 0:
        raise ValueError(f"embedding dim must be divisible by 4, got {dim}")
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim / 4))
    rows, cols = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")

    parts = []
    for coordinate in (rows.flatten().float(), cols.flatten().float()):
        angles = coordinate[:, None] * omega[None, :]
        parts += [angles.sin(), angles.cos()]
    return torch.cat(parts, dim=1)


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """``(N, C, H, W)`` -> ``(N, L, C * patch * patch)`` in row-major patch order."""
    n, c, h, w = images.shape
    if h % patch or w % patch:
        raise ValueError(f"image {h}x{w} is not divisible by patch size {patch}")
    grid_h, grid_w = h // patch, w // patch
    x = images.reshape(n, c, grid_h, patch, grid_w, patch)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(n, grid_h * grid_w, c * patch * patch)


def unpatchify(patches: torch.Tensor, patch: int, channels: int, grid_h: int, grid_w: int):
    """Inverse of :func:`patchify`."""
    n = patches.size(0)
    x = patches.reshape(n, grid_h, grid_w, channels, patch, patch)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(n, channels, grid_h * patch, grid_w * patch)


class Block(nn.Module):
    """Pre-norm transformer block: attention, then MLP, both residual."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViTEncoder(nn.Module):
    """Patch embedding, a class token, and a stack of transformer blocks.

    ``forward`` optionally drops a random fraction of the patches *before* the
    blocks run. That is the whole trick that makes MAE pretraining cheap: with
    75% masking the attention runs on a quarter of the sequence.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError(f"image size {image_size} is not divisible by patch {patch_size}")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.grid = image_size // patch_size
        self.num_patches = self.grid**2
        self.patch_dim = in_channels * patch_size**2

        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer(
            "pos_embed", sincos_pos_embed(dim, self.grid, self.grid).unsqueeze(0), persistent=False
        )
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """Keep a random subset of tokens per sample.

        Implemented by sorting uniform noise, which gives an independent random
        permutation per image and, as a side effect, ``ids_restore`` - the
        permutation the decoder needs to put the tokens back in image order.

        Returns ``(kept_tokens, mask, ids_restore)`` where ``mask`` is 1 on the
        patches that were removed.
        """
        n, length, dim = x.shape
        keep = max(1, int(round(length * (1.0 - mask_ratio))))

        noise = torch.rand(n, length, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :keep]
        kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dim))

        mask = torch.ones(n, length, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return kept, mask, ids_restore

    def forward(self, images: torch.Tensor, mask_ratio: float = 0.0):
        """Returns ``(tokens, mask, ids_restore)``; ``tokens[:, 0]`` is the class token."""
        x = self.patch_embed(images).flatten(2).transpose(1, 2)  # (N, L, D)
        x = x + self.pos_embed

        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = torch.zeros(x.size(0), x.size(1), device=x.device)
            ids_restore = (
                torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
            )

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore


class MAE(nn.Module):
    """Masked autoencoder: the shared encoder plus a small pixel decoder."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        decoder_dim: int = 64,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        norm_pixel_loss: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ViTEncoder(image_size, patch_size, in_channels, dim, depth, heads)
        self.norm_pixel_loss = norm_pixel_loss

        self.decoder_embed = nn.Linear(dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer(
            "decoder_pos_embed",
            sincos_pos_embed(decoder_dim, self.encoder.grid, self.encoder.grid).unsqueeze(0),
            persistent=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_dim, decoder_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.encoder.patch_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward_decoder(self, tokens: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Re-insert mask tokens, un-shuffle, decode to pixels. Returns ``(N, L, patch_dim)``."""
        x = self.decoder_embed(tokens)
        length = ids_restore.size(1)
        missing = length + 1 - x.size(1)

        mask_tokens = self.mask_token.expand(x.size(0), missing, -1)
        patches = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        # Every masked position gets the *same* token; only the position embedding
        # added below tells the decoder which patch it is being asked about.
        patches = torch.gather(
            patches, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.size(2))
        )
        patches = patches + self.decoder_pos_embed

        x = torch.cat([x[:, :1, :], patches], dim=1)
        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:, :]

    def patch_targets(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patchified pixels, plus the per-patch mean and std used to normalise them."""
        target = patchify(images, self.encoder.patch_size)
        mean = target.mean(dim=-1, keepdim=True)
        std = (target.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        return target, mean, std

    def forward(self, images: torch.Tensor, mask_ratio: float = 0.75):
        """Returns ``(loss, prediction, mask)``. Loss is over masked patches only."""
        tokens, mask, ids_restore = self.encoder(images, mask_ratio)
        prediction = self.forward_decoder(tokens, ids_restore)

        target, mean, std = self.patch_targets(images)
        if self.norm_pixel_loss:
            # Predicting each patch's *normalised* pixels removes the easy part of
            # the problem (its average brightness) and measurably improves the
            # features. It also means the raw prediction is not directly viewable,
            # which is what ``reconstruct`` below deals with.
            target = (target - mean) / std

        per_patch = (prediction - target).pow(2).mean(dim=-1)
        loss = (per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, prediction, mask

    @torch.no_grad()
    def reconstruct(
        self, images: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(masked_input, reconstruction)`` as images in ``[0, 1]``.

        When ``norm_pixel_loss`` is on the decoder predicts normalised pixels, so
        the per-patch mean and std of the *original* patch are put back to make a
        picture. Those statistics are ground truth the model never saw, so treat
        the figure as a qualitative check on structure, not on absolute colour -
        the reference implementation's demo does exactly the same thing.
        """
        patch = self.encoder.patch_size
        channels = self.encoder.in_channels
        grid = self.encoder.grid

        target, mean, std = self.patch_targets(images)
        pixels = prediction * std + mean if self.norm_pixel_loss else prediction
        pixels = pixels.clamp(0, 1)

        keep = (1 - mask).unsqueeze(-1)
        filled = keep * target + (1 - keep) * pixels
        masked_input = keep * target + (1 - keep) * 0.5  # grey where the model is blind

        return (
            unpatchify(masked_input, patch, channels, grid, grid),
            unpatchify(filled, patch, channels, grid, grid),
        )


class ViTClassifier(nn.Module):
    """The same encoder with a linear head on the class token.

    ``freeze_encoder=True`` gives the linear probe, ``False`` the fine-tune. Both
    are run from a pretrained *and* a random-init encoder in ``evaluate.py``,
    because "MAE reaches X%" only means something next to "the same ViT trained
    from scratch under the same budget reaches Y%".
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = ViTEncoder(image_size, patch_size, in_channels, dim, depth, heads)
        self.head = nn.Linear(dim, num_classes)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def load_encoder(self, state_dict: dict) -> None:
        self.encoder.load_state_dict(state_dict)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                tokens, _, _ = self.encoder(images)
        else:
            tokens, _, _ = self.encoder(images)
        return self.head(tokens[:, 0])


def build_mae(
    image_size: int = 32,
    patch_size: int = 4,
    in_channels: int = 3,
    dim: int = 128,
    depth: int = 4,
    heads: int = 4,
    decoder_dim: int = 64,
    decoder_depth: int = 2,
    norm_pixel_loss: bool = True,
) -> MAE:
    return MAE(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        dim=dim,
        depth=depth,
        heads=heads,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        norm_pixel_loss=norm_pixel_loss,
    )


def build_classifier(
    num_classes: int,
    image_size: int = 32,
    patch_size: int = 4,
    in_channels: int = 3,
    dim: int = 128,
    depth: int = 4,
    heads: int = 4,
    freeze_encoder: bool = False,
) -> ViTClassifier:
    return ViTClassifier(
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        dim=dim,
        depth=depth,
        heads=heads,
        freeze_encoder=freeze_encoder,
    )


if __name__ == "__main__":
    for channels, size, patch in ((3, 32, 4), (1, 28, 4), (3, 64, 8)):
        model = build_mae(image_size=size, patch_size=patch, in_channels=channels)
        images = torch.rand(4, channels, size, size)
        loss, prediction, mask = model(images, mask_ratio=0.75)
        masked, filled = model.reconstruct(images, prediction, mask)
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.parameters()) - encoder_params
        print(
            f"C={channels} {size}x{size} patch {patch}: {model.encoder.num_patches:3d} patches, "
            f"{int(mask[0].sum().item()):3d} masked  loss {loss.item():.4f}  "
            f"recon {tuple(filled.shape)}  encoder {encoder_params:,} decoder {decoder_params:,}"
        )
        assert masked.shape == images.shape

    # Round-trip check: patchify and unpatchify must be exact inverses.
    images = torch.rand(2, 3, 32, 32)
    restored = unpatchify(patchify(images, 4), 4, 3, 8, 8)
    print("\npatchify round-trip max error:", (restored - images).abs().max().item())

    mae = build_mae()
    clf = build_classifier(10)
    clf.load_encoder(mae.encoder.state_dict())
    print("encoder transfers to the classifier:", tuple(clf(torch.rand(2, 3, 32, 32)).shape))
