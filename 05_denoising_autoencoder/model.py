"""A small U-Net for denoising, and the same network without its skip
connections as a control.

Denoising looks like a job for an autoencoder, but a plain bottleneck
autoencoder is badly suited to it: forcing the image through a narrow code
destroys exactly the fine detail you are trying to recover, so the output comes
back clean *and* blurred.

A U-Net keeps the encoder-decoder shape but adds skip connections from each
encoder level to the matching decoder level. High-frequency detail travels
straight across at full resolution, while the deeper, coarser levels decide
what is signal and what is noise. ``skips=False`` builds the same network
without those connections, so ``compare.py`` measures what they are worth.

Images are expected in ``[0, 1]``; the output layer is a sigmoid.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x 2, the standard U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDenoiser(nn.Module):
    """Three-level U-Net. Fully convolutional, so any size divisible by 4 works."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        skips: bool = True,
        predict_residual: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.skips = skips
        self.predict_residual = predict_residual

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.bottleneck = DoubleConv(c2, c3)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 * 2 if skips else c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 * 2 if skips else c1, c1)

        self.out_conv = nn.Conv2d(c1, in_channels, 1)
        self.activation = nn.Identity() if predict_residual else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)  # full resolution
        e2 = self.enc2(self.pool(e1))  # half
        b = self.bottleneck(self.pool(e2))  # quarter

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1) if self.skips else d2)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1) if self.skips else d1)

        out = self.activation(self.out_conv(d1))
        if self.predict_residual:
            # Learn the noise and subtract it: an easier target than the image,
            # because the residual is close to zero almost everywhere.
            out = (x - out).clamp(0, 1)
        return out


MODELS = {
    "unet": {"skips": True},
    "unet_noskip": {"skips": False},
}


def build_model(
    name: str = "unet",
    in_channels: int = 3,
    base_channels: int = 32,
    predict_residual: bool = False,
) -> UNetDenoiser:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    return UNetDenoiser(
        in_channels=in_channels,
        base_channels=base_channels,
        predict_residual=predict_residual,
        **MODELS[name],
    )


if __name__ == "__main__":
    for channels in (1, 3):
        for name in MODELS:
            model = build_model(name, channels)
            dummy = torch.rand(2, channels, 32, 32)
            params = sum(p.numel() for p in model.parameters())
            print(f"C={channels} {name:12s} out {tuple(model(dummy).shape)}  params {params:,}")
    # Fully convolutional: a network trained on 32x32 patches runs on anything.
    big = build_model("unet", 1)
    print("64x64 input ->", tuple(big(torch.rand(1, 1, 64, 64)).shape))
