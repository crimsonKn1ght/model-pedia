"""A small ResNet encoder, and the two self-supervised objectives wrapped around
it: SimCLR (contrastive) and BYOL (no negatives at all).

    python model.py

Both methods share the same encoder and the same two-view input, and differ only
in what they do with the two embeddings:

**SimCLR** pulls the two views of an image together and pushes them away from
every other image in the batch. The negatives are what stop it from collapsing
to a constant, which is also why it wants a large batch - the batch *is* the
negative set.

**BYOL** has no negatives. One view goes through the online network and a
predictor; the other goes through a *target* network that is an exponential
moving average of the online one. The online network is trained to predict the
target's embedding, and collapse is avoided by the asymmetry alone: the target
lags behind and receives no gradient, so there is no shared shortcut for the two
branches to agree on. It is a genuinely surprising result, and it works.

Everything below the projection head is what gets kept. The head is discarded
after pretraining - features from the layer beneath it consistently probe better,
because the head is trained to throw away exactly the information the
augmentations were supposed to make irrelevant.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 convolutions and an identity shortcut, as in a CIFAR ResNet."""

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
    """Three stages of width ``w, 2w, 4w`` with a global average pool on top.

    Deliberately small - the point of the project is the training signal, not the
    backbone, and a bigger encoder only makes the CPU run longer. ``width=32``
    gives a 128-dimensional feature and about 1.2M parameters.
    """

    def __init__(self, in_channels: int = 3, width: int = 32, blocks_per_stage: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        stages = []
        channels = width
        for stage, stride in enumerate((1, 2, 2)):
            out_channels = width * 2**stage
            for block in range(blocks_per_stage):
                stages.append(BasicBlock(channels, out_channels, stride if block == 0 else 1))
                channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.stages(self.stem(x))).flatten(1)


class ProjectionHead(nn.Module):
    """MLP with a batch norm in the middle; used by both methods.

    BYOL in particular relies on this batch norm - without it the online and
    target branches can agree on a constant, and the loss happily goes to zero
    while the features carry nothing.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """SimCLR's normalised temperature-scaled cross entropy.

    Both inputs are L2-normalised ``(N, D)`` embeddings. The ``2N`` views form
    one classification problem: for each view, pick its partner out of the other
    ``2N - 1`` views. Written as a cross entropy over the similarity matrix with
    the diagonal masked out, which is all the original loss actually is.
    """
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    similarity = (z @ z.t()) / temperature
    similarity.fill_diagonal_(-torch.inf)  # a view is not its own positive
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(similarity, targets)


class SimCLR(nn.Module):
    def __init__(
        self,
        encoder: SmallResNet,
        proj_hidden: int = 256,
        proj_dim: int = 64,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(encoder.feature_dim, proj_hidden, proj_dim)
        self.temperature = temperature

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.encoder(x)), dim=1)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        return nt_xent_loss(self.embed(view1), self.embed(view2), self.temperature)

    def update_target(self, *_args) -> None:
        """No target network; kept so the training loop treats both methods alike."""


class BYOL(nn.Module):
    """Bootstrap Your Own Latent: predict a lagging copy of yourself.

    ``momentum`` is ramped from its base value to 1.0 over training in the paper.
    Doing the same here matters more than it looks: early on the target has to
    move quickly to be worth predicting, and late on it has to be stable or the
    prediction target keeps sliding out from under the online network.
    """

    def __init__(
        self,
        encoder: SmallResNet,
        proj_hidden: int = 256,
        proj_dim: int = 64,
        pred_hidden: int = 128,
        momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(encoder.feature_dim, proj_hidden, proj_dim)
        self.predictor = ProjectionHead(proj_dim, pred_hidden, proj_dim)
        self.base_momentum = momentum

        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for parameter in list(self.target_encoder.parameters()) + list(
            self.target_projector.parameters()
        ):
            parameter.requires_grad_(False)

    def _online(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.projector(self.encoder(x)))

    @torch.no_grad()
    def _target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_projector(self.target_encoder(x))

    @torch.no_grad()
    def update_target(self, progress: float = 0.0) -> None:
        """EMA step. ``progress`` runs 0 -> 1 over the whole run."""
        momentum = 1.0 - (1.0 - self.base_momentum) * (1.0 + torch.cos(torch.pi * torch.tensor(progress))) / 2
        momentum = float(momentum)
        pairs = [
            (self.target_encoder, self.encoder),
            (self.target_projector, self.projector),
        ]
        for target, online in pairs:
            for target_param, online_param in zip(target.parameters(), online.parameters()):
                target_param.mul_(momentum).add_(online_param.detach(), alpha=1.0 - momentum)
            # Buffers (batch-norm statistics) are copied rather than averaged.
            for target_buffer, online_buffer in zip(target.buffers(), online.buffers()):
                target_buffer.copy_(online_buffer)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        # Symmetric: each view predicts the other, which doubles the signal per
        # step for one extra forward pass.
        loss = _byol_loss(self._online(view1), self._target(view2))
        loss = loss + _byol_loss(self._online(view2), self._target(view1))
        return loss.mean()


def _byol_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``2 - 2 cos`` between prediction and (detached) target."""
    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target.detach(), dim=1)
    return 2 - 2 * (prediction * target).sum(dim=1)


METHODS = {
    "simclr": "contrastive, needs in-batch negatives",
    "byol": "predictive, no negatives, EMA target network",
}


def build_encoder(in_channels: int = 3, width: int = 32, blocks_per_stage: int = 2) -> SmallResNet:
    return SmallResNet(in_channels=in_channels, width=width, blocks_per_stage=blocks_per_stage)


def build_model(
    method: str = "simclr",
    in_channels: int = 3,
    width: int = 32,
    blocks_per_stage: int = 2,
    proj_hidden: int = 256,
    proj_dim: int = 64,
    temperature: float = 0.5,
    momentum: float = 0.99,
) -> nn.Module:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {sorted(METHODS)}")
    encoder = build_encoder(in_channels, width, blocks_per_stage)
    if method == "simclr":
        return SimCLR(encoder, proj_hidden, proj_dim, temperature)
    return BYOL(encoder, proj_hidden, proj_dim, proj_hidden // 2, momentum)


if __name__ == "__main__":
    for channels, size in ((3, 32), (1, 28)):
        view1 = torch.rand(8, channels, size, size)
        view2 = torch.rand(8, channels, size, size)
        for method in METHODS:
            model = build_model(method, in_channels=channels)
            loss = model(view1, view2)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"C={channels} {size}x{size} {method:7s} loss {loss.item():6.3f}  "
                f"feature {model.encoder.feature_dim:4d}  trainable {trainable:,}"
            )

    encoder = build_encoder(3, 32)
    print("\nencoder only:", sum(p.numel() for p in encoder.parameters()), "parameters")
    print("features    :", tuple(encoder(torch.rand(2, 3, 64, 64)).shape), "for a 64x64 input")
