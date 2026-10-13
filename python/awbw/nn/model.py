"""The critic: a residual network over tiles that values a position for the player to move.

The input is ``encode_batch``'s output. Index planes (terrain, unit, ...) are
embedded and summed per tile, quantity planes (HP, fuel, ...) are scaled to
[0, 1] and projected, and the globals are projected and added to every tile.

Tiles off the board (terrain 0) are zero at every layer: blocks mask what
their convolutions spread off the board, and pooling only reads the board. So
a board gets the same value in any bucket it fits and wherever it sits in it,
which is what lets training flip and transpose whole buckets.

Every few blocks also add a summary of the whole board (masked mean and max
of some channels, as in KataGo's global pooling), since who is ahead depends
on all of it.

Each block reads its input through an RMS norm over each tile's channels
(pre-norm) and starts as the identity (its last convolution is zero). Without
the norm (``prenorm=False``, the first critics) activations grew about tenfold
per block and the trained network ignored the board. BatchNorm would make a
value depend on the rest of its batch and count the padding in its
statistics; a per-tile LayerNorm more than doubled the step time on Apple
GPUs, while the RMS norm costs nothing once compiled. A LayerNorm sits before
the head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .. import _core

TERRAIN = 0  # plane index; terrain 0 is off the board
LOSS, DRAW, WIN = 0, 1, 2  # output classes, for the player to move
_FAR = 1e4  # subtracted off the board before a max; fits in float16


@dataclass(frozen=True)
class CriticConfig:
    channels: int = 64
    blocks: int = 6
    pool_every: int = 2  # every Nth block mixes in a summary of the board
    pool_channels: int = 32
    head_channels: int = 32
    head_hidden: int = 128
    prenorm: bool = True  # RMS-normalize each block's input


def board_mask(planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1 on the board's tiles [B, 1, S, S], and each board's tile count [B, 1]."""
    mask = (planes[:, TERRAIN : TERRAIN + 1] != 0).float()
    return mask, mask.sum((2, 3)).clamp_min(1)


def masked_pool(x: torch.Tensor, mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """[B, 3C]: per channel, the mean over the board, the mean scaled by board size, and the max."""
    mean = (x * mask).sum((2, 3)) / area
    size = (area.sqrt() - 20) / 10  # 0 for a 20x20 board
    peak = (x - (1 - mask) * _FAR).amax((2, 3))
    return torch.cat([mean, mean * size, peak], 1)


class ChannelNorm(nn.LayerNorm):
    """LayerNorm over the channels of each tile of [B, C, S, S]."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TileNorm(nn.Module):
    """RMS norm over the channels of each tile of [B, C, S, S], with a learned gain; zero stays zero."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))  # 1-d: no weight decay
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps) * self.weight.view(1, -1, 1, 1)


class Embed(nn.Module):
    """Planes and globals to [B, C, S, S] features, zero off the board."""

    def __init__(self, channels: int):
        super().__init__()
        planes = [(int(n), bool(cat)) for _, n, cat in _core.PLANES]
        cat = [i for i, (_, c) in enumerate(planes) if c]
        qty = [i for i, (_, c) in enumerate(planes) if not c]
        sizes = [planes[i][0] for i in cat]
        offsets = [sum(sizes[:j]) for j in range(len(sizes))]
        self.register_buffer("cat_planes", torch.tensor(cat), persistent=False)
        self.register_buffer("qty_planes", torch.tensor(qty), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets).view(1, -1, 1, 1), persistent=False)
        scale = [1.0 / max(planes[i][0] - 1, 1) for i in qty]
        self.register_buffer("qty_scale", torch.tensor(scale).view(1, -1, 1, 1), persistent=False)
        self.table = nn.Embedding(sum(sizes), channels)
        nn.init.normal_(self.table.weight, std=0.5)
        self.qty = nn.Conv2d(len(qty), channels, 1, bias=False)
        self.globals = nn.Linear(_core.NUM_GLOBALS, channels)

    def forward(self, planes: torch.Tensor, globals_: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        idx = planes.index_select(1, self.cat_planes).long() + self.offsets
        x = self.table(idx[:, 0])
        for j in range(1, idx.shape[1]):
            x = x + self.table(idx[:, j])
        # [B, C, S, S], copied out of channels-last: on Apple GPUs a channels-last
        # convolution's backward pass is about 3x slower.
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x + self.qty(planes.index_select(1, self.qty_planes).float() * self.qty_scale)
        x = x + self.globals(globals_)[:, :, None, None]
        return x * mask


class Block(nn.Module):
    """Pre-activation residual block; with ``pool_channels``, conv1's output is
    biased by a summary of the board. Expects x zero off the board, so that
    silu(x) is too."""

    def __init__(self, channels: int, pool_channels: int = 0, prenorm: bool = True):
        super().__init__()
        self.norm = TileNorm(channels) if prenorm else nn.Identity()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)  # each block starts as the identity
        assert self.conv2.bias is not None
        nn.init.zeros_(self.conv2.bias)
        self.pool_channels = pool_channels
        self.pool = nn.Linear(3 * pool_channels, channels) if pool_channels else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm(x)))
        if self.pool is not None:
            g = F.silu(h[:, : self.pool_channels])
            h = h + self.pool(masked_pool(g, mask, area))[:, :, None, None]
        h = self.conv2(F.silu(h) * mask)
        return x + h * mask


class ValueHead(nn.Module):
    """Win/draw/loss logits from the board summary and the globals."""

    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.norm = ChannelNorm(cfg.channels)
        self.conv = nn.Conv2d(cfg.channels, cfg.head_channels, 1)
        self.fc1 = nn.Linear(3 * cfg.head_channels + _core.NUM_GLOBALS, cfg.head_hidden)
        self.fc2 = nn.Linear(cfg.head_hidden, 3)

    def forward(self, x: torch.Tensor, globals_: torch.Tensor, mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv(F.silu(self.norm(x))))  # pooling reads only the board
        summary = torch.cat([masked_pool(h, mask, area), globals_], 1)
        return self.fc2(F.silu(self.fc1(summary)))


class CriticNet(nn.Module):
    """Encoded states to win/draw/loss logits [B, 3] (LOSS, DRAW, WIN) for the player to move."""

    def __init__(self, cfg: CriticConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or CriticConfig()
        self.embed = Embed(cfg.channels)
        self.blocks = nn.ModuleList(
            Block(cfg.channels, cfg.pool_channels if (i + 1) % cfg.pool_every == 0 else 0, cfg.prenorm)
            for i in range(cfg.blocks)
        )
        self.value = ValueHead(cfg)

    def forward(self, planes: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        mask, area = board_mask(planes)
        x = self.embed(planes, globals_, mask)
        for block in self.blocks:
            x = block(x, mask, area)
        return self.value(x, globals_, mask, area)


def value_of(logits: torch.Tensor) -> torch.Tensor:
    """Expected result in [-1, 1] for the player to move: P(win) - P(loss)."""
    p = logits.float().softmax(-1)
    return p[:, WIN] - p[:, LOSS]
