"""Networks over tiles: the critic, which values a position for the player to move,
and the policy-value network, which also scores the micro-steps of a decision.

The input is ``encode_batch``'s output. Index planes (terrain, unit, ...) are
embedded and summed per tile, quantity planes (HP, fuel, ...) are scaled to
[0, 1] and projected, and the globals are projected and added to every tile.

Tiles off the board (terrain 0) are zero at every layer: blocks mask what
their convolutions spread off the board, and pooling only reads the board. So
a board gets the same value in any bucket it fits and wherever it sits in it,
which is what lets training flip and transpose whole buckets.

With ``ratings`` the globals also get the players' ratings (``rating_features``):
whether they are known, their mean and the mover's lead. Training hides them
at random (``TrainConfig.rating_dropout``), so the same network values a
position with ratings, for two given players, or without them, for players
of unknown strength. Self-play can give both sides the same rating.

Every few blocks also add a summary of the whole board (masked mean and max
of some channels, as in KataGo's global pooling), since who is ahead depends
on all of it.

``PolicyValueNet`` has the same trunk and value head, reads the decision in
progress too (``encode_policy_batch``: the pick so far, the phase and one
plane per channel marking where its step is legal), and adds a policy head:
a logit per (channel, tile) from each tile's features and one per non-spatial
id (End Turn, the powers, Build + type) from the board summary, laid out as
``_core.policy_index`` says. With ``PolicyConfig.movement`` it also reads the
movement planes (turns from factories and to reach tiles, core/encode.h), and
with ``PolicyConfig.aux`` a head like the value head predicts what happens
next (``loader.aux_targets``): only training reads it.

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

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .. import _core
from ..data.loader import AUX_BINARY, AUX_REGRESSION

TERRAIN = 0  # plane index; terrain 0 is off the board
LOSS, DRAW, WIN = 0, 1, 2  # output classes, for the player to move
_FAR = 1e4  # subtracted off the board before a max; fits in float16
RATING_CENTER, RATING_SCALE = 900, 200  # Global League ratings: 5% below 706, 5% above 1,182
NUM_RATING_FEATURES = 3


@dataclass(frozen=True)
class CriticConfig:
    channels: int = 64
    blocks: int = 6
    pool_every: int = 2  # every Nth block mixes in a summary of the board
    pool_channels: int = 32
    head_channels: int = 32
    head_hidden: int = 128
    prenorm: bool = True  # RMS-normalize each block's input
    ratings: bool = True  # read the players' ratings


def rating_features(rating: torch.Tensor | None, n: int, device: torch.device) -> torch.Tensor:
    """[N, NUM_RATING_FEATURES] from ratings [N, 2] (the player to move's, then the
    opponent's; negative = unknown) or None: 1 if both are known, then their scaled
    mean and the mover's scaled lead; all 0 unless both are known."""
    if rating is None:
        return torch.zeros(n, NUM_RATING_FEATURES, device=device)
    r = rating.to(device=device, dtype=torch.float32)
    known = (r >= 0).all(1, keepdim=True).float()
    mean = (r.mean(1, keepdim=True) - RATING_CENTER) / RATING_SCALE
    lead = (r[:, :1] - r[:, 1:]) / RATING_SCALE
    return torch.cat([known, mean, lead], 1) * known


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
    """Planes and globals to [B, C, S, S] features, zero off the board. ``planes``:
    (name, number of values, categorical) of each input plane."""

    def __init__(self, channels: int, num_globals: int = _core.NUM_GLOBALS, planes: tuple | None = None):
        super().__init__()
        spec = [(int(n), bool(c)) for _, n, c in (planes or _core.PLANES)]
        cat = [i for i, (_, c) in enumerate(spec) if c]
        qty = [i for i, (_, c) in enumerate(spec) if not c]
        sizes = [spec[i][0] for i in cat]
        offsets = [sum(sizes[:j]) for j in range(len(sizes))]
        self.register_buffer("cat_planes", torch.tensor(cat), persistent=False)
        self.register_buffer("qty_planes", torch.tensor(qty), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets).view(1, -1, 1, 1), persistent=False)
        scale = [1.0 / max(spec[i][0] - 1, 1) for i in qty]
        self.register_buffer("qty_scale", torch.tensor(scale).view(1, -1, 1, 1), persistent=False)
        self.table = nn.Embedding(sum(sizes), channels)
        nn.init.normal_(self.table.weight, std=0.5)
        self.qty = nn.Conv2d(len(qty), channels, 1, bias=False)
        self.globals = nn.Linear(num_globals, channels)

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
    """Win/draw/loss logits (or ``outputs`` others) from the board summary and the globals."""

    def __init__(self, cfg: CriticConfig | PolicyConfig, num_globals: int = _core.NUM_GLOBALS, outputs: int = 3):
        super().__init__()
        self.norm = ChannelNorm(cfg.channels)
        self.conv = nn.Conv2d(cfg.channels, cfg.head_channels, 1)
        self.fc1 = nn.Linear(3 * cfg.head_channels + num_globals, cfg.head_hidden)
        self.fc2 = nn.Linear(cfg.head_hidden, outputs)

    def forward(self, x: torch.Tensor, globals_: torch.Tensor, mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv(F.silu(self.norm(x))))  # pooling reads only the board
        summary = torch.cat([masked_pool(h, mask, area), globals_], 1)
        return self.fc2(F.silu(self.fc1(summary)))


class CriticNet(nn.Module):
    """Encoded states, and optionally the players' ratings, to win/draw/loss logits
    [B, 3] (LOSS, DRAW, WIN) for the player to move."""

    def __init__(self, cfg: CriticConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or CriticConfig()
        num_globals = _core.NUM_GLOBALS + (NUM_RATING_FEATURES if cfg.ratings else 0)
        self.embed = Embed(cfg.channels, num_globals)
        self.blocks = nn.ModuleList(
            Block(cfg.channels, cfg.pool_channels if (i + 1) % cfg.pool_every == 0 else 0, cfg.prenorm)
            for i in range(cfg.blocks)
        )
        self.value = ValueHead(cfg, num_globals)

    def forward(self, planes: torch.Tensor, globals_: torch.Tensor, rating: torch.Tensor | None = None) -> torch.Tensor:
        """``rating`` [B, 2]: the player to move's, then the opponent's (negative =
        unknown); None = both unknown. Ignored without ``cfg.ratings``."""
        if self.cfg.ratings:
            extra = rating_features(rating, len(globals_), globals_.device).to(globals_.dtype)
            globals_ = torch.cat([globals_, extra], 1)
        mask, area = board_mask(planes)
        x = self.embed(planes, globals_, mask)
        for block in self.blocks:
            x = block(x, mask, area)
        return self.value(x, globals_, mask, area)


def value_of(logits: torch.Tensor) -> torch.Tensor:
    """Expected result in [-1, 1] for the player to move: P(win) - P(loss)."""
    p = logits.float().softmax(-1)
    return p[:, WIN] - p[:, LOSS]


# --- Policy and value --------------------------------------------------------

POLICY_PLANES = tuple(_core.PLANES) + tuple(_core.SELECTION_PLANES)
NUM_POLICY_PLANES = len(POLICY_PLANES)
MOVEMENT_PLANES = tuple(_core.MOVEMENT_PLANES)  # after POLICY_PLANES, with PolicyConfig.movement
NUM_AUX_REGRESSION, NUM_AUX_BINARY = len(AUX_REGRESSION), len(AUX_BINARY)
NUM_POLICY_GLOBALS = _core.NUM_GLOBALS + _core.NUM_SELECTION_GLOBALS
LEGAL_PLANES = slice(_core.NUM_PLANES + 1, _core.NUM_PLANES + 1 + _core.NUM_CHANNELS)  # in the input


@dataclass(frozen=True)
class PolicyConfig:
    channels: int = 64
    blocks: int = 6
    pool_every: int = 2
    pool_channels: int = 32
    head_channels: int = 32  # value head
    head_hidden: int = 128
    policy_channels: int = 64  # per-tile hidden layer of the policy head
    ratings: bool = True
    movement: bool = False  # read the movement planes
    aux: bool = False  # predict the auxiliary targets


class PolicyHead(nn.Module):
    """Logits per (channel, tile) [B, NUM_CHANNELS, S, S] and per non-spatial id
    [B, NUM_NONSPATIAL_IDS]."""

    def __init__(self, cfg: PolicyConfig, num_globals: int):
        super().__init__()
        self.norm = ChannelNorm(cfg.channels)
        self.conv1 = nn.Conv2d(cfg.channels, cfg.policy_channels, 1)
        self.conv2 = nn.Conv2d(cfg.policy_channels, _core.NUM_CHANNELS, 1)
        self.fc1 = nn.Linear(3 * cfg.policy_channels + num_globals, cfg.head_hidden)
        self.fc2 = nn.Linear(cfg.head_hidden, _core.NUM_NONSPATIAL_IDS)

    def forward(
        self, x: torch.Tensor, globals_: torch.Tensor, mask: torch.Tensor, area: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.silu(self.conv1(F.silu(self.norm(x)))) * mask
        spatial = self.conv2(h)
        summary = torch.cat([masked_pool(h, mask, area), globals_], 1)
        return spatial, self.fc2(F.silu(self.fc1(summary)))


class PolicyValueNet(nn.Module):
    """Encoded decisions (``encode_policy_batch``), and optionally the players'
    ratings, to policy logits and win/draw/loss logits [B, 3] for the player to move.

    ``forward`` returns the policy as its spatial part [B, NUM_CHANNELS, S, S] and
    its non-spatial part [B, NUM_NONSPATIAL_IDS] (``flat_policy`` joins them in
    ``policy_index`` order), so augmentation can map the spatial logits back.
    """

    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or PolicyConfig()
        num_globals = NUM_POLICY_GLOBALS + (NUM_RATING_FEATURES if cfg.ratings else 0)
        self.embed = Embed(cfg.channels, num_globals, POLICY_PLANES + (MOVEMENT_PLANES if cfg.movement else ()))
        self.blocks = nn.ModuleList(
            Block(cfg.channels, cfg.pool_channels if (i + 1) % cfg.pool_every == 0 else 0) for i in range(cfg.blocks)
        )
        self.policy = PolicyHead(cfg, num_globals)
        self.value = ValueHead(cfg, num_globals)
        self.aux = ValueHead(cfg, num_globals, NUM_AUX_REGRESSION + NUM_AUX_BINARY) if cfg.aux else None

    def forward(
        self, planes: torch.Tensor, globals_: torch.Tensor, rating: torch.Tensor | None = None, aux: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """(spatial policy logits, non-spatial policy logits, value logits); ``rating``
        as in CriticNet. With ``aux`` (and ``cfg.aux``) a fourth: the auxiliary
        predictions [B, NUM_AUX_REGRESSION + NUM_AUX_BINARY], the scaled changes of
        AUX_REGRESSION and then logits for AUX_BINARY."""
        if self.cfg.ratings:
            extra = rating_features(rating, len(globals_), globals_.device).to(globals_.dtype)
            globals_ = torch.cat([globals_, extra], 1)
        mask, area = board_mask(planes)
        x = self.embed(planes, globals_, mask)
        for block in self.blocks:
            x = block(x, mask, area)
        spatial, nonspatial = self.policy(x, globals_, mask, area)
        value = self.value(x, globals_, mask, area)
        if aux:
            if self.aux is None:
                raise ValueError("this network has no auxiliary head (PolicyConfig.aux)")
            return spatial, nonspatial, value, self.aux(x, globals_, mask, area)
        return spatial, nonspatial, value


def flat_policy(spatial: torch.Tensor, nonspatial: torch.Tensor) -> torch.Tensor:
    """[B, policy_size(S, S)] logits, in ``_core.policy_index`` order."""
    return torch.cat([spatial.flatten(1), nonspatial], 1)


def legal_mask(planes: torch.Tensor, nonspatial: torch.Tensor) -> torch.Tensor:
    """[B, policy_size(S, S)] bool: the legal steps, from the input's legal planes
    and the non-spatial legality ``encode_policy_batch`` gives."""
    return torch.cat([planes[:, LEGAL_PLANES].flatten(1), nonspatial], 1) != 0


def masked_log_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Log-probabilities over the legal steps; -inf elsewhere."""
    return logits.float().masked_fill(~legal, -math.inf).log_softmax(-1)
