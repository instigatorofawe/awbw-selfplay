"""Training batches: raw states encoded by the engine, grouped by board size.

Boards are padded to square buckets (BUCKETS) so every batch has one shape,
and a square window keeps the board inside it under the dihedral flips and
transposes used for augmentation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .. import _core
from .shards import Dataset, Shard

BUCKETS = (16, 24, 32, 40)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def bucket_for(height: int, width: int) -> int:
    """Side of the smallest square bucket that holds the board."""
    side = max(height, width)
    for b in BUCKETS:
        if side <= b:
            return b
    raise ValueError(f"a {width}x{height} board is larger than every bucket")


def in_validation(game_ids: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Whether each game is in the validation split: a fixed hash of its id, so a
    game stays on one side across rebuilds and dataset sizes."""
    h = (np.asarray(game_ids).astype(np.uint64) * _GOLDEN) >> np.uint64(40)
    return (h % np.uint64(1_000_000)) < np.uint64(round(fraction * 1_000_000))


def encode(states: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Planes [N, NUM_PLANES, size, size] and globals [N, NUM_GLOBALS] for raw states."""
    states = np.ascontiguousarray(states, dtype=np.uint8)
    planes = np.empty((len(states), _core.NUM_PLANES, size, size), dtype=np.uint8)
    globals_ = np.empty((len(states), _core.NUM_GLOBALS), dtype=np.float32)
    _core.encode_batch(states, planes, globals_)
    return planes, globals_


@dataclass(slots=True)
class Batch:
    planes: np.ndarray  # uint8 [B, NUM_PLANES, S, S]
    globals: np.ndarray  # float32 [B, NUM_GLOBALS]
    z: np.ndarray  # int8 [B]: result for the player to move
    game_id: np.ndarray  # int32 [B]
    map_id: np.ndarray  # int32 [B]
    day: np.ndarray  # int16 [B]
    rating: np.ndarray  # int16 [B, 2]: the player to move's, then the opponent's; -1 unknown


def batches(
    ds: Dataset,
    batch_size: int,
    *,
    split: str = "train",
    val_fraction: float = 0.05,
    keep: Callable[[Shard], np.ndarray] | None = None,
    shuffle: bool = True,
    seed: int = 0,
    shards_at_once: int = 4,
    drop_last: bool = False,
) -> Iterator[Batch]:
    """One pass over the dataset.

    ``split`` is "train", "val" or "all". ``keep`` narrows the samples further,
    e.g. by rating or map: it gets a shard and returns a bool mask over its
    samples. Shuffling mixes ``shards_at_once`` shards at a time.
    """
    if split not in ("train", "val", "all"):
        raise ValueError(f"unknown split {split!r}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds.shards)) if shuffle else np.arange(len(ds.shards))
    for start in range(0, len(order), shards_at_once):
        loaded = [ds.load(int(i)) for i in order[start : start + shards_at_once]]
        parts = [_select(sh, split, val_fraction, keep) for sh in loaded]
        state = np.concatenate([sh.state[i] for sh, i in zip(loaded, parts, strict=True)])
        if not len(state):
            continue
        meta = {
            name: np.concatenate([fn(sh)[i] for sh, i in zip(loaded, parts, strict=True)])
            for name, fn in _SAMPLE_FIELDS.items()
        }
        sizes = np.array([bucket_for(*ds.map_size[int(m)]) for m in meta["map_id"]])
        idx = rng.permutation(len(state)) if shuffle else np.arange(len(state))
        for size in BUCKETS:
            chosen = idx[sizes[idx] == size]
            for b in range(0, len(chosen), batch_size):
                sel = chosen[b : b + batch_size]
                if drop_last and len(sel) < batch_size:
                    continue
                planes, globals_ = encode(state[sel], size)
                yield Batch(planes=planes, globals=globals_, **{k: v[sel] for k, v in meta.items()})


def _select(sh: Shard, split: str, val_fraction: float, keep: Callable[[Shard], np.ndarray] | None) -> np.ndarray:
    mask = np.ones(len(sh), dtype=bool)
    if split != "all":
        val = in_validation(sh.g_id, val_fraction)[sh.game]
        mask &= val if split == "val" else ~val
    if keep is not None:
        mask &= keep(sh)
    return np.flatnonzero(mask)


def _ratings(sh: Shard) -> np.ndarray:
    r = sh.g_rating[sh.game]
    mover = sh.to_play.astype(np.intp)
    rows = np.arange(len(sh))
    return np.stack([r[rows, mover], r[rows, 1 - mover]], axis=1)


_SAMPLE_FIELDS: dict[str, Callable[[Shard], np.ndarray]] = {
    "z": lambda sh: sh.z,
    "game_id": lambda sh: sh.g_id[sh.game],
    "map_id": lambda sh: sh.map_id,
    "day": lambda sh: sh.day,
    "rating": _ratings,
}
