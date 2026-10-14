"""Training batches: raw states encoded by the engine, grouped by board size.

Boards are padded to square buckets (BUCKETS) so every batch has one shape,
and a square window keeps the board inside it under the dihedral flips and
transposes used for augmentation.

``batches`` gives states (the critic's input); ``decision_batches`` gives the
micro-step decisions of a v1 dataset's actions (the policy's), each with the
step taken.
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


def state_map_ids(states: np.ndarray) -> np.ndarray:
    """Map id of each raw state [N, STATE_BYTES]: ``GameState`` starts with it, as a
    little-endian uint32."""
    return np.ascontiguousarray(np.asarray(states, dtype=np.uint8)[:, :4]).view("<u4").ravel()


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
    samples. Shuffling mixes ``shards_at_once`` shards at a time, and the
    batches of different buckets among them.
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
        todo = []
        for size in BUCKETS:
            chosen = idx[sizes[idx] == size]
            todo += [(size, chosen[b : b + batch_size]) for b in range(0, len(chosen), batch_size)]
        if drop_last:
            todo = [(size, sel) for size, sel in todo if len(sel) == batch_size]
        for j in rng.permutation(len(todo)) if shuffle else range(len(todo)):
            size, sel = todo[j]
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


# --- Decisions (v1 datasets) -------------------------------------------------

PHASE_IDLE, PHASE_SELECTED, PHASE_MOVED, PHASE_FACTORY = range(4)  # _core.PHASES
_BUILD = _core.ACTIONS["Build"]


def encode_decisions(
    states: np.ndarray, picks: np.ndarray, n_picks: np.ndarray, size: int, targets: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Planes [N, NUM_POLICY_PLANES, size, size], globals [N, NUM_POLICY_GLOBALS],
    non-spatial legality [N, NUM_NONSPATIAL_IDS] and, with ``targets``, the
    policy index of the step taken (-1 if it isn't legal), for the decisions
    reached in each state by its first ``n_picks`` of ``picks`` [N, 3]."""
    n = len(states)
    planes = np.empty((n, _core.NUM_PLANES + _core.NUM_SELECTION_PLANES, size, size), dtype=np.uint8)
    globals_ = np.empty((n, _core.NUM_GLOBALS + _core.NUM_SELECTION_GLOBALS), dtype=np.float32)
    nonspatial = np.empty((n, _core.NUM_NONSPATIAL_IDS), dtype=np.uint8)
    target = np.empty(n, dtype=np.int64) if targets else None
    _core.encode_policy_batch(
        np.ascontiguousarray(states, dtype=np.uint8),
        np.ascontiguousarray(picks, dtype=np.uint16),
        np.ascontiguousarray(n_picks, dtype=np.uint8),
        planes,
        globals_,
        nonspatial,
        target,
    )
    return planes, globals_, nonspatial, target


@dataclass(slots=True)
class DecisionBatch:
    planes: np.ndarray  # uint8 [B, NUM_POLICY_PLANES, S, S]
    globals: np.ndarray  # float32 [B, NUM_POLICY_GLOBALS]
    nonspatial: np.ndarray  # uint8 [B, NUM_NONSPATIAL_IDS]: which non-spatial ids are legal
    target: np.ndarray  # int64 [B]: policy index of the step taken
    step: np.ndarray  # uint8 [B]: the decision's place in its action (0 first)
    phase: np.ndarray  # uint8 [B]: PHASE_*
    n_legal: np.ndarray  # int16 [B]: legal steps
    kind: np.ndarray  # uint8 [B]: the action taken (ACTIONS)
    turn_start: np.ndarray  # bool [B]: the turn's first decision
    row: np.ndarray  # int64 [B]: the action's sample, unique in the dataset (shard << 32 | row)
    z: np.ndarray  # int8 [B]
    game_id: np.ndarray
    map_id: np.ndarray
    day: np.ndarray
    rating: np.ndarray  # int16 [B, 2], as in Batch


def row_hash(sh: Shard, rows: np.ndarray) -> np.ndarray:
    """A fixed value in [0, 1) per sample, from its game, turn and action."""
    key = sh.g_id[sh.game[rows]].astype(np.uint64) << np.uint64(24)
    assert sh.action is not None
    key ^= (sh.turn[rows].astype(np.uint64) << np.uint64(12)) ^ sh.action[rows].astype(np.uint64)
    return ((key * _GOLDEN) >> np.uint64(40)).astype(np.float64) / float(1 << 24)


def decision_batches(
    ds: Dataset,
    batch_size: int,
    *,
    split: str = "train",
    val_fraction: float = 0.05,
    keep: Callable[[Shard], np.ndarray] | None = None,
    shuffle: bool = True,
    seed: int = 0,
    shards_at_once: int = 8,
    drop_last: bool = False,
    fraction: float = 1.0,
    row_fraction: float = 1.0,
    forced: bool = False,
    steps: tuple[int, ...] = (0, 1, 2),
) -> Iterator[DecisionBatch]:
    """One pass over the decisions of a v1 dataset's labelled actions.

    Each action gives a decision per micro-step, the forced ones (a single
    legal step) left out unless ``forced``; ``steps`` narrows them to some of
    each action's (0 is its first decision). ``fraction`` takes a random share
    of the decisions on each pass, ``row_fraction`` a fixed share of the
    actions (all of their decisions; for validation). ``split`` and ``keep``
    are as in ``batches``. Shuffling mixes ``shards_at_once`` shards at a
    time; only the states their chosen decisions need are kept.
    """
    if not ds.has_actions:
        raise ValueError(f"{ds.root}: not a v1 dataset (no action labels)")
    if split not in ("train", "val", "all"):
        raise ValueError(f"unknown split {split!r}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds.shards)) if shuffle else np.arange(len(ds.shards))
    for start in range(0, len(order), shards_at_once):
        parts = []
        for i in order[start : start + shards_at_once]:
            sh = ds.load(int(i))
            if (
                part := _decisions(sh, int(i), split, val_fraction, keep, fraction, row_fraction, forced, steps, rng)
            ) is not None:
                parts.append(part)
        if not parts:
            continue
        offsets = np.cumsum([0] + [len(p["states"]) for p in parts[:-1]])
        states = np.concatenate([p.pop("states") for p in parts])
        for p, off in zip(parts, offsets, strict=True):
            p["state"] = p["state"] + off
        cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        sizes = np.array([bucket_for(*ds.map_size[int(m)]) for m in cols["map_id"]])
        idx = rng.permutation(len(sizes)) if shuffle else np.arange(len(sizes))
        todo = []
        for size in BUCKETS:
            chosen = idx[sizes[idx] == size]
            todo += [(size, chosen[b : b + batch_size]) for b in range(0, len(chosen), batch_size)]
        if drop_last:
            todo = [(size, sel) for size, sel in todo if len(sel) == batch_size]
        for j in rng.permutation(len(todo)) if shuffle else range(len(todo)):
            size, sel = todo[j]
            planes, globals_, nonspatial, target = encode_decisions(
                states[cols["state"][sel]], cols["steps"][sel], cols["step"][sel], size
            )
            assert target is not None
            ok = target >= 0  # a step the engine no longer takes is left out
            yield DecisionBatch(
                planes=planes[ok],
                globals=globals_[ok],
                nonspatial=nonspatial[ok],
                target=target[ok],
                **{k: cols[k][sel][ok] for k in _DECISION_FIELDS},
            )


_DECISION_FIELDS = ("step", "phase", "n_legal", "kind", "turn_start", "row", "z", "game_id", "map_id", "day", "rating")


def _decisions(
    sh: Shard,
    shard: int,
    split: str,
    val_fraction: float,
    keep: Callable[[Shard], np.ndarray] | None,
    fraction: float,
    row_fraction: float,
    forced: bool,
    steps: tuple[int, ...],
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    """A shard's chosen decisions, each with its state and labels."""
    assert sh.steps is not None and sh.n_steps is not None and sh.n_legal is not None
    assert sh.kind is not None and sh.action is not None
    rows = _select(sh, split, val_fraction, keep)
    rows = rows[sh.n_steps[rows] > 0]
    if row_fraction < 1:
        rows = rows[row_hash(sh, rows) < row_fraction]
    pairs = [
        (r, np.full(len(r), k, dtype=np.uint8))
        for k in steps
        if len(r := rows[(sh.n_steps[rows] > k) & (forced | (sh.n_legal[rows, k] > 1))])
    ]
    if not pairs:
        return None
    row = np.concatenate([r for r, _ in pairs])
    step = np.concatenate([k for _, k in pairs])
    if fraction < 1:
        chosen = rng.random(len(row)) < fraction
        row, step = row[chosen], step[chosen]
        if not len(row):
            return None
    uniq, inv = np.unique(row, return_inverse=True)
    kind = sh.kind[row]
    phase = np.where(step == 0, PHASE_IDLE, np.where(step == 2, PHASE_MOVED, PHASE_SELECTED))
    phase = np.where((step == 1) & (kind == _BUILD), PHASE_FACTORY, phase).astype(np.uint8)
    out = {
        "states": sh.state[uniq],  # each decision's is states[state]
        "state": inv.astype(np.int64),
        "steps": sh.steps[row],
        "step": step,
        "phase": phase,
        "n_legal": sh.n_legal[row, step].astype(np.int16),
        "kind": kind,
        "turn_start": (sh.action[row] == 0) & (step == 0),
        "row": (np.int64(shard) << np.int64(32)) | row.astype(np.int64),
    }
    for name, fn in _SAMPLE_FIELDS.items():
        out[name] = fn(sh)[row]
    return out
