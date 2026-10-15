"""Training batches: raw states encoded by the engine, grouped by board size.

Boards are padded to square buckets (BUCKETS) so every batch has one shape,
and a square window keeps the board inside it under the dihedral flips and
transposes used for augmentation.

``batches`` gives states (the critic's input); ``decision_batches`` gives the
micro-step decisions of a v1 dataset's actions (the policy's), each with the
step taken and optionally with what happens next in the game (``aux_targets``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .. import _core
from .shards import Dataset, Shard

BUCKETS = (16, 24, 32, 40)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SUBSET = np.uint64(0xD1B54A32D192ED03)


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


def in_subset(game_ids: np.ndarray, fraction: float) -> np.ndarray:
    """Whether each game is in a ``fraction`` share of the games, by a hash of its
    id unrelated to ``in_validation``'s, so smaller shares are parts of larger ones
    (a quarter of the games is half of the half) and hold validation games evenly."""
    h = (np.asarray(game_ids).astype(np.uint64) * _SUBSET) >> np.uint64(40)
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
    states: np.ndarray,
    picks: np.ndarray,
    n_picks: np.ndarray,
    size: int,
    targets: bool = True,
    movement: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Planes [N, NUM_POLICY_PLANES, size, size], globals [N, NUM_POLICY_GLOBALS],
    non-spatial legality [N, NUM_NONSPATIAL_IDS] and, with ``targets``, the
    policy index of the step taken (-1 if it isn't legal), for the decisions
    reached in each state by its first ``n_picks`` of ``picks`` [N, 3]. With
    ``movement`` the MOVEMENT_PLANES follow the others."""
    n = len(states)
    n_planes = _core.NUM_PLANES + _core.NUM_SELECTION_PLANES + (_core.NUM_MOVEMENT_PLANES if movement else 0)
    planes = np.empty((n, n_planes, size, size), dtype=np.uint8)
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
        movement,
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
    # With ``aux`` (aux_targets):
    aux_reg: np.ndarray | None = None  # float32 [B, len(AUX_REGRESSION)]
    aux_bin: np.ndarray | None = None  # float32 [B, len(AUX_BINARY)]: 0 / 1
    aux_mask: np.ndarray | None = None  # bool [B, len(AUX_BINARY)]: where aux_bin counts
    group: int = 0  # the pass's group of shards it came from (decision_batches)


# --- Auxiliary targets ---------------------------------------------------------
#
# What happens next in the game, for heads that train the network alongside the
# policy and value (as KataGo's do) rather than as rewards. For the player to
# move, "me", and "opp":
#
# * the change in army value and in unit count from the decision to the start
#   of the turn 1, 2 and 6 player-turns on (the end of my turn, after their
#   reply, three days on) and to the game's last state; a game that ends sooner
#   gives its last state. Scaled by about their spread over the v1 dataset.
# * the CO power bars: whether my COP / SCOP is charged at the start of my next
#   turn and theirs at the start of theirs, and whether I use a power before
#   my turn ends and they do in their next turn. Left out (aux_mask) where the
#   game has no powers, the CO has no such power, or the game ends first.
AUX_HORIZONS = (1, 2, 6, None)  # player-turns on; None: the game's last state
_VALUE_SCALE = (10_000, 12_000, 20_000, 45_000)
_UNITS_SCALE = (2, 2, 5, 9)
AUX_REGRESSION = tuple(
    f"{who}.{what}@{h or 'end'}" for h in AUX_HORIZONS for what in ("value", "units") for who in ("me", "opp")
)
AUX_BINARY = (
    "me.cop_ready@2",
    "me.scop_ready@2",
    "opp.cop_ready@1",
    "opp.scop_ready@1",
    "me.power_used@1",
    "opp.power_used@2",
)
_STAT = {name: i for i, name in enumerate(_core.PLAYER_STATS)}


def aux_targets(sh: Shard, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(AUX_REGRESSION float32 [N, 16], AUX_BINARY as 0 / 1 float32 [N, 6], and where
    each binary target counts, bool [N, 6]) for the shard's samples ``rows``."""
    assert sh.action is not None
    stats = _core.player_stats_batch(np.ascontiguousarray(sh.state))  # [S, 2, PLAYER_STATS]: all rows, no copy
    starts = np.flatnonzero(sh.action == 0)  # turn starts, in game and turn order
    key = sh.game[starts].astype(np.int64) << 16 | sh.turn[starts]
    assert (np.diff(sh.g_start) > 0).all()
    last = np.r_[sh.g_start[1:], len(sh)] - 1  # each game's last sample
    n = len(rows)
    game, turn = sh.game[rows].astype(np.int64), sh.turn[rows].astype(np.int64)
    me = sh.to_play[rows].astype(np.intp)
    opp = 1 - me
    end = last[game]
    at: dict[int | None, tuple[np.ndarray, np.ndarray]] = {None: (end, np.zeros(n, bool))}
    for h in AUX_HORIZONS[:-1]:
        assert h is not None
        want = game << 16 | (turn + h)
        j = np.minimum(np.searchsorted(key, want), len(key) - 1)
        found = key[j] == want
        at[h] = (np.where(found, starts[j], end), ~found)

    def stat(r: np.ndarray, who: np.ndarray, name: str) -> np.ndarray:
        return stats[r, who, _STAT[name]].astype(np.int64)

    reg = np.empty((n, len(AUX_REGRESSION)), np.float32)
    k = 0
    for hi, h in enumerate(AUX_HORIZONS):
        later, _ = at[h]
        for name, scale in (("army_value", _VALUE_SCALE[hi]), ("units", _UNITS_SCALE[hi])):
            for who in (me, opp):
                reg[:, k] = (stat(later, who, name) - stat(rows, who, name)) / scale
                k += 1

    powers = stat(rows, me, "use_powers") != 0
    s1, past1 = at[1]
    s2, past2 = at[2]
    out = np.zeros((n, len(AUX_BINARY)), np.float32)
    mask = np.zeros((n, len(AUX_BINARY)), bool)
    for c, (r, who, past, level) in enumerate(
        ((s2, me, past2, "cop"), (s2, me, past2, "scop"), (s1, opp, past1, "cop"), (s1, opp, past1, "scop"))
    ):
        cost = stat(r, who, f"{level}_cost")
        out[:, c] = stat(r, who, "power") >= cost
        mask[:, c] = powers & (cost > 0) & ~past
    out[:, 4] = stat(s1, me, "power_uses") > stat(rows, me, "power_uses")
    mask[:, 4] = powers & (stat(rows, me, "scop_cost") > 0) & ~past1
    out[:, 5] = stat(s2, opp, "power_uses") > stat(s1, opp, "power_uses")
    mask[:, 5] = powers & (stat(rows, opp, "scop_cost") > 0) & ~past2
    return reg, out, mask


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
    movement: bool = False,
    aux: bool = False,
    memo: list[tuple[np.ndarray, dict[str, np.ndarray]]] | None = None,
    skip_groups: int = 0,
) -> Iterator[DecisionBatch]:
    """One pass over the decisions of a v1 dataset's labelled actions.

    Each action gives a decision per micro-step, the forced ones (a single
    legal step) left out unless ``forced``; ``steps`` narrows them to some of
    each action's (0 is its first decision). ``fraction`` takes a random share
    of the decisions on each pass, ``row_fraction`` a fixed share of the
    actions (all of their decisions; for validation). ``split`` and ``keep``
    are as in ``batches``. Shuffling mixes ``shards_at_once`` shards at a
    time; only the states their chosen decisions need are kept. The next
    group of shards loads in a thread while this one's batches are used.
    ``movement`` adds the movement planes, ``aux`` the auxiliary targets.
    Each batch knows its ``group``; ``skip_groups`` leaves out the first ones,
    as a resumed pass does, and the rest come as they would have.

    ``memo`` holds a fixed choice of decisions (no ``shuffle``, ``fraction``
    1) in memory, for passes that repeat: a whole pass given an empty list
    fills it, and a pass given a filled one reads no shards. Every pass
    given the same memo must choose the same way.
    """
    if not ds.has_actions:
        raise ValueError(f"{ds.root}: not a v1 dataset (no action labels)")
    if split not in ("train", "val", "all"):
        raise ValueError(f"unknown split {split!r}")
    if memo is not None and (shuffle or fraction < 1 or skip_groups):
        raise ValueError("a memo holds a fixed choice of decisions: no shuffle, fraction 1, no groups skipped")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds.shards)) if shuffle else np.arange(len(ds.shards))
    groups = [order[start : start + shards_at_once] for start in range(0, len(order), shards_at_once)]
    # A generator per group, so loads can run ahead and skipped groups change nothing after them.
    jobs = list(zip(range(len(groups)), groups, rng.spawn(len(groups)), strict=True))[skip_groups:]

    def load(job: tuple[int, np.ndarray, np.random.Generator]) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        _, shards, group_rng = job
        parts = []
        for i in shards:
            sh = ds.load(int(i))
            args = (split, val_fraction, keep, fraction, row_fraction, forced, steps, aux, group_rng)
            if (part := _decisions(sh, int(i), *args)) is not None:
                parts.append(part)
        if not parts:
            return None
        offsets = np.cumsum([0] + [len(p["states"]) for p in parts[:-1]])
        states = np.concatenate([p.pop("states") for p in parts])
        for p, off in zip(parts, offsets, strict=True):
            p["state"] = p["state"] + off
        return states, {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    filling = memo is not None and not memo
    if memo:
        stream = ((g, rng, loaded) for g, loaded in enumerate(memo))
    else:
        stream = (
            (g, group_rng, loaded) for (g, _, group_rng), loaded in zip(jobs, _read_ahead(load, jobs), strict=True)
        )
    held = []
    for group, group_rng, loaded in stream:
        if loaded is None:
            continue
        if filling:
            held.append(loaded)
        states, cols = loaded
        sizes = np.array([bucket_for(*ds.map_size[int(m)]) for m in cols["map_id"]])
        idx = group_rng.permutation(len(sizes)) if shuffle else np.arange(len(sizes))
        todo = []
        for size in BUCKETS:
            chosen = idx[sizes[idx] == size]
            todo += [(size, chosen[b : b + batch_size]) for b in range(0, len(chosen), batch_size)]
        if drop_last:
            todo = [(size, sel) for size, sel in todo if len(sel) == batch_size]
        for j in group_rng.permutation(len(todo)) if shuffle else range(len(todo)):
            size, sel = todo[j]
            planes, globals_, nonspatial, target = encode_decisions(
                states[cols["state"][sel]], cols["steps"][sel], cols["step"][sel], size, movement=movement
            )
            assert target is not None
            ok = target >= 0  # a step the engine no longer takes is left out
            yield DecisionBatch(
                planes=planes[ok],
                globals=globals_[ok],
                nonspatial=nonspatial[ok],
                target=target[ok],
                group=group,
                **{k: cols[k][sel][ok] for k in _DECISION_FIELDS + (_AUX_FIELDS if aux else ())},
            )
    if filling and memo is not None:
        memo.extend(held)  # only once the pass is whole


_DECISION_FIELDS = ("step", "phase", "n_legal", "kind", "turn_start", "row", "z", "game_id", "map_id", "day", "rating")
_AUX_FIELDS = ("aux_reg", "aux_bin", "aux_mask")


def _read_ahead[T, R](fn: Callable[[T], R], items: list[T]) -> Iterator[R]:
    """``fn`` of each item in turn, the next one computed in a thread while the
    caller uses this one (decompression and numpy release the GIL)."""
    with ThreadPoolExecutor(1) as pool:
        ahead = [pool.submit(fn, item) for item in items[:1]]
        for item in items[1:]:
            ahead.append(pool.submit(fn, item))
            yield ahead.pop(0).result()
        if ahead:
            yield ahead.pop().result()


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
    aux: bool,
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
    if aux:
        out["aux_reg"], out["aux_bin"], out["aux_mask"] = aux_targets(sh, row)
    return out
