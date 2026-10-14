"""Datasets on disk: raw engine states plus labels, in compressed numpy shards.

A dataset directory holds:

* ``meta.json``: format, counts and how the dataset was built;
* ``maps.npz``: AWBW terrain ids [height, width] of every map used, keyed by map id,
  so a dataset can be read without the replay cache;
* ``shard-00000.npz``, ...: whole games, so no game straddles two shards.

Per sample (one row each): ``state`` (uint8 [N, STATE_BYTES], the raw
``GameState``), ``game`` (index into the shard's game table), ``turn`` (0 for
the game's first turn), ``day``, ``to_play`` and ``z`` (the result for the
player to move: +1 won, -1 lost, 0 draw).

A v1 dataset (``meta["rows"] == "actions"``) has a sample for every action
taken (``extract.extract_v1``), each with ``action`` (actions taken before it
in its turn, so 0 is the turn's start), ``kind`` (the action taken, ACTIONS
value, or NO_ACTION), ``steps`` [N, 3] (its micro-step ids), ``n_steps`` (0 =
unlabelled) and ``n_legal`` [N, 3] (ids legal at each step; 1 = forced).

Per game (``g_`` prefix): ``g_id``, ``g_map``, ``g_winner`` (player index, 2 =
draw), ``g_end`` (index into END_ACTIONS), ``g_days``, ``g_co`` [G, 2] (AWBW CO
ids) and ``g_rating`` [G, 2], both by player index (0 moves first), and
``g_start`` (its first sample). A rating is the player's rating when the game
started (after their latest game that ended on an earlier day; -1 unknown):
the mooo row of the game itself rates them after it, which gives the result
away (``sources.RatingHistory``).

States are stored, never features, so changing the encoder needs no rebuild;
``state_bytes`` in the metadata guards against a changed GameState layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import _core
from ..maps import load_map
from ..net import write_atomic
from ..replay.extract import GameSamples

FORMAT = "awbw-shards"
VERSION = 2  # 2: ratings from before the game (v1's were after it)
END_ACTIONS = ("Resign", "Capt", "Fire", "Delete", "other")
UNKNOWN_RATING = -1


@dataclass(slots=True)
class Shard:
    state: np.ndarray
    game: np.ndarray
    turn: np.ndarray
    day: np.ndarray
    to_play: np.ndarray
    z: np.ndarray
    g_id: np.ndarray
    g_map: np.ndarray
    g_winner: np.ndarray
    g_end: np.ndarray
    g_days: np.ndarray
    g_co: np.ndarray
    g_rating: np.ndarray
    g_start: np.ndarray
    # v1 datasets only.
    action: np.ndarray | None = None
    kind: np.ndarray | None = None
    steps: np.ndarray | None = None
    n_steps: np.ndarray | None = None
    n_legal: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.state)

    @property
    def map_id(self) -> np.ndarray:
        """Map of each sample."""
        return self.g_map[self.game]


def write_npz(path: Path, arrays: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


class ShardWriter:
    """Collect games and write them out ``games_per_shard`` at a time."""

    def __init__(self, root: Path, games_per_shard: int = 1000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.glob("shard-*.npz")):
            raise FileExistsError(f"{self.root} already holds shards")
        self.games_per_shard = games_per_shard
        self.shards: list[str] = []
        self.games = self.samples = 0
        self.maps: set[int] = set()
        self._pending: list[tuple[GameSamples, tuple[int, int]]] = []

    def add(self, g: GameSamples, ratings: tuple[int, int] = (UNKNOWN_RATING, UNKNOWN_RATING)) -> None:
        self._pending.append((g, ratings))
        self.maps.add(g.map_id)
        if len(self._pending) >= self.games_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        name = f"shard-{len(self.shards):05d}.npz"
        arrays = shard_arrays(self._pending)
        write_npz(self.root / name, arrays)
        self.shards.append(name)
        self.games += len(self._pending)
        self.samples += len(arrays["state"])
        self._pending.clear()

    def close(self, **info: object) -> dict:
        """Write the last shard, the maps and the metadata; returns the metadata."""
        self._flush()
        return write_meta(self.root, self.shards, self.games, self.samples, self.maps, **info)


def shard_arrays(games: list[tuple[GameSamples, tuple[int, int]]]) -> dict[str, np.ndarray]:
    """A shard's arrays for some games, each with its players' ratings."""
    gs = [g for g, _ in games]
    counts = np.array([len(g.states) for g in gs])
    end = {name: i for i, name in enumerate(END_ACTIONS)}
    arrays = {
        "state": np.concatenate([g.states for g in gs]),
        "game": np.repeat(np.arange(len(gs), dtype=np.int32), counts),
        "turn": np.concatenate(
            [
                np.arange(n, dtype=np.int16) if g.actions is None else g.actions.turn
                for g, n in zip(gs, counts, strict=True)
            ]
        ),
        "day": np.concatenate([g.day for g in gs]),
        "to_play": np.concatenate([g.to_play for g in gs]),
        "z": np.concatenate([g.outcome for g in gs]),
        "g_id": np.array([g.game_id for g in gs], dtype=np.int32),
        "g_map": np.array([g.map_id for g in gs], dtype=np.int32),
        "g_winner": np.array([g.winner for g in gs], dtype=np.uint8),
        "g_end": np.array([end.get(g.end_action, end["other"]) for g in gs], dtype=np.uint8),
        "g_days": np.array([g.days for g in gs], dtype=np.int16),
        "g_co": np.array([g.cos for g in gs], dtype=np.uint8).reshape(-1, 2),
        "g_rating": np.array([r for _, r in games], dtype=np.int16).reshape(-1, 2),
        "g_start": (np.cumsum(counts) - counts).astype(np.int32),
    }
    labelled = [g.actions for g in gs if g.actions is not None]
    if labelled:
        if len(labelled) != len(gs):
            raise ValueError("a shard's games must all have action labels or none")
        for name in ("action", "kind", "steps", "n_steps", "n_legal"):
            arrays[name] = np.concatenate([getattr(a, name) for a in labelled])
    return arrays


def write_meta(root: Path, shards: list[str], games: int, samples: int, maps: set[int], **info: object) -> dict:
    """Write a dataset's maps and metadata, once its shards are written; returns the metadata."""
    terrain = {str(m): np.asarray(load_map(m).terrain, dtype=np.int16) for m in sorted(maps)}
    write_npz(root / "maps.npz", terrain)
    meta = {
        "format": FORMAT,
        "version": VERSION,
        "state_bytes": _core.STATE_BYTES,
        "games": games,
        "samples": samples,
        "shards": shards,
        **info,
    }
    write_atomic(root / "meta.json", (json.dumps(meta, indent=2) + "\n").encode())
    return meta


class Dataset:
    """A dataset directory. Opening it registers its maps with the engine."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        if self.meta.get("format") != FORMAT or self.meta.get("version") != VERSION:
            raise ValueError(f"{self.root}: not a {FORMAT} v{VERSION} dataset")
        if self.meta["state_bytes"] != _core.STATE_BYTES:
            raise ValueError(
                f"{self.root}: states are {self.meta['state_bytes']} bytes but the engine's are "
                f"{_core.STATE_BYTES}; rebuild the dataset"
            )
        self.map_size: dict[int, tuple[int, int]] = {}  # map id -> (height, width)
        with np.load(self.root / "maps.npz") as maps:
            for key in maps.files:
                terrain = np.ascontiguousarray(maps[key], dtype=np.int16)
                _core.register_map(int(key), terrain)  # refuses an id already taken by other terrain
                self.map_size[int(key)] = terrain.shape

    @property
    def has_actions(self) -> bool:
        """A v1 dataset: a sample per action taken, with its micro-steps."""
        return self.meta.get("rows") == "actions"

    @property
    def shards(self) -> list[Path]:
        return [self.root / name for name in self.meta["shards"]]

    def load(self, shard: Path | int) -> Shard:
        path = self.shards[shard] if isinstance(shard, int) else shard
        with np.load(path) as f:
            return Shard(**{k: f[k] for k in f.files})
