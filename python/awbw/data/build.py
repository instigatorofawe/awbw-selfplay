"""Building datasets from replays.

A v1 dataset has a sample for every action taken, about 490 per game, so its
states are some 30 times v0's. Workers extract the replays and write whole
shards themselves, and send back only counts, so neither moving the states
between processes nor compressing them runs in one process.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..replay.extract import DRAW, GameSamples, extract_v1
from ..replay.sources import IndexEntry, RatingHistory
from .shards import UNKNOWN_RATING, shard_arrays, write_meta, write_npz


def game_ratings(g: GameSamples, index: dict[int, IndexEntry], history: RatingHistory) -> tuple[tuple[int, int], str]:
    """Both players' ratings from before the game (UNKNOWN_RATING if unknown), by
    player index, and how they were found.

    The index lists players in turn order; a row whose result disagrees with
    the replay's is not trusted. The row itself rates players after the game,
    which gives the result away, so the ratings are each player's latest from
    before the game started (``RatingHistory.before``).
    """
    unknown = (UNKNOWN_RATING, UNKNOWN_RATING)
    if (entry := index.get(g.game_id)) is None:
        return unknown, "not in an index"
    if g.winner != DRAW and tuple(p.lost for p in entry.players) != (g.winner != 0, g.winner != 1):
        return unknown, "index result disagrees; dropped"
    before = [history.before(p.name, g.start_date) if g.start_date else None for p in entry.players]
    r0, r1 = (UNKNOWN_RATING if r is None else r for r in before)
    return (r0, r1), "known" if None not in before else "a player has no earlier indexed game"


@dataclass
class BuildStats:
    shards: list[str] = field(default_factory=list)
    games: int = 0
    samples: int = 0
    labelled: int = 0  # samples with an action label
    steps: int = 0  # micro-steps of those actions
    forced: int = 0  # steps with a single legal id
    unstepped: int = 0
    hp_unknown: int = 0
    maps: set[int] = field(default_factory=set)
    rejected: Counter = field(default_factory=Counter)
    ratings: Counter = field(default_factory=Counter)
    ends: Counter = field(default_factory=Counter)

    def add(self, other: BuildStats) -> None:
        self.shards += other.shards
        for name in ("games", "samples", "labelled", "steps", "forced", "unstepped", "hp_unknown"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.maps |= other.maps
        self.rejected.update(other.rejected)
        self.ratings.update(other.ratings)
        self.ends.update(other.ends)


_index: dict[int, IndexEntry] = {}
_history = RatingHistory([])


def _init(index: dict[int, IndexEntry], history: RatingHistory) -> None:
    global _index, _history
    _index, _history = index, history


def _build_shard(job: tuple[Path, str, list[Path]]) -> BuildStats:
    root, name, paths = job
    st = BuildStats()
    games: list[tuple[GameSamples, tuple[int, int]]] = []
    for path in paths:
        g = extract_v1(path)
        if isinstance(g, str):
            st.rejected[g] += 1
            continue
        ratings, how = game_ratings(g, _index, _history)
        st.ratings[how] += 1
        st.ends[g.end_action] += 1
        st.unstepped += g.unstepped
        st.hp_unknown += g.hp_unknown
        st.maps.add(g.map_id)
        a = g.actions
        assert a is not None
        taken = np.arange(3)[None, :] < a.n_steps[:, None]
        st.labelled += int((a.n_steps > 0).sum())
        st.steps += int(taken.sum())
        st.forced += int((taken & (a.n_legal == 1)).sum())
        games.append((g, ratings))
    if games:
        arrays = shard_arrays(games)
        write_npz(root / name, arrays)
        st.shards.append(name)
        st.games = len(games)
        st.samples = len(arrays["state"])
    return st


def build_v1(
    paths: list[Path],
    out: Path,
    index: dict[int, IndexEntry],
    history: RatingHistory,
    *,
    workers: int | None = None,
    replays_per_shard: int = 100,
    log: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
    **info: object,
) -> tuple[dict, BuildStats]:
    """Build a v1 dataset in ``out`` from replay files; returns its metadata and counts.

    ``index`` (2-player mooo rows by game id) and ``history`` give the
    players' ratings (``game_ratings``). Each shard holds the games kept from
    ``replays_per_shard`` consecutive replays.
    """
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("shard-*.npz")):
        raise FileExistsError(f"{out} already holds shards")
    jobs = [
        (out, f"shard-{i:05d}.npz", paths[start : start + replays_per_shard])
        for i, start in enumerate(range(0, len(paths), replays_per_shard))
    ]
    total = BuildStats()
    done = 0
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(index, history)) as pool:
        for st in pool.map(_build_shard, jobs):
            total.add(st)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                log(f"[{done}/{len(jobs)} shards] {total.games:,} games, {total.samples:,} samples")
    meta = write_meta(
        out,
        total.shards,
        total.games,
        total.samples,
        total.maps,
        rows="actions",
        labelled=total.labelled,
        steps=total.steps,
        forced_steps=total.forced,
        unstepped_turns=total.unstepped,
        hp_beliefs_reset=total.hp_unknown,
        rejected=dict(total.rejected.most_common()),
        ratings=dict(total.ratings),
        ended_by=dict(total.ends.most_common()),
        **info,
    )
    return meta, total
