"""Matches between agents from fixed starts, and ratings from their results.

``match`` plays each start twice with seats swapped. ``ladder`` rates every
checkpoint of a self-play run against one fixed baseline, appending each
result to the run's ``ladder.jsonl``, so the M5 exit criterion (Elo rising
across checkpoints) can be read off it.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import _core
from .agents import DRAW, Agent, GameResult, play
from .data.loader import in_validation
from .data.shards import Dataset
from .league import std_rules


def validation_starts(ds: Dataset, n: int) -> list[_core.GameState]:
    """The first states of the dataset's first ``n`` validation games, under the
    Global League's standard rules (``std_rules``)."""
    starts: list[_core.GameState] = []
    for i in range(len(ds.shards)):
        if len(starts) >= n:
            break
        sh = ds.load(i)
        for g in np.flatnonzero(in_validation(sh.g_id))[: n - len(starts)]:
            starts.append(std_rules(_core.GameState.from_bytes(sh.state[sh.g_start[g]].tobytes())))
    return starts


@dataclass(slots=True)
class MatchResult:
    won: int
    lost: int
    drawn: int
    by_seat: tuple[list[GameResult], list[GameResult]]  # me as player 1, then as player 2

    @property
    def games(self) -> int:
        return self.won + self.lost + self.drawn

    @property
    def score(self) -> float:
        return (self.won + 0.5 * self.drawn) / max(self.games, 1)


def match(me: Agent, them: Agent, starts: list[_core.GameState], max_days: int = 30) -> MatchResult:
    """Each start played twice, ``me`` moving first, then second."""
    first = play(starts, (me, them), max_days=max_days)
    second = play(starts, (them, me), max_days=max_days)
    won = lost = drawn = 0
    for seat, results in ((0, first), (1, second)):
        for r in results:
            if r.winner == DRAW:
                drawn += 1
            elif r.winner == seat:
                won += 1
            else:
                lost += 1
    return MatchResult(won, lost, drawn, (first, second))


def elo(score: float, games: int) -> tuple[float, float]:
    """Elo difference for a score in [0, 1] over ``games`` and its standard error
    (the score clamped half a game from 0 and 1)."""
    s = min(max(score, 0.5 / games), 1 - 0.5 / games)
    diff = 400 * math.log10(s / (1 - s))
    se = 400 / math.log(10) * math.sqrt(s * (1 - s) / games) / (s * (1 - s))
    return diff, se


_STEP = re.compile(r"net-(\d+)\.pt$")


def checkpoints(run: Path, every: int = 0) -> list[tuple[int, Path]]:
    """A run's ``net-NNNNNN.pt`` by step, those at multiples of ``every`` only (0: all)."""
    found = []
    for p in run.glob("net-*.pt"):
        if m := _STEP.search(p.name):
            step = int(m.group(1))
            if not every or step % every == 0:
                found.append((step, p))
    return sorted(found)


def ladder(
    run: Path,
    baseline: Path,
    make_agent: Callable[[Path, int], Agent],
    starts: list[_core.GameState],
    *,
    every: int = 0,
    max_days: int = 30,
    settings: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Rate each checkpoint of ``run`` not yet rated against ``baseline`` with
    these ``settings``, appending to ``run/ladder.jsonl``; returns the new entries.
    ``make_agent(path, seed)`` builds the agents."""
    settings = {"baseline": str(baseline), "starts": len(starts), "max_days": max_days, **(settings or {})}
    out = run / "ladder.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            e = json.loads(line)
            if e["settings"] == settings:
                done.add(e["checkpoint"])
    new = []
    for step, path in checkpoints(run, every):
        if path.name in done:
            continue
        t0 = time.monotonic()
        r = match(make_agent(path, 0), make_agent(baseline, 1), starts, max_days)
        diff, se = elo(r.score, r.games)
        entry = {
            "checkpoint": path.name,
            "step": step,
            "won": r.won,
            "lost": r.lost,
            "drawn": r.drawn,
            "score": r.score,
            "elo": diff,
            "elo_se": se,
            "seconds": time.monotonic() - t0,
            "settings": settings,
        }
        log(f"{path.name}: {r.won}-{r.lost}-{r.drawn}, score {r.score:.3f}, Elo {diff:+.0f} ± {se:.0f}")
        with out.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        new.append(entry)
    return new
