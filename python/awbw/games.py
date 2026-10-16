"""Recorded games: the first state and every micro-step taken. The luck lives in
the state (``GameState.rng``), so the steps replay a game exactly; a record
keeps nothing else it needs but the map's terrain.

A record file is JSON Lines, a game a line, with binary parts zlib-compressed
and base64-encoded ("packed"):

* ``format``: 1; ``source``: what played it ("self-play", "arena"); ``info``:
  free-form (a run's training step, seeds, ...)
* ``map``: the AWBW map id; ``terrain``: its AWBW terrain ids, packed int16
  [height, width], with ``height`` and ``width``
* ``start``: the first state, packed STATE_BYTES
* ``steps``: the micro-step ids, packed little-endian uint16
* ``final``: a hash of the state the steps end at (``state_hash``); a replay
  that ends elsewhere was recorded under other engine rules
* ``winner`` (player index, 2 a draw), ``days``, ``decided`` ("game over",
  "day limit" or "resigned"), ``actions``
* ``players``: what played each side, player 0 (who moves first) then 1
* ``searched`` (optional): each decision the players looked at, packed
  ``SEARCHED``: its place in ``steps`` (``len(steps)`` where the mover
  resigned), the value from the mover's side (searched, if it was), the
  network's, the simulations run and whether it was a full search (self-play's
  samples)
"""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _core

if TYPE_CHECKING:
    from .agents import GameResult

FORMAT = 1
SEARCHED = np.dtype([("step", "<i4"), ("value", "<f4"), ("net_value", "<f4"), ("sims", "<u2"), ("full", "u1")])


@dataclass(slots=True)
class GameLog:
    start: _core.GameState
    steps: np.ndarray  # uint16 [n]
    winner: int
    days: int
    decided: str
    actions: int = 0
    players: tuple[str, str] = ("", "")
    searched: np.ndarray | None = None  # SEARCHED [k]
    final: str = ""  # state_hash of the last state; "" unknown
    source: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def map_id(self) -> int:
        return int(self.start.map_id)


def state_hash(s: _core.GameState) -> str:
    return hashlib.blake2b(s.to_bytes(), digest_size=8).hexdigest()


def _pack(b: bytes) -> str:
    return base64.b64encode(zlib.compress(b, 6)).decode("ascii")


def _unpack(text: str) -> bytes:
    return zlib.decompress(base64.b64decode(text))


def searched_array(step, value, net_value, sims, full=None) -> np.ndarray:
    a = np.zeros(len(step), SEARCHED)
    a["step"], a["value"], a["net_value"], a["sims"] = step, value, net_value, sims
    if full is not None:
        a["full"] = full
    return a


def from_record(
    rec: _core.GameRecord, players: tuple[str, str] = ("", ""), source: str = "self-play", **info: Any
) -> GameLog:
    """A self-play game (``BatchedSearch.take_records``) as a log."""
    s = rec.searched
    decided = "resigned" if rec.resigned else "day limit" if rec.capped else "game over"
    return GameLog(
        start=rec.start_state,
        steps=np.asarray(rec.steps, np.uint16),
        winner=int(rec.winner),
        days=int(rec.days),
        decided=decided,
        actions=int(rec.actions),
        players=players,
        searched=searched_array(s["step"], s["value"], s["net_value"], s["sims"], s["full"]),
        final=state_hash(rec.final_state),
        source=source,
        info=info,
    )


def from_result(r: GameResult, players: tuple[str, str] = ("", ""), source: str = "arena", **info: Any) -> GameLog:
    """A game ``agents.play`` played as a log."""
    notes = r.notes
    return GameLog(
        start=r.start,
        steps=np.asarray(r.steps, np.uint16),
        winner=int(r.winner),
        days=int(r.days),
        decided=r.decided,
        actions=int(r.actions),
        players=players,
        searched=searched_array(
            [i for i, _ in notes],
            [n.value for _, n in notes],
            [n.net_value for _, n in notes],
            [n.sims for _, n in notes],
        )
        if notes
        else None,
        final=state_hash(r.final),
        source=source,
        info=info,
    )


def to_json(log: GameLog) -> dict[str, Any]:
    md = _core.find_map(log.map_id)
    if md is None:
        raise ValueError(f"map {log.map_id} is not registered")
    terrain = np.ascontiguousarray(md.terrain(), "<i2")
    d: dict[str, Any] = {
        "format": FORMAT,
        "source": log.source,
        "info": log.info,
        "map": log.map_id,
        "height": int(terrain.shape[0]),
        "width": int(terrain.shape[1]),
        "terrain": _pack(terrain.tobytes()),
        "start": _pack(log.start.to_bytes()),
        "steps": _pack(np.asarray(log.steps, "<u2").tobytes()),
        "final": log.final,
        "winner": log.winner,
        "days": log.days,
        "decided": log.decided,
        "actions": log.actions,
        "players": list(log.players),
    }
    if log.searched is not None:
        d["searched"] = _pack(np.ascontiguousarray(log.searched, SEARCHED).tobytes())
    return d


def from_json(d: dict[str, Any]) -> GameLog:
    """A log from ``to_json``'s dict, registering its map if the engine hasn't it."""
    if d.get("format") != FORMAT:
        raise ValueError(f"unknown game record format {d.get('format')!r}")
    if _core.find_map(d["map"]) is None:
        terrain = np.frombuffer(_unpack(d["terrain"]), "<i2").reshape(d["height"], d["width"])
        _core.register_map(int(d["map"]), np.ascontiguousarray(terrain, np.int16))
    searched = None
    if "searched" in d:
        searched = np.frombuffer(_unpack(d["searched"]), SEARCHED).copy()
    return GameLog(
        start=_core.GameState.from_bytes(_unpack(d["start"])),
        steps=np.frombuffer(_unpack(d["steps"]), "<u2").astype(np.uint16),
        winner=int(d["winner"]),
        days=int(d["days"]),
        decided=str(d["decided"]),
        actions=int(d.get("actions", 0)),
        players=tuple(d.get("players") or ("", "")),  # type: ignore[arg-type]
        searched=searched,
        final=str(d.get("final", "")),
        source=str(d.get("source", "")),
        info=dict(d.get("info") or {}),
    )


def append(path: Path, logs: Iterable[GameLog]) -> int:
    """Add games to a record file; returns how many."""
    lines = [json.dumps(to_json(log), separators=(",", ":")) + "\n" for log in logs]
    if lines:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.writelines(lines)
    return len(lines)


def read(path: Path) -> Iterator[GameLog]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield from_json(json.loads(line))


@dataclass(slots=True)
class Move:
    """A completed action of a replayed game."""

    first: int  # index in steps of its first micro-step
    last: int  # and of the one that completed it
    action: tuple  # as _core.step_action gives it: (kind, unit slot, to, target, arg)
    mover: int
    status: int  # OK, or TRAPPED: the unit ran into a hidden enemy
    state: _core.GameState  # after it; the one before is the previous move's (or the start)


@dataclass(slots=True)
class Replayed:
    moves: list[Move]
    matches: bool  # the steps ended at the recorded final state (True if none was recorded)


def replay(log: GameLog) -> Replayed:
    """Take the log's steps from its first state. A step the engine refuses
    raises ValueError: the game was recorded under other rules."""
    s = log.start.copy()
    sel = _core.Selection()
    moves: list[Move] = []
    first = 0
    for i, step_id in enumerate(log.steps.tolist()):
        action = _core.step_action(s, sel, step_id)
        mover = int(s.active)
        st = _core.step(s, sel, step_id)
        if st not in (_core.OK, _core.TRAPPED):
            raise ValueError(f"step {i} ({step_id}) was refused: {_core.status_name(st)}; recorded under other rules?")
        if action is not None:
            moves.append(Move(first, i, tuple(action), mover, int(st), s.copy()))
            first = i + 1
    return Replayed(moves, not log.final or state_hash(s) == log.final)
