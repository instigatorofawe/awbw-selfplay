"""Check snapshot loading and movement against replays.

Every turn-start snapshot is loaded into an engine GameState and checked for
consistency. Logged moves are checked against the engine's movement rules:
the path must be legal, its cost must equal the fuel spent, and the
cheapest-path search must reach the destination at no more than that cost.

Moves are checked against the turn-start state only while that state still
decides their legality (check-turns applies every action): up to the first
action that can change movement (a fight, which may remove blocking enemies;
a CO power; a pipe seam attack; a resupply).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .. import _core, engine
from ..maps import map_path
from .combat_check import _view
from .parse import ReplayFormatError, load_replay

_STOP = frozenset(["Fire", "Power", "AttackSeam", "Supply", "Repair"])


@dataclass(slots=True)
class MoveReport:
    snapshots: int = 0
    snapshot_errors: list[str] = field(default_factory=list)
    moves: int = 0
    moves_ok: int = 0
    failures: list[str] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)

    def add(self, other: MoveReport) -> None:
        self.snapshots += other.snapshots
        self.snapshot_errors += other.snapshot_errors
        self.moves += other.moves
        self.moves_ok += other.moves_ok
        self.failures += other.failures
        self.skipped.update(other.skipped)


def _moves(action: dict) -> dict | None:
    move = action if action.get("action") == "Move" else action.get("Move")
    return move if isinstance(move, dict) and "paths" in move else None


def check_replay(path: Path) -> MoveReport:
    rep = MoveReport()
    try:
        replay = load_replay(path)
    except ReplayFormatError as e:
        rep.snapshot_errors.append(f"{path.name}: parse error: {e}")
        return rep
    if len(replay.player_ids) != 2 or replay.game.fog or not map_path(replay.game.map_id).exists():
        rep.skipped["replay"] += 1
        return rep
    md = engine.map_data(replay.game.map_id)
    gid = replay.game.id
    for turn in replay.turns:
        rep.snapshots += 1
        try:
            state = engine.state_from_turn(replay, turn, md)
        except ValueError as e:
            rep.snapshot_errors.append(f"{gid} day {turn.day}: {e}")
            continue
        slot_of = {state.unit(s).id: s for s in state.unit_slots()}
        for action in turn.actions or []:
            if action.get("action") in _STOP:
                break
            move = _moves(action)
            if move is None:
                continue
            unit = _view(move["unit"])
            steps = [(int(p["x"]), int(p["y"])) for p in _view(move["paths"])]
            slot = slot_of.get(int(unit["units_id"]))
            if slot is None or len(steps) < 2:
                rep.skipped["new unit" if slot is None else "no movement"] += 1
                continue
            rep.moves += 1
            before = state.unit(slot)
            cost = _core.path_cost(state, slot, steps)
            spent = before.fuel - int(unit["units_fuel"])
            dest = steps[-1]
            best = {(x, y): c for x, y, c in _core.reachable(state, slot)}.get(dest)
            where = f"{gid} day {turn.day} unit {before.id} {steps[0]}->{dest}"
            if cost < 0:
                rep.failures.append(f"{where}: engine rejects the logged path {steps}")
            elif cost != spent:
                rep.failures.append(f"{where}: path cost {cost} but {spent} fuel spent")
            elif best is None or best > cost:
                rep.failures.append(f"{where}: search cost {best} for a path costing {cost}")
            else:
                rep.moves_ok += 1
    return rep
