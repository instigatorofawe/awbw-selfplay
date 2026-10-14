"""Step through replays with the engine and compare against AWBW.

Each turn starts from its turn-start snapshot. The logged actions are applied
with the engine, and the result is checked against what the log shows after
each action (positions, fuel, fight results, ammo, power meters, capture
points, funds) and, after the turn ends, against the next turn's snapshot:
every unit, both players and every property.

Fights are forced to the logged result. The log only shows displayed HP, so a
fight can leave a unit at several internal HPs; the checker tries them in
turn (most likely first, guided by the next snapshot). Each unit's HP carries
the set of such choices it depends on, and a mismatch goes back only to the
choices it depends on (conflict-directed backjumping): a later fight's
result, or a final HP. Every other mismatch is the same for all choices and
is reported at once.

Turns are skipped while the engine lacks the rules they need: rain and snow
move costs, and CO powers with immediate effects.

``step_turn`` runs the same check from a given start state, to carry the
engine's HP beliefs (what watching every turn tells players) into the next
snapshot, and can record each action with the state it was taken in (policy
training labels).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import _core, engine, rules
from ..maps import map_path
from .combat_check import _view, unit_id
from .model import Replay, Turn
from .parse import ReplayFormatError, load_replay

# A unit's internal HP, HP belief and hidden-HP belief (GameState.hp_belief and
# hidden_belief), by AWBW unit id.
Beliefs = tuple[int, list[int], list[int]]


@dataclass(slots=True)
class Labelled:
    """An action a player took, with the state they took it in."""

    state: bytes  # the raw GameState before the action
    kind: int  # ACTIONS value
    steps: list[int]  # the micro-step ids that play it (legal.h)
    n_legal: list[int]  # how many ids were legal at each of those steps


A = _core.ACTIONS
_FLAGS = _core.UNIT_MOVED | _core.UNIT_HIDDEN | _core.UNIT_CAPTURING
_SEARCH_BUDGET = 100_000  # fight results tried per turn

# CO powers whose immediate effects the engine doesn't model yet, by AWBW CO
# id: 1 = COP, 2 = SCOP.
_UNMODELLED_POWERS: dict[int, tuple[int, ...]] = {}


# Powers whose effect lands at the start of the owner's next turn: skip the
# opponent's turn while one is on.
_UNMODELLED_CARRY_OVER: frozenset[tuple[int, int]] = frozenset()
_POWER_LEVEL = {"N": 0, "Y": 1, "S": 2}
_WEATHERS = {"C": "clear", "R": "rain", "S": "snow"}


def _shown(hp: int) -> int:
    return (hp + 9) // 10


def _logged_hp(value: Any) -> int | None:
    """A logged displayed HP, or None where the viewer isn't shown it ("?", Sonja's)."""
    return None if value == "?" else int(value)


class Diverged(Exception):
    """The engine and the replay disagree."""

    def __init__(self, category: str, detail: str, index: int, deps: frozenset[int] = frozenset()):
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail
        self.index = index  # action index in the turn; len(actions) for the end-of-turn check
        self.deps = deps  # fight choices (step numbers) the mismatch depends on


@dataclass(slots=True)
class TurnCheck:
    game_id: int
    day: int
    ok: bool | None  # None: not checked (see category)
    category: str = ""
    detail: str = ""
    action: int = -1
    # Missiles the engine aims where AWBW did, per power ("Rachel SCOP"): [matched, logged]
    aims: dict[str, list[int]] = field(default_factory=dict)


@dataclass(slots=True)
class StepReport:
    replays: int = 0
    replays_skipped: Counter = field(default_factory=Counter)
    turns: int = 0
    ok: int = 0
    actions_ok: int = 0  # actions applied in turns that matched
    skipped: Counter = field(default_factory=Counter)  # turns, by reason
    diverged: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)
    aims: dict[str, list[int]] = field(default_factory=dict)  # see TurnCheck.aims

    def add_turn(self, t: TurnCheck, n_actions: int) -> None:
        self.turns += 1
        if t.ok:
            self.ok += 1
            self.actions_ok += n_actions
            self._add_aims(t.aims)
        else:
            (self.skipped if t.ok is None else self.diverged)[t.category] += 1
            ex = self.examples.setdefault(t.category, [])
            if len(ex) < 5:
                ex.append(f"{t.game_id} day {t.day} action {t.action}: {t.detail}")

    def _add_aims(self, aims: dict[str, list[int]]) -> None:
        for k, (hit, n) in aims.items():
            tot = self.aims.setdefault(k, [0, 0])
            tot[0] += hit
            tot[1] += n

    def add(self, other: StepReport) -> None:
        self._add_aims(other.aims)
        self.replays += other.replays
        self.replays_skipped.update(other.replays_skipped)
        self.turns += other.turns
        self.ok += other.ok
        self.actions_ok += other.actions_ok
        self.skipped.update(other.skipped)
        self.diverged.update(other.diverged)
        for k, v in other.examples.items():
            ex = self.examples.setdefault(k, [])
            ex.extend(v[: max(0, 5 - len(ex))])


@dataclass(slots=True)
class _Step:
    """One engine action built from the log."""

    name: str  # logged action
    index: int  # position in the turn's action list
    log: dict
    uid: int = 0  # acting unit (AWBW id); 0 for player actions
    path: list[tuple[int, int]] | None = None
    moved: dict | None = None  # logged unit after its move
    target: tuple[int, int] | None = None
    other: int = 0  # second unit involved (AWBW id): cargo, repaired unit, defender


@dataclass(slots=True)
class _Ctx:
    """Search bookkeeping for one turn."""

    nodes: int = 0


_NONE: frozenset[int] = frozenset()


def _xy(p: dict) -> tuple[int, int]:
    return int(p["x"]), int(p["y"])


def _move_part(action: dict) -> dict | None:
    mv = action if action.get("action") == "Move" else action.get("Move")
    return mv if isinstance(mv, dict) and "paths" in mv else None


def _own_path(mv: dict, player_id: int) -> list[tuple[int, int]]:
    """The mover's own view of its path: the global view leaves out the steps
    a hidden unit took unseen."""
    paths = mv["paths"]
    own = paths.get(str(player_id)) if isinstance(paths, dict) else None
    return [_xy(p) for p in (own if isinstance(own, list) and own else _view(paths))]


class _TurnStepper:
    # What the end-of-turn check compares of each unit: (name, index into unit_rows).
    _END_FIELDS: tuple[tuple[str, Any], ...] = (
        ("type", 1),
        ("owner", 2),
        ("position", slice(3, 5)),
        ("hp", 5),
        ("fuel", 6),
        ("ammo", 7),
    )

    def __init__(self, replay: Replay, md: _core.MapData, turn: Turn, nxt: Turn | None):
        self.replay = replay
        self.md = md
        self.turn = turn
        self.nxt = nxt
        self.r = rules.load()
        self.index = engine.turn_order(replay)
        self.unit_type = {u.name: u.index for u in self.r.units}
        self.actions = turn.actions or []
        self.steps = self._steps()
        self.expected = engine.state_from_turn(replay, nxt, md) if nxt is not None else None
        # Final HP of every unit in the next snapshot, to order fight candidates.
        self.final_hp = {u.id: u.hp for u in nxt.units} if nxt is not None else {}
        self.next_player = self.index[nxt.player_id] if nxt is not None else -1
        # Missile aims by action index (the last search branch to pass a power overwrites).
        self.aims: dict[int, tuple[str, int, int]] = {}
        # Each unit's (internal HP, HP belief) at the end of a turn that matched, by AWBW id.
        self.end_beliefs: dict[int, Beliefs] | None = None
        self.unchecked: frozenset[int] = frozenset()  # units (AWBW ids) left out of the end-of-turn check
        self.check_legal = False  # also check each logged action is among the legal ones
        # With a dict: each applied action by log index, with the state before it.
        # A search branch that fails is overwritten by the one that passes.
        self.trace: dict[int, Labelled] | None = None

    def _hp_seen(self, uid: int, hp: int) -> int | None:
        """What the end-of-turn check compares of a unit's HP."""
        return hp

    # -- log -> steps ------------------------------------------------------

    def _steps(self) -> list[_Step]:
        steps: list[_Step] = []
        acts = self.actions
        i = 0
        while i < len(acts):
            a = acts[i]
            name = str(a.get("action"))
            mv = _move_part(a)
            step = _Step(name, i, a)
            if mv is not None:
                unit = _view(mv["unit"])
                step.uid = int(unit["units_id"])
                step.path = _own_path(mv, int(unit["units_players_id"]))
                step.moved = unit
            if name == "Move" and i + 1 < len(acts) and acts[i + 1].get("action") == "Unload":
                nxt = acts[i + 1]
                if int(_view(nxt.get("transportID"))) == step.uid:
                    # The transport's move and its first unload are one engine action.
                    un = _view(nxt["unit"])
                    step.name = "Unload"
                    step.log = nxt
                    step.other = int(un["units_id"])
                    step.target = (int(un["units_x"]), int(un["units_y"]))
                    step.index = i + 1
                    steps.append(step)
                    i += 2
                    continue
            if (
                name == "Move"
                and i + 1 < len(acts)
                and acts[i + 1].get("action") in ("Repair", "Supply")
                and _move_part(acts[i + 1]) is None
            ):
                nxt = acts[i + 1]
                if unit_id(nxt[nxt["action"]]["unit"]) == step.uid:
                    # Logged as a move and then the action on its own: one engine action.
                    step.name = str(nxt["action"])
                    step.log = nxt
                    step.index = i + 1
                    if step.name == "Repair":
                        step.other = int(_view(nxt["Repair"]["repaired"])["units_id"])
                    steps.append(step)
                    i += 2
                    continue
            if name == "Unload":
                un = _view(a["unit"])
                step.uid = int(_view(a["transportID"]))
                step.other = int(un["units_id"])
                step.target = (int(un["units_x"]), int(un["units_y"]))
            elif name == "Fire":
                info = _view(a["Fire"]["combatInfoVision"])["combatInfo"]
                step.uid = int(info["attacker"]["units_id"])
                step.other = int(info["defender"]["units_id"])
                step.target = (int(info["defender"]["units_x"]), int(info["defender"]["units_y"]))
            elif name == "Capt" and not step.uid:
                b = a["Capt"]["buildingInfo"]
                step.target = (int(b["buildings_x"]), int(b["buildings_y"]))
            elif name in ("Supply", "Repair", "Hide", "Unhide") and not step.uid:
                if _view(a[name].get("unit")) is not None:
                    step.uid = unit_id(a[name]["unit"])
            elif name == "AttackSeam":
                sa = a["AttackSeam"]
                info = _view(sa["unit"])["combatInfo"]
                step.uid = int(info["units_id"])
                step.target = (int(sa["seamX"]), int(sa["seamY"]))
            elif name == "Delete":
                step.uid = int(_view(a["Delete"]["unitId"]))
            if name == "Repair":
                step.other = int(_view(a["Repair"]["repaired"])["units_id"])
            steps.append(step)
            i += 1
        return steps

    # -- helpers -----------------------------------------------------------

    def _slot(self, state: _core.GameState, alias: dict[int, int], uid: int, index: int) -> int:
        slot = state.find_unit(alias.get(uid, uid))
        if slot < 0:
            raise Diverged("unknown unit", f"unit {uid} is not in the engine state", index)
        return slot

    def _apply(self, state: _core.GameState, index: int, kind: str, label: bool = True, **kw: Any) -> int:
        """Apply a logged action; ``label``: a player's decision (to record in ``trace``)."""
        if self.check_legal or (label and self.trace is not None):
            walked = _walk_steps(state, kind, kw)
            if walked is None and self.check_legal:
                raise Diverged(f"illegal: {kind}", f"not among the legal actions: {kw}", index)
            if label and self.trace is not None:
                if walked is None:
                    self.trace.pop(index, None)  # left unlabelled
                else:
                    self.trace[index] = Labelled(state.to_bytes(), A[kind], *walked)
        st = _core.apply(state, A[kind], **kw)
        if st not in (_core.OK, _core.TRAPPED):
            raise Diverged(f"{kind}: {_core.status_name(st)}", f"engine rejects {kind} {kw}", index)
        return st

    def _expect(self, what: str, got: Any, want: Any, index: int, deps: frozenset[int] = _NONE) -> None:
        if got != want:
            raise Diverged(what, f"engine {got}, replay {want}", index, deps)

    def _check_moved(self, state, alias, step: _Step) -> None:
        if step.moved is None or step.name == "Join":
            return
        slot = state.find_unit(alias.get(step.uid, step.uid))
        if slot < 0:
            return  # died in the action (counterattack)
        u = state.unit(slot)
        name = self.r.units[u.type].name
        x, y = self.md_xy(u.tile)
        self._expect(
            f"move position: {name}", (x, y), (int(step.moved["units_x"]), int(step.moved["units_y"])), step.index
        )
        self._check_supplies(state, slot, int(step.moved["units_fuel"]), None, "move", step.index)

    def _check_supplies(self, state, slot: int, fuel: int | None, ammo: int | None, what: str, index: int) -> None:
        """Check a unit's fuel and ammo (None: not logged) against the log."""
        u = state.unit(slot)
        name = self.r.units[u.type].name
        if fuel is not None:
            self._expect(f"{what} fuel: {name}", u.fuel, fuel, index)
        if ammo is not None:
            self._expect(f"{what} ammo: {name}", u.ammo, ammo, index)

    def md_xy(self, tile: int) -> tuple[int, int]:
        return tile % self.md.width, tile // self.md.width

    # -- search ------------------------------------------------------------

    def run(self, start: _core.GameState | None = None) -> None:
        """Step the turn from ``start`` (default: its snapshot); raises Diverged or _Skip."""
        state = start.copy() if start is not None else engine.state_from_turn(self.replay, self.turn, self.md)
        self._run(0, state, {}, {}, _Ctx())

    def _run(self, i: int, state, alias: dict[int, int], deps: dict[int, frozenset[int]], ctx: _Ctx) -> None:
        """Apply steps i.. and check the end of the turn. `deps` maps a unit to
        the fight choices its HP depends on."""
        while i < len(self.steps):
            step = self.steps[i]
            if step.name != "Fire":
                if self._step(state, alias, step):
                    return
                i += 1
                continue
            pre = deps.get(step.uid, _NONE) | deps.get(step.other, _NONE)
            cands = self._fire_candidates(state, alias, step, pre)
            if len(cands) == 1:
                if pre:
                    deps = {**deps, step.uid: pre, step.other: pre}
                self._apply_fight(state, alias, step, cands[0])
                i += 1
                continue
            here = pre | {i}
            conflict: set[int] = set()
            worst: Diverged | None = None
            for cand in cands:
                ctx.nodes += 1
                if ctx.nodes > _SEARCH_BUDGET:
                    raise Diverged("search budget", "too many internal-HP choices", step.index)
                s2, a2 = state.copy(), dict(alias)
                d2 = {**deps, step.uid: here, step.other: here}
                try:
                    self._apply_fight(s2, a2, step, cand)
                    self._run(i + 1, s2, a2, d2, ctx)
                    return
                except Diverged as e:
                    if i not in e.deps:
                        raise  # the same whatever this fight left
                    conflict |= e.deps
                    if worst is None or e.index > worst.index:
                        worst = e
            assert worst is not None
            raise Diverged(worst.category, worst.detail, worst.index, frozenset(conflict - {i}))
        self._end(state, alias, deps)

    # -- fights ------------------------------------------------------------

    def _fire_candidates(self, state, alias, step: _Step, pre: frozenset[int]) -> list[tuple[int, ...]]:
        """Arguments for ``_fire`` that give the logged result, most likely first."""
        info = _view(step.log["Fire"]["combatInfoVision"])["combatInfo"]
        want = (int(info["attacker"]["units_hit_points"]), int(info["defender"]["units_hit_points"]))
        slot = self._slot(state, alias, step.uid, step.index)
        here = self.md_xy(state.unit(slot).tile)
        dest = step.path[-1] if step.path else here
        assert step.target is not None
        outs = _core.fight_outcomes(state, slot, dest, step.target)
        cands = [(a, d, p) for a, d, p in outs if (_shown(a), _shown(d)) == want]
        if not cands:
            shown = sorted({(_shown(a), _shown(d)) for a, d, _ in outs})
            u = state.unit(slot)
            raise Diverged(
                f"fire result: {self.r.units[u.type].name}",
                f"logged {want}, engine allows {shown}",
                step.index,
                pre,
            )
        att_owner = state.unit(slot).owner

        def score(c: tuple[int, int, float]) -> tuple[int, float]:
            a, d, p = c
            miss = 0
            for uid, hp, owner in ((step.uid, a, att_owner), (step.other, d, 1 - att_owner)):
                final = self.final_hp.get(uid)
                if hp == 0 or final is None:
                    continue
                if owner == self.next_player:  # may be repaired at the start of its turn
                    miss += not (final == hp or final - hp in (10, 20, 30) or (final == 100 and hp >= 70))
                else:
                    miss += final != hp
            return miss, -p

        cands.sort(key=score)
        return [(a, d) for a, d, _ in cands]

    def _apply_fight(self, state, alias, step: _Step, cand: tuple[int, ...]) -> None:
        """Apply a fight as one of ``_fire_candidates``."""
        self._fire(state, alias, step, *cand)

    def _fire(self, state, alias, step: _Step, att_hp: int, def_hp: int) -> None:
        slot = self._slot(state, alias, step.uid, step.index)
        dslot = self._slot(state, alias, step.other, step.index)
        att_player, def_player = state.unit(slot).owner, state.unit(dslot).owner
        self._apply(
            state,
            step.index,
            "Fire",
            unit=slot,
            to=step.path[-1] if step.path else None,
            target=step.target,
            path=step.path,
            att_hp=att_hp,
            def_hp=def_hp,
        )
        self._check_moved(state, alias, step)
        fire = step.log["Fire"]
        info = _view(fire["combatInfoVision"])["combatInfo"]
        for uid, side in ((step.uid, "attacker"), (step.other, "defender")):
            s = state.find_unit(alias.get(uid, uid))
            logged_ammo = info[side].get("units_ammo")
            if s >= 0 and logged_ammo is not None:
                # AWBW can log negative ammo for a unit that fired with none left.
                self._check_supplies(state, s, None, max(0, int(logged_ammo)), "fire", step.index)
        cv = fire.get("copValues") or {}
        players = state.players
        for side, player in (("attacker", att_player), ("defender", def_player)):
            if isinstance(cv.get(side), dict):
                co = self.r.cos[players[player].co].name if players[player].co in self.r.cos else players[player].co
                self._expect(f"fire meter: {co}", players[player].power, int(cv[side]["copValue"]), step.index)

    # -- other actions -----------------------------------------------------

    def _step(self, state, alias, step: _Step) -> bool:
        """Apply one non-fight step. Returns True if the turn ends here (resignation)."""
        name, i = step.name, step.index
        a = step.log
        to = step.path[-1] if step.path else None
        if name == "Move":
            slot = self._slot(state, alias, step.uid, i)
            self._apply(state, i, "Wait", unit=slot, to=to, path=step.path)
            self._check_moved(state, alias, step)
        elif name == "Unload":
            slot = self._slot(state, alias, step.uid, i)
            cargo = self._slot(state, alias, step.other, i)
            held = list(state.unit(slot).cargo)
            if cargo not in held:
                raise Diverged("unload: not carried", f"unit {step.other} is not in transport {step.uid}", i)
            self._apply(state, i, "Unload", unit=slot, to=to, target=step.target, arg=held.index(cargo), path=step.path)
            self._check_moved(state, alias, step)
        elif name == "Capt":
            if step.uid:
                slot = self._slot(state, alias, step.uid, i)
            else:
                assert step.target is not None
                slot = state.unit_at(*step.target)
            self._apply(state, i, "Capture", unit=slot, to=to, path=step.path)
            self._check_moved(state, alias, step)
            b = a["Capt"]["buildingInfo"]
            x, y = int(b["buildings_x"]), int(b["buildings_y"])
            prop = self._prop_at(x, y)
            over = a["Capt"].get("eliminated")
            if isinstance(over, dict):  # the capture ends the game; AWBW logs the points unreset
                winners = (over.get("GameOver") or {}).get("winners") or []
                self._expect("capture win", state.winner, self.index[int(winners[0])] if winners else -1, i)
                return True
            self._expect("capture points", state.prop_points(prop), int(b["buildings_capture"]), i)
            if int(b["buildings_capture"]) == 20 and b.get("buildings_players_id"):
                self._expect("capture owner", state.prop_owner(prop), self.index[int(b["buildings_players_id"])], i)
        elif name == "Build":
            g = _view(a.get("newUnit"))
            if not isinstance(g, dict) or "units_id" not in g:
                return False  # a build that didn't happen; the snapshot check confirms nothing changed
            xy = (int(g["units_x"]), int(g["units_y"]))
            self._apply(state, i, "Build", target=xy, arg=self.unit_type[g["units_name"]])
            alias[int(g["units_id"])] = state.unit(state.unit_at(*xy)).id
        elif name == "Load":
            slot = self._slot(state, alias, step.uid, i)
            self._apply(state, i, "Load", unit=slot, to=to, path=step.path)
            self._check_moved(state, alias, step)
        elif name == "Join":
            slot = self._slot(state, alias, step.uid, i)
            self._apply(state, i, "Join", unit=slot, to=to, path=step.path)
            j = a["Join"]
            self._expect("join funds", state.players[state.active].funds, int(_view(j["newFunds"])), i)
            g = _view(j["unit"])
            into = self._slot(state, alias, int(g["units_id"]), i)
            u = state.unit(into)
            if (shown := _logged_hp(g["units_hit_points"])) is not None:
                self._expect("join hp", _shown(u.hp), shown, i)
            self._expect("join fuel", u.fuel, int(g["units_fuel"]), i)
            self._expect("join ammo", u.ammo, int(g["units_ammo"]), i)
        elif name == "Supply":
            slot = self._slot(state, alias, step.uid, i) if step.uid else self._supplier(state, alias, step)
            self._apply(state, i, "Supply", unit=slot, to=to, path=step.path)
            self._check_moved(state, alias, step)
        elif name == "Repair":
            slot = self._slot(state, alias, step.uid, i)
            target = state.unit(self._slot(state, alias, step.other, i))
            self._apply(state, i, "Repair", unit=slot, to=to, target=self.md_xy(target.tile), path=step.path)
            self._check_moved(state, alias, step)
            rep = a["Repair"]
            self._expect("repair funds", state.players[state.active].funds, int(_view(rep["funds"])), i)
            got = state.unit(self._slot(state, alias, step.other, i))
            if (shown := _logged_hp(_view(rep["repaired"])["units_hit_points"])) is not None:
                self._expect("repair hp", _shown(got.hp), shown, i)
        elif name in ("Hide", "Unhide"):
            slot = self._slot(state, alias, step.uid, i)
            self._apply(state, i, name, unit=slot, to=to, path=step.path)
            self._check_moved(state, alias, step)
        elif name == "AttackSeam":
            slot = self._slot(state, alias, step.uid, i)
            self._apply(state, i, "AttackSeam", unit=slot, to=to, target=step.target, path=step.path)
            self._check_moved(state, alias, step)
            assert step.target is not None
            sa = a["AttackSeam"]
            self._expect(
                "seam hp", state.prop_points(self._prop_at(*step.target)), max(0, int(sa["buildings_hit_points"])), i
            )
            info = _view(sa["unit"])["combatInfo"]
            # AWBW can log -1 here: it spends ammo the unit doesn't have.
            self._expect("seam ammo", state.unit(slot).ammo, max(0, int(info["units_ammo"])), i)
        elif name == "Delete":
            slot = self._slot(state, alias, step.uid, i)
            self._apply(state, i, "Delete", unit=slot)
        elif name == "Power":
            self._power(state, alias, a, i)
        elif name == "End":
            self._apply(state, i, "EndTurn")
            info = a["updatedInfo"]
            self._expect("next funds", state.players[state.active].funds, int(_view(info["nextFunds"])), i)
            repaired = info.get("repaired")
            # The repaired units are the next player's; the global view hides
            # Sonja's HP ("?"), her own view doesn't.
            own = repaired.get(str(info.get("nextPId"))) if isinstance(repaired, dict) else None
            for row in own if isinstance(own, list) else _view(repaired) or []:
                if not str(row.get("units_hit_points")).isdigit():
                    continue
                s = state.find_unit(alias.get(int(row["units_id"]), int(row["units_id"])))
                if s >= 0:
                    u = state.unit(s)
                    self._expect(
                        f"repaired hp: {self.r.units[u.type].name}", _shown(u.hp), int(row["units_hit_points"]), i
                    )
        elif name == "Resign":
            return True
        else:
            raise Diverged(f"unsupported action {name}", "no engine action for it", i)
        return False

    def _power(self, state, alias: dict[int, int], a: dict, i: int) -> None:
        p = state.players[state.active]
        level = 2 if a.get("coPower") == "S" else 1
        what = f"{self.r.cos[p.co].name} {'SCOP' if level == 2 else 'COP'}"
        if level in _UNMODELLED_POWERS.get(p.co, ()):
            raise _Skip(f"power: {what}")
        missiles = [(int(c["x"]), int(c["y"])) for c in a.get("missileCoords") or []]
        if missiles:  # how AWBW's aim compares with the engine's (not a divergence)
            aimed = _core.missile_targets(state, level)
            self.aims[i] = (what, sum(x == y for x, y in zip(aimed, missiles, strict=False)), len(missiles))
        before = set(state.unit_slots())
        self._apply(state, i, "Power", arg=level, missiles=missiles or None)
        self._expect("power meter", state.players[state.active].power, int(a["playersCOP"]), i)
        weather = a.get("weather")
        if isinstance(weather, dict):
            self._expect(f"{what}: weather", _core.WEATHERS[state.weather], _WEATHERS[weather["weatherCode"]], i)
        added = _view(a.get("unitAdd"))
        if isinstance(added, dict):  # Sensei's footsoldiers
            new = {self.md_xy(state.unit(s).tile): s for s in set(state.unit_slots()) - before}
            for row in added.get("units") or []:
                xy = (int(row["units_x"]), int(row["units_y"]))
                if xy not in new:
                    raise Diverged(f"{what}: unit not placed", f"no new unit at {xy}", i)
                alias[int(row["units_id"])] = state.unit(new.pop(xy)).id
            if new:
                raise Diverged(f"{what}: extra unit", f"engine places units at {sorted(new)}", i)
        replaced = _view(a.get("unitReplace"))
        rows = replaced.get("units") if isinstance(replaced, dict) else None
        for row in rows or []:
            slot = state.find_unit(alias.get(int(row["units_id"]), int(row["units_id"])))
            if slot < 0:
                continue
            u = state.unit(slot)
            name = self.r.units[u.type].name
            # Sonja hides HP from her opponent: the log shows 0 (powers never kill).
            if "units_hit_points" in row and (_logged_hp(row["units_hit_points"]) or 0) > 0:
                self._expect(f"{what}: hp of {name}", _shown(u.hp), int(row["units_hit_points"]), i)
            if "units_fuel" in row:
                self._expect(f"{what}: fuel of {name}", u.fuel, int(row["units_fuel"]), i)
            if "units_ammo" in row:
                self._expect(f"{what}: ammo of {name}", u.ammo, int(row["units_ammo"]), i)
            if "units_moved" in row:
                moved = int(row["units_moved"])  # 0: may act again (Eagle), -1: stunned (Von Bolt)
                flag = _core.UNIT_STUNNED if moved < 0 else _core.UNIT_MOVED
                self._expect(f"{what}: moved flag of {name}", u.flags & flag, flag if moved else 0, i)
        players = _view(a.get("playerReplace"))
        for pid, row in (players or {}).items() if isinstance(players, dict) else ():
            pi = self.index.get(int(pid))
            if pi is None or not isinstance(row, dict):
                continue
            if "players_co_power" in row:
                self._expect(f"{what}: power meter", state.players[pi].power, int(row["players_co_power"]), i)
            if "players_funds" in row:
                self._expect(f"{what}: funds", state.players[pi].funds, int(row["players_funds"]), i)

    def _supplier(self, state, alias: dict[int, int], step: _Step) -> int:
        """The unit that supplied where it stood: the active player's unmoved APC next
        to every unit supplied."""
        rows = [int(r) for r in step.log["Supply"].get("rows") or []]
        tiles = [state.unit(self._slot(state, alias, uid, step.index)).tile for uid in rows]
        apc = self.unit_type["APC"]
        found = []
        for slot in state.unit_slots():
            u = state.unit(slot)
            if u.owner != state.active or u.type != apc or u.flags & _core.UNIT_MOVED:
                continue
            x, y = self.md_xy(u.tile)
            if all(abs(x - tx) + abs(y - ty) == 1 for tx, ty in map(self.md_xy, tiles)):
                found.append(slot)
        if len(found) != 1:
            raise Diverged("supply: supplier", f"{len(found)} APCs could have supplied {rows}", step.index)
        return found[0]

    def _prop_at(self, x: int, y: int) -> int:
        for p, (px, py, _, _) in enumerate(self.md.props()):
            if (px, py) == (x, y):
                return p
        raise Diverged("no property", f"no property at {(x, y)}", -1)

    # -- end of turn -------------------------------------------------------

    def _end(self, state, alias: dict[int, int], deps: dict[int, frozenset[int]]) -> None:
        n = len(self.actions)
        if not self.steps or self.steps[-1].name != "End":
            if self.expected is None:
                return  # the game ended in this turn
            # Cut off at the max turn length: AWBW ends the turn itself.
            self._apply(state, n, "EndTurn", label=False)
        exp = self.expected
        if exp is None:
            return
        awbw_id = {v: k for k, v in alias.items()}
        self._expect("day", state.day, exp.day, n)
        self._expect("active player", state.active, exp.active, n)
        for pi, (got, want) in enumerate(zip(state.players, exp.players, strict=True)):
            for f in ("funds", "power", "cop_cost", "scop_cost", "power_on", "eliminated"):
                self._expect(f"player {f}", (pi, getattr(got, f)), (pi, getattr(want, f)), n)
        for p in range(self.md.n_props):
            xy = self.md.props()[p][:2]
            self._expect("property owner", (xy, state.prop_owner(p)), (xy, exp.prop_owner(p)), n)
            self._expect("property points", (xy, state.prop_points(p)), (xy, exp.prop_points(p)), n)
        mine = {awbw_id.get(r[0], r[0]): r for r in state.unit_rows()}
        theirs = {r[0]: r for r in exp.unit_rows()}
        for uid in self.unchecked:
            mine.pop(uid, None)
            theirs.pop(uid, None)
        for uid in sorted(mine.keys() | theirs.keys()):
            if uid not in theirs:
                name = self.r.units[mine[uid][1]].name
                raise Diverged(
                    f"extra unit: {name}",
                    f"unit {uid} at {mine[uid][3:5]} survives in the engine",
                    n,
                    deps.get(uid, _NONE),
                )
            if uid not in mine:
                name = self.r.units[theirs[uid][1]].name
                raise Diverged(
                    f"missing unit: {name}",
                    f"unit {uid} at {theirs[uid][3:5]} is gone in the engine",
                    n,
                    deps.get(uid, _NONE),
                )
            g, w = mine[uid], theirs[uid]
            name = self.r.units[w[1]].name
            for f, k in self._END_FIELDS:
                got, want = (self._hp_seen(uid, g[k]), self._hp_seen(uid, w[k])) if f == "hp" else (g[k], w[k])
                self._expect(
                    f"unit {f}: {name}", (uid, got), (uid, want), n, deps.get(uid, _NONE) if f == "hp" else _NONE
                )
            self._expect(f"unit flags: {name}", (uid, g[8] & _FLAGS), (uid, w[8] & _FLAGS), n)
            cargo = tuple(awbw_id.get(c, c) for c in g[9:11])
            self._expect(f"unit cargo: {name}", (uid, cargo), (uid, tuple(w[9:11])), n)
        self.end_beliefs = {
            awbw_id.get(r[0], r[0]): (r[5], state.hp_belief(slot), state.hidden_belief(slot))
            for r, slot in zip(state.unit_rows(), state.unit_slots(), strict=True)
        }


class _SpectatorStepper(_TurnStepper):
    """A turn as a spectator of a live game sees it (``step_spectator_turn``): units'
    displayed HP only, and none at all for Sonja's.

    Internal HPs are unknown, so a fight is tried from every pair of internal
    HPs the two units' beliefs allow (for a Sonja unit whose display her
    opponent doesn't know, every display her hidden belief allows). Beliefs
    only read what both players see, so any choice that gives the logged
    result, power meters and funds updates them the same; the most likely goes
    first. A capture or the repairs at a turn's start show a Sonja unit's
    display through capture points and funds, so those choose it too.

    The page's log for a game in progress leaves out supplies, so a unit's
    fuel and ammo are taken from the log and the end-of-turn check skips them.
    """

    _END_FIELDS = tuple(f for f in _TurnStepper._END_FIELDS if f[0] not in ("fuel", "ammo"))
    _MAX_REPAIR_CHOICES = 1000

    def __init__(
        self,
        replay: Replay,
        md: _core.MapData,
        turn: Turn,
        expected: _core.GameState,
        hidden: frozenset[int],
        unchecked: frozenset[int],
    ):
        super().__init__(replay, md, turn, None)
        self.expected = expected
        self.next_player = expected.active
        self.hidden = hidden
        self.unchecked = unchecked
        self.survivors = {r[0] for r in expected.unit_rows()}
        # Units each step's action list mentions from there on, by step index.
        self.mentioned_after: dict[int, set[int]] = {}
        seen: set[int] = set()
        for k in range(len(self.actions) - 1, -1, -1):
            self.mentioned_after[k] = set(seen)
            seen |= _unit_ids(self.actions[k])

    def _hp_seen(self, uid: int, hp: int) -> int | None:
        return None if uid in self.hidden else _shown(hp)

    def _check_supplies(self, state, slot: int, fuel: int | None, ammo: int | None, what: str, index: int) -> None:
        u = state.unit(slot)
        state.set_supplies(slot, u.fuel if fuel is None else fuel, u.ammo if ammo is None else ammo)

    def _display_unknown(self, state, slot: int) -> bool:
        u = state.unit(slot)
        return state.hides_hp(u.owner) and state.hidden_belief(slot).count(0) < _core.HP_BINS - 1

    def _hp_choices(self, state, slot: int) -> list[tuple[int, float]]:
        """(internal HP, probability) the unit's beliefs allow."""
        u = state.unit(slot)
        w = state.hp_belief(slot)
        here = _shown(u.hp)
        if not self._display_unknown(state, slot):
            total = sum(w)
            return [(10 * (here - 1) + 1 + k, p / total) for k, p in enumerate(w) if p]
        hidden = state.hidden_belief(slot)
        out = []
        for d, pd in enumerate(hidden, 1):
            if not pd:
                continue
            within = w if d == here else [1] * _core.HP_BINS
            total = sum(within) * sum(hidden)
            out += [(10 * (d - 1) + 1 + k, pd * p / total) for k, p in enumerate(within) if p]
        return out

    def _display_choices(self, state, slot: int) -> list[tuple[int, float]]:
        """(internal HP, probability) with one HP per display the beliefs allow: the
        unit's own HP for its current display."""
        best: dict[int, tuple[int, float]] = {}
        here = state.unit(slot).hp
        for hp, p in self._hp_choices(state, slot):
            d = _shown(hp)
            keep = here if d == _shown(here) else best.get(d, (hp, 0.0))[0]
            best[d] = (keep, best.get(d, (hp, 0.0))[1] + p)
        return sorted(best.values(), key=lambda c: -c[1])

    def _fire_candidates(self, state, alias, step: _Step, pre: frozenset[int]) -> list[tuple[int, ...]]:
        info = _view(step.log["Fire"]["combatInfoVision"])["combatInfo"]
        want = (_logged_hp(info["attacker"]["units_hit_points"]), _logged_hp(info["defender"]["units_hit_points"]))
        # A unit whose HP is hidden survived if it is still there later.
        later = self.mentioned_after.get(step.index, set()) | self.survivors
        alive = (step.uid in later, step.other in later)
        slot = self._slot(state, alias, step.uid, step.index)
        dslot = self._slot(state, alias, step.other, step.index)
        here = self.md_xy(state.unit(slot).tile)
        dest = step.path[-1] if step.path else here
        assert step.target is not None
        scored: list[tuple[float, tuple[int, int, int, int]]] = []
        for att_start, pa in self._hp_choices(state, slot):
            for def_start, pd in self._hp_choices(state, dslot):
                s2 = state.copy()
                s2.set_hp(slot, att_start)
                s2.set_hp(dslot, def_start)
                for a, d, p in _core.fight_outcomes(s2, slot, dest, step.target):
                    if all(
                        (_shown(hp) == w) if w is not None else (hp > 0 or not lives)
                        for hp, w, lives in zip((a, d), want, alive, strict=True)
                    ):
                        scored.append((pa * pd * p, (att_start, def_start, a, d)))
        scored.sort(key=lambda c: -c[0])
        cands: list[tuple[int, ...]] = []
        for _, cand in scored:  # keep those the rest of the log agrees with: ammo, meters, funds
            try:
                self._apply_fight(state.copy(), dict(alias), step, cand)
            except Diverged:
                continue
            cands.append(cand)
        if not cands:
            u = state.unit(slot)
            raise Diverged(
                f"fire result: {self.r.units[u.type].name}", f"no internal HPs give the logged {want}", step.index, pre
            )
        return cands

    def _apply_fight(self, state, alias, step: _Step, cand: tuple[int, ...]) -> None:
        att_start, def_start, att_hp, def_hp = cand
        state.set_hp(self._slot(state, alias, step.uid, step.index), att_start)
        state.set_hp(self._slot(state, alias, step.other, step.index), def_start)
        self._fire(state, alias, step, att_hp, def_hp)

    def _step(self, state, alias, step: _Step) -> bool:
        if step.name == "Capt":
            slot = self._slot(state, alias, step.uid, step.index) if step.uid else state.unit_at(*step.target)
            if slot >= 0 and self._display_unknown(state, slot):
                return self._choose(
                    state, alias, step, [[(slot, hp, p)] for hp, p in self._display_choices(state, slot)]
                )
        if step.name == "End" and state.hides_hp(1 - state.active):
            return self._choose_repairs(state, alias, step)
        return super()._step(state, alias, step)

    def _choose(self, state, alias, step: _Step, choices: list[list[tuple[int, int, float]]]) -> bool:
        """Apply the step after setting units' HPs as the first of ``choices`` (each a
        list of (slot, HP, probability)) that the log agrees with, most likely first."""

        def weight(c: list[tuple[int, int, float]]) -> float:
            w = 1.0
            for _, _, p in c:
                w *= p
            return w

        worst: Diverged | None = None
        for choice in sorted(choices, key=lambda c: -weight(c)):
            s2, a2 = state.copy(), dict(alias)
            for slot, hp, _ in choice:
                s2.set_hp(slot, hp)
            try:
                ended = super()._step(s2, a2, step)
            except Diverged as e:
                worst = worst or e
                continue
            state.assign(s2)
            alias.clear()
            alias.update(a2)
            return ended
        assert worst is not None
        raise worst

    def _choose_repairs(self, state, alias, step: _Step) -> bool:
        """End the turn, choosing the displays of the next player's Sonja units whose
        display is unknown and that stand on their own properties (where they may be
        repaired) to agree with the funds and HPs logged."""
        nxt = 1 - state.active
        props = {self.md.props()[p][:2] for p in range(self.md.n_props) if state.prop_owner(p) == nxt}
        slots = [
            sl
            for sl in state.unit_slots()
            if state.unit(sl).owner == nxt
            and self.md_xy(state.unit(sl).tile) in props
            and self._display_unknown(state, sl)
            and not state.unit(sl).flags & _core.UNIT_CARRIED
        ]
        if not slots:
            return super()._step(state, alias, step)
        per_unit = [[(sl, hp, p) for hp, p in self._display_choices(state, sl)] for sl in slots]
        keep = len(per_unit[0])
        while keep > 1 and _product(min(len(c), keep) for c in per_unit) > self._MAX_REPAIR_CHOICES:
            keep -= 1
        combos: list[list[tuple[int, int, float]]] = [[]]
        for opts in per_unit:
            combos = [[*c, o] for c in combos for o in opts[:keep]]
        return self._choose(state, alias, step, combos)


def _product(values) -> int:
    out = 1
    for v in values:
        out *= v
    return out


_ID_KEYS = frozenset(
    ("units_id", "unitId", "deletedId", "loadedId", "transportId", "transportID", "joinID", "joinId", "loaded")
)


def _unit_ids(obj: Any) -> set[int]:
    """AWBW unit ids a log entry refers to."""
    out: set[int] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            v = _view(v)
            if isinstance(v, (dict, list)):
                out |= _unit_ids(v)
            elif (k in _ID_KEYS or k in ("unit", "transport")) and str(v).isdigit():
                out.add(int(v))
    elif isinstance(obj, list):
        for v in obj:
            out |= _unit_ids(v)
    return out


def _walk_steps(state: _core.GameState, kind: str, kw: dict[str, Any]) -> tuple[list[int], list[int]] | None:
    """The micro-step ids that play the action and how many ids were legal at each,
    or None if one of them isn't legal."""
    args = {k: kw[k] for k in ("unit", "to", "target", "arg") if k in kw}
    ids = _core.action_steps(state, A[kind], **args)
    sel = _core.Selection()
    n_legal = []
    for n, i in enumerate(ids):
        legal = _core.legal_ids(state, sel)
        j = int(legal.searchsorted(i))
        if j == len(legal) or legal[j] != i:
            return None
        n_legal.append(len(legal))
        if n + 1 < len(ids):
            _core.step(state, sel, i)  # a pick: only the selection changes
    return ids, n_legal


class _Skip(Exception):
    pass


def check_turns(replay: Replay, md: _core.MapData) -> list[tuple[TurnCheck, int]]:
    """Check every turn of a replay, and that each logged action is legal; returns
    (result, number of actions) per turn."""
    out: list[tuple[TurnCheck, int]] = []
    gid = replay.game.id
    order = replay.player_ids
    for k, turn in enumerate(replay.turns):
        nxt = replay.turns[k + 1] if k + 1 < len(replay.turns) else None
        n = len(turn.actions or [])
        res = TurnCheck(gid, turn.day, None)
        j = (order.index(turn.player_id) + 1) % len(order)
        if turn.actions is None:
            res.category = "no actions logged"
        elif nxt is not None and (nxt.day, nxt.player_id) != (turn.day + (j == 0), order[j]):
            res.category = "next snapshot missing"
        elif carried := [
            p for p in turn.players if (p.co_id, _POWER_LEVEL.get(p.co_power_on, 0)) in _UNMODELLED_CARRY_OVER
        ]:
            res.category = f"power on: {rules.load().cos[carried[0].co_id].name}"
        else:
            try:
                stepper = _TurnStepper(replay, md, turn, nxt)
                stepper.check_legal = True
                stepper.run()
                res.ok = True
                for what, hit, n in stepper.aims.values():
                    tot = res.aims.setdefault(what, [0, 0])
                    tot[0] += hit
                    tot[1] += n
            except _Skip as e:
                res.category = str(e)
            except Diverged as e:
                res.ok = False
                res.category, res.detail, res.action = e.category, e.detail, e.index
            except ValueError as e:  # a snapshot the engine can't load
                res.category = f"snapshot: {e}"
        out.append((res, n))
    return out


def step_turn(
    replay: Replay,
    md: _core.MapData,
    k: int,
    start: _core.GameState,
    trace: dict[int, Labelled] | None = None,
) -> tuple[_core.GameState, dict[int, Beliefs] | None, str]:
    """Step turn ``k`` of a replay from ``start`` (its snapshot, with HP beliefs) and
    check it against turn ``k + 1``'s snapshot.

    Returns that snapshot's state; each unit's internal HP and HP beliefs at the
    end of turn ``k`` by AWBW unit id, or None if the turn couldn't be stepped
    or didn't match; and why not ("" if it matched). Raises ValueError if the
    next snapshot can't be loaded.

    With ``trace``, each action the player took is put there by log index
    (``Labelled``); its contents mean nothing unless the turn matched.
    """
    turn, nxt = replay.turns[k], replay.turns[k + 1]
    order = replay.player_ids
    j = (order.index(turn.player_id) + 1) % len(order)
    stepper = _TurnStepper(replay, md, turn, nxt)
    stepper.trace = trace
    assert stepper.expected is not None
    if turn.actions is None:
        return stepper.expected, None, "no actions logged"
    if (nxt.day, nxt.player_id) != (turn.day + (j == 0), order[j]):
        return stepper.expected, None, "next snapshot missing"
    try:
        stepper.run(start)
    except _Skip as e:
        return stepper.expected, None, str(e)
    except Diverged as e:
        return stepper.expected, None, e.category
    except (KeyError, TypeError, ValueError, IndexError) as e:  # a log entry missing what the checker reads
        return stepper.expected, None, f"unreadable log ({type(e).__name__})"
    return stepper.expected, stepper.end_beliefs, ""


def step_last_turn(replay: Replay, md: _core.MapData, start: _core.GameState, trace: dict[int, Labelled]) -> str:
    """Step a replay's last turn, the one the game ends in, from ``start``, recording
    its actions in ``trace`` as step_turn does. No snapshot follows it, so only
    the log checks it. Returns why it couldn't be stepped ("" if it was)."""
    turn = replay.turns[-1]
    if turn.actions is None:
        return "no actions logged"
    stepper = _TurnStepper(replay, md, turn, None)
    stepper.trace = trace
    try:
        stepper.run(start)
    except _Skip as e:
        return str(e)
    except Diverged as e:
        return e.category
    except (KeyError, TypeError, ValueError, IndexError) as e:
        return f"unreadable log ({type(e).__name__})"
    return ""


def step_spectator_turn(
    replay: Replay,
    md: _core.MapData,
    k: int,
    start: _core.GameState,
    expected: _core.GameState,
    *,
    hidden: frozenset[int] = frozenset(),
    unchecked: frozenset[int] = frozenset(),
) -> tuple[dict[int, Beliefs] | None, str]:
    """Step turn ``k`` of a game as a spectator sees it (``_SpectatorStepper``), from
    ``start`` (with HP beliefs), and check it against ``expected``, the next turn's
    start as shown: displayed HP only, and none for units in ``hidden``. Units in
    ``unchecked`` (AWBW ids) are left out of that check.

    Returns each unit's internal HP (one the beliefs allow) and beliefs at the
    end of the turn by AWBW id, or None if the turn couldn't be stepped or
    didn't match; and why not ("" if it matched).
    """
    turn = replay.turns[k]
    if turn.actions is None:
        return None, "no actions logged"
    stepper = _SpectatorStepper(replay, md, turn, expected, hidden, unchecked)
    try:
        stepper.run(start)
    except _Skip as e:
        return None, str(e)
    except Diverged as e:
        return None, e.category
    except (KeyError, TypeError, ValueError, IndexError) as e:  # a log entry missing what the checker reads
        return None, f"unreadable log ({type(e).__name__})"
    return stepper.end_beliefs, ""


def check_file(path: Path) -> StepReport:
    rep = StepReport()
    try:
        replay = load_replay(path)
    except ReplayFormatError:
        rep.replays_skipped["parse error"] += 1
        return rep
    if len(replay.player_ids) != 2 or replay.game.fog or not map_path(replay.game.map_id).exists():
        rep.replays_skipped["not 2 players, fog, or map not cached"] += 1
        return rep
    rep.replays = 1
    md = engine.map_data(replay.game.map_id)
    for res, n in check_turns(replay, md):
        rep.add_turn(res, n)
    return rep
