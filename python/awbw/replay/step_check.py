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

A = _core.ACTIONS
_FLAGS = _core.UNIT_MOVED | _core.UNIT_HIDDEN | _core.UNIT_CAPTURING
_SEARCH_BUDGET = 20000  # fight results tried per turn

# CO powers whose immediate effects the engine doesn't model yet, by AWBW CO
# id: 1 = COP, 2 = SCOP. Powers that only change modifiers (and Colin's and
# Sami's) are modelled.
_UNMODELLED_POWERS: dict[int, tuple[int, ...]] = {
    1: (1, 2),  # Andy: heal
    5: (1, 2),  # Drake: damage, fuel, rain
    9: (1, 2),  # Olaf: snow, damage
    10: (2,),  # Eagle: refresh
    12: (1, 2),  # Hawke: damage and heal
    13: (1, 2),  # Sensei: spawn units
    14: (1, 2),  # Jess: resupply
    19: (1, 2),  # Sasha: meter drain; funds from damage dealt
    23: (1,),  # Kindle: urban damage
    28: (2,),  # Rachel: missiles
    29: (1, 2),  # Sturm: meteor
    30: (2,),  # Von Bolt: missile and stun
}


# Powers whose effect lands at the start of the owner's next turn: skip the
# opponent's turn while one is on.
_UNMODELLED_CARRY_OVER = frozenset([(19, 2)])  # Sasha's War Bonds pay out then
_POWER_LEVEL = {"N": 0, "Y": 1, "S": 2}


def _shown(hp: int) -> int:
    return (hp + 9) // 10


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

    def add_turn(self, t: TurnCheck, n_actions: int) -> None:
        self.turns += 1
        if t.ok:
            self.ok += 1
            self.actions_ok += n_actions
        else:
            (self.skipped if t.ok is None else self.diverged)[t.category] += 1
            ex = self.examples.setdefault(t.category, [])
            if len(ex) < 5:
                ex.append(f"{t.game_id} day {t.day} action {t.action}: {t.detail}")

    def add(self, other: StepReport) -> None:
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

    def _apply(self, state: _core.GameState, index: int, kind: str, **kw: Any) -> int:
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
        self._expect(f"move fuel: {name}", u.fuel, int(step.moved["units_fuel"]), step.index)

    def md_xy(self, tile: int) -> tuple[int, int]:
        return tile % self.md.width, tile // self.md.width

    # -- search ------------------------------------------------------------

    def run(self) -> None:
        state = engine.state_from_turn(self.replay, self.turn, self.md)
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
                self._fire(state, alias, step, *cands[0])
                i += 1
                continue
            here = pre | {i}
            conflict: set[int] = set()
            worst: Diverged | None = None
            for att_hp, def_hp in cands:
                ctx.nodes += 1
                if ctx.nodes > _SEARCH_BUDGET:
                    raise Diverged("search budget", "too many internal-HP choices", step.index)
                s2, a2 = state.copy(), dict(alias)
                d2 = {**deps, step.uid: here, step.other: here}
                try:
                    self._fire(s2, a2, step, att_hp, def_hp)
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

    def _fire_candidates(self, state, alias, step: _Step, pre: frozenset[int]) -> list[tuple[int, int]]:
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
                u = state.unit(s)
                self._expect(f"fire ammo: {self.r.units[u.type].name}", u.ammo, int(logged_ammo), step.index)
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
            self._expect("join hp", _shown(u.hp), int(g["units_hit_points"]), i)
            self._expect("join fuel", u.fuel, int(g["units_fuel"]), i)
            self._expect("join ammo", u.ammo, int(g["units_ammo"]), i)
        elif name == "Supply":
            slot = self._slot(state, alias, step.uid, i)
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
            self._expect("repair hp", _shown(got.hp), int(_view(rep["repaired"])["units_hit_points"]), i)
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
            p = state.players[state.active]
            level = 2 if a.get("coPower") == "S" else 1
            if level in _UNMODELLED_POWERS.get(p.co, ()):
                raise _Skip(f"power: {self.r.cos[p.co].name} {'SCOP' if level == 2 else 'COP'}")
            self._apply(state, i, "Power", arg=level)
            self._expect("power meter", state.players[state.active].power, int(a["playersCOP"]), i)
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
            self._apply(state, n, "EndTurn")
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
            fields = (("type", 1), ("owner", 2), ("position", slice(3, 5)), ("hp", 5), ("fuel", 6), ("ammo", 7))
            for f, k in fields:
                self._expect(
                    f"unit {f}: {name}", (uid, g[k]), (uid, w[k]), n, deps.get(uid, _NONE) if f == "hp" else _NONE
                )
            self._expect(f"unit flags: {name}", (uid, g[8] & _FLAGS), (uid, w[8] & _FLAGS), n)
            cargo = tuple(awbw_id.get(c, c) for c in g[9:11])
            self._expect(f"unit cargo: {name}", (uid, cargo), (uid, tuple(w[9:11])), n)


class _Skip(Exception):
    pass


def check_turns(replay: Replay, md: _core.MapData) -> list[tuple[TurnCheck, int]]:
    """Check every turn of a replay; returns (result, number of actions) per turn."""
    out: list[tuple[TurnCheck, int]] = []
    gid = replay.game.id
    for k, turn in enumerate(replay.turns):
        nxt = replay.turns[k + 1] if k + 1 < len(replay.turns) else None
        n = len(turn.actions or [])
        res = TurnCheck(gid, turn.day, None)
        if turn.actions is None:
            res.category = "no actions logged"
        elif turn.weather != "C" or (nxt is not None and nxt.weather != "C"):
            res.category = "weather"
        elif carried := [
            p for p in turn.players if (p.co_id, _POWER_LEVEL.get(p.co_power_on, 0)) in _UNMODELLED_CARRY_OVER
        ]:
            res.category = f"power on: {rules.load().cos[carried[0].co_id].name}"
        else:
            try:
                _TurnStepper(replay, md, turn, nxt).run()
                res.ok = True
            except _Skip as e:
                res.category = str(e)
            except Diverged as e:
                res.ok = False
                res.category, res.detail, res.action = e.category, e.detail, e.index
            except ValueError as e:  # a snapshot the engine can't load
                res.category = f"snapshot: {e}"
        out.append((res, n))
    return out


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
