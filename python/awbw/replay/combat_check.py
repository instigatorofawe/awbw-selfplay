"""Check the engine's damage formula against every attack logged in replays.

Each turn is replayed from its turn-start snapshot, which gives every unit's
exact internal HP. Logged combat results only give displayed HP, so after a
fight a unit's internal HP is known only up to a set of candidates: the
results of the engine's ``combat_outcomes`` that match the log. An attack
passes if some candidate pair of pre-fight HPs can produce the logged result.

This needs only the combat core, not the full engine: positions, ammo, CO
powers and property ownership are tracked from the log itself.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import _core, rules
from ..maps import MapInfo, load_map, map_path
from .model import Replay, Turn
from .parse import ReplayFormatError, load_replay

HP_RANGE = frozenset(range(1, 101))
_HIDDEN = ("Y", "D")  # awbwUnit.sub_dive values for a dived Sub / hidden Stealth


@dataclass(slots=True)
class _Unit:
    id: int
    type: int  # engine unit index
    owner: int  # player id
    x: int
    y: int
    hp: frozenset[int]  # candidate internal HPs
    ammo: int
    carried: bool
    hidden: bool = False  # dived Sub / hidden Stealth


@dataclass(slots=True)
class _Player:
    id: int
    co: int
    country: int
    power: int  # 0 none, 1 COP, 2 SCOP
    funds: int
    colin_atk: int = 0  # Colin's SCOP attack bonus, fixed by his funds when it starts


@dataclass(slots=True)
class AttackCheck:
    game_id: int
    day: int
    ok: bool | None  # None: skipped (see reason)
    reason: str = ""
    attacker: str = ""
    defender: str = ""
    att_co: int = 0
    att_power: int = 0
    def_co: int = 0
    def_power: int = 0
    dist: int = 0
    att_hp: tuple[int, ...] = ()  # candidate internal HPs before the fight
    def_hp: tuple[int, ...] = ()
    logged: tuple[int, int] = (0, 0)  # displayed HP after: attacker, defender
    predicted: tuple[tuple[int, int], ...] = ()  # displayed results the engine allows
    detail: dict[str, Any] = field(default_factory=dict)


def _view(obj: Any) -> Any:
    """AWBW logs a copy of most payloads per viewer ("global" plus player ids).

    Return the global copy, with fields it hides ("?", as for Sonja's HP)
    filled in from the players' copies. A hidden Stealth's move has no global
    copy and an empty string for the opponent, so empty values are filled too.
    """
    if not (isinstance(obj, dict) and obj and all(k == "global" or str(k).isdigit() for k in obj)):
        return obj
    views = sorted(obj.items(), key=lambda kv: kv[0] != "global")
    merged = views[0][1]
    for _, other in views[1:]:
        merged = _fill(merged, other)
    return merged


def unit_id(payload: Any) -> int:
    """A logged unit reference: an id, or (in some Unhide logs) the whole unit."""
    v = _view(payload)
    return int(v["units_id"]) if isinstance(v, dict) else int(v)


def _fill(a: Any, b: Any) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _fill(a.get(k), b.get(k)) if k in b else a[k] for k in a} | {
            k: v for k, v in b.items() if k not in a
        }
    return b if a in ("?", "", None) else a


def _displayed(hp: int) -> int:
    return (hp + 9) // 10


class _TurnState:
    def __init__(self, replay: Replay, turn: Turn, game_map: MapInfo):
        self.r = rules.load()
        self.game_id = replay.game.id
        self.day = turn.day
        self.active = turn.player_id
        self.unit_index = {u.name: u.index for u in self.r.units}
        country_owner = {p.countries_id: p.id for p in turn.players}
        self.players = {
            p.id: _Player(p.id, p.co_id, p.countries_id, {"N": 0, "Y": 1, "S": 2}.get(p.co_power_on, 0), p.funds)
            for p in turn.players
        }
        self.terrain_id: dict[tuple[int, int], int] = {}
        for y in range(game_map.height):
            for x in range(game_map.width):
                self.terrain_id[(x, y)] = int(game_map.terrain[y, x])
        self.owner: dict[tuple[int, int], int] = {}
        for b in turn.buildings:
            self.terrain_id[(b.x, b.y)] = b.terrain_id
            t = self.r.terrain.get(b.terrain_id)
            if t is not None and t.country in country_owner:
                self.owner[(b.x, b.y)] = country_owner[t.country]
        self.units: dict[int, _Unit] = {}
        for u in turn.units:
            self.units[u.id] = _Unit(
                u.id,
                self.unit_index[u.name],
                u.players_id,
                u.x,
                u.y,
                frozenset([u.hp]),
                u.ammo,
                u.carried,
                u.sub_dive in _HIDDEN,
            )

    # -- terrain ---------------------------------------------------------------

    def _kind(self, x: int, y: int) -> tuple[int, int]:
        t = self.r.terrain[self.terrain_id[(x, y)]]
        return self.r.terrain_kinds.index(t.kind), t.defense

    def _towers(self, pid: int) -> int:
        return sum(1 for pos, o in self.owner.items() if o == pid and self._kind_name(pos) == "comtower")

    def _urban(self, pid: int) -> int:
        urban = rules.TERRAIN_GROUPS["urban"]
        return sum(1 for pos, o in self.owner.items() if o == pid and self._kind_name(pos) in urban)

    def _kind_name(self, pos: tuple[int, int]) -> str:
        return self.r.terrain[self.terrain_id[pos]].kind

    def combatant(self, u: _Unit, hp: int) -> _core.Combatant:
        p = self.players[u.owner]
        kind, stars = self._kind(u.x, u.y)
        extra = 0
        if p.power == 2 and p.co == 15:  # Colin: +3% attack per 1000 funds at activation
            extra = p.colin_atk
        elif p.power == 2 and p.co == 23:  # Kindle: +3% attack per owned urban property
            extra = 3 * self._urban(u.owner)
        # AWBW can log ammo -1 (after spending ammo the unit didn't have).
        ammo = max(0, u.ammo)
        return _core.Combatant(u.type, hp, ammo, p.co, p.power, kind, stars, self._towers(u.owner), extra, u.hidden)

    # -- actions ---------------------------------------------------------------

    def _moved_unit(self, payload: dict) -> None:
        g = _view(payload.get("unit"))
        if not isinstance(g, dict) or "units_id" not in g:
            return
        uid = int(g["units_id"])
        u = self.units.get(uid)
        if u is None:
            return
        u.x, u.y = int(g["units_x"]), int(g["units_y"])
        u.ammo = int(g["units_ammo"])
        u.carried = g.get("units_carried") == "Y"
        u.hidden = g.get("units_sub_dive") in _HIDDEN

    def apply(self, action: dict) -> Iterator[AttackCheck]:
        kind = action.get("action")
        move = action.get("Move")
        if isinstance(move, dict):
            self._moved_unit(move)
        if kind == "Move":
            self._moved_unit(action)
        elif kind == "Build":
            g = _view(action["newUnit"])
            if not isinstance(g, dict):  # AWBW sometimes logs a build with no unit
                return
            uid = int(g["units_id"])
            self.units[uid] = _Unit(
                uid,
                self.unit_index[g["units_name"]],
                int(g["units_players_id"]),
                int(g["units_x"]),
                int(g["units_y"]),
                frozenset([round(float(g["units_hit_points"]) * 10)]),
                int(g["units_ammo"]),
                False,
            )
            p = self.players[int(g["units_players_id"])]
            co = self.r.cos.get(p.co)
            p.funds -= int(g["units_cost"]) * (co.state(p.power).cost if co else 100) // 100
        elif kind == "Fire":
            yield self._fire(action["Fire"])
        elif kind == "Capt":
            info = action["Capt"]["buildingInfo"]
            if int(info["buildings_capture"]) == 20 and info.get("buildings_players_id"):
                pos = (int(info["buildings_x"]), int(info["buildings_y"]))
                self.owner[pos] = int(info["buildings_players_id"])
                if "terrain_id" in info:
                    self.terrain_id[pos] = int(info["terrain_id"])
        elif kind == "Join":
            j = action["Join"]
            gone = int(_view(j["joinID"]))
            g = _view(j["unit"])
            self.units.pop(gone, None)
            u = self.units.get(int(g["units_id"]))
            if u is not None:
                u.hp = self._with_display(HP_RANGE, int(g["units_hit_points"]))
                u.ammo = int(g["units_ammo"])
            p = self.players[int(j["playerId"])]
            p.funds = int(_view(j["newFunds"]))
        elif kind == "Load":
            u = self.units.get(int(_view(action["Load"]["loaded"])))
            if u is not None:
                u.carried = True
        elif kind == "Unload":
            self._moved_unit(action)
        elif kind == "Repair":
            rep = _view(action["Repair"]["repaired"])
            u = self.units.get(int(rep["units_id"]))
            if u is not None:
                u.hp = self._with_display(frozenset(min(100, h + 10) for h in u.hp), int(rep["units_hit_points"]))
            self.players[self.active].funds = int(_view(action["Repair"]["funds"]))
        elif kind == "Supply":
            for uid in action["Supply"].get("rows") or []:
                u = self.units.get(int(uid))
                if u is not None:
                    u.ammo = self.r.units[u.type].ammo
        elif kind in ("Hide", "Unhide"):
            u = self.units.get(unit_id(action[kind]["unit"]))
            if u is not None:
                u.hidden = kind == "Hide"
        elif kind == "Delete":
            self.units.pop(int(_view(action["Delete"]["unitId"])), None)
        elif kind == "Power":
            self._power(action)
        elif kind == "AttackSeam":
            seam = action["AttackSeam"]
            g = _view(seam.get("unit"))
            info = g.get("combatInfo") if isinstance(g, dict) else None
            if isinstance(info, dict):
                u = self.units.get(int(info["units_id"]))
                if u is not None:
                    u.ammo = int(info["units_ammo"])
            if int(seam.get("buildings_hit_points", 1)) <= 0:
                pos = (int(seam["seamX"]), int(seam["seamY"]))
                self.terrain_id[pos] = 115 if int(seam["buildings_terrain_id"]) == 113 else 116

    @staticmethod
    def _with_display(cands: frozenset[int], shown: int) -> frozenset[int]:
        kept = frozenset(h for h in cands if _displayed(h) == shown)
        return kept or frozenset(h for h in HP_RANGE if _displayed(h) == shown)

    def _power(self, action: dict) -> None:
        pid = int(action["playerID"])
        p = self.players[pid]
        p.power = 1 if action.get("coPower") == "Y" else 2
        if p.co == 15 and p.power == 2:
            p.colin_atk = 3 * p.funds // 1000
        if p.co == 15 and p.power == 1:  # Gold Rush
            p.funds = p.funds * 3 // 2
        change = action.get("hpChange")
        if isinstance(change, dict):
            for key, sign in (("hpGain", 1), ("hpLoss", -1)):
                spec = change.get(key)
                if not isinstance(spec, dict):
                    continue
                delta = int(spec["hp"]) * 10
                targets = {int(x) for x in spec.get("players") or []}
                for u in self.units.values():
                    if u.owner in targets:
                        u.hp = frozenset(max(1, min(100, h + delta)) if sign > 0 else max(1, h + delta) for h in u.hp)
        replace = _view(action.get("unitReplace"))
        if isinstance(replace, dict):
            for row in replace.get("units") or []:
                u = self.units.get(int(row["units_id"]))
                if u is None:
                    continue
                if "units_hit_points" in row:
                    shown = int(row["units_hit_points"])
                    if not 1 <= shown <= 10:  # hidden (Sonja): power damage of unknown size
                        u.hp = frozenset(max(1, h - 10 * k) for h in u.hp for k in range(10))
                        continue
                    # Power damage comes in whole HP: find the drop that matches the display.
                    u.hp = frozenset(
                        h2 for h in u.hp for h2 in [max(1, h - 10 * k) for k in range(10)] if _displayed(h2) == shown
                    ) or self._with_display(HP_RANGE, shown)
                if "units_ammo" in row:
                    u.ammo = int(row["units_ammo"])

    def _fire(self, fire: dict) -> AttackCheck:
        civ = _view(fire.get("combatInfoVision"))
        info = civ.get("combatInfo") if isinstance(civ, dict) else None
        check = AttackCheck(self.game_id, self.day, None)
        if not isinstance(info, dict) or not isinstance(info.get("attacker"), dict):
            check.reason = "no combat info"
            return check
        a_log, d_log = info["attacker"], info["defender"]
        att = self.units.get(int(a_log["units_id"]))
        de = self.units.get(int(d_log["units_id"]))
        if att is None or de is None:
            check.reason = "unknown unit"
            return check
        ap, dp = self.players[att.owner], self.players[de.owner]
        dist = abs(att.x - de.x) + abs(att.y - de.y)
        logged = (int(a_log["units_hit_points"]), int(d_log["units_hit_points"]))
        check = dataclasses.replace(
            check,
            attacker=self.r.units[att.type].name,
            defender=self.r.units[de.type].name,
            att_co=ap.co,
            att_power=ap.power,
            def_co=dp.co,
            def_power=dp.power,
            dist=dist,
            att_hp=tuple(sorted(att.hp)),
            def_hp=tuple(sorted(de.hp)),
            logged=logged,
        )
        predicted: set[tuple[int, int]] = set()
        new_att: set[int] = set()
        new_def: set[int] = set()
        for ha in att.hp:
            for hd in de.hp:
                for aa, dd, _ in _core.combat_outcomes(self.combatant(att, ha), self.combatant(de, hd), dist):
                    shown = (_displayed(aa), _displayed(dd))
                    predicted.add(shown)
                    if shown == logged:
                        new_att.add(aa)
                        new_def.add(dd)
        check.predicted = tuple(sorted(predicted))
        check.ok = bool(new_att)
        if not predicted:
            check.reason = "engine says it cannot attack"
        kind, _ = self._kind(att.x, att.y)
        dkind, _ = self._kind(de.x, de.y)
        check.detail = {
            "att_terrain": self.r.terrain_kinds[kind],
            "def_terrain": self.r.terrain_kinds[dkind],
            "att_towers": self._towers(att.owner),
            "def_towers": self._towers(de.owner),
            "att_ammo": att.ammo,
            "def_ammo": de.ammo,
        }
        # Carry on with whatever is consistent; if nothing is, trust the log's display.
        att.hp = frozenset(new_att) if new_att else self._with_display(HP_RANGE, logged[0])
        de.hp = frozenset(new_def) if new_def else self._with_display(HP_RANGE, logged[1])
        att.ammo = int(a_log["units_ammo"]) if a_log.get("units_ammo") is not None else att.ammo
        de.ammo = int(d_log["units_ammo"]) if d_log.get("units_ammo") is not None else de.ammo
        for u, shown in ((att, logged[0]), (de, logged[1])):
            if shown <= 0:
                self.units.pop(u.id, None)
        return check


def check_replay(replay: Replay, game_map: MapInfo) -> list[AttackCheck]:
    """Check every logged attack in a replay."""
    out: list[AttackCheck] = []
    colin_atk: dict[int, int] = {}  # carried across turns while the SCOP lasts
    for turn in replay.turns:
        if not turn.actions:
            continue
        state = _TurnState(replay, turn, game_map)
        for pid, atk in colin_atk.items():
            state.players[pid].colin_atk = atk
        for action in turn.actions:
            out.extend(state.apply(action))
        colin_atk = {p.id: p.colin_atk for p in state.players.values()}
    return out


def check_file(path: Path) -> tuple[list[AttackCheck], str | None]:
    """Check one replay zip. Returns the checks, or why the replay was skipped."""
    try:
        replay = load_replay(path)
    except ReplayFormatError as e:
        return [], f"parse error: {e}"
    if len(replay.player_ids) != 2:
        return [], "not 2 players"
    if replay.game.fog:
        return [], "fog"
    if not map_path(replay.game.map_id).exists():
        return [], "map not cached"
    return check_replay(replay, load_map(replay.game.map_id)), None
