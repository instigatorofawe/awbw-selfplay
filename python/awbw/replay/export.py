"""Engine games as AWBW replays, for the AWBW Replay Player (``write`` makes the zip).

A replay keeps a snapshot per turn start (an ``awbwGame``) and each turn's
logged actions. The Replay Player's parser refuses fields it doesn't know, so
snapshots carry AWBW's fields in AWBW's order, as measured over the newest
replays (``tests/python/test_cached_replays.py`` checks them field by field).

The engine state holds nearly every field. ``Snapshots`` works out the rest
from the states the turn just ended went through:

* ``fired`` on the waiting player's units: whether each acted in its turn,
  which the engine forgets at End Turn. AWBW's value has quirks (a transport
  that only moved stays 0, and so on) that the Replay Player doesn't show, so
  the cached-replay check leaves this one field out.
* A building's ``last_capture``: 0 once it changes hands, 20 once its capture
  points change or the unit on it leaves or dies, otherwise as it was (every
  change in 200 replays). The Replay Player only uses it to word a capture.

``ActionLog`` writes each action's log entries from the states before and
after it, in AWBW's shapes (``test_actions_export_as_awbw_logs_them`` checks
them against real logs, given the path the player took). Some of what the
logs showed:

* A moved unit is logged in three identical views, one per player and
  "global", ordered by player id with "global" just before the first
  player who isn't moving. Its movement and range include its CO's bonuses,
  and its vision is one less in rain.
* ``dist`` is the movement cost. A unit the counterattack destroys moves
  with ``discovered`` []; destroyed units show 0 HP and 0 ammo.
* Repaired and supplied units are listed by id, a unit once per APC next to
  it.
* An HQ (or the last lab) taken to end the game is logged as it was: the
  loser's, its points spent below zero (Victory March takes 30), and only the
  loser's income. A neutral lab taken logs an empty ``onElimination``.

Exported games get ids no real game has (1,000,000,000 and up), and players
get user id -1, which the Replay Player doesn't look up on AWBW.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cache
from typing import Any

from .. import _core, rules
from ..games import GameLog, replay
from ..phpser import PhpObject
from .write import action_line

FIRST_GAME_ID = 1_000_000_000
NO_USER = -1  # the Replay Player asks AWBW for no username
FULL_CAPTURE = 20
TIMERS = {"timers_initial": 14400, "timers_increment": 2880, "timers_max_turn": 10080}
TURN_CLOCK = 1_036_800  # seconds; AWBW's at a game's start with these timers
POWER_ON = ("N", "Y", "S")
WEATHER_CODES = ("C", "R", "S")
# Terrain AWBW lists as buildings: properties, pipe seams (rubble once broken)
# and missile silos, loaded or empty. Some maps also list their pipes; the
# export doesn't.
PAYING = frozenset(["city", "base", "airport", "port", "hq"])  # properties that pay income
BUILDING_KINDS = frozenset(["city", "base", "airport", "port", "hq", "lab", "comtower", "seam", "silo", "silo_empty"])
SECOND_WEAPON = frozenset(["Tank", "Md.Tank", "Neotank", "Mega Tank", "Mech", "B-Copter", "Piperunner"])
SYMBOLS = {"foot": "G", "vehicle": "G", "copter": "M", "plane": "M", "naval": "S"}
RUBBLE = {113: 115, 114: 116}  # pipe seam -> its rubble


def _date(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def co_image(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower()) + ".png"


@cache
def _terrain_ids() -> dict[tuple[str, int | None], int]:
    """(kind, owning country or None) -> AWBW terrain id, for buildings."""
    out: dict[tuple[str, int | None], int] = {}
    for t in rules.load().terrain.values():
        if t.kind in BUILDING_KINDS:
            out.setdefault((t.kind, t.country), t.id)
    return out


@dataclass(frozen=True, slots=True)
class GameMeta:
    """What a replay says about its game beyond the engine state."""

    id: int
    name: str
    players: tuple[int, int] = (1, 2)  # AWBW player ids, player 0 (who moves first) then 1
    users: tuple[int, int] = (NO_USER, NO_USER)
    names: tuple[str, str] = ("Player 1", "Player 2")  # for AWBW's messages ("... has resigned!")
    start: datetime = datetime(2026, 1, 1)
    turn_minutes: int = 1  # time between turn starts, for the snapshots' dates


@dataclass(slots=True)
class _Building:
    id: int
    x: int
    y: int
    terrain: int  # the map's terrain id
    kind: str
    prop: int  # engine property index, or -1 (an empty silo)
    last_capture: int = FULL_CAPTURE
    points: int = FULL_CAPTURE
    owner: int = _core.NO_OWNER


@dataclass(slots=True)
class Snapshots:
    """A game's turn-start snapshots, in order (``take`` each in turn)."""

    md: _core.MapData
    meta: GameMeta
    buildings: list[_Building] = field(default_factory=list)
    turns: int = 0

    def __post_init__(self) -> None:
        r = rules.load()
        terrain = self.md.terrain()
        prop_at = {(x, y): p for p, (x, y, _, _) in enumerate(self.md.props())}
        tiles = []
        for y in range(self.md.height):
            for x in range(self.md.width):
                t = r.terrain.get(int(terrain[y, x]))
                if t is not None and t.kind in BUILDING_KINDS:
                    tiles.append((x, y, t.id, t.kind))
        tiles.sort()
        self.buildings = [
            _Building(i + 1, x, y, tid, kind, prop_at.get((x, y), -1)) for i, (x, y, tid, kind) in enumerate(tiles)
        ]

    def take(self, s: _core.GameState, turn: Sequence[_core.GameState] = ()) -> PhpObject:
        """The snapshot of a turn start. ``turn``: the states the turn just ended
        went through, from its start to just before End Turn (none before the
        first snapshot)."""
        acted = acted_ids(turn[-1]) if turn else set()
        vacated = self._vacated(turn)
        meta = self.meta
        when = meta.start + timedelta(minutes=meta.turn_minutes * self.turns)
        first = self.turns == 0
        self.turns += 1
        r = rules.load()
        active = int(s.active)
        pid = meta.players
        players = []
        for i, p in enumerate(s.players[:2]):
            co = r.cos[p.co]
            players.append(
                PhpObject(
                    "awbwPlayer",
                    {
                        "id": pid[i],
                        "users_id": meta.users[i],
                        "games_id": meta.id,
                        "countries_id": int(p.country),
                        "co_id": int(p.co),
                        "funds": int(p.funds),
                        "turn": None,
                        "email": None,
                        "uniq_id": None,
                        "eliminated": "Y" if p.eliminated else "N",
                        "last_read": _date(when),
                        "last_read_broadcasts": None,
                        "emailpress": None,
                        "signature": None,
                        "co_power": int(p.power),
                        "co_power_on": POWER_ON[p.power_on],
                        "order": i + 1,
                        "accept_draw": "N",
                        "co_max_power": int(p.cop_cost),
                        "co_max_spower": int(p.scop_cost),
                        "co_image": co_image(co.name),
                        "team": str(pid[i]),
                        "aet_count": 0,
                        "turn_start": _date(when),
                        "turn_clock": TURN_CLOCK,
                        "tags_co_id": None,
                        "tags_co_power": None,
                        "tags_co_max_power": None,
                        "tags_co_max_spower": None,
                        "interface": "N",
                    },
                )
            )
        weather_owner = int(s.weather_owner)
        settings = s.settings
        return PhpObject(
            "awbwGame",
            {
                "id": meta.id,
                "name": meta.name,
                "password": None,
                "creator": meta.users[0],
                "start_date": _date(meta.start),
                "end_date": None,
                "activity_date": _date(when),
                "maps_id": int(s.map_id),
                "weather_type": "Clear",
                "weather_start": pid[weather_owner] if weather_owner < 2 else None,
                "weather_code": WEATHER_CODES[s.weather],
                "win_condition": None,
                "turn": pid[active],
                "day": int(s.day),
                "active": "Y",
                "funds": int(settings.funds_per_property),
                "capture_win": int(settings.capture_win),
                "fog": "N",
                "comment": None,
                "type": "N",
                "boot_interval": -1,
                "starting_funds": 0,
                "official": "N",
                "min_rating": 0,
                "max_rating": None,
                "league": None,
                "team": "N",
                "aet_interval": -1,
                "aet_date": _date(when),
                "use_powers": "Y" if settings.use_powers else "N",
                "players": players,
                "buildings": self._buildings(s, when, first, vacated),
                "units": self._units(s, acted),
                **TIMERS,
            },
        )

    def _vacated(self, turn: Sequence[_core.GameState]) -> set[int]:
        """Ids of the buildings whose unit left or died during the turn."""
        out: set[int] = set()
        before: dict[int, int] = {}
        for s in turn:
            for b in self.buildings:
                slot = s.unit_at(b.x, b.y)
                now = s.unit(slot).id if slot >= 0 else 0
                if before.get(b.id, 0) and before[b.id] != now:
                    out.add(b.id)
                before[b.id] = now
        return out

    def _buildings(self, s: _core.GameState, when: datetime, first: bool, vacated: set[int]) -> list[PhpObject]:
        ids = _terrain_ids()
        out = []
        for b in self.buildings:
            if b.kind == "seam":
                hp = s.prop_points(b.prop)
                terrain, points = (b.terrain, hp) if hp else (RUBBLE[b.terrain], b.points)
            elif b.kind == "silo":
                terrain, points = (b.terrain if s.prop_points(b.prop) else ids[("silo_empty", None)]), FULL_CAPTURE
            elif b.prop < 0:  # an empty silo
                terrain, points = b.terrain, FULL_CAPTURE
            else:
                owner = s.prop_owner(b.prop)
                points = s.prop_points(b.prop)
                country = s.players[owner].country if owner < 2 else None
                terrain = ids[(b.kind, country)]
                if first:
                    pass
                elif owner != b.owner:
                    b.last_capture = 0
                elif points != b.points or b.id in vacated:
                    b.last_capture = FULL_CAPTURE
                b.owner = owner
            b.points = points
            out.append(
                PhpObject(
                    "awbwBuilding",
                    {
                        "id": b.id,
                        "games_id": self.meta.id,
                        "terrain_id": terrain,
                        "x": b.x,
                        "y": b.y,
                        "capture": points,
                        "last_capture": b.last_capture,
                        "last_updated": _date(when),
                    },
                )
            )
        return out

    def _units(self, s: _core.GameState, acted: set[int]) -> list[PhpObject]:
        active = int(s.active)
        rows = sorted(s.unit_rows(), key=lambda u: (u[3], u[4], bool(u[8] & _core.UNIT_CARRIED), u[0]))
        out = []
        for row in rows:
            if row[2] == active:  # only a unit Von Bolt's SCOP stunned starts its turn spent
                spent = int(bool(row[8] & (_core.UNIT_MOVED | _core.UNIT_STUNNED)))
                moved, fired = spent, spent
            else:
                moved, fired = 0, int(row[0] in acted)
            out.append(PhpObject("awbwUnit", _unit_fields(self.meta, row, moved, fired, row[5] / 10)))
        return out


def _unit_fields(meta: GameMeta, row: tuple, moved: int, fired: int, hit_points: float) -> dict:
    """A unit as AWBW's ``awbwUnit`` has it; ``row`` as ``GameState.unit_rows`` gives it."""
    uid, typ, owner, x, y, _, fuel, ammo, flags, cargo1, cargo2 = row
    t = rules.load().units[typ]
    return {
        "id": uid,
        "games_id": meta.id,
        "players_id": meta.players[owner],
        "name": t.name,
        "movement_points": t.move,
        "vision": t.vision,
        "fuel": fuel,
        "fuel_per_turn": t.fuel_per_day,
        "sub_dive": "Y" if flags & _core.UNIT_HIDDEN else "N",
        "ammo": ammo,
        "short_range": t.min_range if t.indirect else 0,
        "long_range": t.max_range if t.indirect else 0,
        "second_weapon": "Y" if t.name in SECOND_WEAPON else "N",
        "symbol": SYMBOLS[t.cls],
        "cost": t.cost,
        "movement_type": t.move_type,
        "x": x,
        "y": y,
        "moved": moved,
        "capture": int(bool(flags & _core.UNIT_CAPTURING)),
        "fired": fired,
        "hit_points": hit_points,
        "cargo1_units_id": cargo1,
        "cargo2_units_id": cargo2,
        "carried": "Y" if flags & _core.UNIT_CARRIED else "N",
    }


def acted_ids(s: _core.GameState) -> set[int]:
    """Ids of the active player's units that have acted this turn (call just before End Turn)."""
    active = int(s.active)
    return {u[0] for u in s.unit_rows() if u[2] == active and u[8] & _core.UNIT_MOVED}


def snapshots(log: GameLog, meta: GameMeta) -> list[PhpObject]:
    """A recorded game's turn-start snapshots: its start, then after each End Turn."""
    md = _core.find_map(log.map_id)
    if md is None:
        raise ValueError(f"map {log.map_id} is not registered")
    snaps = Snapshots(md, meta)
    out = [snaps.take(log.start)]
    turn = [log.start]
    for mv in replay(log).moves:
        if mv.action[0] == _core.ACTIONS["EndTurn"]:
            out.append(snaps.take(mv.state, turn))
            turn = [mv.state]
        else:
            turn.append(mv.state)
    return out


def action_lines(log: GameLog, meta: GameMeta) -> list[bytes]:
    """A recorded game's action log: a line per turn, each action as AWBW logs it."""
    md = _core.find_map(log.map_id)
    if md is None:
        raise ValueError(f"map {log.map_id} is not registered")
    actions = ActionLog(md, meta, {(b.x, b.y): b.id for b in Snapshots(md, meta).buildings})
    lines = []
    before = start = log.start
    turn: list[Json] = []
    for mv in replay(log).moves:
        turn += actions.entries(before, mv.state, mv.action, mv.status)
        before = mv.state
        if mv.action[0] == A["EndTurn"]:
            lines.append(action_line(meta.players[start.active], start.day, turn))
            turn, start = [], mv.state
    if log.decided == "resigned":  # not a step: the player to move resigned
        turn += actions.entries(before, before, (A["Resign"], -1, None, None, 0), _core.OK)
    if turn:
        lines.append(action_line(meta.players[start.active], start.day, turn))
    return lines


def shown_hp(hp: int) -> int:
    """HP as AWBW shows it, from the internal 0 to 100."""
    return (hp + 9) // 10


Json = dict[str, Any]
Action = tuple[int, int, tuple[int, int] | None, tuple[int, int] | None, int]  # as _core.step_action gives it
A = _core.ACTIONS
NAMES = {v: k for k, v in A.items()}


@dataclass(slots=True)
class ActionLog:
    """AWBW's logged JSON for each engine action, built from the states before and
    after it (``entries``). ``buildings``: AWBW building id by tile."""

    md: _core.MapData
    meta: GameMeta
    buildings: dict[tuple[int, int], int]
    turns: int = 0  # End Turns so far, for the next turn's start date
    _path: Sequence[tuple[int, int]] | None = None
    _trapped: bool = False

    def entries(
        self,
        before: _core.GameState,
        after: _core.GameState,
        action: Action,
        status: int,
        path: Sequence[tuple[int, int]] | None = None,
    ) -> list[Json]:
        """The log entries of one action: most are one, an Unload after moving is
        the transport's Move and then the Unload. ``path``: the path the unit
        moved along, if not the one the engine takes (``move_path``)."""
        kind, slot, to, target, arg = action
        self._path = path
        self._trapped = status == _core.TRAPPED
        if kind == A["EndTurn"]:
            return [self._end(before, after)]
        if kind == A["Resign"]:
            return [self._resign(before)]
        if kind == A["Build"]:
            assert target is not None
            return [self._build(before, after, target)]
        if kind == A["Delete"]:
            delete: Json = {"action": "Delete", "unitId": {"global": before.unit(slot).id}}
            if self._ended(after):
                delete["eliminated"] = self._rout(after)
            return [{"action": "Delete", "Delete": delete}]
        assert to is not None
        move = self._move(before, slot, to, alone=kind == A["Wait"])
        if kind == A["Wait"]:
            assert move is not None
            return [move]
        nested: Json | list = move if move is not None else []  # [] for a unit acting where it stands
        if kind == A["Fire"]:
            assert target is not None
            if move is not None and after.find_unit(before.unit(slot).id) < 0:
                move["discovered"] = []  # a unit the counterattack destroyed
            fire = self._fire(before, after, slot, to, target)
            if self._ended(after):
                fire["eliminated"] = self._rout(after)
            return [{"action": "Fire", "Move": nested, "Fire": fire}]
        if kind == A["Capture"]:
            return [{"action": "Capt", "Move": nested, "Capt": self._capture(before, after, slot, to)}]
        if kind == A["Load"]:
            load = {
                "action": "Load",
                "loaded": {"global": before.unit(slot).id},
                "transport": {"global": before.unit(before.unit_at(*to)).id},
            }
            return [{"action": "Load", "Move": nested, "Load": load}]
        if kind == A["Join"]:
            return [{"action": "Join", "Move": nested, "Join": self._join(before, after, slot, to)}]
        if kind == A["Unload"]:
            assert target is not None
            unload = self._unload(before, after, slot, target, arg)
            return [unload] if move is None else [move, unload]
        raise ValueError(f"no AWBW log entry for {NAMES[kind]}")

    # -- pieces ------------------------------------------------------------

    def _pid(self, player: int) -> int:
        return self.meta.players[player]

    def _row(self, s: _core.GameState, slot: int) -> tuple:
        uid = s.unit(slot).id
        return next(r for r in s.unit_rows() if r[0] == uid)

    def _unit(self, s: _core.GameState, slot: int, row: tuple, moved: int, fired: int) -> Json:
        """The unit in ``slot`` as an action logs it: as in a snapshot, but with
        shown HP, movement and range with its CO's bonuses, and vision one less
        in rain."""
        r = rules.load()
        f = _unit_fields(self.meta, row, moved, fired, shown_hp(row[5]))
        f["movement_points"] = _core.move_points(s, slot)
        if WEATHER_CODES[s.weather] == "R":
            f["vision"] = max(1, f["vision"] - 1)
        if f["long_range"]:
            p = s.players[row[2]]
            f["long_range"] += r.cos[p.co].state(p.power_on).range[row[1]]
        out: Json = {"0": f["id"]}
        out.update({"units_" + k: v for k, v in f.items()})
        out["countries_code"] = rules.load().countries[s.players[row[2]].country].code
        return out

    def _views(self, mover: int, value: Any) -> Json:
        """A value every player sees alike, keyed as AWBW keys a moved unit: by
        player id in id order, with "global" just before the first player that
        isn't the mover."""
        out: Json = {}
        for pid in sorted(self.meta.players):
            if pid != self._pid(mover) and "global" not in out:
                out["global"] = value
            out[str(pid)] = value
        return out

    def _move(self, before: _core.GameState, slot: int, to: tuple[int, int], alone: bool) -> Json | None:
        """The Move entry, or None for a unit acting where it stands (``alone``:
        the move is the whole action, logged even in place)."""
        if self._path is None:
            plan = _core.move_path(before, slot, to)
            if plan is None:
                raise ValueError(f"unit {before.unit(slot).id} can't reach {to}")
            path, stop, trapped, fuel = plan
        else:
            path, stop, trapped = list(self._path), len(self._path) - 1, self._trapped
            fuel = _core.path_cost(before, slot, path)
        if len(path) == 1 and not alone:
            return None
        mover = int(before.active)
        row = list(self._row(before, slot))
        row[3], row[4] = path[stop]
        row[6] -= fuel
        if stop:
            row[8] &= ~_core.UNIT_CAPTURING
        return {
            "action": "Move",
            "unit": self._views(mover, self._unit(before, slot, tuple(row), 1, 0)),
            "paths": {"global": [{"unit_visible": True, "x": x, "y": y} for x, y in path[: stop + 1]]},
            "dist": fuel,
            "trapped": trapped,
            "discovered": {str(self._pid(mover)): None},
        }

    def _fire(
        self, before: _core.GameState, after: _core.GameState, slot: int, to: tuple[int, int], target: tuple[int, int]
    ) -> Json:
        mover = int(before.active)
        rows = {r[0]: r for r in after.unit_rows()}

        def side(s: int, at: tuple[int, int]) -> Json:
            u = before.unit(s)
            row = rows.get(u.id)
            ammo, hp = (row[7], shown_hp(row[5])) if row else (0, 0)  # destroyed
            return {"units_ammo": ammo, "units_hit_points": hp, "units_id": u.id, "units_x": at[0], "units_y": at[1]}

        info: Json = {"attacker": side(slot, to), "defender": side(before.unit_at(*target), target)}
        players = after.players
        bonds = [p for p in (mover, 1 - mover) if _war_bonds(before, p) and players[p].funds > before.players[p].funds]
        if bonds:  # Sasha's SCOP: funds from the damage she deals (logged when she gains some)
            info["gainedFunds"] = {
                str(self._pid(p)): players[p].funds - before.players[p].funds if p in bonds else None
                for p in (mover, 1 - mover)
            }
        return {
            "action": "Fire",
            "combatInfoVision": {"global": {"hasVision": True, "combatInfo": info}},
            "copValues": {
                k: {"playerId": self._pid(p), "copValue": int(players[p].power), "tagValue": None}
                for k, p in (("attacker", mover), ("defender", 1 - mover))
            },
        }

    def _capture(self, before: _core.GameState, after: _core.GameState, slot: int, at: tuple[int, int]) -> Json:
        r = rules.load()
        mover = int(before.active)
        x, y = at
        prop = next(i for i, (px, py, _, _) in enumerate(self.md.props()) if (px, py) == at)
        bid = self.buildings[at]
        was = before.prop_owner(prop)
        kind = r.terrain_kinds[self.md.props()[prop][2]]
        hq_taken = self._ended(after) and kind in ("hq", "lab")
        if hq_taken:
            # AWBW logs the HQ (or last lab) a game ends on as it was: the
            # loser's, its points spent below zero, and only the loser's income.
            t = r.terrain[_terrain_ids()[(kind, before.players[was].country)]]
            info: Json = {
                "0": bid,
                "buildings_id": bid,
                "buildings_x": x,
                "buildings_y": y,
                "buildings_capture": before.prop_points(prop) - self._capture_points(before, slot),
                "terrain_id": t.id,
                "terrain_name": t.name,
                "terrain_defense": t.defense,
                "buildings_players_id": self._pid(mover),
                "buildings_team": str(self._pid(mover)),
            }
            income: Json | None = {str(self._pid(was)): {"player": self._pid(was), "income": self._income(before, was)}}
        elif after.prop_owner(prop) == mover:
            t = r.terrain[_terrain_ids()[(kind, after.players[mover].country)]]
            info = {
                "0": bid,
                "buildings_id": bid,
                "buildings_x": x,
                "buildings_y": y,
                "buildings_capture": FULL_CAPTURE,
                "terrain_id": t.id,
                "terrain_name": t.name,
                "terrain_defense": t.defense,
                "buildings_players_id": self._pid(mover),
                "buildings_team": str(self._pid(mover)),
            }
            changed = sorted({mover} | ({was} if was < 2 else set()), key=self._pid)
            income = {str(self._pid(p)): {"player": self._pid(p), "income": self._income(after, p)} for p in changed}
        else:
            info = {
                "buildings_capture": after.prop_points(prop),
                "buildings_id": bid,
                "buildings_x": x,
                "buildings_y": y,
                "buildings_team": str(self._pid(was)) if was < 2 else None,
            }
            income = None
        out: Json = {
            "action": "Capt",
            "buildingInfo": info,
            "vision": {"global": {"onCapture": {"x": x, "y": y}}},
            "income": income,
        }
        if kind == "lab" and was >= 2 and after.prop_owner(prop) == mover:
            # AWBW checks a lab's previous owner for elimination, even nobody.
            out["onElimination"] = {str(self._pid(mover)): {"units": [], "buildings": []}}
        if self._ended(after):
            loser = 1 - mover
            if hq_taken:
                # The loser's buildings as they were before the capture.
                held = sorted(
                    (px, py) for i, (px, py, _, _) in enumerate(self.md.props()) if before.prop_owner(i) == loser
                )
                out["onElimination"] = {
                    str(self._pid(mover)): {
                        "units": [],
                        "buildings": [
                            {"buildings_players_id": self._pid(loser), "buildings_x": bx, "buildings_y": by}
                            for bx, by in held
                        ],
                    }
                }
            out["eliminated"] = {
                "eliminatedByPId": self._pid(mover),
                "message": f"{self.meta.names[loser]} was eliminated by capture!",
                "playerId": self._pid(loser),
                "GameOver": self._game_over(after),
            }
        return out

    def _capture_points(self, before: _core.GameState, slot: int) -> int:
        """The capture points AWBW logs a unit taking."""
        p = before.players[before.unit(slot).owner]
        co = rules.load().cos[p.co]
        if co.name == "Sami" and p.power_on == 2:  # Victory March: AWBW logs 30 whatever the HP
            return 30
        return shown_hp(before.unit(slot).hp) * co.state(p.power_on).capture // 100

    def _income(self, s: _core.GameState, player: int) -> int:
        r = rules.load()
        paying = sum(
            1
            for i, (_, _, kind, _) in enumerate(self.md.props())
            if s.prop_owner(i) == player and r.terrain_kinds[kind] in PAYING
        )
        return paying * (s.settings.funds_per_property + r.cos[s.players[player].co].d2d.income_bonus)

    def _build(self, before: _core.GameState, after: _core.GameState, at: tuple[int, int]) -> Json:
        mover = int(before.active)
        new = after.unit_at(*at)
        unit = self._unit(after, new, self._row(after, new), 1, 1)
        return {"action": "Build", "newUnit": {"global": unit}, "discovered": {str(self._pid(mover)): None}}

    def _join(self, before: _core.GameState, after: _core.GameState, slot: int, at: tuple[int, int]) -> Json:
        mover = int(before.active)
        joined = after.unit_at(*at)
        unit = self._unit(after, joined, self._row(after, joined), 1, 1)
        return {
            "action": "Join",
            "newFunds": {"global": int(after.players[mover].funds)},
            "playerId": self._pid(mover),
            "unit": {"global": unit},
            "joinID": {"global": before.unit(slot).id},
        }

    def _unload(
        self, before: _core.GameState, after: _core.GameState, slot: int, at: tuple[int, int], arg: int
    ) -> Json:
        mover = int(before.active)
        cargo = before.unit(slot).cargo[arg]
        dropped = after.find_unit(before.unit(cargo).id)
        unit = self._unit(after, dropped, self._row(after, dropped), 1, 1)
        return {
            "action": "Unload",
            "unit": {"global": unit},
            "transportID": before.unit(slot).id,
            "discovered": {str(self._pid(mover)): None},
        }

    def _end(self, before: _core.GameState, after: _core.GameState) -> Json:
        r = rules.load()
        self.turns += 1
        nxt = int(after.active)
        repaired = []
        supplied: list[str] = []
        by_tile = {(row[3], row[4]): row for row in after.unit_rows() if not row[8] & _core.UNIT_CARRIED}
        props = {(px, py): (i, r.terrain_kinds[k]) for i, (px, py, k, _) in enumerate(self.md.props())}
        for xy in sorted(self.buildings):
            row = by_tile.get(xy)
            if row is None or row[2] != nxt or xy not in props:
                continue
            prop, kind = props[xy]
            if after.prop_owner(prop) == nxt and _services(kind, r.units[row[1]].cls):
                repaired.append({"units_id": str(row[0]), "units_hit_points": shown_hp(row[5])})
        repaired.sort(key=lambda u: int(u["units_id"]))
        for row in by_tile.values():  # APCs resupply the units next to them
            if row[2] == nxt and r.units[row[1]].supplies:
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    v = by_tile.get((row[3] + dx, row[4] + dy))
                    if v is not None and v[2] == nxt:
                        supplied.append(str(v[0]))  # once per APC
        supplied.sort(key=int)
        when = self.meta.start + timedelta(minutes=self.meta.turn_minutes * self.turns)
        return {
            "action": "End",
            "updatedInfo": {
                "event": "NextTurn",
                "nextPId": self._pid(nxt),
                "nextFunds": {"global": int(after.players[nxt].funds)},
                "nextTimer": TURN_CLOCK,
                "nextWeather": WEATHER_CODES[after.weather],
                "supplied": {"global": supplied},
                "repaired": {"global": repaired},
                "day": int(after.day),
                "nextTurnStart": _date(when),
            },
        }

    def _resign(self, before: _core.GameState) -> Json:
        mover = int(before.active)
        return {
            "action": "Resign",
            "Resign": {
                "action": "Resign",
                "playerId": self._pid(mover),
                "message": f"{self.meta.names[mover]} has resigned!",
            },
            "GameOver": self._game_over(before, winner=1 - mover),
        }

    def _rout(self, after: _core.GameState) -> Json:
        loser = 1 - int(after.winner)
        return {
            "message": f"{self.meta.names[loser]} was eliminated by rout!",
            "playerId": self._pid(loser),
            "GameOver": self._game_over(after),
        }

    def _ended(self, s: _core.GameState) -> bool:
        return s.winner in (0, 1)

    def _game_over(self, s: _core.GameState, winner: int | None = None) -> Json:
        """AWBW's GameOver for a game ``winner`` (default: the state's) has won."""
        if winner is None:
            winner = int(s.winner)
        when = self.meta.start + timedelta(minutes=self.meta.turn_minutes * self.turns)
        return {
            "day": int(s.day),
            "gameEndDate": when.strftime("%Y-%m-%d"),
            "losers": [self._pid(1 - winner)],
            "message": f"The game is over! {self.meta.names[winner]} is the winner!",
            "winners": [self._pid(winner)],
        }


def _war_bonds(s: _core.GameState, player: int) -> bool:
    p = s.players[player]
    return p.power_on == 2 and rules.load().cos[p.co].name == "Sasha"


def _services(kind: str, cls: str) -> bool:
    """Whether a property of ``kind`` repairs a unit of class ``cls``."""
    if kind in ("city", "hq", "base"):
        return cls in ("foot", "vehicle")
    return (kind, cls) in (("airport", "copter"), ("airport", "plane"), ("port", "naval"))
