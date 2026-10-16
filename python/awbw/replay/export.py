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

Exported games get ids no real game has (1,000,000,000 and up), and players
get user id -1, which the Replay Player doesn't look up on AWBW.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cache

from .. import _core, rules
from ..games import GameLog, replay
from ..phpser import PhpObject

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
        r = rules.load()
        active = int(s.active)
        rows = sorted(s.unit_rows(), key=lambda u: (u[3], u[4], bool(u[8] & _core.UNIT_CARRIED), u[0]))
        out = []
        for uid, typ, owner, x, y, hp, fuel, ammo, flags, cargo1, cargo2 in rows:
            t = r.units[typ]
            if owner == active:  # only a unit Von Bolt's SCOP stunned starts its turn spent
                spent = int(bool(flags & (_core.UNIT_MOVED | _core.UNIT_STUNNED)))
                moved, fired = spent, spent
            else:
                moved, fired = 0, int(uid in acted)
            out.append(
                PhpObject(
                    "awbwUnit",
                    {
                        "id": uid,
                        "games_id": self.meta.id,
                        "players_id": self.meta.players[owner],
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
                        "hit_points": hp / 10,
                        "cargo1_units_id": cargo1,
                        "cargo2_units_id": cargo2,
                        "carried": "Y" if flags & _core.UNIT_CARRIED else "N",
                    },
                )
            )
        return out


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
