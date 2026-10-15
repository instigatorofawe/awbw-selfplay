"""Bridge between parsed replays and the C++ engine state."""

from __future__ import annotations

import numpy as np

from . import _core, rules
from .maps import MapInfo, load_map
from .replay.model import GameInfo, Replay, Turn

_POWER_ON = {"N": 0, "Y": 1, "S": 2}
_WEATHER = {"C": 0, "R": 1, "S": 2}  # AWBW weather_code -> _core.WEATHERS index
_HIDDEN = ("Y", "D")  # awbwUnit.sub_dive for a dived Sub / hidden Stealth
_CAPTURABLE = frozenset({"airport", "base", "city", "comtower", "hq", "lab", "port"})
_SAME_TILE = {"rubble": "seam", "silo_empty": "silo"}  # a pipe seam broken, a silo fired


def register(game_map: MapInfo) -> _core.MapData:
    """Register a map with the engine (idempotent) and return it."""
    return _core.register_map(game_map.id, np.ascontiguousarray(game_map.terrain, dtype=np.int16))


def map_data(map_id: int) -> _core.MapData:
    """The engine's copy of a cached AWBW map, registering it on first use."""
    found = _core.find_map(map_id)
    return found if found is not None else register(load_map(map_id))


def map_edit(replay: Replay) -> str | None:
    """How the replay's first snapshot disagrees with the cached map, or None.

    AWBW lets an author edit a map after games were played on it and keeps
    only the latest version; even the site's own game pages show older games
    on the new terrain. A snapshot lists the game's buildings (properties,
    pipes, seams and silos) but no other terrain, so the version a game was
    played on can't be rebuilt: such games are left out. Each listed building
    must match the map's tile, and each property on the map must be listed.
    """
    if not replay.turns:
        return None
    rules_terrain = rules.load().terrain

    def kind(terrain_id: int) -> str | None:
        t = rules_terrain.get(int(terrain_id))
        return None if t is None else _SAME_TILE.get(t.kind, t.kind)

    terrain = load_map(replay.game.map_id).terrain
    height, width = terrain.shape
    listed = set()
    for b in replay.turns[0].buildings:
        if not (0 <= b.x < width and 0 <= b.y < height):
            return f"a building off the map at ({b.x}, {b.y})"
        listed.add((b.x, b.y))
        game, now = kind(b.terrain_id), kind(terrain[b.y, b.x])
        if game != now:
            return f"{game} at ({b.x}, {b.y}), where the map has {now}"
    for y in range(height):
        for x in range(width):
            if (x, y) not in listed and (now := kind(terrain[y, x])) in _CAPTURABLE:
                return f"the map's {now} at ({x}, {y}) isn't in the game"
    return None


def settings_for(game: GameInfo) -> _core.Settings:
    s = _core.Settings()
    s.funds_per_property = game.funds_per_property
    s.capture_win = min(game.capture_win, 1000)
    s.use_powers = int(game.use_powers)
    return s


def turn_order(replay: Replay) -> dict[int, int]:
    """AWBW player id -> engine player index (0 moves first)."""
    return {pid: i for i, pid in enumerate(replay.player_ids)}


def state_from_turn(
    replay: Replay, turn: Turn, md: _core.MapData | None = None, settings: _core.Settings | None = None
) -> _core.GameState:
    """The engine state at the start of one turn of a replay (``settings`` defaults to the game's)."""
    if md is None:
        md = map_data(replay.game.map_id)
    r = rules.load()
    index = turn_order(replay)
    unit_type = {u.name: u.index for u in r.units}
    players = np.zeros((2, 8), dtype=np.int64)
    for p in turn.players:
        players[index[p.id]] = (
            p.co_id,
            p.countries_id,
            p.funds,
            p.co_power,
            p.co_max_power,
            p.co_max_spower,
            _POWER_ON.get(p.co_power_on, 0),
            int(p.eliminated),
        )
    buildings = np.array([(b.x, b.y, b.terrain_id, b.capture) for b in turn.buildings], dtype=np.int64).reshape(-1, 4)
    rows = []
    for u in turn.units:
        flags = (
            (_core.UNIT_MOVED if u.moved else 0)
            | (_core.UNIT_HIDDEN if u.sub_dive in _HIDDEN else 0)
            | (_core.UNIT_CAPTURING if u.capturing else 0)
        )
        cargo = u.cargo
        rows.append(
            (u.id, unit_type[u.name], index[u.players_id], u.x, u.y, u.hp, u.fuel, u.ammo, flags, cargo[0], cargo[1])
        )
    units = np.array(rows, dtype=np.int64).reshape(-1, 11)
    return _core.load_snapshot(
        md,
        settings if settings is not None else settings_for(replay.game),
        turn.day,
        index[turn.player_id],
        players,
        buildings,
        units,
        weather=_WEATHER[turn.weather],
        weather_owner=index.get(turn.weather_start, -1) if turn.weather_start is not None else -1,
    )
