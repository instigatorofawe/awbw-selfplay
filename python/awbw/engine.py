"""Bridge between parsed replays and the C++ engine state."""

from __future__ import annotations

import numpy as np

from . import _core, rules
from .maps import MapInfo, load_map
from .replay.model import GameInfo, Replay, Turn

_POWER_ON = {"N": 0, "Y": 1, "S": 2}
_WEATHER = {"C": 0, "R": 1, "S": 2}  # AWBW weather_code -> _core.WEATHERS index
_HIDDEN = ("Y", "D")  # awbwUnit.sub_dive for a dived Sub / hidden Stealth


def register(game_map: MapInfo) -> _core.MapData:
    """Register a map with the engine (idempotent) and return it."""
    return _core.register_map(game_map.id, np.ascontiguousarray(game_map.terrain, dtype=np.int16))


def map_data(map_id: int) -> _core.MapData:
    """The engine's copy of a cached AWBW map, registering it on first use."""
    found = _core.find_map(map_id)
    return found if found is not None else register(load_map(map_id))


def settings_for(game: GameInfo) -> _core.Settings:
    s = _core.Settings()
    s.funds_per_property = game.funds_per_property
    s.capture_win = min(game.capture_win, 1000)
    s.use_powers = int(game.use_powers)
    return s


def turn_order(replay: Replay) -> dict[int, int]:
    """AWBW player id -> engine player index (0 moves first)."""
    return {pid: i for i, pid in enumerate(replay.player_ids)}


def state_from_turn(replay: Replay, turn: Turn, md: _core.MapData | None = None) -> _core.GameState:
    """The engine state at the start of one turn of a replay."""
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
        settings_for(replay.game),
        turn.day,
        index[turn.player_id],
        players,
        buildings,
        units,
        weather=_WEATHER[turn.weather],
        weather_owner=index.get(turn.weather_start, -1) if turn.weather_start is not None else -1,
    )
