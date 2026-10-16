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


def map_countries(game_map: MapInfo) -> list[int]:
    """The map's country slots in turn order: the countries its properties and
    predeployed units belong to, by AWBW country id (player 1 takes the first)."""
    r = rules.load()
    by_code = {c.code: c.id for c in r.countries.values()}
    found = {t.country for tid in np.unique(game_map.terrain).tolist() if (t := r.terrain.get(int(tid))) and t.country}
    found |= {by_code[u.country_code] for u in game_map.predeployed if u.country_code in by_code}
    return sorted(found)


def _lookup(table: dict, key: int | str, what: str) -> int:
    """The id of a CO or country given by id, name or (a country) code, any case."""
    for k, v in table.items():
        names = (str(v.name).lower(), getattr(v, "code", ""))
        if key == k or (isinstance(key, str) and key.lower() in names):
            return int(k)
    raise ValueError(f"no {what} {key!r}")


def new_game(
    game_map: MapInfo | int,
    cos: tuple[int | str, int | str],
    countries: tuple[int | str, int | str] | None = None,
    settings: _core.Settings | None = None,
    starting_funds: int = 0,
    seed: int | None = None,
) -> _core.GameState:
    """AWBW's first state of a game on a two-player map, as its replays' first
    snapshots have it: day 1, clear weather, player 1 (the map's first country
    slot) to move with a day's income, player 2 with none (both with
    ``starting_funds`` on top, AWBW's option); each slot's
    properties and predeployed units (full fuel and ammo) to its player;
    capture points 20 and pipe seams 99 HP; power meters empty.

    ``cos`` and ``countries`` (AWBW ids, names or country codes) are player
    1's then player 2's; countries default to the map's slots (players may take
    any). ``seed`` reseeds the luck.
    """
    game_map = load_map(game_map) if isinstance(game_map, int) else game_map
    r = rules.load()
    md = register(game_map)
    slots = map_countries(game_map)
    if len(slots) != 2:
        raise ValueError(f"map {game_map.id} has {len(slots)} countries; only two-player maps are played")
    co_ids = [_lookup(r.cos, c, "CO") for c in cos]
    country_ids = slots if countries is None else [_lookup(r.countries, c, "country") for c in countries]
    if country_ids[0] == country_ids[1]:
        raise ValueError("the players need different countries")
    settings = settings if settings is not None else _core.Settings()
    rank = {cid: i for i, cid in enumerate(slots)}
    building_id = {}
    for t in r.terrain.values():
        building_id.setdefault((t.kind, t.country), t.id)
    buildings, paying = [], [0, 0]
    for x, y, kind, _ in md.props():
        tid = int(game_map.terrain[y, x])
        t = r.terrain[tid]
        name = r.terrain_kinds[kind]
        if t.country:
            owner = rank[t.country]
            tid = building_id[(t.kind, country_ids[owner])]
            paying[owner] += name in ("city", "base", "airport", "port", "hq")
        buildings.append((x, y, tid, 99 if name == "seam" else 20))
    players = np.zeros((2, 8), dtype=np.int64)
    for p in (0, 1):
        co = co_ids[p]
        players[p] = (co, country_ids[p], starting_funds, 0, _core.cop_cost(co, 0), _core.scop_cost(co, 0), 0, 0)
    players[0, 2] += paying[0] * (settings.funds_per_property + r.cos[co_ids[0]].d2d.income_bonus)
    by_code = {c.code: c.id for c in r.countries.values()}
    rows = []
    for i, u in enumerate(game_map.predeployed):
        ut = r.unit(u.unit_id)
        owner = rank[by_code[u.country_code]]
        rows.append((i + 1, ut.index, owner, u.x, u.y, u.hp * 10, ut.fuel, ut.ammo, 0, 0, 0))
    s = _core.load_snapshot(
        md,
        settings,
        1,
        0,
        players,
        np.array(buildings, dtype=np.int64).reshape(-1, 4),
        np.array(rows, dtype=np.int64).reshape(-1, 11),
    )
    if seed is not None:
        s.seed(seed)
    return s
