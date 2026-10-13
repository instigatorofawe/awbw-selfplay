"""Games in progress on the official site, as engine states for the critic.

A public game's page (``game.php``) shows its settings, and the page's replay
viewer (``api/game/load_replay.php``) serves the state at the start of every
turn played so far, in the page's format, without a login. Turn starts are
what the critic was trained on (replay snapshots), so a game is valued at the
start of its current turn, whatever its player has done since.

A spectator is shown less than a replay records:

* HP is the displayed 1-10, not the exact value. A unit showing d gets 10d - 4,
  about the average exact HP behind that display in replays, or 100 for a 10.
  Below 10 its HP belief (what the critic reads) allows every internal HP
  behind the display equally; a player who watched every fight would know
  more, as the replay datasets do, but stepping a live game's history isn't
  done yet. Sonja's units show "?" outside her team and count as full.
* Hidden Subs and Stealths are left out unless the other player can see them,
  but each player's unit count and army value include them. Each one left out
  is put back, hidden, on a tile out of the other player's sight, with the HP
  its army value implies. The encoder then counts it in the totals and keeps
  it off the other player's board, as for a replay. The player to move's own
  units are all on its board, though, so one of those put back is in the
  wrong place there.

Turns that have ended never change, so they are cached under
``cache/live/<game id>/``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _core, engine, rules
from .maps import MapInfo, fetch_map, load_map
from .net import Http, cache_dir, write_atomic
from .replay.model import BuildingState, GameInfo, PlayerState, Replay, Turn, UnitState

AWBW = "https://awbw.amarriner.com"
GAME_URL = AWBW + "/game.php?games_id={game_id}"
REPLAY_API = AWBW + "/api/game/load_replay.php"
_KEEP = ("players", "buildings", "units", "players_units_count", "gameWeather", "currentTurnPId")


class LiveUnavailable(RuntimeError):
    """The site would not serve the game or a turn of it."""


class Unsupported(ValueError):
    """A game the engine doesn't model (fog, tags, more than two players)."""


def game_id_from(text: str) -> int:
    """A game id from an id or a game page URL."""
    if m := re.search(r"games_id=(\d+)", text):
        return int(m.group(1))
    if text.strip().isdigit():
        return int(text)
    raise ValueError(f"not a game id or a game page URL: {text!r}")


@dataclass(slots=True)
class LiveGame:
    """A game page: settings, players and where the game stands."""

    game: GameInfo
    names: dict[int, str]  # player id -> username
    order: tuple[int, ...]  # player ids in turn order
    day: int
    to_move: int  # player id whose turn it is
    acted: int  # units of the player to move that have acted this turn
    ended: bool
    tags: bool
    day_limit: int  # 0 = off
    unit_limit: int  # 0 = off
    bans: tuple[str, ...]  # unit names

    def check_supported(self) -> None:
        if self.game.fog:
            raise Unsupported(f"game {self.game.id} has fog")
        if self.tags:
            raise Unsupported(f"game {self.game.id} has tags")
        if len(self.order) != 2:
            raise Unsupported(f"game {self.game.id} has {len(self.order)} players")

    def settings(self) -> _core.Settings:
        s = engine.settings_for(self.game)
        s.day_limit = self.day_limit
        s.unit_limit = self.unit_limit
        r = rules.load()
        s.banned_units = sum(1 << r.unit(name).index for name in self.bans)
        return s


def _js_value(html: str, name: str) -> Any:
    """The value of a one-line ``let``/``const`` in the page's script, or None."""
    m = re.search(rf"^\s*(?:let|const|var) {name} = (.*?);\s*$", html, re.M)
    return json.loads(m.group(1)) if m else None


def _int_or(text: str | None, default: int) -> int:
    return int(text) if text is not None and text.strip().isdigit() else default


def parse_page(html: str) -> LiveGame:
    game_id = _js_value(html, "gameId")
    if game_id is None:
        raise LiveUnavailable("not a game page (is the game public?)")
    shown = dict(re.findall(r"<td class=small_text[^>]*>([^<:]+):</td><td class=small_text>([^<]*)</td>", html))
    players = _js_value(html, "playersInfo") or {}
    by_order = sorted(players.values(), key=lambda p: int(p["players_order"]))
    to_move = int(_js_value(html, "currentTurn"))
    units = _js_value(html, "unitsInfo") or {}  # [] when there are none
    units = units.values() if isinstance(units, dict) else []
    title = re.search(r"<title>\s*Game - (.*?) - AWBW\s*</title>", html, re.S)
    game = GameInfo(
        id=int(game_id),
        name=title.group(1) if title else "",
        map_id=int(_js_value(html, "mapId")),
        funds_per_property=_int_or(shown.get("Funds"), 1000),
        starting_funds=_int_or(shown.get("Starting Funds"), 0),
        capture_win=int(_js_value(html, "gameCaptureLimit") or 1000),
        fog=_js_value(html, "gameFog") == "Y",
        use_powers=_js_value(html, "gameCop") != "N",
        weather_type=shown.get("Weather", "Clear"),
        type="",
        official=False,
        team=False,
        league=None,
        start_date=None,
        min_rating=None,
        max_rating=None,
    )
    return LiveGame(
        game=game,
        names={int(p["players_id"]): str(p["users_username"]) for p in by_order},
        order=tuple(int(p["players_id"]) for p in by_order),
        day=int(_js_value(html, "gameDay")),
        to_move=to_move,
        acted=sum(1 for u in units if int(u["units_players_id"]) == to_move and int(u.get("units_moved") or 0)),
        ended=bool(_js_value(html, "gameEndDate")),
        tags=shown.get("Tags", "Off") != "Off",
        day_limit=_int_or(shown.get("Day Limit"), 0),
        unit_limit=_int_or(shown.get("Unit Limit"), 0),
        bans=tuple(_js_value(html, "banUnits") or ()),
    )


def fetch_game(game_id: int, http: Http) -> LiveGame:
    html = http.get(GAME_URL.format(game_id=game_id)).decode("utf-8", "replace")
    try:
        return parse_page(html)
    except LiveUnavailable as e:
        raise LiveUnavailable(f"game {game_id}: {e}") from None


# --- turns -----------------------------------------------------------------


def _turn_path(game_id: int, index: int) -> Path:
    return cache_dir() / "live" / str(game_id) / f"turn-{index:04d}.json"


def _post_turn(game: LiveGame, index: int, http: Http, initial: bool = False) -> dict[str, Any]:
    payload = {"gameId": game.game.id, "turn": index, "turnPId": game.to_move, "turnDay": game.day, "initial": initial}
    raw = http.post_json(REPLAY_API, payload)
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("err") or "gameState" not in data:
        why = data.get("message") if isinstance(data, dict) else raw[:200]
        raise LiveUnavailable(f"game {game.game.id} turn {index}: {why}")
    return data


def _load_turn(game: LiveGame, index: int, http: Http, last: int) -> dict[str, Any]:
    """Turn ``index``'s start (day and gameState); ended turns come from the cache."""
    path = _turn_path(game.game.id, index)
    if index < last and path.exists():
        return json.loads(path.read_text())
    data = _post_turn(game, index, http)
    kept = {"day": data["day"], "gameState": {k: data["gameState"].get(k) for k in _KEEP}}
    if index < last:
        write_atomic(path, json.dumps(kept).encode())
    return kept


def fetch_turn_data(game: LiveGame, http: Http, *, history: bool = False) -> list[tuple[int, dict[str, Any]]]:
    """(turn index, data) for the current turn, or with ``history`` for every turn so far."""
    # Two players alternate, so the current turn is usually this one; the
    # response's day selector, which lists every turn, says for sure.
    guess = 2 * (game.day - 1) + game.order.index(game.to_move)
    try:
        current = _post_turn(game, guess, http, initial=True)
    except LiveUnavailable:
        guess, current = 0, _post_turn(game, 0, http, initial=True)
    count = len(current.get("daySelector") or ())
    if count == 0:
        raise LiveUnavailable(f"game {game.game.id}: no turns listed")
    if count != guess + 1:
        current = _post_turn(game, count - 1, http)
    last = count - 1
    out = [(i, _load_turn(game, i, http, last)) for i in range(last)] if history else []
    out.append((last, {"day": current["day"], "gameState": {k: current["gameState"].get(k) for k in _KEEP}}))
    return out


# --- states ----------------------------------------------------------------


@dataclass(slots=True)
class LiveTurn:
    index: int  # turn number in the game, from 0
    turn: Turn
    put_back: list[UnitState] = field(default_factory=list)  # hidden units the page left out
    unplaced: Counter[tuple[int, str]] = field(default_factory=Counter)  # (player id, unit) left out, not put back
    estimated_hp: int = 0  # damaged units whose HP comes from the displayed HP
    unknown_hp: int = 0  # units whose HP is hidden (Sonja), counted as full
    display_only: set[int] = field(default_factory=set)  # ids of units known only by their displayed HP


def hp_from_display(shown: Any) -> int | None:
    """Internal HP for the HP a spectator is shown: exact if it has a decimal,
    else about the average behind that display, or None if it is hidden."""
    try:
        value = float(shown)
    except (TypeError, ValueError):
        return None
    if value != int(value):
        return round(value * 10)
    return 100 if value >= 10 else max(10 * int(value) - 4, 1)


def _displayed(hp: int) -> int:
    return -(-hp // 10)


def _player(p: dict[str, Any]) -> PlayerState:
    return PlayerState(
        id=int(p["players_id"]),
        users_id=int(p.get("users_id") or 0),
        countries_id=int(p["players_countries_id"]),
        co_id=int(p["players_co_id"]),
        tags_co_id=None,
        order=int(p["players_order"]),
        team=str(p.get("players_team") or ""),
        funds=int(p["players_funds"]),
        eliminated=p.get("players_eliminated") == "Y",
        co_power=int(p.get("players_co_power") or 0),
        co_max_power=int(p.get("players_co_max_power") or 0),
        co_max_spower=int(p.get("players_co_max_spower") or 0),
        co_power_on=str(p.get("players_co_power_on") or "N").upper(),
    )


def _strip(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {k.removeprefix(prefix): v for k, v in d.items()}


def _weather_owner(code: str, players: list[PlayerState]) -> int | None:
    """Player id whose power set the weather: Olaf's snow or Drake's Typhoon."""
    r = rules.load()
    olaf, drake = (next(c.id for c in r.cos.values() if c.name == name) for name in ("Olaf", "Drake"))
    for p in players:
        if code == "S" and p.co_id == olaf and p.co_power_on in ("Y", "S"):
            return p.id
        if code == "R" and p.co_id == drake and p.co_power_on == "S":
            return p.id
    return None


def _hiding_tile(
    name: str, owner: int, units: list[UnitState], buildings: list[BuildingState], info: MapInfo
) -> tuple[int, int] | None:
    """A free tile the unit can stand on, away from the other player's units
    and off properties, so the other player can't see it there."""
    r = rules.load()
    move_type = r.unit(name).move_type
    costs = r.move_costs["clear"]
    taken = {(u.x, u.y) for u in units if not u.carried} | {(b.x, b.y) for b in buildings}
    foes = [(u.x, u.y) for u in units if u.players_id != owner and not u.carried]
    best, best_dist = None, 1
    for y in range(info.height):
        for x in range(info.width):
            terrain = r.terrain.get(int(info.terrain[y, x]))
            if terrain is None or (x, y) in taken or move_type not in costs.get(terrain.kind, {}):
                continue
            dist = min((abs(x - fx) + abs(y - fy) for fx, fy in foes), default=info.width + info.height)
            if dist > best_dist:
                best, best_dist = (x, y), dist
    return best


def _put_back_hidden(lt: LiveTurn, counts: dict[str, Any], info: MapInfo, hp_known: dict[int, bool]) -> None:
    r = rules.load()
    units = lt.turn.units
    next_id = max((u.id for u in units), default=0) + 1
    for pid_text, c in (counts or {}).items():
        pid = int(pid_text)
        theirs = [u for u in units if u.players_id == pid]
        listed = c.get("count") or {}
        missing = Counter({k: int(v) for k, v in (listed.items() if isinstance(listed, dict) else ())})
        missing -= Counter(u.name for u in theirs)
        if not missing:
            continue
        hideable = Counter({k: n for k, n in missing.items() if k in ("Sub", "Stealth")})
        for k, n in (missing - hideable).items():
            lt.unplaced[pid, k] += n
        names = list(hideable.elements())
        if not names:
            continue
        # The army value counts displayed HP: what is left is the left-out units'.
        cost = [r.unit(k).cost for k in names]
        shown = sum(u.cost * _displayed(u.hp) // 10 for u in theirs)
        left = int(c.get("value") or 0) - shown
        tenths = round(10 * left / sum(cost)) if hp_known.get(pid, True) else 10
        hp = hp_from_display(min(max(tenths, 1), 10))
        assert hp is not None
        for k in names:
            tile = _hiding_tile(k, pid, units, lt.turn.buildings, info)
            if tile is None:
                lt.unplaced[pid, k] += 1
                continue
            t = r.unit(k)
            u = UnitState(
                id=next_id,
                players_id=pid,
                name=k,
                x=tile[0],
                y=tile[1],
                hp=hp,
                fuel=t.fuel,
                ammo=t.ammo,
                moved=False,
                capturing=False,
                fired=False,
                carried=False,
                cargo=(0, 0),
                sub_dive="Y",
                movement_points=t.move,
                vision=t.vision,
                fuel_per_turn=t.fuel_per_day,
                short_range=t.min_range,
                long_range=t.max_range,
                second_weapon=False,
                cost=t.cost,
                movement_type=t.move_type,
                symbol="",
            )
            next_id += 1
            units.append(u)
            lt.put_back.append(u)
            if hp < 100:
                lt.display_only.add(u.id)


def live_turn(index: int, data: dict[str, Any], info: MapInfo) -> LiveTurn:
    """A turn's start, from the replay API, in the replay model (see the module doc)."""
    gs = data["gameState"]
    players = [_player(p) for p in gs["players"].values()]
    buildings = [
        BuildingState.from_php(_strip(b, "buildings_")) for column in gs["buildings"].values() for b in column.values()
    ]
    units, estimated, unknown, hp_known, display_only = [], 0, 0, {}, set()
    for raw in gs["units"].values() if isinstance(gs["units"], dict) else ():  # [] when there are none
        f = _strip(raw, "units_")
        hp = hp_from_display(f["hit_points"])
        if hp is None:
            hp, unknown = 100, unknown + 1
            hp_known[int(f["players_id"])] = False
        elif hp < 100 and float(f["hit_points"]).is_integer():
            estimated += 1
            display_only.add(int(f["id"]))
        units.append(UnitState.from_php(f | {"hit_points": hp / 10}))
    weather = str(gs["gameWeather"]["code"])
    turn = Turn(
        day=int(data["day"]),
        player_id=int(gs["currentTurnPId"]),
        weather=weather,
        players=players,
        buildings=buildings,
        units=units,
        weather_start=_weather_owner(weather, players),
    )
    lt = LiveTurn(index=index, turn=turn, estimated_hp=estimated, unknown_hp=unknown, display_only=display_only)
    _put_back_hidden(lt, gs.get("players_units_count") or {}, info, hp_known)
    return lt


def state(game: LiveGame, lt: LiveTurn) -> _core.GameState:
    """The engine state at the start of a live turn."""
    md = engine.map_data(game.game.map_id)
    s = engine.state_from_turn(Replay(game.game, [lt.turn]), lt.turn, md, game.settings())
    for slot in s.unit_slots():
        if s.unit(slot).id in lt.display_only:
            s.set_hp_unknown(slot)
    if problem := s.check():
        raise LiveUnavailable(f"game {game.game.id} turn {lt.index}: the engine rejects the state ({problem})")
    return s


def fetch(game_id: int, http: Http, *, history: bool = False) -> tuple[LiveGame, list[LiveTurn]]:
    """A public game's page and its current turn's start (or every turn's, with ``history``)."""
    game = fetch_game(game_id, http)
    game.check_supported()
    fetch_map(game.game.map_id, http)
    info = load_map(game.game.map_id)
    engine.register(info)
    return game, [live_turn(i, data, info) for i, data in fetch_turn_data(game, http, history=history)]
