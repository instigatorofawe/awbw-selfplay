"""Games in progress on the official site, as engine states for the critic.

A public game's page (``game.php``) shows its settings, and the page's replay
viewer (``api/game/load_replay.php``) serves the state at the start of every
turn played so far, in the page's format, without a login. Turn starts are
what the critic was trained on (replay snapshots), so a game is valued at the
start of its current turn, whatever its player has done since.

A spectator is shown less than a replay records:

* HP is the displayed 1-10, not the exact value. A unit showing d gets 10d - 4,
  about the average exact HP behind that display in replays, or 100 for a 10.
  Sonja's units show "?" outside her team: they count as full.
* Hidden Subs and Stealths are left out unless the other player can see them,
  but each player's unit count and army value include them. Each one left out
  is put back, hidden, on a tile out of the other player's sight, with the HP
  its army value implies. The encoder then counts it in the totals and keeps
  it off the other player's board, as for a replay. The player to move's own
  units are all on its board, though, so one of those put back is in the
  wrong place there.

What the critic reads of HP is each unit's HP belief: what a player who
watched every turn knows. ``states`` gets it by stepping the game's turns from
the start, as the replay datasets do, with each turn's log from the replay
viewer (``replay_actions`` puts it in the replay file's format). A spectator's
log shows displayed HP and "?" for Sonja's, as her opponent sees them, so
stepping chooses internal HPs (and her displays) that the beliefs allow and
the log agrees with (``step_check.step_spectator_turn``). With Sonja to move,
her own units show the displays it chose. ``state`` alone knows only the
page: below 10 every internal HP behind a display is equally likely, and any
display of Sonja's to her opponent.

The viewer's log for a game in progress leaves out supplies (an APC's refill
shows only in the fuel logged next), and a finished game's viewer shows its
hidden units and Sonja's HP in the turn starts (and its log as the player
whose turn it was saw it).

Turns that have ended never change, so they are cached under
``cache/live/<game id>/``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _core, engine, rules
from .maps import MapInfo, PredeployedUnit, fetch_map, load_map
from .net import Http, cache_dir, write_atomic
from .replay.combat_check import _view
from .replay.model import BuildingState, GameInfo, PlayerState, Replay, Turn, UnitState
from .replay.step_check import Beliefs, step_spectator_turn

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
    predeployed: list[PredeployedUnit] = field(default_factory=list)  # the map's units, once it is loaded

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


def _kept(data: dict[str, Any]) -> dict[str, Any]:
    """What a turn's response is kept for: its day, start state and actions."""
    gs = data["gameState"]
    return {"day": data["day"], "gameState": {k: gs.get(k) for k in _KEEP}, "actions": data.get("actions") or []}


def _load_turn(game: LiveGame, index: int, http: Http, last: int) -> dict[str, Any]:
    """Turn ``index``'s start and actions; ended turns come from the cache."""
    path = _turn_path(game.game.id, index)
    if index < last and path.exists():
        kept = json.loads(path.read_text())
        if "actions" in kept:  # cached before actions were kept: fetch again
            return kept
    kept = _kept(_post_turn(game, index, http))
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
    out.append((last, _kept(current)))
    return out


# --- actions ---------------------------------------------------------------

# Actions a unit can take after moving: the replay logs the move inside them.
_AFTER_MOVE = ("Fire", "Capt", "Load", "Join", "Supply", "Repair", "Hide", "Unhide", "AttackSeam")


def _views(value: Any) -> dict[str, Any]:
    """A value as the replay logs one that differs by viewer: the spectator's is the global copy."""
    return {"global": value}


def _mover(kind: str, a: dict[str, Any]) -> int | None:
    """The unit an after-move action is by, if the action says."""
    if kind in ("Fire", "AttackSeam"):
        return int(a["attacker"]["units_id"])
    key = {"Load": "loadedId", "Join": "joinId", "Repair": "unitId", "Hide": "unitId", "Unhide": "unitId"}.get(kind)
    return int(a[key]) if key and a.get(key) is not None else None


def _after_move(kind: str, a: dict[str, Any], mover: int | None) -> dict[str, Any]:
    """An after-move action's own part, as the replay logs it."""
    rest = {k: v for k, v in a.items() if k != "action"}
    if kind == "Fire":
        info = {"attacker": rest.pop("attacker"), "defender": rest.pop("defender")}
        return {"action": kind, "combatInfoVision": _views({"hasVision": True, "combatInfo": info})} | rest
    if kind == "Capt":
        if "newIncome" in rest:
            rest["income"] = rest.pop("newIncome")
        return {"action": kind} | rest
    if kind == "Load":
        return {"action": kind, "loaded": rest["loadedId"], "transport": rest["transportId"]}
    if kind == "Join":
        return {
            "action": kind,
            "newFunds": _views(rest["newFunds"]),
            "playerId": rest.get("playerId"),
            "unit": _views(rest["joinedUnit"]),
            "joinID": rest["joinId"],
        }
    if kind == "Supply":
        return {"action": kind, "unit": _views(mover), "rows": rest.get("suppliedIds") or []}
    if kind == "Repair":
        return {
            "action": kind,
            "unit": _views(rest["unitId"]),
            "repaired": _views(rest["repairedUnit"]),
            "funds": _views(rest["funds"]),
        }
    if kind in ("Hide", "Unhide"):
        return {"action": kind, "unit": _views(rest["unitId"]), "vision": _views(rest.get("vision"))}
    assert kind == "AttackSeam"
    att = rest.pop("attacker")
    out = {
        "action": kind,
        "unit": _views({"hasVision": True, "combatInfo": att}),
        "buildings_hit_points": rest.pop("seamHp"),
        "buildings_terrain_id": rest.pop("seamTerrainId", None),
    }
    return out | rest


def _plain(kind: str, a: dict[str, Any]) -> dict[str, Any]:
    """An action that is not part of a move, as the replay logs it."""
    rest = {k: v for k, v in a.items() if k != "action"}
    if kind == "Build":
        return {"action": kind, "newUnit": _views(rest.get("newUnit"))} | {
            k: v for k, v in rest.items() if k != "newUnit"
        }
    if kind == "Unload":
        return {
            "action": kind,
            "unit": _views(rest["unloadedUnit"]),
            "transportID": _views(rest["transportId"]),
            "discovered": rest.get("discovered"),
        }
    if kind == "Delete":
        return {"action": kind, "Delete": {"action": kind, "unitId": _views(rest["deletedId"])}}
    if kind == "Power":
        out: dict[str, Any] = {"action": kind}
        for k, v in rest.items():
            if k == "playerId":
                out["playerID"] = v
            elif k == "copValue":
                out["playersCOP"] = v
            elif k in ("unitReplace", "playerReplace", "unitAdd"):
                out[k] = _views(v)
            else:
                out[k] = v
        return out
    if kind == "NextTurn":
        info = {"event": "NextTurn"} | rest
        for k in ("nextFunds", "repaired", "supplied"):
            if k in info:
                info[k] = _views(info[k])
        return {"action": "End", "updatedInfo": info}
    if kind == "Resign":
        over = rest.pop("GameOver", None)
        out = {"action": kind, "Resign": {"action": kind} | rest}
        return out | ({"GameOver": over} if over is not None else {})
    return dict(a)


def replay_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A turn's actions from the replay API (a spectator's view, one entry per step)
    in the replay file's format, as the turn stepper reads it.

    The API lists a move and what the unit did after it as two entries, and a
    unit acting where it stands without the move; the replay nests the move in
    the action. Fields that the replay logs per viewer get the spectator's copy
    as their global one, so ``_view`` reads them as it reads a replay's.
    """
    out: list[dict[str, Any]] = []
    move: dict[str, Any] | None = None  # the last move, while the next entry may be its unit's action
    unit: dict[str, Any] = {}  # the unit that made it
    for a in actions:
        kind = str(a.get("action"))
        if move is not None and kind in _AFTER_MOVE:
            mover = _mover(kind, a)
            if mover is None and kind == "Capt":
                b = a["buildingInfo"]
                ends_there = (int(unit["units_x"]), int(unit["units_y"])) == (
                    int(b["buildings_x"]),
                    int(b["buildings_y"]),
                )
                mover = int(unit["units_id"]) if ends_there else None
            if mover is None and kind == "Supply" and unit.get("units_name") == "APC":
                mover = int(unit["units_id"])  # the API names no supplier: the APC that just moved
            if mover == int(unit["units_id"]):
                out.append({"action": kind, "Move": move, kind: _after_move(kind, a, mover)})
                move = None
                continue
        if move is not None:
            out.append(move)
            move = None
        if kind == "Move":
            unit = a["unit"]
            rest = {k: v for k, v in a.items() if k not in ("action", "unit", "path")}
            move = {"action": "Move", "unit": _views(unit), "paths": _views(a.get("path") or [])} | rest
        elif kind in _AFTER_MOVE:
            out.append({"action": kind, "Move": [], kind: _after_move(kind, a, _mover(kind, a))})
        else:
            out.append(_plain(kind, a))
    if move is not None:
        out.append(move)
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
    hidden_hp: set[int] = field(default_factory=set)  # ids of units whose HP is hidden (Sonja)


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
            if not hp_known.get(pid, True):
                lt.hidden_hp.add(u.id)
            elif hp < 100:
                lt.display_only.add(u.id)


def live_turn(index: int, data: dict[str, Any], info: MapInfo) -> LiveTurn:
    """A turn's start, from the replay API, in the replay model (see the module doc)."""
    gs = data["gameState"]
    players = [_player(p) for p in gs["players"].values()]
    buildings = [
        BuildingState.from_php(_strip(b, "buildings_")) for column in gs["buildings"].values() for b in column.values()
    ]
    units, estimated, hp_known, display_only, hidden_hp = [], 0, {}, set(), set()
    for raw in gs["units"].values() if isinstance(gs["units"], dict) else ():  # [] when there are none
        f = _strip(raw, "units_")
        hp = hp_from_display(f["hit_points"])
        if hp is None:
            hp = 100
            hidden_hp.add(int(f["id"]))
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
        actions=replay_actions(data["actions"]) if "actions" in data else None,
        weather_start=_weather_owner(weather, players),
    )
    lt = LiveTurn(
        index=index,
        turn=turn,
        estimated_hp=estimated,
        unknown_hp=len(hidden_hp),
        display_only=display_only,
        hidden_hp=hidden_hp,
    )
    _put_back_hidden(lt, gs.get("players_units_count") or {}, info, hp_known)
    return lt


def state(game: LiveGame, lt: LiveTurn, *, put_back: bool = True) -> _core.GameState:
    """The engine state at the start of a live turn, knowing only what the page shows
    (``states`` adds what watching the turns before tells). Without ``put_back``,
    the hidden units the page left out stay out."""
    md = engine.map_data(game.game.map_id)
    turn = lt.turn
    if not put_back and lt.put_back:
        left_out = {u.id for u in lt.put_back}
        turn = dataclasses.replace(turn, units=[u for u in turn.units if u.id not in left_out])
    s = engine.state_from_turn(Replay(game.game, [turn]), turn, md, game.settings())
    for slot in s.unit_slots():
        if s.unit(slot).id in lt.display_only:
            s.set_hp_unknown(slot)
        if s.unit(slot).id in lt.hidden_hp and s.hides_hp(s.unit(slot).owner):
            s.set_hidden_unknown(slot)  # any display, to her opponent
    if problem := s.check():
        raise LiveUnavailable(f"game {game.game.id} turn {lt.index}: the engine rejects the state ({problem})")
    return s


# --- beliefs ---------------------------------------------------------------


def _sure(k: int) -> list[int]:
    """A belief sure of its k-th value."""
    return [255 if i == k else 0 for i in range(_core.HP_BINS)]


def _starting_beliefs(game: LiveGame, first: LiveTurn) -> dict[int, Beliefs]:
    """The units on the board when the game starts are the map's, at 10 x the HP the
    map gives them (Sonja's too, though the page hides it): everyone knows their HP."""
    on_map = {(p.x, p.y): p.hp for p in game.predeployed}
    out: dict[int, Beliefs] = {}
    for u in first.turn.units:
        shown = on_map.get((u.x, u.y)) if u.id in first.hidden_hp else _displayed(u.hp)
        if shown:
            out[u.id] = (10 * shown, _sure(_core.HP_BINS - 1), _sure(shown - 1))
    return out


def _beliefs_of(s: _core.GameState) -> dict[int, Beliefs]:
    return {s.unit(i).id: (s.unit(i).hp, s.hp_belief(i), s.hidden_belief(i)) for i in s.unit_slots()}


def _give_beliefs(s: _core.GameState, lt: LiveTurn, beliefs: dict[int, Beliefs]) -> None:
    """Give the units of a turn's start the internal HP and beliefs ``beliefs`` has
    for them (by AWBW id), where their shown HP agrees."""
    left_out = {u.id for u in lt.put_back}
    for slot in s.unit_slots():
        u = s.unit(slot)
        if u.id in left_out or u.id not in beliefs:
            continue
        hp, w, hidden = beliefs[u.id]
        if u.id not in lt.hidden_hp and _displayed(hp) != _displayed(u.hp):
            continue
        hides = s.hides_hp(u.owner)
        s.set_hp_unknown(slot)
        if hides:
            s.set_hidden_unknown(slot)
        s.set_hp(slot, hp)
        s.set_hp_belief(slot, w)
        if hides:
            s.set_hidden_belief(slot, hidden)


def _hp_touched(actions: list[dict[str, Any]]) -> set[int] | None:
    """AWBW ids of the units whose HP a turn's log may have changed, or None for any."""
    ids: set[int] = set()
    for a in actions:
        kind = a.get("action")
        if kind == "Fire":
            info = _view(a["Fire"]["combatInfoVision"])["combatInfo"]
            ids |= {int(info["attacker"]["units_id"]), int(info["defender"]["units_id"])}
        elif kind == "Repair":
            ids.add(int(_view(a["Repair"]["repaired"])["units_id"]))
        elif kind == "Join":
            ids |= {int(a["Join"]["joinID"]), int(_view(a["Join"]["unit"])["units_id"])}
        elif kind == "Power":
            if a.get("hpChange") or a.get("missileCoords"):
                return None
            rows = (_view(a.get("unitReplace")) or {}).get("units") or []
            ids |= {int(r["units_id"]) for r in rows if "units_hit_points" in r}
        elif kind == "End":
            ids |= {int(r["units_id"]) for r in _view(a["updatedInfo"].get("repaired")) or []}
    return ids


@dataclass(slots=True)
class History:
    """How stepping a game's turns went (``states``)."""

    stepped: int = 0  # turns stepped to the next turn's start
    failed: Counter[str] = field(default_factory=Counter)  # turns the engine couldn't step, by why


def states(game: LiveGame, turns: list[LiveTurn]) -> tuple[list[_core.GameState], History]:
    """The engine state at the start of each of a game's turns (every turn from the
    first, in order), with HP beliefs from stepping the turns before as a spectator
    (``step_check.step_spectator_turn``): what a player who watched the whole game
    knows of each unit's internal HP, and of Sonja's units' displays.

    Hidden units the page left out are left out of the stepping too. A turn the
    engine can't step leaves each unit whose HP its log may have changed knowing
    only what the page shows; the others keep what they had.
    """
    md = engine.map_data(game.game.map_id)
    replay = Replay(game.game, [lt.turn for lt in turns])
    out: list[_core.GameState] = []
    report = History()
    known = _starting_beliefs(game, turns[0])
    for k, lt in enumerate(turns):
        s = state(game, lt)
        _give_beliefs(s, lt, known)
        out.append(s)
        if k + 1 == len(turns):
            break
        start = s
        if lt.put_back:
            start = state(game, lt, put_back=False)
            _give_beliefs(start, lt, known)
        nxt = turns[k + 1]
        end, why = step_spectator_turn(
            replay,
            md,
            k,
            start,
            state(game, nxt),
            hidden=frozenset(nxt.hidden_hp),
            unchecked=frozenset(u.id for u in nxt.put_back),
        )
        if end is not None:
            report.stepped += 1
            known = end
            continue
        report.failed[why] += 1
        touched = _hp_touched(lt.turn.actions or [])
        before = _beliefs_of(start)
        known = {} if touched is None else {uid: b for uid, b in before.items() if uid not in touched}
    return out, report


def fetch(game_id: int, http: Http, *, history: bool = False) -> tuple[LiveGame, list[LiveTurn]]:
    """A public game's page and its current turn's start (or every turn's, with
    ``history``, which ``states`` needs for beliefs)."""
    game = fetch_game(game_id, http)
    game.check_supported()
    fetch_map(game.game.map_id, http)
    info = load_map(game.game.map_id)
    engine.register(info)
    game.predeployed = list(info.predeployed)
    return game, [live_turn(i, data, info) for i, data in fetch_turn_data(game, http, history=history)]
