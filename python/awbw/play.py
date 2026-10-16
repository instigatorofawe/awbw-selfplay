"""Local play against a network (PLAN.md §8 "Local play against the network").

``Game`` is one game between a person and an agent. It holds the state, the
person's action in progress (a ``Selection`` and the picks so far), the state
before each action, so that undo steps back an action at a time (the luck
too: the RNG is in the state), and every step taken, for the record.

``Game.view()`` is everything the page draws: the board, the players, and the
person's legal picks decoded into tiles and menu entries, with move paths,
attack ranges and damage previews, so the page has no rules of its own.
``serve`` puts a game behind a local HTTP server (``Server``) with a JSON API
and the page.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import _core, rules
from .agents import Agent, Decision, GameResult, Note, adjudicate
from .games import GameLog, Move, from_result
from .viewer import _Describer, _frame, board_data, page

CHANNEL = {name: i for i, name in enumerate(_core.CHANNELS)}
TARGETED = ("Fire", "AttackSeam", "Repair", "Launch", "Unload1", "Unload2")  # at the tile acted on
MENU = ("Fire", "AttackSeam", "Capture", "Launch", "Repair", "Supply", "Load", "Join")
MENU_LAST = ("Unload1", "Unload2", "Hide", "Unhide", "Explode", "Wait")
LABELS = {"AttackSeam": "Fire (seam)", "Launch": "Launch missile"}
PAYING = ("city", "base", "airport", "port", "hq")  # properties that pay income (core/map.cpp)
_FACTORY_KINDS = {"base": "base", "city": "base", "airport": "airport", "port": "port"}  # a city: Hachi's SCOP


def decode(md: _core.MapData, step_id: int) -> tuple[str, tuple[int, int], int]:
    """A micro-step id as (its channel, or "EndTurn", "COP", "SCOP" or "Build"; its
    tile (x, y), (-1, -1) for the last four; the unit type Build makes, else -1)."""
    if step_id >= _core.ID_BUILD:
        return "Build", (-1, -1), step_id - _core.ID_BUILD
    if step_id >= _core.ID_END_TURN:
        return ("EndTurn", "COP", "SCOP")[step_id - _core.ID_END_TURN], (-1, -1), -1
    channel, tile = divmod(step_id, _core.MAX_TILES)
    return _core.CHANNELS[channel], _xy(md, tile), -1


def step_id(md: _core.MapData, channel: str, x: int, y: int) -> int:
    return CHANNEL[channel] * _core.MAX_TILES + y * md.width + x


def _xy(md: _core.MapData, tile: int) -> tuple[int, int]:
    return tile % md.width, tile // md.width


def _co_state(s: _core.GameState, player: int) -> rules.CoState:
    p = s.players[player]
    return rules.load().cos[int(p.co)].state(int(p.power_on))


def attack_range(s: _core.GameState, slot: int, sources: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The tiles a unit could fire on from ``sources``: any of them for a
    direct unit, only its own tile for an indirect one (which can't move and
    fire); [] for a unit with no weapon it can use."""
    md = _core.find_map(s.map_id)
    u = s.unit(slot)
    r = rules.load()
    ut = r.units[u.type]
    att = _core.combatant(s, slot)
    if all(_core.weapon(att, t) < 0 for t in range(len(r.units))):
        return []
    lo, hi = ut.min_range, ut.max_range
    if hi > 1:
        hi = max(lo, hi + _co_state(s, u.owner).range[u.type])
        sources = [_xy(md, u.tile)]
    out: set[tuple[int, int]] = set()
    for sx, sy in sources:
        for dy in range(-hi, hi + 1):
            for dx in range(-hi, hi + 1):
                d = abs(dx) + abs(dy)
                x, y = sx + dx, sy + dy
                if lo <= d <= hi and 0 <= x < md.width and 0 <= y < md.height:
                    out.add((x, y))
    return sorted(out, key=lambda t: (t[1], t[0]))


def build_cost(s: _core.GameState, player: int, unit_type: int) -> int:
    """A unit's price for the player now (the engine's: cost x the CO's cost % / 100)."""
    return rules.load().units[unit_type].cost * _co_state(s, player).cost // 100


def factory_types(kind: str, banned: int) -> list[int]:
    """The unit types a factory of ``kind`` builds (a city counts as a base, for
    Hachi's SCOP), banned ones left out."""
    builds = {
        "base": lambda m: m not in ("A", "S", "L"),
        "airport": lambda m: m == "A",
        "port": lambda m: m in ("S", "L"),
    }[_FACTORY_KINDS[kind]]
    return [u.index for u in rules.load().units if builds(u.move_type) and not banned >> u.index & 1]


def _shown(hp: int) -> int:
    return (hp + 9) // 10


def fight_preview(s: _core.GameState, slot: int, at: tuple[int, int], target: tuple[int, int]) -> dict[str, Any]:
    """An attack as the players can foresee it, knowing only displayed HP: the
    damage range in % (the formula at the luck rolls' bounds), and the target's
    displayed HP after and the attacker's after the counter, with their chances
    over the luck and the HPs the units' displays allow."""
    defender = s.unit_at(*target)
    att, dfd = _core.combatant(s, slot, at), _core.combatant(s, defender)
    co = _co_state(s, int(s.unit(slot).owner))
    damage = [_core.damage(att, dfd, -co.luck_bad), _core.damage(att, dfd, co.luck_good)]
    after: dict[int, float] = {}
    counter: dict[int, float] = {}
    for a, b, p in _core.seen_fight_outcomes(s, slot, at, target):
        after[_shown(b)] = after.get(_shown(b), 0.0) + p
        counter[_shown(a)] = counter.get(_shown(a), 0.0) + p
    return {
        "damage": damage,
        "hp": _shown(s.unit(defender).hp),
        "after": sorted([k, round(v, 4)] for k, v in after.items()),
        "own": _shown(s.unit(slot).hp),
        "counter": sorted([k, round(v, 4)] for k, v in counter.items()),
        "kill": round(after.get(0, 0.0), 4),
    }


@dataclass(slots=True)
class _Before:
    """The game just before an action, for undo."""

    state: _core.GameState
    steps: int
    notes: int
    log: int
    mover: int


class Game:
    """A game between a person (``human``: 0 moves first, 1 second) and an agent.

    ``on_over`` gets the finished game's log once, when it ends.
    """

    def __init__(
        self,
        start: _core.GameState,
        human: int,
        agent: Agent,
        labels: tuple[str, str],
        on_over: Callable[[GameLog], None] | None = None,
        info: dict[str, Any] | None = None,
    ):
        md = _core.find_map(start.map_id)
        if md is None:
            raise ValueError(f"map {start.map_id} is not registered")
        self.md = md
        self.start = start.copy()
        self.human = human
        self.agent = agent
        self.labels = labels
        self.on_over = on_over
        self.info = info or {}
        self.state = start.copy()
        self.sel = _core.Selection()
        self.picks: list[int] = []  # the person's picks in the action in progress, not yet in steps
        self.steps: list[int] = []
        self.notes: list[tuple[int, Note]] = []
        self.history: list[_Before] = []
        self.log: list[dict[str, Any]] = []  # each action: mover, day, text, marks (viewer's), path, kind
        self.resigned = -1  # the player who resigned
        self.recorded = False
        self.describer = _Describer(md)
        self.lock = threading.RLock()

    # The game's course

    @property
    def over(self) -> bool:
        return self.state.winner != _core.NO_OWNER or self.resigned >= 0

    @property
    def winner(self) -> int:
        if self.resigned >= 0:
            return 1 - self.resigned
        return int(self.state.winner) if self.state.winner != _core.NO_OWNER else -1

    @property
    def human_to_move(self) -> bool:
        return not self.over and int(self.state.active) == self.human

    def pick(self, step: int) -> dict[str, Any] | None:
        """The person's next micro-step; the action it completes (as ``think``
        gives it), or None for a pick. ValueError if it isn't legal now."""
        with self.lock:
            if not self.human_to_move:
                raise ValueError("it isn't your turn")
            if step not in _core.legal_ids(self.state, self.sel):
                raise ValueError(f"step {step} isn't legal now")
            if _core.step_action(self.state, self.sel, step) is None:
                _core.step(self.state, self.sel, step)
                self.picks.append(step)
                return None
            picks, self.picks = self.picks, []
            return self._complete(picks, step)

    def cancel(self) -> None:
        """Drop the person's action in progress (picks don't touch the state)."""
        with self.lock:
            self.sel = _core.Selection()
            self.picks = []

    def undo(self) -> bool:
        """Step back to before the person's last action, undoing the agent's
        actions since; False if there is none or the game is over."""
        with self.lock:
            if self.over or not any(b.mover == self.human for b in self.history):
                return False
            while True:
                b = self.history.pop()
                if b.mover == self.human:
                    break
            self.state = b.state.copy()
            del self.steps[b.steps :]
            del self.notes[b.notes :]
            del self.log[b.log :]
            self.sel = _core.Selection()
            self.picks = []
            return True

    def resign(self) -> None:
        with self.lock:
            if not self.human_to_move:
                raise ValueError("you can resign on your own turn")
            self.cancel()
            self.resigned = self.human
            self.log.append({"m": self.human, "d": int(self.state.day), "t": "resigns", "k": [], "path": []})
            self._finish()

    def think(self) -> dict[str, Any] | None:
        """The agent's next action: it decides each step with a choice, forced
        steps taken for it; the action as the log keeps it, or None if it isn't
        the agent's turn."""
        with self.lock:
            if self.over or int(self.state.active) == self.human:
                return None
            sel, picks = _core.Selection(), []
            while True:
                legal = _core.legal_ids(self.state, sel)
                step = int(legal[0])
                if len(legal) > 1:
                    (step,) = self.agent.choose([Decision(self.state, list(picks), legal)])
                    noted = getattr(self.agent, "notes", None)
                    if noted:
                        self.notes.append((len(self.steps) + len(picks), noted[0]))
                if _core.step_action(self.state, sel, step) is not None:
                    self.sel = sel
                    return self._complete(picks, step)
                _core.step(self.state, sel, step)
                picks.append(step)

    def _complete(self, picks: list[int], step: int) -> dict[str, Any]:
        """Take the step that completes an action, after ``picks`` (taken on
        self.sel already); keeps the state before for undo, and logs it."""
        s = self.state
        mover = int(s.active)
        idle = s.copy()
        action = _core.step_action(s, self.sel, step)
        assert action is not None
        kind, slot, to, _, _ = action
        path: list[tuple[int, int]] = []
        if to is not None and slot is not None and slot >= 0:
            mp = _core.move_path(s, slot, tuple(to))
            path = mp[0][: mp[1] + 1] if mp else []
        self.history.append(_Before(idle, len(self.steps), len(self.notes), len(self.log), mover))
        status = _core.step(s, self.sel, step)
        if status not in (_core.OK, _core.TRAPPED):
            raise RuntimeError(f"step {step} was refused: {_core.status_name(status)}")
        self.steps.extend(picks)
        self.steps.append(step)
        self.sel = _core.Selection()
        text, marks = self.describer.describe(idle, Move(0, 0, tuple(action), mover, int(status), s.copy()))
        entry = {
            "m": mover,
            "d": int(idle.day),
            "t": text,
            "k": marks,
            "path": [list(p) for p in path],
            "kind": int(kind),
        }
        self.log.append(entry)
        if self.over:
            self._finish()
        return entry

    def _finish(self) -> None:
        if self.recorded or self.on_over is None:
            return
        self.recorded = True
        self.on_over(self.record())

    def record(self) -> GameLog:
        """The game so far as a log (``games``); unfinished, it's decided as a day limit would be."""
        with self.lock:
            s = self.state
            decided = "resigned" if self.resigned >= 0 else "game over" if self.over else "day limit"
            winner = self.winner if self.over else adjudicate(self.md, s)
            actions = sum(1 for e in self.log if e.get("kind") is not None)
            result = GameResult(
                winner, int(s.day), decided, actions, s.copy(), self.start.copy(), list(self.steps), list(self.notes)
            )
            return from_result(result, self.labels, "local", **self.info)

    # What the page draws

    def board(self) -> dict[str, Any]:
        """The game's fixed data for board.js, and who plays which side."""
        return board_data(self.md, self.start, self.labels, moves=True) | {"human": self.human}

    def view(self) -> dict[str, Any]:
        with self.lock:
            s = self.state
            stats = _core.player_stats_batch(np.frombuffer(s.to_bytes(), np.uint8)[None])[0]
            army = _core.PLAYER_STATS.index("army_value")
            kinds = rules.load().terrain_kinds
            players = []
            for p in range(2):
                pays = sum(
                    1 for i, (_, _, k, _) in enumerate(self.md.props()) if s.prop_owner(i) == p and kinds[k] in PAYING
                )
                income = pays * (s.settings.funds_per_property + _co_state(s, p).income_bonus)
                players.append({"income": income, "army": int(stats[p, army])})
            # the agent's latest value for its position, from the person's side
            value = round(-self.notes[-1][1].value, 4) if self.notes else None
            return {
                "frame": _frame(s, self.md.n_props),
                "players": players,
                "phase": _core.PHASES[self.sel.phase],
                "options": self.options() if self.human_to_move else None,
                "over": self.over,
                "winner": self.winner,
                "decided": "resigned" if self.resigned >= 0 else "game over" if self.over else "",
                "can_undo": not self.over and any(b.mover == self.human for b in self.history),
                "log": self.log[-60:],
                "actions": len(self.log),
                "value": value,
            }

    def options(self) -> dict[str, Any]:
        """The person's legal picks from the selection, decoded for the page."""
        s, md, sel = self.state, self.md, self.sel
        legal = [int(i) for i in _core.legal_ids(s, sel)]
        phase = _core.PHASES[sel.phase]
        if phase == "idle":
            out: dict[str, Any] = {"select": [], "delete": [], "cop": None, "scop": None, "end_turn": None}
            out["unmoved"] = 0  # the person's units that can still act, for End Turn's confirmation
            for i in legal:
                ch, at, _ = decode(md, i)
                if ch == "Select":
                    out["select"].append([*at, i])
                    slot = s.unit_at(*at)
                    if slot >= 0 and not s.unit(slot).flags & _core.UNIT_MOVED:
                        out["unmoved"] += 1
                elif ch == "Delete":
                    out["delete"].append([*at, i])
                elif ch in ("COP", "SCOP"):
                    out[ch.lower()] = i
                elif ch == "EndTurn":
                    out["end_turn"] = i
            return out
        if phase == "factory":
            at = _xy(md, sel.tile)
            kind = rules.load().terrain_kinds[md.props()[self.describer.prop_at[at]][2]]
            ids = {decode(md, i)[2]: i for i in legal}
            builds = []
            for ut in factory_types(kind, s.settings.banned_units):
                cost = build_cost(s, int(s.active), ut)
                builds.append({"type": ut, "name": rules.load().units[ut].name, "cost": cost, "id": ids.get(ut)})
            return {"factory": list(at), "builds": builds}
        slot = sel.unit
        here = _xy(md, s.unit(slot).tile)
        if phase == "selected":
            moves, dests = [], []
            for i in legal:
                ch, at, _ = decode(md, i)
                if ch == "Move":
                    mp = _core.move_path(s, slot, at)
                    moves.append({"to": list(at), "id": i, "path": [list(p) for p in mp[0]] if mp else [list(at)]})
                    dests.append(at)
            delete = step_id(md, "Delete", *here)
            return {
                "unit": list(here),
                "moves": moves,
                "attack": [list(t) for t in attack_range(s, slot, dests)],
                "delete": delete if delete in _core.legal_ids(s, _core.Selection()) else None,
            }
        # moved: the action menu at the destination
        dest = _xy(md, sel.tile)
        by_channel: dict[str, list[tuple[tuple[int, int], int]]] = {}
        for i in legal:
            ch, at, _ = decode(md, i)
            by_channel.setdefault(ch, []).append((at, i))
        unit_name = rules.load().units[s.unit(slot).type].name
        menu = []
        for ch in MENU + MENU_LAST:
            if ch not in by_channel:
                continue
            entry: dict[str, Any] = {"channel": ch, "name": self._label(ch, unit_name, slot)}
            if ch in TARGETED:
                targets = []
                for at, i in by_channel[ch]:
                    target: dict[str, Any] = {"at": list(at), "id": i}
                    if ch == "Fire":
                        target["preview"] = fight_preview(s, slot, dest, at)
                    elif ch == "AttackSeam":
                        target["seam"] = _core.seam_damage(_core.combatant(s, slot, dest))
                    targets.append(target)
                entry["targets"] = targets
            else:
                entry["id"] = by_channel[ch][0][1]
            menu.append(entry)
        return {"unit": list(here), "at": list(dest), "menu": menu}

    def _label(self, channel: str, unit_name: str, slot: int) -> str:
        if channel in ("Unload1", "Unload2"):
            cargo = self.state.unit(slot).cargo[int(channel[-1]) - 1]
            return f"Unload {rules.load().units[self.state.unit(cargo).type].name}" if cargo >= 0 else "Unload"
        if channel in ("Hide", "Unhide") and unit_name == "Sub":
            return "Dive" if channel == "Hide" else "Surface"
        return LABELS.get(channel, channel)

    def unit_info(self, x: int, y: int) -> dict[str, Any] | None:
        """Where any unit on (x, y), the opponent's included, could move and fire on its next turn."""
        with self.lock:
            s = self.state
            slot = s.unit_at(x, y)
            if slot < 0:
                return None
            u = s.unit(slot)
            moves = [(tx, ty) for tx, ty, _ in _core.reachable(s, slot)]
            stops = [t for t in moves if s.unit_at(*t) in (-1, slot)]
            return {
                "unit": [x, y],
                "owner": int(u.owner),
                "moves": [list(t) for t in moves],
                "attack": [list(t) for t in attack_range(s, slot, stops)],
            }


class Server(ThreadingHTTPServer):
    """A local server for one game at a time. ``new_game(params)`` makes the
    next one when the page asks (params from the page's request)."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], game: Game, new_game: Callable[[dict[str, Any]], Game] | None = None):
        super().__init__(address, _Handler)
        self.game = game
        self.new_game = new_game

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/"


class _Handler(BaseHTTPRequestHandler):
    @property
    def _server(self) -> Server:
        assert isinstance(self.server, Server)
        return self.server

    def log_message(self, format: str, *args: Any) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, code: int = 200) -> None:
        self._send(code, json.dumps(data, separators=(",", ":")).encode(), "application/json")

    def do_GET(self) -> None:
        url = urlparse(self.path)
        game = self._server.game
        if url.path == "/":
            self._send(200, page("play.html", game.board()).encode(), "text/html; charset=utf-8")
        elif url.path == "/api/board":
            self._json(game.board())
        elif url.path == "/api/view":
            self._json(game.view())
        elif url.path == "/api/unit":
            q = parse_qs(url.query)
            try:
                x, y = int(q["x"][0]), int(q["y"][0])
            except (KeyError, ValueError):
                self._json({"error": "x and y are needed"}, 400)
                return
            self._json(game.unit_info(x, y))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "the body isn't JSON"}, 400)
            return
        game = self._server.game
        try:
            done = None
            if url.path == "/api/pick":
                done = game.pick(int(body["id"]))
            elif url.path == "/api/cancel":
                game.cancel()
            elif url.path == "/api/undo":
                if not game.undo():
                    raise ValueError("nothing to undo")
            elif url.path == "/api/resign":
                game.resign()
            elif url.path == "/api/think":
                done = game.think()
            elif url.path == "/api/new":
                if self._server.new_game is None:
                    raise ValueError("this server plays one game")
                self._server.game = game = self._server.new_game(body)
            else:
                self._json({"error": "not found"}, 404)
                return
        except (KeyError, ValueError) as e:
            self._json({"error": str(e)}, 409)
            return
        self._json({"done": done, "view": game.view()})


def serve(game: Game, port: int = 0, new_game: Callable[[dict[str, Any]], Game] | None = None) -> Server:
    """A server for ``game`` on 127.0.0.1 (port 0: any free one); call
    ``serve_forever`` on it, or run it in a thread."""
    return Server(("127.0.0.1", port), game, new_game)


def start_for(
    base: _core.GameState,
    rng: np.random.Generator,
    map_id: int | None = None,
    cos: tuple[int | str, int | str] | None = None,
    countries: tuple[int | str, int | str] | None = None,
) -> _core.GameState:
    """A first state (``engine.new_game``) from a real game's setup, ``base``:
    its map, COs, countries and settings, each replaced where given. On
    another map, COs not given are drawn at random (Sonja aside, as in
    training) and countries default to the map's."""
    from .engine import new_game

    same_map = map_id is None or map_id == base.map_id
    if cos is None:
        if same_map:
            cos = (int(base.players[0].co), int(base.players[1].co))
        else:
            pool = sorted(c.id for c in rules.load().cos.values() if c.name != "Sonja")
            cos = (int(rng.choice(pool)), int(rng.choice(pool)))
    if countries is None and same_map:
        countries = (int(base.players[0].country), int(base.players[1].country))
    settings = base.settings if same_map else _core.Settings()
    return new_game(base.map_id if map_id is None else map_id, cos, countries, settings, seed=int(rng.integers(2**31)))


def record_to(path: Path) -> Callable[[GameLog], None]:
    """An ``on_over`` that appends finished games to a record file."""
    from .games import append

    def write(log: GameLog) -> None:
        append(path, [log])

    return write
