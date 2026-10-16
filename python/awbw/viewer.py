"""A recorded game (``games``) as one self-contained HTML page that steps
through it action by action: the board, each player's funds and power, the
action in words, and the value the players saw at each decision.

``game_view`` replays the game into plain data (a frame per action);
``write_page`` puts it into ``viewer.html``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import _core, rules, sprites
from .games import SEARCHED, GameLog, Move, replay

ACTION_NAMES = {v: k for k, v in _core.ACTIONS.items()}
KIND_NAMES = {"hq": "HQ", "comtower": "com tower", "silo": "missile silo"}


def _map_name(map_id: int) -> str:
    from .maps import load_map, map_path

    try:
        return load_map(map_id).name if map_path(map_id).exists() else ""
    except (OSError, ValueError):
        return ""


class _Describer:
    def __init__(self, md: _core.MapData):
        self.md = md
        r = rules.load()
        self.unit_names = [u.name for u in r.units]
        self.co_names = {c.id: c.name for c in r.cos.values()}
        self.kinds = r.terrain_kinds
        self.props = md.props()
        self.prop_at = {(x, y): p for p, (x, y, _, _) in enumerate(self.props)}

    def xy(self, tile: int) -> tuple[int, int]:
        return tile % self.md.width, tile // self.md.width

    def unit(self, s: _core.GameState, slot: int) -> tuple[str, int, int]:
        """(type name, displayed HP, id) of the unit in a slot."""
        u = s.unit(slot)
        return self.unit_names[u.type], (u.hp + 9) // 10, int(u.id)

    def after_hp(self, after: _core.GameState, uid: int) -> int:
        """Displayed HP of the unit with id ``uid`` after the action; 0 if destroyed."""
        slot = after.find_unit(uid)
        return (after.unit(slot).hp + 9) // 10 if slot >= 0 else 0

    def describe(self, before: _core.GameState, m: Move) -> tuple[str, list[list[int]]]:
        """The action in words, and the tiles to mark: [x, y, role] with role 0
        the unit's start, 1 where it went, 2 what it acted on."""
        kind, slot, to, target, arg = m.action
        name = ACTION_NAMES.get(kind, str(kind))
        marks: list[list[int]] = []
        who, hp, uid = "", 0, 0
        if slot is not None and slot >= 0:
            uname, hp, uid = self.unit(before, slot)
            fx, fy = self.xy(before.unit(slot).tile)
            marks.append([fx, fy, 0])
            who = f"{uname} ({fx},{fy})"
            if to is not None and tuple(to) != (fx, fy):
                who += f" → ({to[0]},{to[1]})"
            if to is not None:
                marks.append([to[0], to[1], 1])
        if target is not None:
            marks.append([target[0], target[1], 2])
        if name == "EndTurn":
            text = "ends the turn"
        elif name == "Resign":
            text = "resigns"
        elif name == "Power":
            co = self.co_names.get(before.players[m.mover].co, "")
            text = f"{co} activates the {'Super CO Power' if arg == 2 else 'CO Power'}"
        elif name == "Build":
            text = f"builds {self.unit_names[arg]} at ({target[0]},{target[1]})"
        elif name == "Delete":
            text = f"scraps {who}"
        elif name == "Fire":
            d = before.unit_at(target[0], target[1])
            dname, dhp, did = self.unit(before, d)
            dafter = self.after_hp(m.state, did)
            aafter = self.after_hp(m.state, uid)
            text = f"{who} attacks {dname} ({target[0]},{target[1]}): {dhp} → {dafter if dafter else 'destroyed'}"
            if aafter != hp:
                text += f"; counter {hp} → {aafter if aafter else 'destroyed'}"
        elif name == "Capture":
            p = self.prop_at.get(tuple(to))
            kind = self.kinds[self.props[p][2]] if p is not None else "property"
            what = KIND_NAMES.get(kind, kind)
            if p is not None and m.state.prop_owner(p) == m.mover:
                text = f"{who} captures the {what}"
            else:
                left = m.state.prop_points(p) if p is not None else 0
                text = f"{who} captures the {what}: {left} left"
        elif name == "Load":
            t = before.unit_at(to[0], to[1])
            text = f"{who} boards the {self.unit(before, t)[0]}"
        elif name == "Join":
            t = before.unit_at(to[0], to[1])
            text = f"{who} joins the {self.unit(before, t)[0]}"
        elif name == "Unload":
            cargo = before.unit(slot).cargo[arg]
            cname = self.unit(before, cargo)[0] if cargo != _core.NO_UNIT else "cargo"
            text = f"{who} drops {cname} at ({target[0]},{target[1]})"
        elif name == "Repair":
            t = before.unit_at(target[0], target[1])
            text = f"{who} repairs the {self.unit(before, t)[0]}"
        elif name == "AttackSeam":
            text = f"{who} attacks the pipe seam ({target[0]},{target[1]})"
        elif name == "Launch":
            text = f"{who} fires the silo at ({target[0]},{target[1]})"
        elif name in ("Wait", "Supply", "Hide", "Unhide", "Explode"):
            verb = {
                "Wait": "waits",
                "Supply": "resupplies",
                "Hide": "hides",
                "Unhide": "surfaces",
                "Explode": "explodes",
            }
            text = f"{who} {verb[name]}"
        else:
            text = f"{who} {name}"
        if m.status == _core.TRAPPED:
            text += " (trapped by a hidden unit)"
        return text, marks


def _frame(s: _core.GameState, n_props: int) -> dict[str, Any]:
    return {
        "d": int(s.day),
        "a": int(s.active),
        "w": int(s.weather),
        "p": [[p.funds, p.power, p.power_on, p.cop_cost, p.scop_cost] for p in s.players[:2]],
        "u": [list(row) for row in s.unit_rows()],
        "o": [s.prop_owner(p) for p in range(n_props)],
        "c": [s.prop_points(p) for p in range(n_props)],
    }


def board_data(
    md: _core.MapData,
    start: _core.GameState,
    labels: tuple[str, str],
    weathers: tuple[int, ...] = (0, 1, 2),
    moves: bool = False,
) -> dict[str, Any]:
    """What board.js draws a game's frames with: the map, unit names, each
    player's label, CO and country, and the sprites if there is a local copy
    (``sprites.for_board``: textures for ``weathers``, move animations if
    ``moves``)."""
    r = rules.load()
    co_names = {c.id: c.name for c in r.cos.values()}
    players = []
    for label, p in zip(labels, start.players[:2], strict=True):
        country = r.countries.get(p.country)
        players.append(
            {
                "name": label,
                "co": co_names.get(p.co, str(p.co)),
                "country": country.name if country else "",
                "code": country.code if country else "",
            }
        )
    terrain_kinds = {}
    for tid in np.unique(md.terrain()).tolist():
        t = r.terrain.get(int(tid))
        terrain_kinds[int(tid)] = [t.kind, t.name, t.defense] if t else ["plain", str(tid), 1]
    return {
        "map": {
            "id": md.id,
            "name": _map_name(md.id),
            "width": md.width,
            "height": md.height,
            "terrain": md.terrain().tolist(),
            "terrain_kinds": terrain_kinds,
            "props": [list(p) for p in md.props()],
        },
        "kinds": list(r.terrain_kinds),
        "units": [u.name for u in r.units],
        "unit_stats": [[u.fuel, u.ammo] for u in r.units],
        "countries": {c.id: c.code for c in r.countries.values()},
        "players": players,
        "sprites": sprites.for_board(md, (start.players[0].country, start.players[1].country), weathers, moves),
    }


def game_view(log: GameLog, title: str = "") -> dict[str, Any]:
    """The game as plain data for viewer.html."""
    replayed = replay(log)
    md = _core.find_map(log.map_id)
    assert md is not None
    describer = _Describer(md)
    frames = [_frame(log.start, md.n_props) | {"t": "start", "m": -1, "k": []}]
    before = log.start
    searched = log.searched if log.searched is not None else np.zeros(0, SEARCHED)
    at = 0  # next searched decision
    for mv in replayed.moves:
        text, marks = describer.describe(before, mv)
        f = _frame(mv.state, md.n_props) | {"t": text, "m": mv.mover, "k": marks}
        while at < len(searched) and searched[at]["step"] < mv.first:
            at += 1
        if at < len(searched) and searched[at]["step"] <= mv.last:
            sign = 1 if mv.mover == 0 else -1
            f["v"] = round(sign * float(searched[at]["value"]), 4)
            f["nv"] = round(sign * float(searched[at]["net_value"]), 4)
            f["n"] = int(searched[at]["sims"])
        frames.append(f)
        before = mv.state
    if log.decided == "resigned":
        loser = 1 - log.winner if log.winner in (0, 1) else int(before.active)
        frames.append(_frame(before, md.n_props) | {"t": "resigns", "m": loser, "k": []})
    weathers = tuple(sorted({f["w"] for f in frames}))
    return board_data(md, log.start, log.players, weathers) | {
        "title": title,
        "result": {"winner": log.winner, "days": log.days, "decided": log.decided},
        "source": log.source,
        "info": log.info,
        "matches": replayed.matches,
        "frames": frames,
    }


def page(template: str, data: Any) -> str:
    """An HTML template from this package with board.js and ``data`` (JSON) put in."""
    here = Path(__file__).parent
    html = (here / template).read_text().replace("/*BOARD_JS*/", (here / "board.js").read_text())
    return html.replace("/*GAME_DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))


def write_page(view: dict[str, Any], out: Path) -> Path:
    out.write_text(page("viewer.html", view))
    return out
