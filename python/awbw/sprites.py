"""The AWBW Replay Player's art on the boards of the viewer and the play page.

The art is Nintendo's, so none of it is in the repo: the boards draw it from
a local copy in ``cache/sprites/`` if there is one, and are coloured squares
otherwise. The copy is the Player's embedded resources laid out as its own
folders: ``Textures/Map/<skin>/...``, ``Textures/Units/<Country>/...``,
``Textures/UI/...`` and the index files ``Json/{Tiles,Buildings,Units,
Countries}.json`` that name each AWBW terrain, building and unit's textures.

``for_board`` picks what one game's board needs and returns it as data URIs,
with the tables board.js draws from. Terrain is the Player's Classic skin and
buildings its AW2 skin, its defaults. Sea and shoal take the Player's
coastline pieces from their neighbours: a sea tile shows land on each side
that is land and on each corner that is land with water on both sides next
to it (shoal, reef and bridges are water); a shoal shows land toward land and
bridges, open water toward sea, reef and the map's edge, and nothing toward
more shoal. That rule picks the piece the Player's own test renders show on
all 2,176 sea and shoal tiles of its five test maps.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import _core, rules
from .net import cache_dir

MAP_SKIN, BUILDING_SKIN = "Classic", "AW2"
_WEATHERS = ("Clear", "Rain", "Snow")  # in _core.WEATHERS order
_PROPS_ON_PLAIN = ("city", "base", "airport", "port", "hq", "comtower", "lab", "silo", "seam")
_RUBBLE = {113: 115, 114: 116}  # a pipe seam's terrain when broken
_MOVES = {"S": "MoveSideAnimation", "U": "MoveUpAnimation", "D": "MoveDownAnimation"}
_UI = {
    "body": "Arrow_Body",
    "curve": "Arrow_Curved",
    "tip": "Arrow_Tip",
    "capture": "Capturing",
    "cargo": "HasCargo",
    "dive": "Dive",
    "stun": "Stun",
    "fuel": "LowFuel",
    "ammo": "LowAmmo",
}

Ref = list[Any]  # [[texture per weather], index into the timings or -1]


def root() -> Path:
    return cache_dir() / "sprites"


def _jsonc(path: Path) -> Any:
    """The Player's JSON, which has // and /* */ comments and trailing commas."""
    text, out, i, quoted = path.read_text(encoding="utf-8-sig"), [], 0, False
    while i < len(text):
        c = text[i]
        if quoted:
            out.append(c)
            if c == "\\":
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
            out.append(c)
        elif text.startswith("//", i):
            i = text.find("\n", i)
            if i < 0:
                break
            continue
        elif text.startswith("/*", i):
            i = text.index("*/", i) + 2
            continue
        else:
            out.append(c)
        i += 1
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", "".join(out)))


class _Art:
    """The index files and textures of one copy."""

    def __init__(self, base: Path):
        self.textures = base / "Textures"
        tiles = _jsonc(base / "Json" / "Tiles.json").values()
        self.tiles = {t["AWBWID"]: t for t in tiles}
        self.coast: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}
        for t in tiles:
            group, _, name = t["Textures"]["Clear"].partition("/")
            if group in ("Sea", "Shoal") and t["TerrainType"] == group:
                self.coast[group, frozenset() if name in ("Sea", "C") else frozenset(name.split("-"))] = t
        self.buildings = {b["AWBWID"]: b for b in _jsonc(base / "Json" / "Buildings.json").values()}
        self.units = {u["AWBWId"]: u for u in _jsonc(base / "Json" / "Units.json").values()}
        self.countries = {c["AWBWID"]: c for c in _jsonc(base / "Json" / "Countries.json").values()}
        self._png: dict[str, bytes | None] = {}

    def png(self, key: str) -> bytes | None:
        if key not in self._png:
            p = self.textures / f"{key}.png"
            self._png[key] = p.read_bytes() if p.is_file() else None
        return self._png[key]

    def height(self, key: str) -> int:
        data = self.png(key)
        return int.from_bytes(data[20:24], "big") if data else 16  # PNG's IHDR


_ARTS: dict[Path, _Art] = {}


def _art(base: Path) -> _Art | None:
    if base not in _ARTS and (base / "Json" / "Tiles.json").is_file():
        _ARTS[base] = _Art(base)
    return _ARTS.get(base)


class _Picker:
    """Collects one board's refs, the textures they use and the frame timings."""

    def __init__(self, art: _Art, weathers: tuple[int, ...]):
        self.art, self.weathers = art, weathers
        self.timings: list[list[int]] = []
        self.keys: set[str] = set()

    def ref(self, folder: str, textures: dict[str, str], frames: list[int] | None) -> Ref | None:
        """A ref to an entry's textures under ``folder``: the texture itself when
        ``frames`` is None, else one texture per frame ("-0", "-1", ...)."""
        tails = [""] if frames is None else [f"-{i}" for i in range(len(frames))]
        clear = f"{folder}/{textures['Clear']}"
        if any(self.art.png(clear + t) is None for t in tails):
            return None
        per_weather = []
        for w, name in enumerate(_WEATHERS):
            key = f"{folder}/{textures.get(name, textures['Clear'])}"
            if any(self.art.png(key + t) is None for t in tails):
                key = clear
            per_weather.append(key)
            if w in self.weathers:
                self.keys.update(key + t for t in tails)
        timing = -1
        if frames is not None:
            if frames not in self.timings:
                self.timings.append(frames)
            timing = self.timings.index(frames)
        return [per_weather, timing]

    def unit_ref(self, folder: str, animation: dict[str, Any] | None) -> Ref | None:
        if animation is None:
            return None
        tails = [f"-{i}" for i in range(len(animation["Frames"]))]
        key = f"{folder}/{animation['Texture']}"
        if any(self.art.png(key + t) is None for t in tails):
            return None
        self.keys.update(key + t for t in tails)
        if animation["Frames"] not in self.timings:
            self.timings.append(animation["Frames"])
        return [[key], self.timings.index(animation["Frames"])]

    def key(self, key: str) -> str | None:
        if self.art.png(key) is None:
            return None
        self.keys.add(key)
        return key

    def height(self, ref: Ref | None) -> int:
        if ref is None:
            return 16
        tails = [""] if ref[1] < 0 else [f"-{i}" for i in range(len(self.timings[ref[1]]))]
        return max(self.art.height(k + t) for k in ref[0] for t in tails)


def _coast_class(kind: str | None) -> str:
    return "off" if kind is None else kind if kind in ("sea", "shoal", "reef", "bridge") else "land"


def coast_tokens(kind: str, around: Mapping[str, str | None]) -> frozenset[str]:
    """The Player's coastline piece for a sea or shoal tile, from its
    neighbours' terrain kinds (``around``: N, E, S, W, NE, SE, SW, NW; None
    off the map), as the tokens of its texture's name."""
    c = {d: _coast_class(k) for d, k in around.items()}
    if kind == "sea":
        sides = {d for d in "NESW" if c[d] == "land"}
        water = ("sea", "shoal", "reef", "bridge")
        corners = {d for d in ("NE", "SE", "SW", "NW") if c[d] == "land" and c[d[0]] in water and c[d[1]] in water}
        return frozenset(sides | corners)
    tokens = set()
    for d in "NESW":
        if c[d] in ("land", "bridge"):
            tokens.add(d)
        elif c[d] in ("sea", "reef", "off"):
            tokens.add("A" + d)
    return frozenset(tokens)


_AROUND = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
    "NE": (1, -1),
    "SE": (1, 1),
    "SW": (-1, 1),
    "NW": (-1, -1),
}


def for_board(
    md: _core.MapData, countries: tuple[int, int], weathers: tuple[int, ...] = (0, 1, 2), moves: bool = False
) -> dict[str, Any] | None:
    """What board.js draws one game's board with, or None without a copy.

    ``countries``: each player's AWBW country id; ``weathers``: those whose
    textures to include; ``moves``: include the units' move animations.
    The refs are [[texture per weather], timing]: a texture's key is the
    weather's, plus "-<frame>" at the time when timing (an index into
    ``anim``, frame lengths in ms) is not -1.
    """
    art = _art(root().resolve())
    if art is None:
        return None
    r = rules.load()
    pick = _Picker(art, weathers)
    map_folder, building_folder = f"Map/{MAP_SKIN}", f"Map/{BUILDING_SKIN}"
    terrain = md.terrain()
    kinds = [[r.terrain[int(t)].kind if int(t) in r.terrain else "plain" for t in row] for row in terrain.tolist()]

    def tile_ref(entry: dict[str, Any] | None) -> Ref | None:
        return pick.ref(map_folder, entry["Textures"], None) if entry else None

    def building_ref(terrain_id: int | None) -> Ref | None:
        entry = art.buildings.get(terrain_id) if terrain_id is not None else None
        return pick.ref(building_folder, entry["Textures"], entry.get("frames", [1000])) if entry else None

    plain = tile_ref(art.tiles.get(1))
    ground: list[Ref | None] = []
    for y in range(md.height):
        for x in range(md.width):
            kind = kinds[y][x]
            if kind in _PROPS_ON_PLAIN:
                ground.append(plain)
            elif kind in ("sea", "shoal"):
                around = {}
                for d, (dx, dy) in _AROUND.items():
                    xx, yy = x + dx, y + dy
                    around[d] = kinds[yy][xx] if 0 <= xx < md.width and 0 <= yy < md.height else None
                group = "Sea" if kind == "sea" else "Shoal"
                ground.append(tile_ref(art.coast.get((group, coast_tokens(kind, around)))))
            else:
                ground.append(tile_ref(art.tiles.get(int(terrain[y, x]))))

    by_kind = {(t.kind, t.country): t.id for t in r.terrain.values()}
    props: list[dict[str, Ref | None] | None] = []
    top = [16] + [pick.height(g) for g in ground[: md.width]]
    for x, y, _, _ in md.props():
        tid = int(terrain[y, x])
        kind = kinds[y][x]
        if kind == "seam":
            p = {"n": building_ref(tid), "x": tile_ref(art.tiles.get(_RUBBLE.get(tid, 115)))}
        elif kind == "silo":
            p = {"n": building_ref(tid), "x": tile_ref(art.tiles.get(112))}
        else:
            p = {"n": building_ref(by_kind.get((kind, None)))}
            for player, country in enumerate(countries):
                p[str(player)] = building_ref(by_kind.get((kind, country)))
        props.append(p)
        if y == 0:
            top += [pick.height(ref) for ref in p.values()]

    units, move_refs = [], []
    for country in countries:
        c = art.countries.get(country, {})
        folder = c.get("unitPath", "")
        entries = [art.units.get(u.id) or {} for u in r.units] if folder else [{} for _ in r.units]
        units.append([pick.unit_ref(folder, e.get("IdleAnimation")) for e in entries])
        if moves:
            move_refs.append([{k: pick.unit_ref(folder, e.get(a)) for k, a in _MOVES.items()} for e in entries])
    ui: dict[str, Any] = {name: pick.key(f"UI/{texture}") for name, texture in _UI.items()}
    ui["hp"] = [pick.key(f"UI/Healthv2/{n}") for n in range(10)]
    return {
        "img": {k: "data:image/png;base64," + base64.b64encode(art.png(k) or b"").decode() for k in sorted(pick.keys)},
        "anim": pick.timings,
        "ground": ground,
        "plain": plain,
        "props": props,
        "units": units,
        "moves": move_refs if moves else None,
        "ui": ui,
        "pad": max(top) - 16,
    }
