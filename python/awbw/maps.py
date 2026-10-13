"""AWBW maps from the public map API (``api/map/map_info.php``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .net import Http, cache_dir, write_atomic

MAP_INFO_URL = "https://awbw.amarriner.com/api/map/map_info.php?maps_id={map_id}"


@dataclass(slots=True, frozen=True)
class PredeployedUnit:
    unit_id: int  # AWBW unit type id
    x: int
    y: int
    hp: int
    country_code: str


@dataclass(slots=True)
class MapInfo:
    id: int
    name: str
    author: str
    player_count: int
    width: int
    height: int
    terrain: np.ndarray  # int16 [height, width] of AWBW terrain ids
    predeployed: list[PredeployedUnit]


def parse_map_info(map_id: int, raw: bytes) -> MapInfo:
    d = json.loads(raw)
    if not isinstance(d, dict) or "Terrain Map" not in d:
        raise ValueError(f"map {map_id}: unexpected response {raw[:200]!r}")
    width, height = int(d["Size X"]), int(d["Size Y"])
    terrain = np.zeros((height, width), dtype=np.int16)
    for x, column in enumerate(d["Terrain Map"]):  # the API is column-major
        for y, value in enumerate(column):
            # Non-integer entries are blank tiles in the editor; AWBW treats them as id 0.
            terrain[y, x] = value if isinstance(value, int) else 0
    predeployed = [
        PredeployedUnit(
            int(u["Unit ID"]), int(u["Unit X"]), int(u["Unit Y"]), int(u["Unit HP"]), str(u["Country Code"])
        )
        for u in d.get("Predeployed Units") or []
    ]
    return MapInfo(
        id=map_id,
        name=str(d.get("Name", "")),
        author=str(d.get("Author", "")),
        player_count=int(d.get("Player Count", 0)),
        width=width,
        height=height,
        terrain=terrain,
        predeployed=predeployed,
    )


def map_path(map_id: int) -> Path:
    return cache_dir() / "maps" / f"{map_id}.json"


def fetch_map(map_id: int, http: Http) -> Path:
    """Download a map's JSON into the cache unless it is already there."""
    path = map_path(map_id)
    if not path.exists():
        raw = http.get(MAP_INFO_URL.format(map_id=map_id))
        parse_map_info(map_id, raw)  # refuse to cache error pages
        write_atomic(path, raw)
    return path


def load_map(map_id: int) -> MapInfo:
    return parse_map_info(map_id, map_path(map_id).read_bytes())
