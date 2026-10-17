"""Global League map rotation and settings, from AWBW's league maps API.

Replays don't record unit bans or the unit limit, so for league games these
settings come from here. The rotation changes over time, so each fetch is
kept as a dated snapshot under ``cache/league/``. Self-play and the arena play
under today's standard rules (``std_rules``).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core, rules
from .net import Http, cache_dir, write_atomic

LEAGUE_MAPS_URL = "https://awbw.amarriner.com/api/league/maps.php"
STD_BANS = ("Black Bomb", "Stealth")  # on every std map of the 2026-10-12 rotation
STD_UNIT_LIMIT = 50  # likewise


def std_rules(state: _core.GameState) -> _core.GameState:
    """``state``, changed to the Global League's standard rules of the 2026-10-12
    rotation: Black Bomb and Stealth banned, 50 units a player, no day limit.
    A state from a replay has none of these, which the replay doesn't record."""
    st = state.settings
    st.banned_units = sum(1 << rules.load().unit(name).index for name in STD_BANS)
    st.unit_limit = STD_UNIT_LIMIT
    st.day_limit = 0
    return state


def std_rules_rows(states: np.ndarray) -> np.ndarray:
    """``std_rules`` on each row of packed states [N, STATE_BYTES], in place."""
    for row in states:
        row[:] = np.frombuffer(std_rules(_core.GameState.from_bytes(row.tobytes())).to_bytes(), np.uint8)
    return states


@dataclass(slots=True, frozen=True)
class LeagueTier:
    name: str  # "T0".."T5", "TL"
    enabled: bool  # whether games are played in this tier on this map
    co_ids: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class LeagueMap:
    id: int
    name: str
    mode: str  # "std", "fog" or "hf" (high funds)
    unit_limit: int
    cap_limit: int
    unit_bans: tuple[str, ...]  # unit names, as in `LeagueRotation.units`
    tiers: tuple[LeagueTier, ...]


@dataclass(slots=True)
class LeagueRotation:
    date: str  # when this snapshot was fetched (YYYY-MM-DD)
    maps: list[LeagueMap]
    units: dict[int, str]  # AWBW unit id -> name
    cos: dict[int, str]  # AWBW CO id -> name

    def mode(self, mode: str) -> list[LeagueMap]:
        return [m for m in self.maps if m.mode == mode]


def parse_rotation(raw: bytes, date: str) -> LeagueRotation:
    d = json.loads(raw)
    if not isinstance(d, dict) or "maps" not in d:
        raise ValueError(f"unexpected league maps response: {raw[:200]!r}")
    maps = [
        LeagueMap(
            id=int(m["id"]),
            name=str(m["name"]),
            mode=str(m["type"]),
            unit_limit=int(m["unitLimit"]),
            cap_limit=int(m["capLimit"]),
            unit_bans=tuple(str(u) for u in m.get("unitBans") or []),
            tiers=tuple(
                LeagueTier(str(t["name"]), bool(t["enabled"]), tuple(int(c["id"]) for c in t.get("cos") or []))
                for t in m.get("tiers") or []
            ),
        )
        for m in d["maps"]
    ]
    return LeagueRotation(
        date=date,
        maps=maps,
        units={int(k): str(v["name"]) for k, v in (d.get("units") or {}).items()},
        cos={int(k): str(v["name"]) for k, v in (d.get("cos") or {}).items()},
    )


def rotation_dir() -> Path:
    return cache_dir() / "league"


def fetch_rotation(http: Http, today: dt.date | None = None) -> Path:
    """Fetch the current rotation into a dated snapshot (one per day)."""
    date = (today or dt.date.today()).isoformat()
    path = rotation_dir() / f"maps-{date}.json"
    if not path.exists():
        raw = http.post_json(LEAGUE_MAPS_URL, {"method": "listMaps"})
        parse_rotation(raw, date)  # refuse to cache error responses
        write_atomic(path, raw)
    return path


def load_rotation(path: Path | None = None) -> LeagueRotation:
    """Load a snapshot; by default the most recent one in the cache."""
    if path is None:
        snapshots = sorted(rotation_dir().glob("maps-*.json"))
        if not snapshots:
            raise FileNotFoundError(f"no league rotation snapshots in {rotation_dir()}; run `awbw league`")
        path = snapshots[-1]
    return parse_rotation(path.read_bytes(), path.stem.removeprefix("maps-"))
