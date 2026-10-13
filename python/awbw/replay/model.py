"""Typed view of an AWBW replay: game settings, turn-start snapshots, action log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _yn(value: object) -> bool:
    return value in ("Y", "y")


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


@dataclass(slots=True)
class GameInfo:
    """Per-game settings; constant across turns."""

    id: int
    name: str
    map_id: int
    funds_per_property: int
    starting_funds: int
    capture_win: int  # properties needed to win; AWBW uses 1000 for "off"
    fog: bool
    use_powers: bool
    weather_type: str  # "Clear", "Random", ...
    type: str  # AWBW game type code ("L" league, "N", "A", ...)
    official: bool
    team: bool
    league: str | None
    start_date: str | None
    min_rating: int | None
    max_rating: int | None
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN = frozenset(
        [
            "id",
            "name",
            "maps_id",
            "funds",
            "starting_funds",
            "capture_win",
            "fog",
            "use_powers",
            "weather_type",
            "type",
            "official",
            "team",
            "league",
            "start_date",
            "min_rating",
            "max_rating",
            "players",
            "buildings",
            "units",
        ]
    )

    @classmethod
    def from_php(cls, f: dict[str, Any]) -> GameInfo:
        return cls(
            id=int(f["id"]),
            name=str(f.get("name") or ""),
            map_id=int(f["maps_id"]),
            funds_per_property=int(f["funds"]),
            starting_funds=int(f.get("starting_funds") or 0),
            capture_win=int(f.get("capture_win") or 1000),
            fog=_yn(f.get("fog")),
            use_powers=_yn(f.get("use_powers", "Y")),
            weather_type=str(f.get("weather_type") or "Clear"),
            type=str(f.get("type") or ""),
            official=_yn(f.get("official")),
            team=_yn(f.get("team")),
            league=None if f.get("league") is None else str(f["league"]),
            start_date=None if f.get("start_date") is None else str(f["start_date"]),
            min_rating=_opt_int(f.get("min_rating")),
            max_rating=_opt_int(f.get("max_rating")),
            extra={k: v for k, v in f.items() if k not in cls._KNOWN},
        )


@dataclass(slots=True)
class PlayerState:
    id: int
    users_id: int
    countries_id: int
    co_id: int
    tags_co_id: int | None
    order: int  # turn order: lower moves first
    team: str
    funds: int
    eliminated: bool
    co_power: int  # current power meter, in AWBW's funds-based units
    co_max_power: int  # meter needed for COP
    co_max_spower: int  # meter needed for SCOP
    co_power_on: str  # "N" none, "Y" COP active, "S" SCOP active

    @classmethod
    def from_php(cls, f: dict[str, Any]) -> PlayerState:
        return cls(
            id=int(f["id"]),
            users_id=int(f["users_id"]),
            countries_id=int(f["countries_id"]),
            co_id=int(f["co_id"]),
            tags_co_id=_opt_int(f.get("tags_co_id")),
            order=int(f["order"]),
            team=str(f.get("team") or ""),
            funds=int(f["funds"]),
            eliminated=_yn(f.get("eliminated")),
            co_power=int(f.get("co_power") or 0),
            co_max_power=int(f.get("co_max_power") or 0),
            co_max_spower=int(f.get("co_max_spower") or 0),
            co_power_on=str(f.get("co_power_on") or "N").upper(),
        )


@dataclass(slots=True)
class BuildingState:
    id: int
    terrain_id: int  # AWBW terrain id; encodes building type and owner country
    x: int
    y: int
    capture: int  # remaining capture points (20 = untouched)
    last_capture: int

    @classmethod
    def from_php(cls, f: dict[str, Any]) -> BuildingState:
        return cls(
            id=int(f["id"]),
            terrain_id=int(f["terrain_id"]),
            x=int(f["x"]),
            y=int(f["y"]),
            capture=int(f["capture"]),
            last_capture=int(f.get("last_capture") or 0),
        )


@dataclass(slots=True)
class UnitState:
    id: int
    players_id: int
    name: str
    x: int
    y: int
    hp: int  # internal HP 0..100 (snapshots store hp / 10 with one decimal)
    fuel: int
    ammo: int
    moved: bool
    capturing: bool
    fired: bool
    carried: bool
    cargo: tuple[int, int]  # AWBW ids in cargo slots 1 and 2; 0 = empty (slots don't shift)
    sub_dive: str  # "N", or the dived/hidden marker for subs and stealths
    movement_points: int
    vision: int
    fuel_per_turn: int
    short_range: int
    long_range: int
    second_weapon: bool
    cost: int
    movement_type: str
    symbol: str

    @classmethod
    def from_php(cls, f: dict[str, Any]) -> UnitState:
        cargo = (int(f.get("cargo1_units_id") or 0), int(f.get("cargo2_units_id") or 0))
        return cls(
            id=int(f["id"]),
            players_id=int(f["players_id"]),
            name=str(f["name"]),
            x=int(f["x"]),
            y=int(f["y"]),
            hp=round(float(f["hit_points"]) * 10),
            fuel=int(f["fuel"]),
            ammo=int(f["ammo"]),
            moved=bool(int(f.get("moved") or 0)),
            capturing=bool(int(f.get("capture") or 0)),
            fired=bool(int(f.get("fired") or 0)),
            carried=_yn(f.get("carried")),
            cargo=cargo,
            sub_dive=str(f.get("sub_dive") or "N"),
            movement_points=int(f["movement_points"]),
            vision=int(f["vision"]),
            fuel_per_turn=int(f["fuel_per_turn"]),
            short_range=int(f["short_range"]),
            long_range=int(f["long_range"]),
            second_weapon=_yn(f.get("second_weapon")),
            cost=int(f["cost"]),
            movement_type=str(f["movement_type"]),
            symbol=str(f.get("symbol") or ""),
        )


@dataclass(slots=True)
class Turn:
    """State at the start of one player's turn, plus the actions taken in it."""

    day: int
    player_id: int  # whose turn it is
    weather: str  # weather code ("C" clear, ...)
    players: list[PlayerState]
    buildings: list[BuildingState]
    units: list[UnitState]
    actions: list[dict] | None = None  # None when the log has no entry for this turn


@dataclass(slots=True)
class GameOver:
    day: int
    winners: list[int]  # player ids
    losers: list[int]
    end_action: str  # action that ended the game ("Resign", "Fire", "Capt", ...)


@dataclass(slots=True)
class Replay:
    game: GameInfo
    turns: list[Turn]
    orphan_action_turns: int = 0  # log entries with no matching snapshot

    @property
    def player_ids(self) -> list[int]:
        """Player ids in turn order."""
        return [p.id for p in sorted(self.turns[0].players, key=lambda p: p.order)]

    @property
    def days(self) -> int:
        return self.turns[-1].day

    def game_over(self) -> GameOver | None:
        """Result from the action log, or None if the log never ends the game."""
        for turn in reversed(self.turns):
            for action in reversed(turn.actions or []):
                info = _find_key(action, "GameOver")
                if isinstance(info, dict):
                    return GameOver(
                        day=int(info.get("day") or turn.day),
                        winners=[int(p) for p in info.get("winners") or []],
                        losers=[int(p) for p in info.get("losers") or []],
                        end_action=str(action.get("action")),
                    )
        return None


def _find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    return None
