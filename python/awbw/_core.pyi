"""AWBW engine core (C++)"""

from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


MAX_WIDTH: int = 40

MAX_HEIGHT: int = 40

MAX_TILES: int = 1600

MAX_PLAYERS: int = 2

MAX_UNITS: int = 128

MAX_PROPS: int = 128

STATE_BYTES: int = 4000

NO_UNIT: int = 255

NO_OWNER: int = 255

UNIT_MOVED: int = 1

UNIT_HIDDEN: int = 2

UNIT_CARRIED: int = 4

UNIT_STUNNED: int = 8

UNIT_CAPTURING: int = 16

class MapData:
    """A map's static data, shared by every game on it."""

    @property
    def id(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def n_props(self) -> int: ...

    def props(self) -> list[tuple[int, int, int, int]]:
        """
        (x, y, terrain kind, starting country) for each tile with per-game state.
        """

    def tile(self, x: int, y: int) -> int: ...

def register_map(map_id: int, terrain: Annotated[NDArray[numpy.int16], dict(shape=(None, None), order='C', device='cpu', writable=False)]) -> MapData:
    """
    Build a map from AWBW terrain ids [height, width] and register it under `map_id`.
    """

def find_map(map_id: int) -> MapData:
    """The registered map with this id, or None."""

class Settings:
    """Game options that vary between AWBW games."""

    def __init__(self) -> None: ...

    @property
    def funds_per_property(self) -> int: ...

    @funds_per_property.setter
    def funds_per_property(self, arg: int, /) -> None: ...

    @property
    def capture_win(self) -> int: ...

    @capture_win.setter
    def capture_win(self, arg: int, /) -> None: ...

    @property
    def day_limit(self) -> int: ...

    @day_limit.setter
    def day_limit(self, arg: int, /) -> None: ...

    @property
    def unit_limit(self) -> int: ...

    @unit_limit.setter
    def unit_limit(self, arg: int, /) -> None: ...

    @property
    def use_powers(self) -> int: ...

    @use_powers.setter
    def use_powers(self, arg: int, /) -> None: ...

    @property
    def banned_units(self) -> int: ...

    @banned_units.setter
    def banned_units(self, arg: int, /) -> None: ...

class Unit:
    @property
    def id(self) -> int: ...

    @property
    def tile(self) -> int: ...

    @property
    def type(self) -> int: ...

    @property
    def owner(self) -> int: ...

    @property
    def hp(self) -> int: ...

    @property
    def fuel(self) -> int: ...

    @property
    def ammo(self) -> int: ...

    @property
    def flags(self) -> int: ...

    @property
    def carrier(self) -> int: ...

    @property
    def cargo(self) -> list[int]: ...

class Player:
    @property
    def funds(self) -> int: ...

    @property
    def power(self) -> int: ...

    @property
    def cop_cost(self) -> int: ...

    @property
    def scop_cost(self) -> int: ...

    @property
    def co(self) -> int: ...

    @property
    def country(self) -> int: ...

    @property
    def power_on(self) -> int: ...

    @property
    def eliminated(self) -> int: ...

class GameState:
    """A game position: a fixed-size plain value."""

    def copy(self) -> GameState: ...

    def to_bytes(self) -> bytes: ...

    @staticmethod
    def from_bytes(data: bytes) -> GameState: ...

    @property
    def map_id(self) -> int: ...

    @property
    def settings(self) -> Settings: ...

    @property
    def day(self) -> int: ...

    @property
    def active(self) -> int: ...

    @property
    def winner(self) -> int: ...

    @property
    def players(self) -> list[Player]: ...

    def unit(self, slot: int) -> Unit: ...

    def unit_slots(self) -> list[int]:
        """Slots holding a unit, in slot order."""

    def unit_at(self, x: int, y: int) -> int:
        """Slot of the uncarried unit on a tile, or -1."""

    def prop_owner(self, prop: int) -> int: ...

    def prop_points(self, prop: int) -> int: ...

    def check(self) -> str:
        """Consistency problems, or '' if the state is sound."""

    def __getstate__(self) -> bytes: ...

    def __setstate__(self, arg: bytes, /) -> None: ...

def load_snapshot(map: MapData, settings: Settings, day: int, active: int, players: Annotated[NDArray[numpy.int64], dict(shape=(None, 8), order='C', device='cpu', writable=False)], buildings: Annotated[NDArray[numpy.int64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], units: Annotated[NDArray[numpy.int64], dict(shape=(None, 11), order='C', device='cpu', writable=False)]) -> GameState:
    """
    GameState from a flattened AWBW snapshot. players: [2, 8] (co, country, funds, power, cop_cost, scop_cost, power_on, eliminated); buildings: [n, 4] (x, y, terrain_id, capture); units: [n, 11] (id, type, owner, x, y, hp, fuel, ammo, flags, cargo1_id, cargo2_id).
    """

def reachable(state: GameState, slot: int) -> list[tuple[int, int, int]]:
    """
    (x, y, cost) of every tile the unit can move through this turn, cheapest first.
    """

def path_cost(state: GameState, slot: int, path: Sequence[tuple[int, int]]) -> int:
    """
    Movement cost of a path of (x, y) starting on the unit's tile, or -1 if it is not a legal move.
    """

def move_points(state: GameState, slot: int) -> int:
    """Movement points with CO bonuses, before the fuel cap."""

class Combatant:
    """
    One side of a fight: unit, tile and CO state as the damage formula sees them.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, type: int, hp: int, ammo: int = 0, co: int = 0, power: int = 0, terrain: int = 0, stars: int = 0, towers: int = 0, extra_atk: int = 0, hidden: bool = False) -> None: ...

    @property
    def type(self) -> int:
        """engine unit index"""

    @type.setter
    def type(self, arg: int, /) -> None: ...

    @property
    def hp(self) -> int:
        """internal HP, 1..100"""

    @hp.setter
    def hp(self, arg: int, /) -> None: ...

    @property
    def ammo(self) -> int: ...

    @ammo.setter
    def ammo(self, arg: int, /) -> None: ...

    @property
    def co(self) -> int:
        """AWBW CO id"""

    @co.setter
    def co(self, arg: int, /) -> None: ...

    @property
    def power(self) -> int:
        """0 none, 1 COP, 2 SCOP"""

    @power.setter
    def power(self, arg: int, /) -> None: ...

    @property
    def terrain(self) -> int:
        """terrain kind index of the unit's tile"""

    @terrain.setter
    def terrain(self, arg: int, /) -> None: ...

    @property
    def stars(self) -> int:
        """defense stars of the unit's tile"""

    @stars.setter
    def stars(self, arg: int, /) -> None: ...

    @property
    def towers(self) -> int:
        """comtowers owned by the unit's player"""

    @towers.setter
    def towers(self, arg: int, /) -> None: ...

    @property
    def extra_atk(self) -> int:
        """attack % from state-dependent CO effects"""

    @extra_atk.setter
    def extra_atk(self, arg: int, /) -> None: ...

    @property
    def hidden(self) -> int:
        """dived Sub or hidden Stealth"""

    @hidden.setter
    def hidden(self, arg: int, /) -> None: ...

def attack_value(att: Combatant, defender: Combatant) -> int: ...

def defense_value(defender: Combatant, att: Combatant) -> int: ...

def damage(att: Combatant, defender: Combatant, luck: int, counter: bool = False) -> int:
    """Damage in internal HP for a given net luck roll."""

def can_attack(att: Combatant, defender: Combatant, dist: int) -> bool: ...

def weapon(att: Combatant, def_type: int) -> int:
    """-1 none, 0 ammo weapon, 1 weapon without ammo"""

def combat_outcomes(att: Combatant, defender: Combatant, dist: int) -> list[tuple[int, int, float]]:
    """
    Exact (attacker_hp, defender_hp, probability) results of an attack, counterattack included.
    """
