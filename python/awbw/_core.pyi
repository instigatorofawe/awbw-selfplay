"""AWBW engine core (C++)"""

from typing import overload


MAX_WIDTH: int = 40

MAX_HEIGHT: int = 40

MAX_TILES: int = 1600

MAX_PLAYERS: int = 2

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
