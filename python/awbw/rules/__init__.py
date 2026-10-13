"""AWBW rules tables: units, terrain, damage chart, COs and countries.

The JSON files next to this module are the single source of truth. Python
reads them here, and ``tools/gen_tables.py`` compiles them into the C++
engine's ``src/core/tables.h``.

Unit selectors (used by CO modifiers and transports) name a unit, a class
(``foot``, ``vehicle``, ``copter``, ``plane``, ``naval``) or a group:
``all``, ``ground`` (foot + vehicle), ``air`` (copter + plane), ``direct``
(max range 1), ``indirect`` (max range > 1) and ``transport`` (units that
carry others and have no weapon).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

MOVE_TYPES = ("F", "B", "T", "W", "A", "S", "L", "P")  # AWBW movement_type codes
UNIT_CLASSES = ("foot", "vehicle", "copter", "plane", "naval")
POWER_STATES = ("d2d", "cop", "scop")
WEATHERS = ("clear",)  # rain and snow costs are not tabulated yet
TERRAIN_GROUPS = {"urban": ("city", "base", "airport", "port", "hq", "lab", "comtower")}
POWER_BONUS = 10  # every CO's extra attack and defense % while a power is on
TOWER_ATK = 10  # default attack % per owned comtower


@dataclass(frozen=True, slots=True)
class UnitType:
    index: int  # engine index, in AWBW id order
    id: int  # AWBW unit id
    name: str  # AWBW name, as in snapshots ("Md.Tank", "B-Copter", ...)
    cls: str
    cost: int
    move: int
    move_type: str
    vision: int
    fuel: int
    fuel_per_day: int
    hidden_fuel_per_day: int
    ammo: int
    min_range: int
    max_range: int
    capture: bool
    carry_slots: int
    carry: frozenset[int]  # unit indices this transport can load
    supplies: bool
    repairs: bool
    hides: bool
    explodes: bool

    @property
    def indirect(self) -> bool:
        return self.max_range > 1


@dataclass(frozen=True, slots=True)
class Terrain:
    id: int
    name: str
    kind: str
    defense: int
    country: int | None


@dataclass(frozen=True, slots=True)
class Country:
    id: int
    code: str
    name: str


@dataclass(frozen=True, slots=True)
class CoState:
    """A CO's modifiers in one power state (``d2d``, ``cop`` or ``scop``).

    The universal power bonus and the default comtower attack are not
    included here; see ``POWER_BONUS`` and ``TOWER_ATK``.
    """

    atk: tuple[int, ...]  # attack % by unit index
    defense: tuple[int, ...]
    move: tuple[int, ...]
    range: tuple[int, ...]
    terrain_atk: dict[str, int]  # extra attack % by the attacker's terrain kind
    atk_per_star: int
    star_mult: int
    luck_good: int
    luck_bad: int
    tower_atk: int
    tower_def: int
    def_vs_indirect: int
    counter: int  # counterattack damage %
    capture: int  # capture points %
    cost: int  # build cost %
    move_cost_one: bool
    income_bonus: int
    air_fuel_per_day: int
    repair_bonus: int  # extra internal HP per repair


@dataclass(frozen=True, slots=True)
class Co:
    id: int
    name: str
    cop_stars: int | None
    scop_stars: int | None
    d2d: CoState
    cop: CoState
    scop: CoState
    notes: dict[str, str]

    def state(self, power: int) -> CoState:
        """Modifiers for power state 0 (none), 1 (COP) or 2 (SCOP)."""
        return (self.d2d, self.cop, self.scop)[power]


@dataclass(frozen=True)
class Rules:
    units: tuple[UnitType, ...]
    terrain: dict[int, Terrain]
    terrain_kinds: tuple[str, ...]
    move_costs: dict[str, dict[str, dict[str, int]]]  # weather -> kind -> move type -> cost
    primary: tuple[tuple[int | None, ...], ...]  # [attacker][defender] base damage with the ammo weapon
    secondary: tuple[tuple[int | None, ...], ...]  # same, with the weapon that needs no ammo
    cos: dict[int, Co]
    countries: dict[int, Country]

    def unit(self, key: int | str) -> UnitType:
        """Look up a unit type by AWBW id or name."""
        for u in self.units:
            if key in (u.id, u.name):
                return u
        raise KeyError(key)

    def select(self, selector: str, except_: str | None = None) -> frozenset[int]:
        """Unit indices matching a selector, minus those matching ``except_``."""
        chosen = _select(self.units, self.primary, self.secondary, selector)
        if except_ is not None:
            chosen -= _select(self.units, self.primary, self.secondary, except_)
        return chosen


def _read(name: str) -> Any:
    return json.loads(files(__package__).joinpath(name).read_text(encoding="utf-8"))


def _select(units, primary, secondary, selector: str) -> frozenset[int]:
    def armed(u: UnitType) -> bool:
        return any(x is not None for x in primary[u.index]) or any(x is not None for x in secondary[u.index])

    preds = {
        "all": lambda u: True,
        "ground": lambda u: u.cls in ("foot", "vehicle"),
        "air": lambda u: u.cls in ("copter", "plane"),
        "direct": lambda u: not u.indirect,
        "indirect": lambda u: u.indirect,
        "transport": lambda u: u.carry_slots > 0 and not armed(u),
    }
    if selector in preds:
        pred = preds[selector]
    elif selector in UNIT_CLASSES:
        pred = lambda u: u.cls == selector  # noqa: E731
    elif any(u.name == selector for u in units):
        pred = lambda u: u.name == selector  # noqa: E731
    else:
        raise ValueError(f"unknown unit selector {selector!r}")
    return frozenset(u.index for u in units if pred(u))


@cache
def load() -> Rules:
    """Load and cross-check the rules tables (cached)."""
    raw_units = _read("units.json")["units"]
    names = [u["name"] for u in raw_units]
    index = {n: i for i, n in enumerate(names)}
    assert len(index) == len(names), "duplicate unit names"
    assert [u["id"] for u in raw_units] == sorted(u["id"] for u in raw_units), "units must be in AWBW id order"

    dmg = _read("damage.json")
    assert set(dmg["chart"]) == set(names)
    primary: list[list[int | None]] = [[None] * len(names) for _ in names]
    secondary: list[list[int | None]] = [[None] * len(names) for _ in names]
    classes = {u["name"]: u["class"] for u in raw_units}
    for a, row in dmg["chart"].items():
        ai = index[a]
        has_ammo = raw_units[ai]["ammo"] > 0
        targets = dmg["primary_targets"].get(a)
        for d, value in row.items():
            uses_ammo = has_ammo and (targets is None or classes[d] in targets or d in targets)
            (primary if uses_ammo else secondary)[ai][index[d]] = value
    for a, row in dmg["secondary_fallback"].items():
        for d, value in row.items():
            ai, di = index[a], index[d]
            assert primary[ai][di] is not None and secondary[ai][di] is None, (a, d)
            secondary[ai][di] = value
    primary_t = tuple(tuple(r) for r in primary)
    secondary_t = tuple(tuple(r) for r in secondary)

    # Transports are resolved after every unit exists, since carry uses selectors.
    pending: list[UnitType] = []
    for i, u in enumerate(raw_units):
        lo, hi = u["range"]
        pending.append(
            UnitType(
                index=i,
                id=u["id"],
                name=u["name"],
                cls=u["class"],
                cost=u["cost"],
                move=u["move"],
                move_type=u["move_type"],
                vision=u["vision"],
                fuel=u["fuel"],
                fuel_per_day=u["fuel_per_day"],
                hidden_fuel_per_day=u.get("hidden_fuel_per_day", u["fuel_per_day"]),
                ammo=u["ammo"],
                min_range=lo,
                max_range=hi,
                capture=u.get("capture", False),
                carry_slots=u.get("carry", {}).get("slots", 0),
                carry=frozenset(),
                supplies=u.get("supplies", False),
                repairs=u.get("repairs", False),
                hides=u.get("hides", False),
                explodes=u.get("explodes", False),
            )
        )
        assert u["class"] in UNIT_CLASSES and u["move_type"] in MOVE_TYPES, u["name"]
    units: list[UnitType] = []
    for u, raw in zip(pending, raw_units, strict=True):
        carry: frozenset[int] = frozenset()
        for sel in raw.get("carry", {}).get("units", []):
            carry |= _select(pending, primary_t, secondary_t, sel)
        units.append(dataclasses.replace(u, carry=carry))
    units_t = tuple(units)

    traw = _read("terrain.json")
    move_costs = traw["move_costs"]
    assert tuple(move_costs) == WEATHERS, tuple(move_costs)
    kinds = tuple(move_costs["clear"])
    terrain = {}
    for t in traw["terrain"]:
        assert t["kind"] in kinds, t
        terrain[t["id"]] = Terrain(t["id"], t["name"], t["kind"], t["defense"], t["country"])
    for weather in move_costs.values():
        assert tuple(weather) == kinds
        for costs in weather.values():
            assert set(costs) <= set(MOVE_TYPES)

    countries = {c["id"]: Country(c["id"], c["code"], c["name"]) for c in _read("countries.json")["countries"]}

    cos = {}
    for c in _read("cos.json")["cos"]:
        d2d_raw = c["d2d"]
        states = [_co_state(units_t, primary_t, secondary_t, kinds, d2d_raw)]
        for name in ("cop", "scop"):
            states.append(_co_state(units_t, primary_t, secondary_t, kinds, {**d2d_raw, **c[name]}))
        cop, scop = c["stars"]
        cos[c["id"]] = Co(c["id"], c["name"], cop, scop, states[0], states[1], states[2], c["notes"])

    return Rules(
        units=units_t,
        terrain=terrain,
        terrain_kinds=kinds,
        move_costs=move_costs,
        primary=primary_t,
        secondary=secondary_t,
        cos=cos,
        countries=countries,
    )


_CO_KEYS = frozenset(
    [
        "mods",
        "terrain_atk",
        "atk_per_star",
        "star_mult",
        "luck",
        "tower_atk",
        "tower_def",
        "def_vs_indirect",
        "counter",
        "capture",
        "cost",
        "move_cost_one",
        "income_bonus",
        "air_fuel_per_day",
        "repair_bonus",
    ]
)


def _co_state(units, primary, secondary, kinds, raw: dict[str, Any]) -> CoState:
    unknown = set(raw) - _CO_KEYS
    assert not unknown, unknown
    n = len(units)
    per_unit = {k: [0] * n for k in ("atk", "def", "move", "range")}
    for m in raw.get("mods", []):
        chosen = _select(units, primary, secondary, m["units"])
        if "except" in m:
            chosen -= _select(units, primary, secondary, m["except"])
        assert set(m) <= {"units", "except", "atk", "def", "move", "range"}, m
        for k in per_unit:
            for i in chosen:
                per_unit[k][i] += m.get(k, 0)
    terrain_atk: dict[str, int] = {}
    for key, value in raw.get("terrain_atk", {}).items():
        for kind in TERRAIN_GROUPS.get(key, (key,)):
            assert kind in kinds, kind
            terrain_atk[kind] = value
    good, bad = raw.get("luck", [9, 0])
    return CoState(
        atk=tuple(per_unit["atk"]),
        defense=tuple(per_unit["def"]),
        move=tuple(per_unit["move"]),
        range=tuple(per_unit["range"]),
        terrain_atk=terrain_atk,
        atk_per_star=raw.get("atk_per_star", 0),
        star_mult=raw.get("star_mult", 1),
        luck_good=good,
        luck_bad=bad,
        tower_atk=raw.get("tower_atk", TOWER_ATK),
        tower_def=raw.get("tower_def", 0),
        def_vs_indirect=raw.get("def_vs_indirect", 0),
        counter=raw.get("counter", 100),
        capture=raw.get("capture", 100),
        cost=raw.get("cost", 100),
        move_cost_one=raw.get("move_cost_one", False),
        income_bonus=raw.get("income_bonus", 0),
        air_fuel_per_day=raw.get("air_fuel_per_day", 0),
        repair_bonus=raw.get("repair_bonus", 0),
    )
