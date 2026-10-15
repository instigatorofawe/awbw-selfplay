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

HP_BINS: int = 10

STATE_BYTES: int = 6560

NO_UNIT: int = 255

NO_OWNER: int = 255

UNIT_MOVED: int = 1

UNIT_HIDDEN: int = 2

UNIT_CARRIED: int = 4

UNIT_STUNNED: int = 8

UNIT_CAPTURING: int = 16

STAR_VALUE: int = 90000

WEATHERS: tuple = ('clear', 'rain', 'snow')

ACTIONS: dict = ...

ENCODE_VERSION: int = 4

NUM_PLANES: int = 21

NUM_GLOBALS: int = 104

PLANES: tuple = ...

GLOBAL_NAMES: tuple = ...

def encode_batch(states: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)], planes: Annotated[NDArray[numpy.uint8], dict(shape=(None, None, None, None), order='C', device='cpu')], globals: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')]) -> None:
    """
    Encode raw states [N, STATE_BYTES] for the network, each from its player to move's
    side, into planes [N, NUM_PLANES, H, W] (uint8) and globals [N, NUM_GLOBALS]
    (float32). Each board sits at the top-left of its H x W window, the rest 0. Every
    state's map must be registered and fit; states are checked before encoding.
    PLANES lists each plane's (name, number of values, categorical).
    """

SELECTION_ENCODE_VERSION: int = 1

NUM_SELECTION_PLANES: int = 18

NUM_SELECTION_GLOBALS: int = 29

NUM_NONSPATIAL_IDS: int = 28

SELECTION_PLANES: tuple = ...

SELECTION_GLOBAL_NAMES: tuple = ...

def policy_size(h: int, w: int) -> int:
    """
    Number of policy logits over an h x w window: NUM_CHANNELS * h * w spatial ones,
    then NUM_NONSPATIAL_IDS.
    """

def policy_index(state: GameState, id: int, h: int, w: int) -> int:
    """
    Where micro-step `id` is among the policy logits over an h x w window: the channel
    times h * w plus y * w + x for a spatial id, NUM_CHANNELS * h * w plus its place
    after ID_END_TURN for the others.
    """

MOVEMENT_ENCODE_VERSION: int = 1

NUM_MOVEMENT_PLANES: int = 10

MOVEMENT_PLANES: tuple = ...

def encode_policy_batch(states: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)], picks: Annotated[NDArray[numpy.uint16], dict(shape=(None, 3), order='C', device='cpu', writable=False)], n_picks: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], planes: Annotated[NDArray[numpy.uint8], dict(shape=(None, None, None, None), order='C', device='cpu')], globals: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], nonspatial: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu')], targets: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu')] | None = None, movement: bool = False) -> None:
    """
    Encode decisions for the policy network. Decision i is in state i after its first
    n_picks[i] picks (micro-steps that don't complete an action) of picks[i] [N, 3]:
    planes [N, NUM_PLANES + NUM_SELECTION_PLANES, H, W] (encode_batch's, then
    SELECTION_PLANES), globals [N, NUM_GLOBALS + NUM_SELECTION_GLOBALS] and the
    legality of the non-spatial ids [N, NUM_NONSPATIAL_IDS] (uint8). With `targets`
    [N] (int64), the step taken next, picks[i][n_picks[i]], goes there as its
    policy_index, or -1 if it (or a pick before it) isn't legal; without, a pick
    that isn't legal raises. With `movement`, MOVEMENT_PLANES follow in planes (turns
    from factories and to reach tiles; core/encode.h).
    """

PLAYER_STATS: tuple = ...

def player_stats_batch(states: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)]) -> Annotated[NDArray[numpy.int32], dict(shape=(None, None, None))]:
    """
    Per player of raw states [N, STATE_BYTES], by player index: PLAYER_STATS as int32
    [N, 2, len(PLAYER_STATS)]. The army value is cost x displayed HP / 10 over every
    unit, carried and hidden ones included.
    """

OK: int = 0

TRAPPED: int = 1

def status_name(status: int) -> str:
    """What an apply() status code means."""

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

    def factory_turns(self, prop: int, unit_type: int) -> Annotated[NDArray[numpy.uint8], dict(shape=(None, None))]:
        """
        Turns a unit of `unit_type` needs from factory `prop` (an index into props()) to
        each tile [height, width], on the starting terrain in clear weather without CO
        bonuses; 255 where it can't go.
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
    def cargo(self) -> tuple[int, int]:
        """
        Slots of the units in cargo slots 1 and 2; -1 = empty (slots don't shift).
        """

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
    def power_uses(self) -> int: ...

    @property
    def eliminated(self) -> int: ...

    @property
    def power_atk(self) -> int: ...

class GameState:
    """A game position: a fixed-size plain value."""

    def copy(self) -> GameState: ...

    def assign(self, other: GameState) -> None:
        """Make this state equal to `other`, in place."""

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
    def weather(self) -> int:
        """Index into WEATHERS."""

    @property
    def weather_owner(self) -> int:
        """Player whose power set the weather, or 255 for the game's own weather."""

    @property
    def winner(self) -> int: ...

    def seed(self, seed: int) -> None:
        """Reseed the state's RNG (combat luck)."""

    @property
    def players(self) -> list[Player]: ...

    def unit(self, slot: int) -> Unit: ...

    def hp_belief(self, slot: int) -> list[int]:
        """
        What a player who watched every turn knows of the unit's internal HP: HP_BINS
        weights (0..255, the largest 255), the k-th for internal HP 10 * (displayed - 1) + 1 + k.
        """

    def set_hp_belief(self, slot: int, weights: Sequence[int]) -> None:
        """Replace the unit's HP belief (see hp_belief)."""

    def set_hp_unknown(self, slot: int) -> None:
        """Make every internal HP the unit's display allows equally likely."""

    def set_hp(self, slot: int, hp: int) -> None:
        """
        Set the unit's internal HP, keeping what players know: within its display the HP
        belief must allow it, and a new display makes every HP behind it equally likely.
        For a unit whose owner hides HP, the hidden-HP belief must allow its display.
        """

    def set_supplies(self, slot: int, fuel: int, ammo: int) -> None:
        """Set the unit's fuel and ammo."""

    def hides_hp(self, player: int) -> bool:
        """Whether the player's units show no HP to the other player (Sonja)."""

    def hidden_belief(self, slot: int) -> list[int]:
        """
        What the opponent of a player who hides HP knows of the unit's displayed HP:
        HP_BINS weights (0..255, the largest 255), the k-th for displayed HP k + 1. Only
        kept for units of a player who hides HP (hides_hp).
        """

    def set_hidden_belief(self, slot: int, weights: Sequence[int]) -> None:
        """Replace the unit's hidden-HP belief (see hidden_belief)."""

    def set_hidden_unknown(self, slot: int) -> None:
        """Make every displayed HP equally likely to the unit's opponent."""

    def unit_slots(self) -> list[int]:
        """Slots holding a unit, in slot order."""

    def find_unit(self, id: int) -> int:
        """Slot of the unit with this id, or -1."""

    def unit_rows(self) -> list[tuple[int, int, int, int, int, int, int, int, int, int, int]]:
        """
        (id, type, owner, x, y, hp, fuel, ammo, flags, cargo1_id, cargo2_id) for every unit, in slot order; cargo ids are 0 when empty.
        """

    def unit_at(self, x: int, y: int) -> int:
        """Slot of the uncarried unit on a tile, or -1."""

    def prop_owner(self, prop: int) -> int: ...

    def prop_points(self, prop: int) -> int: ...

    def check(self) -> str:
        """Consistency problems, or '' if the state is sound."""

    def __getstate__(self) -> bytes: ...

    def __setstate__(self, arg: bytes, /) -> None: ...

def load_snapshot(map: MapData, settings: Settings, day: int, active: int, players: Annotated[NDArray[numpy.int64], dict(shape=(None, 8), order='C', device='cpu', writable=False)], buildings: Annotated[NDArray[numpy.int64], dict(shape=(None, 4), order='C', device='cpu', writable=False)], units: Annotated[NDArray[numpy.int64], dict(shape=(None, 11), order='C', device='cpu', writable=False)], weather: int = 0, weather_owner: int = -1) -> GameState:
    """
    GameState from a flattened AWBW snapshot. players: [2, 8] (co, country, funds, power, cop_cost, scop_cost, power_on, eliminated); buildings: [n, 4] (x, y, terrain_id, capture); units: [n, 11] (id, type, owner, x, y, hp, fuel, ammo, flags, cargo1_id, cargo2_id); weather: WEATHERS index; weather_owner: player index whose power set it, or -1.
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

def apply(state: GameState, kind: int, unit: int = -1, to: tuple[int, int] | None = None, target: tuple[int, int] | None = None, arg: int = 0, path: Sequence[tuple[int, int]] | None = None, att_hp: int = -1, def_hp: int = -1, missiles: Sequence[tuple[int, int]] | None = None) -> int:
    """
    Carry out one action for the active player, in place. kind: ACTIONS[name]; unit: slot; to / target: (x, y); path: [(x, y), ...] from the unit's tile to `to` (default: the cheapest path its owner can see); att_hp / def_hp force a fight's result; missiles: a power's missile centres [(x, y), ...] (default: aimed as AWBW does). Returns OK, TRAPPED or an error status (state unchanged); see status_name.
    """

CHANNELS: tuple = ...

NUM_CHANNELS: int = 17

PHASES: tuple = ('idle', 'selected', 'moved', 'factory')

ID_END_TURN: int = 27200

ID_COP: int = 27201

ID_SCOP: int = 27202

ID_BUILD: int = 27203

NUM_ACTION_IDS: int = 27228

class Selection:
    """
    The active player's action in progress, between micro-steps: phase (PHASES index), the
    picked unit's slot (-1 if none) and tile: the unit's own, its destination, or the
    factory's (a map tile index, y * width + x).
    """

    def __init__(self) -> None: ...

    @property
    def phase(self) -> int: ...

    @phase.setter
    def phase(self, arg: int, /) -> None: ...

    @property
    def unit(self) -> int: ...

    @unit.setter
    def unit(self, arg: int, /) -> None: ...

    @property
    def tile(self) -> int: ...

    @tile.setter
    def tile(self, arg: int, /) -> None: ...

    def copy(self) -> Selection: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __repr__(self) -> str: ...

def legal_actions(state: GameState) -> list[tuple[int, int, tuple[int, int] | None, tuple[int, int] | None, int]]:
    """
    Every complete action the player to move can take, Resign aside, as apply()'s
    arguments (kind, unit, to, target, arg): apply(state, *a) takes each one, with OK, or
    TRAPPED when an enemy the player can't see is in the way. [] once the game is over.
    """

def legal_ids(state: GameState, selection: Selection = ...) -> Annotated[NDArray[numpy.int64], dict(shape=(None,))]:
    """
    Legal micro-step ids from `selection`, ascending (int64). An id is
    CHANNELS index * MAX_TILES + tile, or ID_END_TURN, ID_COP, ID_SCOP, ID_BUILD + unit
    type. Every id leads on to a legal action; empty once the game is over.
    """

def state_hash(state: GameState, selection: Selection = ...) -> int:
    """
    64-bit transposition-table key (core/hash.h): equal for states that play the same
    whatever order of actions reached them, since unit slots and ids, the RNG and free
    slots don't count. With a selection, the decision in progress (an idle one gives the
    state's key). The same on every run.
    """

def hash_batch(states: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)]) -> Annotated[NDArray[numpy.uint64], dict(shape=(None,))]:
    """
    state_hash of raw states [N, STATE_BYTES] (uint64 [N]); their maps must be registered,
    and states are checked first.
    """

def legal_mask(state: GameState, selection: Selection = ...) -> Annotated[NDArray[numpy.bool_], dict(shape=(None,))]:
    """legal_ids() as a mask over all NUM_ACTION_IDS ids."""

def step(state: GameState, selection: Selection, id: int, path: Sequence[tuple[int, int]] | None = None, att_hp: int = -1, def_hp: int = -1, missiles: Sequence[tuple[int, int]] | None = None) -> int:
    """
    Take micro-step `id`, in place: a pick advances `selection` and returns OK; a step
    that completes an action applies it (path / att_hp / def_hp / missiles as in apply),
    resets `selection` and returns apply's status. An id that isn't legal returns an
    error and changes nothing.
    """

def step_action(state: GameState, selection: Selection, id: int) -> tuple[int, int, tuple[int, int] | None, tuple[int, int] | None, int] | None:
    """
    The action (as in legal_actions) that step `id` from `selection` would complete, or
    None for a pick. Legality isn't checked.
    """

def action_steps(state: GameState, kind: int, unit: int = -1, to: tuple[int, int] | None = None, target: tuple[int, int] | None = None, arg: int = 0) -> list[int]:
    """
    The micro-step ids that play an action given as apply()'s arguments (so
    action_steps(state, *a) for a in legal_actions(state)); [] for Resign.
    """

def missile_targets(state: GameState, level: int) -> list[tuple[int, int] | None]:
    """
    Where the active player's power `level` (1 COP, 2 SCOP) would aim its missiles now, as AWBW does: [(x, y), ...] (None for a missile with no target); [] for a power without missiles.
    """

def combatant(state: GameState, slot: int, at: tuple[int, int] | None = None) -> Combatant:
    """
    The unit as the damage formula sees it, standing on `at` (default: its tile).
    """

def fight_outcomes(state: GameState, slot: int, from_: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int, float]]:
    """
    (attacker_hp, defender_hp, probability) results of the unit attacking from `from_` the unit on `target`.
    """

class SearchConfig:
    """Settings of a BatchedSearch (core/search.h)."""

    def __init__(self, *, sims: int = 64, fast_sims: int = 0, full_fraction: float = 1.0, considered: int = 16, gumbel_scale: float = 1.0, visit_scale: float = 50.0, value_scale: float = 0.10000000149011612, rescale: bool = False, pick_values: bool = True, max_days: int = 0, max_actions: int = 0, resign_value: float = -0.8999999761581421, resign_turns: int = 0, resign_off: float = 0.10000000149011612, movement: bool = False, seed: int = 0) -> None: ...

    @property
    def sims(self) -> int:
        """simulations of a full search, after the root's own evaluation"""

    @sims.setter
    def sims(self, arg: int, /) -> None: ...

    @property
    def fast_sims(self) -> int:
        """
        self-play: simulations of a fast search, played but not kept as a sample
        """

    @fast_sims.setter
    def fast_sims(self, arg: int, /) -> None: ...

    @property
    def full_fraction(self) -> float:
        """self-play: share of the searches that are full"""

    @full_fraction.setter
    def full_fraction(self, arg: float, /) -> None: ...

    @property
    def considered(self) -> int:
        """steps the root samples to consider (Gumbel top-k)"""

    @considered.setter
    def considered(self, arg: int, /) -> None: ...

    @property
    def gumbel_scale(self) -> float:
        """0: no noise, the root considers the likeliest steps"""

    @gumbel_scale.setter
    def gumbel_scale(self, arg: float, /) -> None: ...

    @property
    def visit_scale(self) -> float:
        """sigma(q) = (visit_scale + most visits) * value_scale * q"""

    @visit_scale.setter
    def visit_scale(self, arg: float, /) -> None: ...

    @property
    def value_scale(self) -> float: ...

    @value_scale.setter
    def value_scale(self, arg: float, /) -> None: ...

    @property
    def rescale(self) -> bool:
        """
        map Q to 0..1 by its spread among a node's steps (mctx's default), not (Q + 1) / 2
        """

    @rescale.setter
    def rescale(self, arg: bool, /) -> None: ...

    @property
    def pick_values(self) -> bool:
        """
        decisions with a unit or factory picked take the network's value; else that of the
        decision they were picked from (the same state)
        """

    @pick_values.setter
    def pick_values(self, arg: bool, /) -> None: ...

    @property
    def max_days(self) -> int:
        """
        a game still going after this day is decided as a day limit decides it; 0: no cap
        """

    @max_days.setter
    def max_days(self, arg: int, /) -> None: ...

    @property
    def max_actions(self) -> int:
        """
        self-play: a game ends after this many actions, decided the same way; 0: no cap
        """

    @max_actions.setter
    def max_actions(self, arg: int, /) -> None: ...

    @property
    def resign_value(self) -> float:
        """
        self-play: a turn whose first searched decision has at most this value counts toward resigning
        """

    @resign_value.setter
    def resign_value(self, arg: float, /) -> None: ...

    @property
    def resign_turns(self) -> int:
        """
        self-play: a player resigns after this many such turns in a row; 0: never
        """

    @resign_turns.setter
    def resign_turns(self, arg: int, /) -> None: ...

    @property
    def resign_off(self) -> float:
        """
        self-play: share of the games that play on to the end, to check the rule
        """

    @resign_off.setter
    def resign_off(self, arg: float, /) -> None: ...

    @property
    def movement(self) -> bool:
        """the network reads the movement planes"""

    @movement.setter
    def movement(self, arg: bool, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

class SearchResult:
    """One searched decision."""

    @property
    def id(self) -> int:
        """the step chosen"""

    @property
    def mover(self) -> int: ...

    @property
    def value(self) -> float:
        """
        the root's searched value (mean over its visits), from the mover's side
        """

    @property
    def net_value(self) -> float:
        """the network's value of the root"""

    @property
    def sims(self) -> int: ...

    @property
    def ids(self) -> NDArray[numpy.int64]:
        """the legal steps, ascending"""

    @property
    def prior(self) -> NDArray[numpy.float32]: ...

    @property
    def policy(self) -> NDArray[numpy.float32]:
        """the improved policy, softmax(logits + sigma(completed Q))"""

    @property
    def visits(self) -> NDArray[numpy.uint32]: ...

    @property
    def q(self) -> NDArray[numpy.float32]:
        """each step's value from the mover's side (0 if unvisited)"""

class GameRecord:
    """How a self-play game ended."""

    @property
    def game(self) -> int: ...

    @property
    def winner(self) -> int:
        """player index, or DRAW"""

    @property
    def days(self) -> int: ...

    @property
    def actions(self) -> int: ...

    @property
    def decisions(self) -> int:
        """searched, full and fast"""

    @property
    def capped(self) -> bool:
        """stopped by max_days or max_actions and decided as a day limit decides"""

    @property
    def resigned(self) -> bool:
        """the loser resigned"""

    @property
    def would_resign(self) -> int:
        """
        a game that played on (resign_off): the player the rule would first have had resign, else -1
        """

    @property
    def would_resign_day(self) -> int:
        """and the day, else -1"""

    @property
    def final_state(self) -> GameState: ...

class BatchedSearch:
    """
    Many Gumbel MCTS searches at once, with the network called in batches (core/search.h):
        while search.collect():
            planes, globals, legal, movers = search.batch()
            search.submit(logits, values)
    Not thread-safe.
    """

    def __init__(self, config: SearchConfig, size: int) -> None:
        """
        Leaves are encoded in a size x size window, which every map searched must fit.
        """

    @property
    def config(self) -> SearchConfig: ...

    @property
    def size(self) -> int: ...

    def add_root(self, state: GameState, selection: Selection = ...) -> int:
        """
        Search the decision `selection` in `state` and keep the result; returns its slot.
        """

    def add_game(self, state: GameState) -> int:
        """
        Play a game from `state` to its end, searching each decision with a choice; returns
        its number. Samples and records come out of take_samples() and take_records().
        """

    def collect(self) -> int:
        """
        Walk each search down to a leaf for the network, finishing the searches that are
        done; returns the number of leaves, 0 once nothing is left to search.
        """

    @property
    def leaves(self) -> int: ...

    def max_legal(self) -> int: ...

    def batch(self) -> tuple:
        """
        The leaves as the policy network reads them: planes [n, P, size, size] uint8 and
        globals [n, NUM_GLOBALS + NUM_SELECTION_GLOBALS] float32 (as encode_policy_batch),
        each leaf's legal steps as policy indices [n, max_legal()] int64 padded with -1,
        and the player deciding at each [n] uint8.
        """

    def submit(self, logits: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], values: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> None:
        """
        The network's logits over each leaf's legal steps [n, >= max_legal()] float32 (in
        batch()'s order), and its value [n] float32 for the player deciding there,
        P(win) - P(loss).
        """

    @property
    def active(self) -> int:
        """searches not finished and games still going"""

    @property
    def slots(self) -> int: ...

    def done(self, slot: int) -> bool: ...

    def result(self, slot: int) -> SearchResult: ...

    def take_samples(self) -> dict:
        """
        Self-play's full searches since the last call, as arrays: game, decision, states
        [N, STATE_BYTES], picks [N, 3] and n_picks (as encode_policy_batch takes them), mover,
        chosen, sims, value, net_value; then each sample's legal steps flattened, sample i's
        from offsets[i] to offsets[i + 1]: ids, index (each id's policy_index in the
        search's size x size window), prior, policy (the improved policy, the training
        target), visits and q.
        """

    def take_records(self) -> list[GameRecord]:
        """Self-play games finished since the last call."""

    def clear(self) -> None:
        """Free every slot."""

DRAW: int = 2

def day_limit_winner(state: GameState) -> int:
    """
    The winner of a game stopped at a day limit: more properties (comtowers and labs
    included), then the larger army value; else DRAW.
    """

def cop_cost(co: int, uses: int) -> int:
    """COP cost after `uses` power uses."""

def scop_cost(co: int, uses: int) -> int:
    """SCOP cost after `uses` power uses."""
