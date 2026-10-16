"""Agents that play micro-steps, and whole games between two of them.

``play`` runs many games at once: every step it gathers each game's pending
decision, asks the agent whose turn it is for all of its decisions in one
call (one network batch for a ``PolicyAgent``), and takes the chosen steps.
A step with a single legal id is taken without asking. A game still going
after ``max_days`` is decided as AWBW decides a day limit: more properties
(comtowers and labs included) wins; then the larger army value; then a draw.
Each result keeps the game's first state and every step taken, and what the
agents made of their decisions (``Note``), for ``games``' records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import _core, rules

DRAW = 2  # GameResult.winner when nobody won


@dataclass(slots=True)
class Decision:
    """A game's pending micro-step decision."""

    state: _core.GameState
    picks: list[int]  # picks made so far in this action (selection.phase > 0)
    legal: np.ndarray  # legal ids, ascending


@dataclass(slots=True)
class Note:
    """What an agent made of a decision: the value for the player deciding
    (searched, if it searched) and the network's, and the simulations run."""

    value: float
    net_value: float
    sims: int = 0


class Agent(Protocol):
    """An agent may also keep ``notes``, a ``Note`` per decision of its last
    ``choose`` call, which ``play`` records."""

    def choose(self, decisions: list[Decision]) -> list[int]:
        """A legal id for each decision."""
        ...


_DELETE = _core.CHANNELS.index("Delete")


class RandomAgent:
    """Uniform among the legal steps, scrapping a unit only if nothing else is legal
    (a uniform mover scraps its own units often enough to lose by rout)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def choose(self, decisions: list[Decision]) -> list[int]:
        out = []
        for d in decisions:
            keep = d.legal[(d.legal >= _core.ID_END_TURN) | (d.legal // _core.MAX_TILES != _DELETE)]
            out.append(int(self.rng.choice(keep if len(keep) else d.legal)))
        return out


class PolicyAgent:
    """Plays a ``nn.policy.Policy``'s prior: its likeliest step, or with
    ``temperature`` > 0 a sample from prior ** (1 / temperature).

    ``ratings``: the (mover's, opponent's) ratings the network is told, e.g. a
    strong player's for both; None leaves them unknown.
    """

    def __init__(self, policy, temperature: float = 0.0, ratings: tuple[int, int] | None = None, seed: int = 0):
        self.policy = policy
        self.temperature = temperature
        self.ratings = ratings
        self.rng = np.random.default_rng(seed)
        self.notes: list[Note] = []

    def choose(self, decisions: list[Decision]) -> list[int]:
        if not decisions:
            return []
        states = np.stack([np.frombuffer(d.state.to_bytes(), np.uint8) for d in decisions])
        picks = np.zeros((len(decisions), 3), np.uint16)
        for i, d in enumerate(decisions):
            picks[i, : len(d.picks)] = d.picks
        n_picks = np.array([len(d.picks) for d in decisions], np.uint8)
        out = []
        scored = self.policy.score(states, picks, n_picks, ratings=self.ratings)
        self.notes = [Note(sc.value, sc.value) for sc in scored]
        for sc in scored:
            if self.temperature <= 0:
                out.append(int(sc.ids[np.argmax(sc.prior)]))
            else:
                w = np.power(sc.prior.astype(np.float64), 1.0 / self.temperature)
                out.append(int(self.rng.choice(sc.ids, p=w / w.sum())))
        return out


@dataclass(slots=True)
class GameResult:
    winner: int  # player index (0 moves first), or DRAW
    days: int
    decided: str  # "game over" or "day limit"
    actions: int
    final: _core.GameState = field(repr=False)
    start: _core.GameState = field(repr=False)
    steps: list[int] = field(repr=False)  # every micro-step, forced ones included
    notes: list[tuple[int, Note]] = field(repr=False)  # (index in steps, note) of the decisions noted


def property_counts(md: _core.MapData, s: _core.GameState) -> list[int]:
    """Properties each player owns, comtowers and labs included (pipe seams aren't owned)."""
    counts = [0, 0]
    for p in range(md.n_props):
        owner = s.prop_owner(p)
        if owner in (0, 1):
            counts[owner] += 1
    return counts


def army_values(s: _core.GameState) -> list[float]:
    """Each player's units' cost x displayed HP / 10, carried units included."""
    cost = [u.cost for u in rules.load().units]
    value = [0.0, 0.0]
    for slot in s.unit_slots():
        u = s.unit(slot)
        value[u.owner] += cost[u.type] * ((u.hp + 9) // 10) / 10
    return value


def adjudicate(md: _core.MapData, s: _core.GameState) -> int:
    """The day-limit winner: more properties, then the larger army; else DRAW."""
    props, army = property_counts(md, s), army_values(s)
    if props[0] != props[1]:
        return 0 if props[0] > props[1] else 1
    if army[0] != army[1]:
        return 0 if army[0] > army[1] else 1
    return DRAW


def play(
    starts: list[_core.GameState],
    agents: tuple[Agent, Agent],
    max_days: int = 30,
    max_actions: int = 20_000,
) -> list[GameResult]:
    """Play a game from each start, ``agents[i]`` moving for player i."""
    states = [s.copy() for s in starts]
    sels = [_core.Selection() for _ in states]
    picks: list[list[int]] = [[] for _ in states]
    steps: list[list[int]] = [[] for _ in states]
    notes: list[list[tuple[int, Note]]] = [[] for _ in states]
    actions = [0] * len(states)
    results: list[GameResult | None] = [None] * len(states)
    maps = {s.map_id: _core.find_map(s.map_id) for s in states}
    live = set(range(len(states)))
    while live:
        pending: tuple[list[int], list[int]] = ([], [])
        asked: tuple[list[Decision], list[Decision]] = ([], [])
        for g in sorted(live):
            s, sel = states[g], sels[g]
            while True:  # take forced steps until a real decision or the end
                done = s.winner != _core.NO_OWNER or s.day > max_days or actions[g] >= max_actions
                if done:
                    over = s.winner != _core.NO_OWNER
                    md = maps[s.map_id]
                    assert md is not None
                    results[g] = GameResult(
                        winner=int(s.winner) if over else adjudicate(md, s),
                        days=min(int(s.day), max_days),
                        decided="game over" if over else "day limit",
                        actions=actions[g],
                        final=s,
                        start=starts[g].copy(),
                        steps=steps[g],
                        notes=notes[g],
                    )
                    live.discard(g)
                    break
                legal = _core.legal_ids(s, sel)
                if len(legal) > 1:
                    pending[s.active].append(g)
                    asked[s.active].append(Decision(s, list(picks[g]), legal))
                    break
                _take(s, sel, picks[g], int(legal[0]), actions, g, steps[g])
        for player in (0, 1):
            agent = agents[player]
            chosen = agent.choose(asked[player])
            noted = getattr(agent, "notes", None)
            if noted is not None and len(noted) == len(chosen):
                for g, note in zip(pending[player], noted, strict=True):
                    notes[g].append((len(steps[g]), note))
            for g, step_id in zip(pending[player], chosen, strict=True):
                _take(states[g], sels[g], picks[g], step_id, actions, g, steps[g])
    return [r for r in results if r is not None]


def _take(
    s: _core.GameState,
    sel: _core.Selection,
    picks: list[int],
    step_id: int,
    actions: list[int],
    g: int,
    steps: list[int],
) -> None:
    steps.append(step_id)
    completes = _core.step_action(s, sel, step_id) is not None
    st = _core.step(s, sel, step_id)
    if st not in (_core.OK, _core.TRAPPED):
        raise RuntimeError(f"step {step_id} was refused: {_core.status_name(st)}")
    if completes:
        picks.clear()
        actions[g] += 1
    else:
        picks.append(step_id)
