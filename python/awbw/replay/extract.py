"""Training samples from replays.

v0 is one sample per turn: the state at the start of the turn, labelled with
the game's result from the side of the player to move. The state is the
turn's snapshot, with each unit's HP belief (what a player who watched every
turn knows of its internal HP; see ``core/belief.h``) carried over from the
engine stepping through the turns before, as ``check-turns`` does.

v1 adds a sample for every action a player took, in the state they took it
in, labelled with the micro-steps that play it (``core/legal.h``) besides the
result: the policy's training data. The engine steps each turn as
``check-turns`` does, so only turns that replay exactly get action samples;
the start of every turn is a sample either way, so v1's turn starts are v0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import _core, engine
from .parse import ReplayFormatError, load_replay
from .stats import summarize_replay, v1_rejection
from .step_check import Beliefs, Labelled, step_last_turn, step_turn

DRAW = 2  # GameSamples.winner when nobody won
NO_ACTION = 255  # GameSamples.kind of a sample with no action label


@dataclass(slots=True)
class ActionLabels:
    """What a v1 game's samples are: each the state before an action."""

    turn: np.ndarray  # int16 [N]: the sample's turn, 0 for the game's first
    action: np.ndarray  # int16 [N]: actions taken before it in its turn
    kind: np.ndarray  # uint8 [N]: the action taken (ACTIONS), NO_ACTION if unlabelled
    steps: np.ndarray  # uint16 [N, 3]: its micro-step ids
    n_steps: np.ndarray  # uint8 [N]: how many; 0 = unlabelled
    n_legal: np.ndarray  # uint16 [N, 3]: legal ids at each step (1 = forced)


@dataclass(slots=True)
class GameSamples:
    game_id: int
    map_id: int
    winner: int  # player index (0 moves first each day), or DRAW
    end_action: str  # what ended the game: "Resign", "Capt", ...
    days: int
    cos: tuple[int, int]  # AWBW CO ids, by player index
    start_date: str | None  # "YYYY-MM-DD HH:MM:SS", as the replay records it
    states: np.ndarray  # uint8 [turns, STATE_BYTES]: each turn's starting state, in order
    to_play: np.ndarray  # uint8 [turns]: player to move in each state
    day: np.ndarray  # int16 [turns]
    # Turns the engine couldn't step (see step_turn), and the units after them whose
    # beliefs had to restart knowing only their displayed HP.
    unstepped: int = 0
    hp_unknown: int = 0
    actions: ActionLabels | None = None  # v1 only; v0 has one sample per turn

    @property
    def outcome(self) -> np.ndarray:
        """Result for the player to move in each state: +1 won, -1 lost, 0 draw."""
        if self.winner == DRAW:
            return np.zeros(len(self.states), dtype=np.int8)
        return np.where(self.to_play == self.winner, 1, -1).astype(np.int8)


def extract_v0(path: str | Path) -> GameSamples | str:
    """A replay's v0 samples, or why the replay is left out."""
    try:
        r = load_replay(path)
    except (ReplayFormatError, OSError) as e:
        return f"parse error ({type(e).__name__})"
    if reason := v1_rejection(summarize_replay(r, path)):
        return reason
    if engine.map_edit(r):
        return "map edited since the game"
    over = r.game_over()
    assert over is not None  # v1_rejection requires one
    order = engine.turn_order(r)
    winners = [order[p] for p in over.winners if p in order]
    md = engine.map_data(r.game.map_id)
    rows, to_play, day = [], [], []
    unstepped = hp_unknown = 0
    try:
        s = engine.state_from_turn(r, r.turns[0], md)
    except ValueError:
        return "snapshot rejected"
    for k in range(len(r.turns)):
        if s.check():
            return "snapshot rejected"
        rows.append(np.frombuffer(s.to_bytes(), np.uint8))
        to_play.append(s.active)
        day.append(s.day)
        if k + 1 == len(r.turns):
            break
        try:
            nxt, beliefs, why = step_turn(r, md, k, s)
        except ValueError:
            return "snapshot rejected"
        unstepped += bool(why)
        hp_unknown += carry_beliefs(nxt, beliefs, s)
        s = nxt
    by_order = sorted(r.turns[0].players, key=lambda p: p.order)
    return GameSamples(
        game_id=r.game.id,
        map_id=r.game.map_id,
        winner=winners[0] if len(winners) == 1 else DRAW,
        end_action=over.end_action,
        days=r.days,
        cos=(by_order[0].co_id, by_order[1].co_id),
        start_date=r.game.start_date,
        states=np.stack(rows),
        to_play=np.array(to_play, dtype=np.uint8),
        day=np.array(day, dtype=np.int16),
        unstepped=unstepped,
        hp_unknown=hp_unknown,
    )


def carry_beliefs(state: _core.GameState, beliefs: dict[int, Beliefs] | None, before: _core.GameState) -> int:
    """Give each unit of ``state`` (a turn's snapshot) its HP beliefs from the end of
    the turn before, ``beliefs`` (by AWBW unit id, from step_turn).

    Without them (that turn couldn't be stepped), a unit keeps its beliefs from
    ``before`` (the previous turn's start) if its HP is unchanged; a new unit at
    full HP was built; any other unit knows only its displayed HP, or, if its
    owner hides HP, nothing to the opponent. Returns how many units that leaves
    knowing only their display.
    """
    if beliefs is None:
        beliefs = {
            before.unit(slot).id: (before.unit(slot).hp, before.hp_belief(slot), before.hidden_belief(slot))
            for slot in before.unit_slots()
        }
    unknown = 0
    for slot in state.unit_slots():
        u = state.unit(slot)
        hp, w, hidden = beliefs.get(u.id, (-1, [], []))
        if hp == u.hp:
            state.set_hp_belief(slot, w)
            if state.hides_hp(u.owner):
                state.set_hidden_belief(slot, hidden)
        elif u.id in beliefs or u.hp < 100:
            state.set_hp_unknown(slot)
            if state.hides_hp(u.owner):
                state.set_hidden_unknown(slot)
            unknown += 1
    return unknown


def extract_v1(path: str | Path) -> GameSamples | str:
    """A replay's v1 samples, or why the replay is left out.

    Each turn gives its starting state, then the state before each further
    action, in order; a sample is labelled with the action taken in it. A turn
    the engine can't step gives its start alone, unlabelled. So does a
    resignation, the one action with no micro-steps; a turn the server cut off
    has no End Turn label either.
    """
    try:
        r = load_replay(path)
    except (ReplayFormatError, OSError) as e:
        return f"parse error ({type(e).__name__})"
    if reason := v1_rejection(summarize_replay(r, path)):
        return reason
    if engine.map_edit(r):
        return "map edited since the game"
    over = r.game_over()
    assert over is not None  # v1_rejection requires one
    order = engine.turn_order(r)
    winners = [order[p] for p in over.winners if p in order]
    md = engine.map_data(r.game.map_id)
    rows: list[bytes] = []
    labels: list[Labelled | None] = []
    turn_of: list[int] = []
    action_of: list[int] = []
    to_play, day = [], []
    unstepped = hp_unknown = 0
    try:
        s = engine.state_from_turn(r, r.turns[0], md)
    except ValueError:
        return "snapshot rejected"
    for k in range(len(r.turns)):
        if s.check():
            return "snapshot rejected"
        trace: dict[int, Labelled] = {}
        last = k + 1 == len(r.turns)
        nxt, beliefs = None, None
        try:
            if last:
                why = step_last_turn(r, md, s, trace)
            else:
                nxt, beliefs, why = step_turn(r, md, k, s, trace)
        except ValueError:
            return "snapshot rejected"
        taken: list[Labelled | None] = [trace[i] for i in sorted(trace)] if not why else []
        start = s.to_bytes()
        if not taken or taken[0] is None or taken[0].state != start:
            taken.insert(0, None)  # the turn's start, unlabelled
        for j, lab in enumerate(taken):
            rows.append(start if lab is None else lab.state)
            labels.append(lab)
            turn_of.append(k)
            action_of.append(j)
            to_play.append(s.active)
            day.append(s.day)
        if last:
            break
        assert nxt is not None
        unstepped += bool(why)
        hp_unknown += carry_beliefs(nxt, beliefs, s)
        s = nxt
    n = len(rows)
    steps = np.zeros((n, 3), dtype=np.uint16)
    n_legal = np.zeros((n, 3), dtype=np.uint16)
    n_steps = np.zeros(n, dtype=np.uint8)
    kind = np.full(n, NO_ACTION, dtype=np.uint8)
    for i, lab in enumerate(labels):
        if lab is not None and lab.steps:
            m = len(lab.steps)
            steps[i, :m] = lab.steps
            n_legal[i, :m] = lab.n_legal
            n_steps[i] = m
            kind[i] = lab.kind
    by_order = sorted(r.turns[0].players, key=lambda p: p.order)
    return GameSamples(
        game_id=r.game.id,
        map_id=r.game.map_id,
        winner=winners[0] if len(winners) == 1 else DRAW,
        end_action=over.end_action,
        days=r.days,
        cos=(by_order[0].co_id, by_order[1].co_id),
        start_date=r.game.start_date,
        states=np.frombuffer(b"".join(rows), np.uint8).reshape(n, -1),
        to_play=np.array(to_play, dtype=np.uint8),
        day=np.array(day, dtype=np.int16),
        unstepped=unstepped,
        hp_unknown=hp_unknown,
        actions=ActionLabels(
            turn=np.array(turn_of, dtype=np.int16),
            action=np.array(action_of, dtype=np.int16),
            kind=kind,
            steps=steps,
            n_steps=n_steps,
            n_legal=n_legal,
        ),
    )
