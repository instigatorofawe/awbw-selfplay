"""Training samples from replays.

v0 is one sample per turn: the state at the start of the turn (a snapshot,
so it needs no engine stepping), labelled with the game's result from the
side of the player to move.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import engine
from .parse import ReplayFormatError, load_replay
from .stats import summarize_replay, v1_rejection

DRAW = 2  # GameSamples.winner when nobody won


@dataclass(slots=True)
class GameSamples:
    game_id: int
    map_id: int
    winner: int  # player index (0 moves first each day), or DRAW
    end_action: str  # what ended the game: "Resign", "Capt", ...
    days: int
    cos: tuple[int, int]  # AWBW CO ids, by player index
    states: np.ndarray  # uint8 [turns, STATE_BYTES]: each turn's starting state, in order
    to_play: np.ndarray  # uint8 [turns]: player to move in each state
    day: np.ndarray  # int16 [turns]

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
    over = r.game_over()
    assert over is not None  # v1_rejection requires one
    order = engine.turn_order(r)
    winners = [order[p] for p in over.winners if p in order]
    md = engine.map_data(r.game.map_id)
    rows, to_play, day = [], [], []
    for turn in r.turns:
        try:
            s = engine.state_from_turn(r, turn, md)
        except ValueError:
            return "snapshot rejected"
        if s.check():
            return "snapshot rejected"
        rows.append(np.frombuffer(s.to_bytes(), np.uint8))
        to_play.append(s.active)
        day.append(s.day)
    by_order = sorted(r.turns[0].players, key=lambda p: p.order)
    return GameSamples(
        game_id=r.game.id,
        map_id=r.game.map_id,
        winner=winners[0] if len(winners) == 1 else DRAW,
        end_action=over.end_action,
        days=r.days,
        cos=(by_order[0].co_id, by_order[1].co_id),
        states=np.stack(rows),
        to_play=np.array(to_play, dtype=np.uint8),
        day=np.array(day, dtype=np.int16),
    )
