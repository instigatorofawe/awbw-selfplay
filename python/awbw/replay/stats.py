"""Per-replay summaries and the dataset filter for the v1 scope."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .parse import ReplayFormatError, load_replay


@dataclass(slots=True)
class ReplaySummary:
    path: str
    error: str | None = None
    game_id: int = 0
    players: int = 0
    fog: bool = False
    weather_type: str = ""
    use_powers: bool = True
    funds_per_property: int = 0
    starting_funds: int = 0
    capture_win: int = 0
    type: str = ""
    team: bool = False
    tag: bool = False
    map_id: int = 0
    days: int = 0
    turns: int = 0
    turns_without_actions: int = 0
    actions: Counter = field(default_factory=Counter)
    end_action: str | None = None
    co_ids: list[int] = field(default_factory=list)


def summarize(path: str | Path) -> ReplaySummary:
    s = ReplaySummary(path=str(path))
    try:
        r = load_replay(path)
    except (ReplayFormatError, OSError) as e:
        s.error = f"{type(e).__name__}: {e}"
        return s
    g = r.game
    first = r.turns[0]
    s.game_id = g.id
    s.players = len(first.players)
    s.fog, s.weather_type, s.use_powers = g.fog, g.weather_type, g.use_powers
    s.funds_per_property, s.starting_funds, s.capture_win = g.funds_per_property, g.starting_funds, g.capture_win
    s.type, s.team, s.map_id = g.type, g.team, g.map_id
    s.tag = any(p.tags_co_id for p in first.players)
    s.days, s.turns = r.days, len(r.turns)
    s.turns_without_actions = sum(t.actions is None for t in r.turns)
    s.actions = Counter(a.get("action", "?") for t in r.turns for a in t.actions or [])
    over = r.game_over()
    s.end_action = over.end_action if over else None
    s.co_ids = [p.co_id for p in sorted(first.players, key=lambda p: p.order)]
    return s


STANDARD_FUNDS_PER_PROPERTY = 1000


def v1_rejection(s: ReplaySummary) -> str | None:
    """Why a replay is outside the v1 scope, or None if it is in scope.

    v1 is standard 2-player games: no tag, no fog, clear weather, powers on,
    1000 funds per property (no high income) and no starting funds. Capture
    limits vary per map in standard play, so any limit is accepted.
    """
    if s.error:
        return "parse error"
    if s.players != 2:
        return "not 2 players"
    if s.tag:
        return "tag"
    if s.team:
        return "team game"
    if s.fog:
        return "fog"
    if s.weather_type != "Clear":
        return f"weather {s.weather_type}"
    if not s.use_powers:
        return "powers off"
    if s.funds_per_property != STANDARD_FUNDS_PER_PROPERTY:
        return f"funds/property {s.funds_per_property}"
    if s.starting_funds:
        return "starting funds"
    if s.end_action is None:
        return "no game-over in log"
    if s.turns_without_actions:
        return "turns missing actions"
    return None
