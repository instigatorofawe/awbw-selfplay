"""Invariants over whatever real replays are in the local cache (skipped if none).

Fetch some with e.g. ``awbw index "2p global league std" --max-pages 1`` and
``awbw fetch --index cache/index/2p_global_league_std.jsonl --limit 50``.
"""

import pytest

from awbw.net import cache_dir
from awbw.replay import load_replay
from awbw.replay.combat_check import check_file
from awbw.replay.move_check import check_replay as check_moves
from awbw.replay.step_check import check_file as check_turns

PATHS = sorted((cache_dir() / "replays").glob("*.zip"))[:200]


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_replay_invariants(path):
    r = load_replay(path)
    order = r.player_ids
    assert len(set(order)) == len(order)

    days = [t.day for t in r.turns]
    assert days == sorted(days) and days[0] == 1

    if len(order) == 2:
        assert [t.player_id for t in r.turns] == [order[i % 2] for i in range(len(r.turns))]

    for turn in r.turns:
        ids = {u.id for u in turn.units}
        assert len(ids) == len(turn.units)
        for u in turn.units:
            assert 1 <= u.hp <= 100
            assert all(c in ids for c in u.cargo if c)
        carried = {u.id for u in turn.units if u.carried}
        assert carried == {c for u in turn.units for c in u.cargo if c}

    # Games cancelled early (day 1-2, no loser on mooo) log no GameOver at all.
    over = r.game_over()
    if over is not None:
        assert set(over.winners) | set(over.losers) <= set(order)
        assert not set(over.winners) & set(over.losers)
    # Multiplayer logs have gaps (timed-out turns, mid-game resignations). In
    # 2-player games a timeout loss leaves the final turn unlogged, and rarely
    # (2 of 2,000 cached games) a mid-game turn has no entry either; the v1
    # filter drops those games.
    if len(order) == 2:
        assert sum(turn.actions is None for turn in r.turns[:-1]) <= 1


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_logged_attacks_match_engine(path):
    checks, _ = check_file(path)
    replay = load_replay(path)
    bad = [c for c in checks if c.ok is False and not known_oddity(replay, c.day)]
    assert not bad, bad[:3]


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_snapshots_load_and_logged_moves_are_legal(path):
    rep = check_moves(path)
    assert not rep.snapshot_errors, rep.snapshot_errors[:3]
    assert not rep.failures, rep.failures[:3]


# Turns where AWBW's result has no rule behind it that we can see.
KNOWN_ODDITIES = {
    # A Black Boat repair leaves a 9.1-9.9 HP unit as it was; 315 others top up to 10.
    "1643462 day 12",
    "1644362 day 16",
    "1645952 day 10",
    "1650385 day 12",
    "1651661 day 19",
    "1653448 day 12",
    "1654751 day 16",
    "1671326 day 9",
    "1674937 day 11",
    "1724707 day 15",
    "1636466 day 13",  # a city's capture points drop to 0 with no capture logged; it is taken next turn
    "1695038 day 12",  # too many fight-result choices for the search budget
    "1728785 day 15",  # needs 508k fight-result tries, over the search budget
}

# Lash's terrain-star attack bonus is missing in every mismatched fight of a
# Lash game (127 of 1.7M attacks), all in turns overlapping 2026-07-27 16:47 to
# 07-28 04:57 (AWBW server time); it holds in her 90,277 other attacks from
# starred terrain.
LASH = 16
LASH_GAP = ("2026-07-27 16", "2026-07-28 06")


def known_oddity(replay, day: int) -> bool:
    if f"{replay.game.id} day {day}" in KNOWN_ODDITIES:
        return True
    if LASH not in {p.co_id for p in replay.turns[0].players}:
        return False
    turns = replay.turns
    for k, t in enumerate(turns):
        if t.day == day:
            ended = turns[k + 1].started if k + 1 < len(turns) else "9999"
            if t.started < LASH_GAP[1] and ended >= LASH_GAP[0]:
                return True
    return False


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_turns_replay_exactly(path):
    rep = check_turns(path)
    replay = load_replay(path)
    bad = [e for kind in rep.diverged for e in rep.examples[kind] if not known_oddity(replay, int(e.split()[2]))]
    assert not bad, bad[:3]
