"""Invariants over whatever real replays are in the local cache (skipped if none).

Fetch some with e.g. ``awbw index "2p global league std" --max-pages 1`` and
``awbw fetch --index cache/index/2p_global_league_std.jsonl --limit 50``.
"""

import pytest

from awbw.net import cache_dir
from awbw.replay import load_replay

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
            assert all(c in ids for c in u.cargo)
        carried = {u.id for u in turn.units if u.carried}
        assert carried == {c for u in turn.units for c in u.cargo}

    # Games cancelled early (day 1-2, no loser on mooo) log no GameOver at all.
    over = r.game_over()
    if over is not None:
        assert set(over.winners) | set(over.losers) <= set(order)
        assert not set(over.winners) & set(over.losers)
    # Multiplayer logs have gaps (timed-out turns, mid-game resignations), so only
    # 2-player games must log every turn and end it explicitly.
    if len(order) == 2:
        for turn in r.turns[:-1]:
            assert turn.actions and turn.actions[-1]["action"] in ("End", "Tag")  # Tag swaps CO, ending the turn
