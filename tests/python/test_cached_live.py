"""Live stepping over finished games whose replay viewer turns and replay are both in
the local cache (skipped if none): the beliefs a spectator gets from stepping should
be the replay's.

Cache a finished game's turns with ``awbw live CKPT GAME`` (it keeps every ended
turn under ``cache/live/<game id>/``) and its replay with ``awbw fetch``.
"""

import json

import numpy as np
import pytest

from awbw import engine, live, rules
from awbw.data.loader import encode
from awbw.maps import load_map
from awbw.net import cache_dir
from awbw.replay import load_replay
from awbw.replay.extract import extract_v0

SONJA = next(c.id for c in rules.load().cos.values() if c.name == "Sonja")


def _games() -> list[int]:
    root = cache_dir() / "live"
    out = []
    for d in sorted(root.glob("*")) if root.exists() else ():
        first = d / "turn-0000.json"
        replay = cache_dir() / "replays" / f"{d.name}.zip"
        if d.name.isdigit() and replay.exists() and first.exists() and "actions" in json.loads(first.read_text()):
            out.append(int(d.name))
    return out


GAMES = _games()


@pytest.mark.skipif(not GAMES, reason="no cached live games with replays")
@pytest.mark.parametrize("game_id", GAMES)
def test_live_beliefs_match_the_replay(game_id):
    r = load_replay(cache_dir() / "replays" / f"{game_id}.zip")
    samples = extract_v0(cache_dir() / "replays" / f"{game_id}.zip")
    if isinstance(samples, str):
        pytest.skip(f"not in the datasets: {samples}")
    info = load_map(r.game.map_id)
    engine.register(info)
    order = tuple(r.player_ids)
    game = live.LiveGame(r.game, {}, order, r.days, order[0], 0, True, False, 0, 0, (), list(info.predeployed))
    paths = sorted((cache_dir() / "live" / str(game_id)).glob("turn-*.json"))
    turns = [live.live_turn(i, json.loads(p.read_text()), info) for i, p in enumerate(paths)]
    states, report = live.states(game, turns)
    assert not report.failed
    n = min(len(states), len(samples.states))
    size = next(b for b in (16, 24, 32, 40, 48, 64) if b >= max(info.width, info.height))
    got = encode(np.stack([np.frombuffer(s.to_bytes(), np.uint8) for s in states[:n]]), size)
    want = encode(samples.states[:n], size)
    same = [bool((got[0][i] == want[0][i]).all() and np.allclose(got[1][i], want[1][i])) for i in range(n)]
    if SONJA in {p.co_id for p in r.turns[0].players}:
        # A fight with a Sonja unit whose display her opponent doesn't know moves the
        # other unit's belief by her true display in the replay, by the likeliest in live.
        assert sum(same) >= 0.8 * n
    else:
        assert all(same), [i for i, ok in enumerate(same) if not ok]
