import json

import numpy as np
import pytest

from awbw import _core, games
from awbw.agents import RandomAgent, play
from awbw.cli import main
from awbw.search import NetEvaluator, SearchAgent, SearchSettings, self_play
from awbw.viewer import game_view, write_page
from test_policy import start
from test_search import _net, md  # noqa: F401 (a fixture)


def _arena_logs(md, n: int = 2) -> list[games.GameLog]:  # noqa: F811
    s = start(md)
    agent = SearchAgent(NetEvaluator(_net()), SearchSettings(sims=4, considered=4, max_days=3), seed=1)
    results = play([s] * n, (agent, RandomAgent(1)), max_days=3)
    return [games.from_result(r, ("search", "random"), seed=1) for r in results]


def test_an_arena_game_replays_from_its_record(md, tmp_path):  # noqa: F811
    s = start(md)
    agent = SearchAgent(NetEvaluator(_net()), SearchSettings(sims=4, considered=4, max_days=3), seed=1)
    results = play([s, s], (agent, RandomAgent(1)), max_days=3)
    path = tmp_path / "games.jsonl"
    assert games.append(path, [games.from_result(r, ("search", "random"), seed=1) for r in results]) == 2
    back = list(games.read(path))
    assert len(back) == 2
    for r, log in zip(results, back, strict=True):
        assert log.players == ("search", "random") and log.source == "arena" and log.info == {"seed": 1}
        assert (log.winner, log.days, log.decided, log.actions) == (r.winner, r.days, r.decided, r.actions)
        assert log.start.to_bytes() == s.to_bytes()
        rp = games.replay(log)
        assert rp.matches and len(rp.moves) == r.actions
        assert rp.moves[-1].state.to_bytes() == r.final.to_bytes()
        # Player 0 searched its every decision with a choice; the random mover keeps no notes.
        assert log.searched is not None and len(log.searched) > 0
        assert (np.diff(log.searched["step"]) > 0).all() and log.searched["step"][-1] < len(log.steps)
        assert (log.searched["sims"] == 4).all() and (np.abs(log.searched["value"]) <= 1).all()
        noted = {int(i) for i in log.searched["step"]}
        movers = {m.mover for m in rp.moves if any(m.first <= i <= m.last for i in noted)}
        assert movers == {0}


def test_a_self_play_game_replays_from_its_record(md):  # noqa: F811
    s = start(md)
    cfg = _core.SearchConfig(sims=4, considered=4, max_days=2, full_fraction=0.5, fast_sims=2, seed=7)
    samples: dict[int, int] = {}
    records = []
    for batch in self_play(NetEvaluator(_net()), iter([s] * 2), cfg, 16, parallel=2):
        for g in batch.samples["game"].tolist():
            samples[g] = samples.get(g, 0) + 1
        records += batch.records
    assert len(records) == 2
    for rec in records:
        log = games.from_json(json.loads(json.dumps(games.to_json(games.from_record(rec, step=5)))))
        assert log.source == "self-play" and log.info == {"step": 5} and log.decided == "day limit"
        assert log.searched is not None
        assert len(log.searched) == rec.decisions and int(log.searched["full"].sum()) == samples.get(rec.game, 0)
        assert set(log.searched["sims"].tolist()) <= {2, 4}
        rp = games.replay(log)
        assert rp.matches and len(rp.moves) == rec.actions
        assert rp.moves[-1].state.to_bytes() == rec.final_state.to_bytes()
    # Games from one start draw their own luck: their first states differ in it.
    assert records[0].start_state.to_bytes() != records[1].start_state.to_bytes()


def test_a_record_carries_its_map(md):  # noqa: F811
    d = games.to_json(_arena_logs(md, 1)[0])
    assert d["format"] == games.FORMAT and (d["height"], d["width"]) == (md.height, md.width)
    d["map"] = 4242  # a map the engine hasn't seen: reading the record registers it
    assert _core.find_map(4242) is None
    games.from_json(d)
    np.testing.assert_array_equal(_core.find_map(4242).terrain(), md.terrain())
    with pytest.raises(ValueError, match="format"):
        games.from_json(d | {"format": 99})


def test_a_replay_under_other_rules_is_caught(md):  # noqa: F811
    log = _arena_logs(md, 1)[0]
    log.final = "0" * 16
    assert not games.replay(log).matches
    log.steps = log.steps.copy()
    log.steps[0] = 65535  # a step no position allows
    with pytest.raises(ValueError, match="refused"):
        games.replay(log)


def test_the_viewer_page(md, tmp_path, capsys):  # noqa: F811
    logs = _arena_logs(md, 2)
    view = game_view(logs[0], title="t")
    frames = view["frames"]
    assert len(frames) == logs[0].actions + 1 and frames[0]["t"] == "start"
    assert all(f["t"] and f["d"] >= 1 for f in frames)
    assert any(f["t"] == "ends the turn" for f in frames) and any(" → " in f["t"] for f in frames)
    valued = [f for f in frames if "v" in f]
    assert valued and all(f["m"] == 0 and abs(f["v"]) <= 1 for f in valued)
    assert view["matches"] and view["map"]["width"] == md.width and len(view["players"]) == 2
    page = write_page(view, tmp_path / "game.html").read_text()
    assert "/*GAME_DATA*/" not in page
    data = page.split("const G = ", 1)[1].split(";\nconst $", 1)[0]
    assert json.loads(data)["frames"] == json.loads(json.dumps(frames))

    path = tmp_path / "games.jsonl"
    games.append(path, logs)
    assert main(["view-game", str(path), "--list"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[1].startswith("    1  map")
    assert main(["view-game", str(path), "--game", "1"]) == 0
    assert (tmp_path / "games-1.html").exists()
    assert main(["view-game", str(path), "--game", "2"]) == 1
