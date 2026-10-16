import dataclasses
import json

import numpy as np
import pytest

from awbw import _core
from awbw.cli import main
from awbw.data.loader import in_validation
from awbw.data.shards import Dataset
from awbw.games import read as read_games
from awbw.games import replay
from awbw.ladder import elo
from awbw.nn.model import PolicyValueNet
from awbw.nn.policy import load_policy, policy_ids, save_policy
from awbw.nn.train_selfplay import (
    MAX_ENTRIES,
    Games,
    SelfPlayConfig,
    Window,
    cycle,
    game_samples,
    human_starts,
    train,
)
from awbw.search import NetEvaluator, selection_of
from test_policy import TINY, W, cache, random_game, start, write_dataset  # noqa: F401 (a fixture)


def _net() -> PolicyValueNet:
    import torch

    torch.manual_seed(0)
    return PolicyValueNet(TINY).eval()


def _chunk(n: int, size: int, first: int) -> dict[str, np.ndarray]:
    """Window rows whose picks[:, 0] number them from ``first``."""
    picks = np.zeros((n, 3), np.uint16)
    picks[:, 0] = np.arange(first, first + n)
    return {
        "states": np.zeros((n, _core.STATE_BYTES), np.uint8),
        "picks": picks,
        "n_picks": np.zeros(n, np.uint8),
        "size": np.full(n, size, np.uint8),
        "z": np.ones(n, np.int8),
        "index": np.zeros((n, MAX_ENTRIES), np.uint16),
        "prob": np.zeros((n, MAX_ENTRIES), np.float16),
    }


def test_window_keeps_the_latest_samples_by_bucket():
    w = Window(5)
    w.add(_chunk(3, 16, 0))
    w.add(_chunk(4, 24, 3))  # drops the two oldest
    assert w.count == 5 and w.added == 7
    held = {b: sorted(w.picks[r, 0].tolist()) for b, r in w.rows().items()}
    assert held == {16: [2], 24: [3, 4, 5, 6]}
    rng = np.random.default_rng(0)
    seen = {16: 0, 24: 0}
    for _ in range(200):
        size, rows = w.sample(8, rng)
        assert (w.size[rows] == size).all() and len(rows) == 8
        seen[size] += 1
    assert 10 < seen[16] < 80  # chosen by the bucket's share, 1 in 5
    w.add(_chunk(7, 16, 7))  # more than it holds: the latest
    assert sorted(w.picks[:, 0].tolist()) == [9, 10, 11, 12, 13]


def test_game_samples_label_results_and_keep_the_likeliest_steps():
    n_steps = [3, MAX_ENTRIES + 8]
    p0 = np.array([0.5, 0.3, 0.2], np.float32)
    p1 = np.linspace(1, 2, n_steps[1]).astype(np.float32)
    p1 /= p1.sum()
    part = {
        "states": np.zeros((2, _core.STATE_BYTES), np.uint8),
        "picks": np.zeros((2, 3), np.uint16),
        "n_picks": np.zeros(2, np.uint8),
        "mover": np.array([0, 1], np.uint8),
        "offsets": np.array([0, 3, 3 + n_steps[1]]),
        "index": np.arange(3 + n_steps[1]),
        "policy": np.concatenate([p0, p1]),
    }
    s = game_samples([part, part], winner=1, size=24)
    assert s["z"].tolist() == [-1, 1, -1, 1] and (s["size"] == 24).all()
    np.testing.assert_allclose(s["prob"][0, :3], p0, atol=1e-3)
    assert s["index"][0, :3].tolist() == [0, 1, 2] and (s["prob"][0, 3:] == 0).all()
    # The likeliest MAX_ENTRIES of a longer policy, renormalized: here its last ones.
    assert sorted(s["index"][1].tolist()) == list(range(3 + 8, 3 + n_steps[1]))
    assert float(s["prob"][1].astype(np.float32).sum()) == pytest.approx(1, abs=1e-2)
    assert game_samples([part], winner=_core.DRAW, size=24)["z"].tolist() == [0, 0]


def test_games_give_sound_samples(cache, tmp_path):  # noqa: F811
    s = start(cache)
    cfg = _core.SearchConfig(sims=4, considered=4, fast_sims=2, full_fraction=0.5, max_days=2, seed=1)
    starts = cycle(np.frombuffer(s.to_bytes(), np.uint8)[None], np.random.default_rng(0))
    games = Games(NetEvaluator(_net()), starts, cfg, 3)
    done = []
    while len(done) < 4:
        done += games.round()
        assert games.active <= 3  # topped up at the next round, so more than 3 games end
    for g in done:
        smp, rec = g.samples, g.record
        n = len(smp["z"])
        assert n > 0 and g.size == 16 and g.first == s.active
        assert rec.winner in (0, 1, _core.DRAW) and rec.capped
        for i in range(n):
            st = _core.GameState.from_bytes(smp["states"][i].tobytes())
            sel = selection_of(st, smp["picks"][i, : smp["n_picks"][i]])
            want = 0 if rec.winner == _core.DRAW else 1 if st.active == rec.winner else -1
            assert smp["z"][i] == want
            k = int((smp["prob"][i] > 0).sum())
            ids = policy_ids(smp["index"][i, :k], W, g.size)
            assert set(ids.tolist()) <= set(_core.legal_ids(st, sel).tolist())
            assert float(smp["prob"][i].astype(np.float32).sum()) == pytest.approx(1, abs=1e-2)
    assert games.evals > 0 and len(games.pending) <= 3


def test_games_resign_when_lost(cache, tmp_path):  # noqa: F811
    s = start(cache)
    cfg = _core.SearchConfig(sims=4, considered=4, max_days=6, resign_turns=2, resign_off=0, seed=1)
    starts = cycle(np.frombuffer(s.to_bytes(), np.uint8)[None], np.random.default_rng(0))

    def lost(planes, globals_, legal, movers):  # player 0 sees every position as lost
        return np.zeros(legal.shape, np.float32), np.where(movers == 0, -1.0, 1.0).astype(np.float32)

    games = Games(lost, starts, cfg, 2)  # type: ignore[arg-type]
    done = []
    while len(done) < 2:
        done += games.round()
    for g in done:
        rec = g.record
        assert rec.resigned and not rec.capped and rec.winner == 1 and rec.would_resign == -1
        assert rec.days == 2 and rec.final_state.winner == _core.NO_OWNER
        movers = [_core.GameState.from_bytes(st.tobytes()).active for st in g.samples["states"]]
        assert g.samples["z"].tolist() == [1 if m == 1 else -1 for m in movers]


def test_games_score_fast_searches_with_the_fast_network(cache, tmp_path):  # noqa: F811
    s = start(cache)
    cfg = _core.SearchConfig(sims=4, considered=4, fast_sims=2, full_fraction=0.5, skip_prior=0.9, max_days=2, seed=1)

    def net(confident: bool):  # sure of each leaf's first legal step, or uniform
        def evaluate(planes, globals_, legal, movers):
            logits = np.zeros(legal.shape, np.float32)
            logits[:, 0] = 10.0 if confident else 0.0
            return logits, np.zeros(len(planes), np.float32)

        return evaluate

    for fast_is_sure in (True, False):
        starts = cycle(np.frombuffer(s.to_bytes(), np.uint8)[None], np.random.default_rng(0))
        games = Games(net(not fast_is_sure), starts, cfg, 3, net(fast_is_sure))  # type: ignore[arg-type]
        done = []
        while len(done) < 3:
            done += games.round()
        assert 0 < games.fast_evals < games.evals
        for g in done:
            rec = g.record
            # Only the fast searches see the fast network's prior, and only they skip.
            fast = rec.decisions - len(g.samples["z"])
            assert fast > 0
            assert rec.skipped == (fast if fast_is_sure else 0)


def test_selfplay_trains_and_resumes(cache, tmp_path):  # noqa: F811
    ids = np.arange(1, 200)
    train_ids = ids[~in_validation(ids)][:2]
    games = [random_game(cache, 40, seed=int(i), game_id=int(i)) for i in train_ids]
    write_dataset(tmp_path / "ds", games)
    assert len(human_starts(Dataset(tmp_path / "ds"))) == 2
    save_policy(tmp_path / "init.pt", _net())
    cfg = SelfPlayConfig(
        sims=4,
        fast_sims=2,
        considered=4,
        full_fraction=0.5,
        max_days=2,
        parallel=4,
        window=500,
        min_window=20,
        batch_size=8,
        warmup_steps=1,
        human_fraction=0.5,
        checkpoint_every=3,
        log_every=0.0,
        max_steps=6,
    )
    out = tmp_path / "run"
    lines: list[str] = []
    train(tmp_path / "init.pt", out, cfg, data=tmp_path / "ds", log=lines.append)
    names = {p.name for p in out.iterdir()}
    assert {"last.pt", "optimizer.pt", "metrics.jsonl", "net-000003.pt", "net-000006.pt"} <= names
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["step"] == 6 and metrics[-1]["games"] > 0 and metrics[-1]["window"] >= 20
    assert 0 <= metrics[-1]["resign_wrong"] <= metrics[-1]["resign_checked"]
    assert sum(m["train"]["steps"] + m["human"]["steps"] for m in metrics) == 6
    assert sum(m["human"]["steps"] for m in metrics) > 0
    # Every finished game is recorded, those without a full search too, and replays.
    logs = list(read_games(out / "games.jsonl"))
    assert len(logs) >= metrics[-1]["games"] and all(replay(log).matches for log in logs)
    assert all(log.source == "self-play" and "step" in log.info for log in logs)
    for p in [out / "last.pt", *out.glob("net-*.pt")]:  # they load: no numpy scalars in their info
        load_policy(p)
    more = dataclasses.replace(cfg, max_steps=8)
    train(tmp_path / "init.pt", out, more, data=tmp_path / "ds", log=lines.append, resume=True)
    assert any(line.startswith("resumed") for line in lines)
    assert json.loads((out / "metrics.jsonl").read_text().splitlines()[-1])["step"] == 8


def test_selfplay_trains_a_fast_network_alongside(cache, tmp_path):  # noqa: F811
    import torch

    ids = np.arange(1, 200)
    train_ids = ids[~in_validation(ids)][:2]
    write_dataset(tmp_path / "ds", [random_game(cache, 40, seed=int(i), game_id=int(i)) for i in train_ids])
    save_policy(tmp_path / "init.pt", _net())
    torch.manual_seed(1)
    fast = PolicyValueNet(dataclasses.replace(TINY, channels=4, blocks=1)).eval()
    save_policy(tmp_path / "fast.pt", fast)
    cfg = SelfPlayConfig(
        sims=4,
        fast_sims=2,
        considered=4,
        full_fraction=0.5,
        skip_prior=0.5,
        max_days=2,
        parallel=4,
        window=500,
        min_window=20,
        batch_size=8,
        warmup_steps=1,
        human_fraction=0.5,
        checkpoint_every=3,
        log_every=0.0,
        max_steps=6,
    )
    out = tmp_path / "run"
    lines: list[str] = []
    train(tmp_path / "init.pt", out, cfg, data=tmp_path / "ds", fast=tmp_path / "fast.pt", log=lines.append)
    assert any("fast searches scored by" in line and "trained alongside" in line for line in lines)
    assert {"fast-last.pt", "fast-optimizer.pt"} <= {p.name for p in out.iterdir()}
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert sum(m["fast_evals_per_s"] for m in metrics) > 0
    skipped = [m["skipped"] for m in metrics if np.isfinite(m["skipped"])]  # NaN: no game ended
    assert skipped and all(0 <= x <= 1 for x in skipped)
    fast_steps = [m for m in metrics if m["train"]["steps"]]
    assert fast_steps and all(np.isfinite(m["fast"]["train"]["policy_ce"]) for m in fast_steps)
    trained, _ = load_policy(out / "fast-last.pt")
    before, after = fast.state_dict(), trained.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)

    # Resuming picks the fast network up where it was.
    more = dataclasses.replace(cfg, max_steps=8)
    train(
        tmp_path / "init.pt", out, more, data=tmp_path / "ds", fast=tmp_path / "fast.pt", log=lines.append, resume=True
    )
    assert json.loads((out / "metrics.jsonl").read_text().splitlines()[-1])["step"] == 8

    # Kept fixed, it isn't saved or changed.
    fixed = tmp_path / "fixed"
    train(
        tmp_path / "init.pt",
        fixed,
        dataclasses.replace(cfg, fast_train=False),
        data=tmp_path / "ds",
        fast=tmp_path / "fast.pt",
        log=lines.append,
    )
    assert any("fixed" in line for line in lines)
    assert not (fixed / "fast-last.pt").exists()
    assert "fast" not in json.loads((fixed / "metrics.jsonl").read_text().splitlines()[-1])

    # A fast network must read the planes the main one does.
    save_policy(tmp_path / "moving.pt", PolicyValueNet(dataclasses.replace(TINY, movement=True)))
    with pytest.raises(ValueError, match="movement"):
        train(tmp_path / "init.pt", tmp_path / "bad", cfg, data=tmp_path / "ds", fast=tmp_path / "moving.pt")


def test_elo_of_a_score():
    assert elo(0.5, 100)[0] == pytest.approx(0)
    up, se = elo(0.75, 100)
    assert up == pytest.approx(400 * np.log10(3)) and elo(0.25, 100)[0] == pytest.approx(-up)
    assert se == pytest.approx(400 / np.log(10) * np.sqrt(0.75 * 0.25 / 100) / (0.75 * 0.25))
    assert np.isfinite(elo(1.0, 10)[0])


def test_arena_and_ladder_commands(cache, tmp_path, capsys):  # noqa: F811
    ids = np.arange(1, 200)
    val = in_validation(ids)
    games = [random_game(cache, 30, seed=int(i), game_id=int(i)) for i in [ids[val][0], ids[~val][0]]]
    write_dataset(tmp_path / "ds", games)
    run = tmp_path / "run"
    run.mkdir()
    save_policy(tmp_path / "base.pt", _net())
    for step in (2, 4):
        save_policy(run / f"net-{step:06d}.pt", _net())
    common = ["--data", str(tmp_path / "ds"), "--games", "1", "--max-days", "2", "--device", "cpu"]
    assert (
        main(["arena", str(run / "net-000002.pt"), "--opponent", str(tmp_path / "base.pt"), "--sims", "2", *common])
        == 0
    )
    assert " of 2 games" in capsys.readouterr().out
    skipping = ["--sims", "2", "--opponent-sims", "2", "--skip-prior", "0.5", "--opponent-skip-prior", "0.5"]
    assert main(["arena", str(run / "net-000002.pt"), "--opponent", str(tmp_path / "base.pt"), *skipping, *common]) == 0
    out = capsys.readouterr().out
    assert "unsearched from 0.5" in out and "network evaluations per decision" in out
    ladder = ["ladder", str(run), "--baseline", str(tmp_path / "base.pt"), *common]
    assert main(ladder) == 0
    entries = [json.loads(line) for line in (run / "ladder.jsonl").read_text().splitlines()]
    assert [e["step"] for e in entries] == [2, 4] and all(e["won"] + e["lost"] + e["drawn"] == 2 for e in entries)
    assert main(ladder) == 0  # rated already
    assert len((run / "ladder.jsonl").read_text().splitlines()) == 2
    assert main([*ladder, "--every", "4", "--sims", "2"]) == 0  # other settings: rated again
    assert len((run / "ladder.jsonl").read_text().splitlines()) == 3
