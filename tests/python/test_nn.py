import dataclasses
import json
import math

import numpy as np
import pytest
import torch

from _factory import GAME_ID, MAP_ID
from awbw import _core
from awbw.cli import main
from awbw.data.loader import encode, in_validation, state_map_ids
from awbw.data.shards import ShardWriter
from awbw.nn.critic import Critic, load_critic, save_critic
from awbw.nn.model import LOSS, RATING_SCALE, WIN, CriticConfig, CriticNet, rating_features, value_of
from awbw.nn.train_critic import TrainConfig, dihedral, metrics, train
from awbw.replay.extract import extract_v0

TINY = CriticConfig(channels=8, blocks=2, pool_every=1, pool_channels=4, head_channels=4, head_hidden=8)


@pytest.fixture
def game(replay_cache):
    g = extract_v0(replay_cache)
    assert not isinstance(g, str), g
    return g


def _random_net() -> CriticNet:
    """A tiny net with every weight random, so no block is the identity."""
    torch.manual_seed(0)
    net = CriticNet(TINY)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.3)
    return net.eval()


def test_value_ignores_padding_placement_and_batch(game):
    net = _random_net()
    planes16, globals_ = encode(game.states, 16)
    planes24 = np.zeros((2, _core.NUM_PLANES, 24, 24), np.uint8)
    planes24[:, :, 7:12, 11:16] = planes16[:, :, :5, :5]  # the 5x5 board, moved inside a larger bucket
    g = torch.from_numpy(globals_)
    with torch.no_grad():
        a = net(torch.from_numpy(planes16), g)
        b = net(torch.from_numpy(planes24), g)
        alone = net(torch.from_numpy(planes16[1:]), g[1:])
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(a[1:], alone, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(a[0], a[1])  # the two sides' views differ
    v = value_of(a)
    assert ((v >= -1) & (v <= 1)).all()


def test_rating_features_need_both_ratings():
    f = rating_features(torch.tensor([[1100, 900], [900, 1100], [1000, -1], [-1, -1]]), 4, torch.device("cpu"))
    assert f[0].tolist() == pytest.approx([1, 100 / RATING_SCALE, 200 / RATING_SCALE])
    assert f[1].tolist() == pytest.approx([1, 100 / RATING_SCALE, -200 / RATING_SCALE])
    assert f[2:].abs().sum() == 0
    assert rating_features(None, 2, torch.device("cpu")).abs().sum() == 0


def test_ratings_move_the_value_and_unknown_means_none(game):
    net = _random_net()
    planes, globals_ = (torch.from_numpy(a) for a in encode(game.states, 16))
    with torch.no_grad():
        none = net(planes, globals_)
        unknown = net(planes, globals_, torch.tensor([[1000, -1], [-1, -1]]))
        rated = net(planes, globals_, torch.tensor([[1200, 800], [800, 1200]]))
    torch.testing.assert_close(none, unknown)
    assert not torch.allclose(none, rated)
    critic = Critic(net)
    np.testing.assert_allclose(
        critic.wdl(game.states, [[1200, 800], [800, 1200]]), rated.softmax(-1).numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(critic.wdl(game.states, (-1, -1)), critic.wdl(game.states), rtol=1e-6)
    equal = critic.wdl(game.states, (1100, 1100))  # broadcast to every state
    assert equal.shape == (2, 3) and not np.allclose(equal, critic.wdl(game.states))


def test_dihedral_gives_eight_distinct_boards():
    board = torch.arange(9, dtype=torch.uint8).view(1, 1, 3, 3)
    seen = {tuple(dihedral(board, k).flatten().tolist()) for k in range(8)}
    assert len(seen) == 8
    assert torch.equal(dihedral(board, 0), board)


def test_metrics_of_sure_and_uniform_predictions():
    z = np.array([1, -1, 1, -1], np.int8)
    day = np.array([1, 2, 10, 30], np.int16)
    sure = np.zeros((4, 3), np.float32)
    sure[np.arange(4), z + 1] = 1
    m = metrics(sure, z, day)
    assert m["loss"] < 1e-6 and m["accuracy"] == 1 and m["brier"] == 0 and m["ece"] == 0
    assert [d["days"] for d in m["by_day"]] == ["1-3", "10-12", "21+"]
    m = metrics(np.full((4, 3), 1 / 3, np.float32), z, day)
    assert m["loss"] == pytest.approx(math.log(3), rel=1e-6)
    assert m["calibration"] == [(pytest.approx(1 / 3), 0.5, 4)]


def test_state_map_ids(game):
    assert state_map_ids(game.states).tolist() == [MAP_ID, MAP_ID]
    assert _core.GameState.from_bytes(game.states[0].tobytes()).map_id == MAP_ID


def test_train_checkpoints_resumes_and_evaluates(game, tmp_path):
    ids = np.arange(GAME_ID, GAME_ID + 200)
    val = in_validation(ids, 0.25)
    chosen = [*ids[val][:3], *ids[~val][:9]]  # 3 validation games, 9 training games
    writer = ShardWriter(tmp_path / "ds", games_per_shard=4)
    for game_id in chosen:
        writer.add(dataclasses.replace(game, game_id=int(game_id)), ratings=(1000, 1100))
    writer.close()
    out = tmp_path / "run"
    cfg = TrainConfig(epochs=2, batch_size=4, warmup_steps=1, val_fraction=0.25, log_every=1)
    lines: list[str] = []
    last = train(tmp_path / "ds", out, cfg, TINY, log=lines.append)
    assert last["n"] == 6 and last["step"] == 8  # 18 training samples: 4 full batches per epoch
    assert {p.name for p in out.iterdir()} >= {"best.pt", "last.pt", "critic-ep01.pt", "critic-ep02.pt"}

    last = train(tmp_path / "ds", out, dataclasses.replace(cfg, epochs=3), log=lines.append, resume=True)
    assert any(line.startswith("resumed") for line in lines)
    assert last["step"] == 12
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [4, 8, 12]
    assert all(r["no_ratings"]["n"] == r["n"] for r in rows)
    assert any(line.startswith("  val with ratings") for line in lines)
    assert any(line.startswith("  val without") for line in lines)

    critic = Critic.load(out / "best.pt", "cpu")
    p = critic.wdl(game.states)
    np.testing.assert_allclose(p.sum(1), 1, rtol=1e-5)
    np.testing.assert_allclose(critic.value(game.states), p[:, WIN] - p[:, LOSS], rtol=1e-6)
    assert torch.equal(critic.net.embed.table.weight, load_critic(out / "best.pt")[0].embed.table.weight)

    with pytest.raises(ValueError, match="no training samples"):
        train(tmp_path / "ds", tmp_path / "none", dataclasses.replace(cfg, min_rating=1050), TINY)


def test_load_refuses_another_encoder(game, tmp_path):
    path = tmp_path / "c.pt"
    save_critic(path, CriticNet(TINY))
    ck = torch.load(path, weights_only=True)
    ck["encoder"]["version"] += 1
    torch.save(ck, path)
    with pytest.raises(ValueError, match="retrain"):
        load_critic(path)


def test_load_reads_checkpoints_from_before_prenorm_and_ratings(game, tmp_path):
    path = tmp_path / "c.pt"
    old = CriticNet(dataclasses.replace(TINY, prenorm=False, ratings=False))
    save_critic(path, old)
    ck = torch.load(path, weights_only=True)
    del ck["config"]["prenorm"], ck["config"]["ratings"]
    torch.save(ck, path)
    net, _ = load_critic(path)
    assert not net.cfg.prenorm and not any("norm" in n for n, _ in net.blocks.named_parameters())
    assert not net.cfg.ratings
    critic = Critic(net)
    np.testing.assert_array_equal(critic.wdl(game.states, (1200, 800)), critic.wdl(game.states))  # ignored


def test_critic_needs_registered_maps(game):
    states = game.states.copy()
    states[:, :4] = np.frombuffer(np.uint32(987_654).tobytes(), np.uint8)
    with pytest.raises(ValueError, match="not registered"):
        Critic(_random_net()).wdl(states)


def test_cli_values_a_replay_and_evaluates(game, replay_cache, tmp_path, capsys):
    ckpt = tmp_path / "c.pt"
    save_critic(ckpt, _random_net())
    assert main(["value", str(ckpt), str(replay_cache), "--device", "cpu"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("P1 won (Resign, day 1)")
    assert len(out) == 2 + len(game.states)
    assert main(["value", str(ckpt), str(replay_cache), "--ratings", "1200", "800", "--device", "cpu"]) == 0
    assert capsys.readouterr().out.splitlines()[2:] != out[2:]

    writer = ShardWriter(tmp_path / "ds")
    writer.add(game)
    writer.close()
    args = ["eval-critic", str(ckpt), "--data", str(tmp_path / "ds"), "--split", "all", "--device", "cpu"]
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "2 all samples" in printed and "without ratings" in printed
