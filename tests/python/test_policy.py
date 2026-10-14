import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from awbw import _core, engine, rules
from awbw.agents import DRAW, PolicyAgent, RandomAgent, adjudicate, army_values, play, property_counts
from awbw.cli import main
from awbw.data.loader import (
    AUX_BINARY,
    AUX_REGRESSION,
    PHASE_FACTORY,
    PHASE_IDLE,
    PHASE_MOVED,
    PHASE_SELECTED,
    aux_targets,
    decision_batches,
    encode_decisions,
    in_subset,
    in_validation,
)
from awbw.data.shards import Dataset, ShardWriter
from awbw.nn.model import (
    MOVEMENT_PLANES,
    NUM_POLICY_GLOBALS,
    NUM_POLICY_PLANES,
    PolicyConfig,
    PolicyValueNet,
    board_attention,
    flat_policy,
    legal_mask,
    masked_log_softmax,
    rope_angles,
    rotate,
)
from awbw.nn.policy import Policy, load_policy, policy_ids, save_policy
from awbw.nn.train_critic import dihedral
from awbw.nn.train_policy import (
    PolicyTrainConfig,
    count_decisions,
    games_keep,
    policy_metrics,
    split_size,
    train,
    undihedral,
)
from awbw.replay.extract import NO_ACTION, ActionLabels, GameSamples

U = {u.name: u.index for u in rules.load().units}
A = _core.ACTIONS
DELETE = _core.CHANNELS.index("Delete")
MAP_ID = 78
W, H = 7, 6
TINY = PolicyConfig(
    channels=8, blocks=2, pool_every=1, pool_channels=4, head_channels=4, head_hidden=8, policy_channels=4
)

# Orange Star HQ and base top left, Blue Moon's bottom right, neutral cities between.
TERRAIN = np.ones((H, W), dtype=np.int16)
for (x, y), t in {(0, 0): 42, (1, 0): 39, (4, 0): 2, (2, 1): 34, (1, 2): 3, (4, 2): 34, (5, 3): 3, (2, 4): 34}.items():
    TERRAIN[y, x] = t
TERRAIN[5, 5], TERRAIN[5, 6] = 44, 47


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A cache holding the test map, registered with the engine."""
    monkeypatch.setenv("AWBW_CACHE", str(tmp_path))
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / f"{MAP_ID}.json").write_text(
        json.dumps({"Name": "t", "Size X": W, "Size Y": H, "Terrain Map": TERRAIN.T.tolist(), "Predeployed Units": []})
    )
    return engine.map_data(MAP_ID)


def start(md) -> _core.GameState:
    props = [(x, y, int(TERRAIN[y, x]), 20) for y in range(H) for x in range(W) if TERRAIN[y, x] >= 34]
    # co, country, funds, power, cop_cost, scop_cost, power_on, eliminated
    players = np.array([[1, 1, 8000, 0, 270000, 630000, 0, 0], [7, 2, 8000, 0, 270000, 540000, 0, 0]])
    rows = [  # id, type, owner, x, y, hp, fuel, ammo, flags, cargo1, cargo2
        (1, U["Infantry"], 0, 1, 1, 100, 99, 0, 0, 0, 0),
        (2, U["Tank"], 0, 2, 2, 100, 70, 9, 0, 0, 0),
        (3, U["Artillery"], 0, 0, 2, 100, 50, 9, 0, 0, 0),
        (4, U["Infantry"], 1, 5, 4, 100, 99, 0, 0, 0, 0),
        (5, U["Tank"], 1, 4, 3, 100, 70, 9, 0, 0, 0),
        (6, U["Mech"], 1, 6, 3, 100, 70, 3, 0, 0, 0),
    ]
    return _core.load_snapshot(
        md, _core.Settings(), 1, 0, players, np.array(props), np.array(rows, dtype=np.int64).reshape(-1, 11)
    )


def random_game(md, n_actions: int, seed: int = 0, game_id: int = 1) -> GameSamples:
    """A game of random micro-steps (no scrapping, End Turn now and then), recorded
    as extract_v1 records a replay."""
    rng = np.random.default_rng(seed)
    s = start(md)
    rows, labels, turn, action, to_play, day = [], [], [], [], [], []
    t = j = 0
    for _ in range(n_actions):
        if s.winner != _core.NO_OWNER:
            break
        sel = _core.Selection()
        before = s.to_bytes()
        taken, counts = [], []
        while True:
            legal = _core.legal_ids(s, sel)
            others = legal[(legal != _core.ID_END_TURN) & (legal // _core.MAX_TILES != DELETE)]
            end = sel.phase == 0 and (not len(others) or rng.random() < 0.1)
            pick = _core.ID_END_TURN if end else int(rng.choice(others))
            act = _core.step_action(s, sel, pick)
            taken.append(pick)
            counts.append(len(legal))
            assert _core.step(s, sel, pick) in (_core.OK, _core.TRAPPED)
            if act is not None:
                break
        rows.append(before)
        labels.append((act[0], taken, counts))
        turn.append(t)
        action.append(j)
        st = _core.GameState.from_bytes(before)
        to_play.append(st.active)
        day.append(st.day)
        t, j = (t + 1, 0) if act[0] == A["EndTurn"] else (t, j + 1)
    n = len(rows)
    steps, n_legal = np.zeros((n, 3), np.uint16), np.zeros((n, 3), np.uint16)
    for i, (_, taken, counts) in enumerate(labels):
        steps[i, : len(taken)], n_legal[i, : len(taken)] = taken, counts
    return GameSamples(
        game_id=game_id,
        map_id=MAP_ID,
        winner=0,
        end_action="Resign",
        days=int(max(day)),
        cos=(1, 7),
        start_date="2026-01-01 00:00:00",
        states=np.frombuffer(b"".join(rows), np.uint8).reshape(n, -1),
        to_play=np.array(to_play, np.uint8),
        day=np.array(day, np.int16),
        actions=ActionLabels(
            turn=np.array(turn, np.int16),
            action=np.array(action, np.int16),
            kind=np.array([k for k, _, _ in labels], np.uint8),
            steps=steps,
            n_steps=np.array([len(x) for _, x, _ in labels], np.uint8),
            n_legal=n_legal,
        ),
    )


def labels(g: GameSamples) -> ActionLabels:
    assert g.actions is not None
    return g.actions


def write_dataset(path, games, per_shard=2):
    writer = ShardWriter(path, games_per_shard=per_shard)
    for g in games:
        writer.add(g, ratings=(1000, 1100))
    writer.close(rows="actions")
    return Dataset(path)


def selection_after(state: bytes, picks) -> tuple[_core.GameState, _core.Selection]:
    s, sel = _core.GameState.from_bytes(state), _core.Selection()
    for pick in picks:
        assert _core.step(s, sel, int(pick)) == _core.OK
    return s, sel


def test_random_games_cover_every_phase(cache):
    g = random_game(cache, 120)
    a = labels(g)
    assert set(a.kind.tolist()) >= {A["Wait"], A["Build"], A["EndTurn"], A["Fire"], A["Capture"]}
    assert (a.n_steps > 0).all() and NO_ACTION not in a.kind


def test_decisions_are_each_steps_state_and_label(cache, tmp_path):
    g = random_game(cache, 120)
    a = labels(g)
    ds = write_dataset(tmp_path / "ds", [g, dataclasses.replace(g, game_id=2)])
    assert ds.has_actions and ds.meta["samples"] == 2 * len(g.states)
    sh = ds.load(0)
    assert sh.steps is not None
    np.testing.assert_array_equal(sh.steps[: len(g.states)], a.steps)
    batches = list(decision_batches(ds, 32, split="all", shuffle=False))
    taken = (np.arange(3)[None] < a.n_steps[:, None]) & (a.n_legal > 1)
    assert sum(len(b.target) for b in batches) == 2 * taken.sum()
    phases = set()
    for b in batches:
        size = b.planes.shape[-1]
        for i in range(len(b.target)):
            row = (int(b.row[i]) & 0xFFFFFFFF) % len(g.states)  # both games are g
            k = int(b.step[i])
            assert int(policy_ids(b.target[i], W, size)) == a.steps[row, k]
            s, sel = selection_after(g.states[row].tobytes(), a.steps[row, :k])
            assert b.phase[i] == sel.phase and b.n_legal[i] == a.n_legal[row, k]
            assert b.turn_start[i] == (a.action[row] == 0 and k == 0)
            legal = legal_mask(torch.from_numpy(b.planes[i : i + 1]), torch.from_numpy(b.nonspatial[i : i + 1]))
            ids = np.sort(policy_ids(np.flatnonzero(legal[0].numpy()), W, size))
            np.testing.assert_array_equal(ids, _core.legal_ids(s, sel))
            phases.add(int(b.phase[i]))
    assert phases == {PHASE_IDLE, PHASE_SELECTED, PHASE_MOVED, PHASE_FACTORY}

    every = sum(len(b.target) for b in decision_batches(ds, 32, split="all", forced=True))
    assert every == 2 * int(a.n_steps.sum())
    firsts = list(decision_batches(ds, 512, split="all", forced=True, steps=(0,)))
    assert sum(len(b.target) for b in firsts) == 2 * len(g.states)
    assert sum(b.turn_start.sum() for b in firsts) == 2 * (a.action == 0).sum()
    some = sum(len(b.target) for b in decision_batches(ds, 32, split="all", fraction=0.5, seed=3))
    assert 0.3 * every < some < 0.7 * every
    fewer = [b.row for b in decision_batches(ds, 32, split="all", row_fraction=0.5, shuffle=False)]
    again = [b.row for b in decision_batches(ds, 32, split="all", row_fraction=0.5, shuffle=False)]
    assert 0 < sum(map(len, fewer)) < 2 * taken.sum() and all(map(np.array_equal, fewer, again))

    val = in_validation(np.array([1, 2]), 0.5)
    expected = sum(taken.sum() for v in val if v)
    assert count_decisions(ds, "val", None, 0.5) == expected


def test_encode_decisions_flags_steps_that_are_not_legal(cache):
    g = random_game(cache, 10)
    a = labels(g)
    picks = np.array([[_core.ID_COP, 0, 0]], np.uint16)  # no power to use
    _, _, _, target = encode_decisions(g.states[:1], picks, np.zeros(1, np.uint8), 16)
    assert target is not None and target.tolist() == [-1]
    with pytest.raises(ValueError, match="pick"):
        encode_decisions(g.states[:1], picks, np.ones(1, np.uint8), 16, targets=False)
    planes, globals_, nonspatial, target = encode_decisions(g.states[:1], a.steps[:1], np.zeros(1, np.uint8), 16)
    assert planes.shape == (1, NUM_POLICY_PLANES, 16, 16) and globals_.shape == (1, NUM_POLICY_GLOBALS)
    assert nonspatial[0, 0] == 1 and target is not None and target[0] >= 0  # End Turn is legal


def test_movement_planes_follow_the_others(cache):
    g = random_game(cache, 10)
    picks, none = np.zeros((1, 3), np.uint16), np.zeros(1, np.uint8)
    plain, *_ = encode_decisions(g.states[:1], picks, none, 16, targets=False)
    planes, *_ = encode_decisions(g.states[:1], picks, none, 16, targets=False, movement=True)
    assert planes.shape == (1, NUM_POLICY_PLANES + _core.NUM_MOVEMENT_PLANES, 16, 16)
    np.testing.assert_array_equal(planes[:, :NUM_POLICY_PLANES], plain)
    move = {name: planes[0, NUM_POLICY_PLANES + i] for i, (name, _, _) in enumerate(MOVEMENT_PLANES)}
    # My Infantry at (1, 1): a turn from my base at (1, 0), three from theirs at (5, 5).
    assert move["home"][1, 1] == 1 + 1 and move["away"][1, 1] == 1 + 3
    assert move["home"][0, 1] == 0  # no unit there
    assert move["reach.me.vehicle"][2, 2] == 1 and move["reach.me.vehicle"][2, 5] == 2  # my Tank
    assert move["reach.opp.foot"][4, 5] == 1  # their Infantry
    base = next(i for i, p in enumerate(cache.props()) if p[:2] == (1, 0))
    turns = cache.factory_turns(base, U["Infantry"])
    assert turns.shape == (H, W) and turns[0, 1] == 0 and turns[1, 1] == 1 and turns[5, 5] == 3
    with pytest.raises(ValueError, match="build"):
        cache.factory_turns(base, U["Fighter"])


def test_aux_targets_follow_the_game(cache, tmp_path):
    g = random_game(cache, 150)
    a = labels(g)
    ds = write_dataset(tmp_path / "ds", [g])
    sh = ds.load(0)
    rows = np.arange(len(g.states))
    reg, binary, mask = aux_targets(sh, rows)
    assert reg.shape == (len(rows), len(AUX_REGRESSION)) and binary.shape == mask.shape == (len(rows), len(AUX_BINARY))

    def state(r):
        return _core.GameState.from_bytes(g.states[r].tobytes())

    def units(s, p):
        return sum(s.unit(i).owner == p for i in s.unit_slots())

    starts = {int(a.turn[r]): int(r) for r in rows if a.action[r] == 0}
    last = len(rows) - 1
    col = {name: i for i, name in enumerate(AUX_REGRESSION)}
    for r in rows[:: max(1, len(rows) // 25)]:
        now, me = state(r), int(g.to_play[r])
        for h, scale_v, scale_u in ((1, 10_000, 2), (2, 12_000, 2), (6, 20_000, 5), (None, 45_000, 9)):
            later = state(starts.get(int(a.turn[r]) + h, last) if h else last)
            for who, p in (("me", me), ("opp", 1 - me)):
                want = (army_values(later)[p] - army_values(now)[p]) / scale_v
                assert reg[r, col[f"{who}.value@{h or 'end'}"]] == pytest.approx(want, abs=1e-6)
                want = (units(later, p) - units(now, p)) / scale_u
                assert reg[r, col[f"{who}.units@{h or 'end'}"]] == pytest.approx(want, abs=1e-6)
        if int(a.turn[r]) + 2 in starts:
            nxt = state(starts[int(a.turn[r]) + 2]).players[me]
            assert mask[r, 0] and binary[r, 0] == (nxt.power >= nxt.cop_cost)
        else:
            assert not mask[r, 0] and not mask[r, 5]
    # Batches carry the same targets for their decisions.
    for b in decision_batches(ds, 64, split="all", shuffle=False, aux=True):
        assert b.aux_reg is not None and b.aux_bin is not None and b.aux_mask is not None
        r = b.row & 0xFFFFFFFF
        np.testing.assert_array_equal(b.aux_reg, reg[r])
        np.testing.assert_array_equal(b.aux_mask, mask[r])
    assert next(iter(decision_batches(ds, 64, split="all"))).aux_reg is None


TINY_TRANSFORMER = dataclasses.replace(TINY, trunk="transformer", heads=2, mlp_ratio=2)


def _random_net(cfg: PolicyConfig = TINY) -> PolicyValueNet:
    torch.manual_seed(0)
    net = PolicyValueNet(cfg)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.3)
    return net.eval()


@pytest.mark.parametrize("cfg", [TINY, TINY_TRANSFORMER], ids=["resnet", "transformer"])
def test_policy_ignores_padding_placement_and_batch(cache, cfg):
    g = random_game(cache, 30)
    a = labels(g)
    net = _random_net(cfg)
    planes16, globals_, nonspatial, _ = encode_decisions(g.states[:4], a.steps[:4], np.array([0, 1, 1, 2]), 16)
    planes24 = np.zeros((4, NUM_POLICY_PLANES, 24, 24), np.uint8)
    planes24[:, :, 5 : 5 + H, 9 : 9 + W] = planes16[:, :, :H, :W]
    g_ = torch.from_numpy(globals_)
    with torch.no_grad():
        s16, n16, v16 = net(torch.from_numpy(planes16), g_)
        s24, n24, v24 = net(torch.from_numpy(planes24), g_)
        _, _, alone = net(torch.from_numpy(planes16[1:]), g_[1:])
    torch.testing.assert_close(v16, v24, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(n16, n24, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s16[:, :, :H, :W], s24[:, :, 5 : 5 + H, 9 : 9 + W], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(v16[1:], alone, rtol=1e-5, atol=1e-6)
    logp = masked_log_softmax(
        flat_policy(s16, n16), legal_mask(torch.from_numpy(planes16), torch.from_numpy(nonspatial))
    )
    torch.testing.assert_close(logp.exp().sum(1), torch.ones(4))


def test_rotary_attention_reads_offsets_only():
    cos, sin = rope_angles(5, 8, torch.device("cpu"))
    q, k = torch.randn(8), torch.randn(8)

    def score(qy, qx, ky, kx):
        return float(rotate(q, cos[qy * 5 + qx], sin[qy * 5 + qx]) @ rotate(k, cos[ky * 5 + kx], sin[ky * 5 + kx]))

    assert score(1, 2, 3, 0) == pytest.approx(score(2, 3, 4, 1), abs=1e-5)
    assert score(1, 2, 3, 0) != pytest.approx(score(1, 2, 0, 3), abs=1e-3)  # another offset


@pytest.mark.parametrize("d", [8, 16, 24, 32])
def test_board_attention_reads_only_the_board(d):
    torch.manual_seed(0)
    q, k, v = (torch.randn(3, 2, 36, d) for _ in range(3))
    off = (torch.rand(3, 1, 36, 1) < 0.4).float()
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=(off == 0).transpose(-1, -2))
    torch.testing.assert_close(board_attention(q, k, v, off), want)


def test_undihedral_inverts_dihedral():
    board = torch.arange(2 * 9, dtype=torch.float32).view(1, 2, 3, 3)
    for k in range(8):
        assert torch.equal(undihedral(dihedral(board, k), k), board)


def test_policy_metrics_by_phase_and_action():
    nll = np.array([0.1, 0.2, 1.0, 0.5])
    correct = np.array([True, True, False, True])
    n_legal = np.array([2, 4, 8, 2], np.int16)
    phase = np.array([PHASE_IDLE, PHASE_SELECTED, PHASE_IDLE, PHASE_MOVED], np.uint8)
    row = np.array([7, 7, 9, 9])
    m = policy_metrics(nll, correct, n_legal, phase, row)
    assert m["n"] == 4 and m["accuracy"] == 0.75 and m["loss"] == pytest.approx(0.45)
    assert m["uniform"] == pytest.approx(np.log([2, 4, 8, 2]).mean())
    assert m["by_phase"]["idle"]["n"] == 2 and m["by_phase"]["moved"]["accuracy"] == 1.0
    assert m["actions"] == 2 and m["action_loss"] == pytest.approx(0.9) and m["action_accuracy"] == 0.5


def test_train_resumes_and_scores_decisions(cache, tmp_path):
    ids = np.arange(1, 200)
    val = in_validation(ids, 0.5)
    games = [random_game(cache, 60, seed=int(i), game_id=int(i)) for i in [*ids[val][:1], *ids[~val][:2]]]
    write_dataset(tmp_path / "ds", games)
    out = tmp_path / "run"
    cfg = PolicyTrainConfig(epochs=1, fraction=1.0, batch_size=16, warmup_steps=1, val_fraction=0.5, val_rows=1.0)
    lines: list[str] = []
    last = train(tmp_path / "ds", out, cfg, TINY, log=lines.append)
    assert last["policy"]["n"] > 0 and last["value"]["n"] == (labels(games[0]).action == 0).sum()
    assert set(last["policy"]["by_phase"]) <= set(_core.PHASES)
    assert {p.name for p in out.iterdir()} >= {"best.pt", "last.pt", "policy-ep01.pt", "metrics.jsonl"}
    last = train(tmp_path / "ds", out, dataclasses.replace(cfg, epochs=2), log=lines.append, resume=True)
    assert any(line.startswith("resumed") for line in lines)
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 2

    policy = Policy.load(out / "best.pt", "cpu")
    g = games[0]
    a = labels(g)
    scored = policy.score(g.states[:3], a.steps[:3], np.array([0, 1, 0]), ratings=(1000, 1000))
    for i, sc in enumerate(scored):
        s, sel = selection_after(g.states[i].tobytes(), a.steps[i, : [0, 1, 0][i]])
        np.testing.assert_array_equal(sc.ids, _core.legal_ids(s, sel))
        assert sc.prior.sum() == pytest.approx(1, rel=1e-5) and -1 <= sc.value <= 1
    assert [len(x.ids) for x in policy.score(g.states[:2])] == [
        len(_core.legal_ids(_core.GameState.from_bytes(r.tobytes()), _core.Selection())) for r in g.states[:2]
    ]


def test_movement_and_aux_train_and_score(cache, tmp_path):
    ids = np.arange(1, 200)
    val = in_validation(ids, 0.5)
    games = [random_game(cache, 60, seed=int(i), game_id=int(i)) for i in [*ids[val][:1], *ids[~val][:2]]]
    write_dataset(tmp_path / "ds", games)
    out = tmp_path / "run"
    cfg = PolicyTrainConfig(
        epochs=1, fraction=1.0, batch_size=16, warmup_steps=1, val_fraction=0.5, val_rows=1.0, log_every=1
    )
    model = dataclasses.replace(TINY, movement=True, aux=True)
    lines: list[str] = []
    last = train(tmp_path / "ds", out, cfg, model, log=lines.append)
    assert set(last["policy"]["aux"]["regression"]) == set(AUX_REGRESSION)
    assert any(line.startswith("  val aux mse") for line in lines)
    assert any(" aux " in line for line in lines if line.startswith("epoch "))
    net, _ = load_policy(out / "best.pt")
    assert net.cfg == model and net.aux is not None
    g = games[0]
    scored = Policy(net).score(g.states[:2])
    assert all(sc.prior.sum() == pytest.approx(1, rel=1e-5) for sc in scored)
    with pytest.raises(ValueError, match="auxiliary"):
        PolicyValueNet(TINY)(
            torch.zeros(1, NUM_POLICY_PLANES, 16, 16, dtype=torch.uint8), torch.zeros(1, NUM_POLICY_GLOBALS), aux=True
        )


def test_training_games_can_be_a_nested_subset(cache, tmp_path):
    many = np.arange(1, 200_000)
    quarter, half = in_subset(many, 0.25), in_subset(many, 0.5)
    assert not (quarter & ~half).any()
    assert half.mean() == pytest.approx(0.5, abs=0.01) and quarter.mean() == pytest.approx(0.25, abs=0.01)
    assert in_validation(many)[half].mean() == pytest.approx(0.05, abs=0.003)  # unrelated to the split
    ids = np.arange(1, 200)
    val = in_validation(ids, 0.5)
    train_ids = [int(i) for i in ids[~val][:8]]
    games = [random_game(cache, 20, seed=i, game_id=i) for i in [int(ids[val][0]), *train_ids]]
    ds = write_dataset(tmp_path / "ds", games)

    def decisions(i: int) -> int:
        a = labels(games[[g.game_id for g in games].index(i)])
        return int(((np.arange(3)[None, :] < a.n_steps[:, None]) & (a.n_legal > 1)).sum())

    def size(keep):
        return split_size(ds, "train", keep, 0.5)

    assert size(None) == (8, sum(map(decisions, train_ids)))
    halved = [i for i in train_ids if in_subset(np.array([i]), 0.5)[0]]
    assert 0 < len(halved) < 8 and size(games_keep(fraction=0.5)) == (len(halved), sum(map(decisions, halved)))
    assert size(games_keep(game_ids=(train_ids[2], train_ids[5])))[0] == 3
    assert size(games_keep(maps=(MAP_ID,))) == size(None)
    assert size(games_keep(exclude_maps=(MAP_ID,))) == (0, 0)
    cfg = PolicyTrainConfig(epochs=1, fraction=1.0, batch_size=16, val_fraction=0.5, hold_out_maps=(MAP_ID,))
    with pytest.raises(ValueError, match="no training decisions"):
        train(tmp_path / "ds", tmp_path / "run", cfg, TINY)


def test_load_policy_refuses_another_encoder(tmp_path):
    net = PolicyValueNet(TINY)
    save_policy(tmp_path / "p.pt", net)
    assert load_policy(tmp_path / "p.pt")[0].cfg == TINY
    # Checkpoints from before the movement planes and auxiliary head still load.
    ck = torch.load(tmp_path / "p.pt", weights_only=True)
    del ck["config"]["movement"], ck["config"]["aux"]
    torch.save(ck, tmp_path / "old.pt")
    assert load_policy(tmp_path / "old.pt")[0].cfg == TINY
    save_policy(tmp_path / "m.pt", PolicyValueNet(dataclasses.replace(TINY, movement=True)))
    ck = torch.load(tmp_path / "m.pt", weights_only=True)
    ck["encoder"]["movement_version"] = -1
    torch.save(ck, tmp_path / "m.pt")
    with pytest.raises(ValueError, match="other input features"):
        load_policy(tmp_path / "m.pt")
    ck = torch.load(tmp_path / "p.pt", weights_only=True)
    ck["encoder"]["selection_version"] = -1
    torch.save(ck, tmp_path / "p.pt")
    with pytest.raises(ValueError, match="other input features"):
        load_policy(tmp_path / "p.pt")


def test_cli_trains_and_evaluates_a_policy(cache, tmp_path, capsys):
    games = [random_game(cache, 40, seed=i, game_id=i) for i in range(1, 5)]
    write_dataset(tmp_path / "ds", games)
    out = tmp_path / "run"
    args = ["--data", str(tmp_path / "ds"), "--device", "cpu"]
    small = ["--epochs", "1", "--fraction", "1", "--channels", "32", "--blocks", "2", "--batch-size", "16"]
    assert main(["train-policy", *args, "--out", str(out), *small, "--val-rows", "1"]) == 0
    assert main(["eval-policy", str(out / "best.pt"), *args, "--split", "all", "--rows", "1"]) == 0
    text = capsys.readouterr().out
    assert "by phase" in text and "value at" in text and "uniform" in text
    some = ["--split", "all", "--rows", "1", "--maps", str(MAP_ID), "--game-ids", "1", "3"]
    assert main(["eval-policy", str(out / "best.pt"), *args, *some]) == 0
    assert "value at" in capsys.readouterr().out
    fewer = ["--games-fraction", "0.9", "--hold-out-maps", "1", "2", "--game-ids", "0", "100"]
    assert main(["train-policy", *args, "--out", str(tmp_path / "fewer"), *small, *fewer]) == 0
    info = load_policy(tmp_path / "fewer" / "best.pt")[1]
    assert info["train"]["hold_out_maps"] == [1, 2] and 0 < info["train_games"] <= 4
    attention = ["--trunk", "transformer", "--heads", "4", "--mlp-ratio", "2", "--recompute"]
    assert main(["train-policy", *args, "--out", str(tmp_path / "tf"), *small, *attention]) == 0
    net = load_policy(tmp_path / "tf" / "best.pt")[0]
    assert (net.cfg.trunk, net.cfg.heads, net.cfg.mlp_ratio) == ("transformer", 4, 2)


def test_games_between_agents_end_or_are_adjudicated(cache):
    s = start(cache)
    assert property_counts(cache, s) == [2, 2]  # each HQ and base
    assert army_values(s) == [1000 + 7000 + 6000, 1000 + 7000 + 3000]
    assert adjudicate(cache, s) == 0  # equal properties: the larger army
    results = play([s, s], (RandomAgent(1), RandomAgent(2)), max_days=4)
    assert len(results) == 2
    for r in results:
        assert r.actions > 0 and r.winner in (0, 1, DRAW)
        assert (r.decided == "day limit") == (r.final.winner == _core.NO_OWNER)
        if r.decided == "day limit":
            assert r.days == 4 and r.final.day == 5 and r.winner == adjudicate(cache, r.final)
    policy = Policy(_random_net())
    for agent in (PolicyAgent(policy), PolicyAgent(policy, temperature=1.0, ratings=(1200, 1200))):
        (r,) = play([s], (agent, RandomAgent(3)), max_days=2)
        assert r.actions > 0 and r.days <= 2
