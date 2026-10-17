import json
import os

import numpy as np
import pytest
import torch

from awbw import _core, engine
from awbw.agents import DRAW, RandomAgent, play
from awbw.data.loader import encode_decisions
from awbw.nn.model import PolicyValueNet
from awbw.nn.policy import Policy, policy_ids
from awbw.search import NetEvaluator, SearchAgent, SearchSettings, run, selection_of, self_play
from test_policy import MAP_ID, TERRAIN, TINY, H, W, start


@pytest.fixture
def md(tmp_path, monkeypatch):
    """test_policy's map, registered with the engine."""
    monkeypatch.setenv("AWBW_CACHE", str(tmp_path))
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / f"{MAP_ID}.json").write_text(
        json.dumps({"Name": "t", "Size X": W, "Size Y": H, "Terrain Map": TERRAIN.T.tolist(), "Predeployed Units": []})
    )
    return engine.map_data(MAP_ID)


def _net(seed: int = 0) -> PolicyValueNet:
    import torch

    torch.manual_seed(seed)
    net = PolicyValueNet(TINY)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.3)
    return net.eval()


def _decisions(s: _core.GameState) -> list[tuple[_core.GameState, list[int]]]:
    """The turn's first decision, and one each of the selected and moved phases."""
    idle = _core.legal_ids(s)
    tank = int(idle[(idle // _core.MAX_TILES == 0) & (idle % _core.MAX_TILES == 2 * W + 2)][0])  # the Tank
    sel = selection_of(s, [tank])
    # A destination with more than Wait to do there.
    move = next(int(m) for m in _core.legal_ids(s, sel) if len(_core.legal_ids(s, selection_of(s, [tank, m]))) > 1)
    return [(s, []), (s, [tank]), (s, [tank, move])]


def test_leaves_are_encoded_as_the_policy_reads_them(md):
    s = start(md)
    search = _core.BatchedSearch(_core.SearchConfig(sims=4), 16)
    for state, picks in _decisions(s):
        search.add_root(state, selection_of(state, picks))
    assert search.collect() == 3  # each root first
    planes, globals_, legal, movers = search.batch()
    picks = np.zeros((3, 3), np.uint16)
    n_picks = np.array([0, 1, 2], np.uint8)
    for i, (_, p) in enumerate(_decisions(s)):
        picks[i, : len(p)] = p
    states = np.stack([np.frombuffer(s.to_bytes(), np.uint8)] * 3)
    want_planes, want_globals, _, _ = encode_decisions(states, picks, n_picks, 16, targets=False)
    np.testing.assert_array_equal(planes, want_planes)
    np.testing.assert_array_equal(globals_, want_globals)
    assert list(movers) == [0, 0, 0]
    # Gathered logits give the prior Policy.score gives.
    net = _net()
    logits, values = NetEvaluator(net)(planes, globals_, legal, movers)
    scored = Policy(net).score(states, picks, n_picks)
    for i, (state, p) in enumerate(_decisions(s)):
        n = int((legal[i] >= 0).sum())
        ids = policy_ids(legal[i, :n], W, 16)
        np.testing.assert_array_equal(ids, _core.legal_ids(state, selection_of(state, p)))
        np.testing.assert_array_equal(ids, scored[i].ids)
        prior = np.exp(logits[i, :n] - logits[i, :n].max())
        np.testing.assert_allclose(prior / prior.sum(), scored[i].prior, rtol=1e-4, atol=1e-6)
        assert values[i] == pytest.approx(scored[i].value, abs=1e-5)
    # Padding the batch (as a compiled evaluator does) changes nothing.
    padded = NetEvaluator(net)
    padded.pad = True
    logits2, values2 = padded(planes, globals_, legal, movers)
    np.testing.assert_allclose(logits2, logits, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(values2, values, rtol=1e-5, atol=1e-6)
    search.submit(logits, values)
    assert search.active == 3


DEVICES = [
    "cpu",
    # GitHub's macOS runners report MPS but are virtual machines that can't run it
    pytest.param(
        "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available() or "CI" in os.environ, reason="no MPS")
    ),
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
]


@pytest.mark.parametrize("device", DEVICES)
def test_queued_batches_score_as_one_at_a_time(md, device):
    s = start(md)
    search = _core.BatchedSearch(_core.SearchConfig(sims=4), 16)
    for state, picks in _decisions(s):
        search.add_root(state, selection_of(state, picks))
    search.collect()
    planes, globals_, legal, movers = search.batch()
    net = _net()
    want = NetEvaluator(net)(planes, globals_, legal, movers)
    evaluator = NetEvaluator(net, device, bf16=False)
    evaluator.pad = True  # as compiled: to pad_to, or else a power of two
    # Several batches queued before any is read, each in two parts (2 leaves and 1).
    evaluator.batch_size = 2
    queued = [evaluator.launch(planes, globals_, legal, movers, pad_to=pad) for pad in (0, 3, 5)]
    assert evaluator.evals == 9 and evaluator.rows == (16 + 16) + (3 + 3) + (5 + 5)
    for pending in reversed(queued):
        logits, values = pending.result()
        np.testing.assert_allclose(logits, want[0], rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(values, want[1], rtol=1e-3, atol=1e-4)


def test_search_results_and_the_agent(md):
    s = start(md)
    evaluator = NetEvaluator(_net())
    search = _core.BatchedSearch(_core.SearchConfig(sims=16, considered=8, seed=3), 16)
    slots = [search.add_root(state, selection_of(state, picks)) for state, picks in _decisions(s)]
    evals = run(search, evaluator)
    assert evals <= 3 * 17  # the roots, then a leaf per simulation that doesn't end the game
    for slot in slots:
        r = search.result(slot)
        assert r.sims == 16 and r.visits.sum() == 16
        assert r.id in r.ids and r.policy.sum() == pytest.approx(1, abs=1e-5)
        assert r.prior.sum() == pytest.approx(1, abs=1e-5)
        assert -1 <= r.value <= 1
    with pytest.raises(IndexError):
        search.result(len(slots))

    agent = SearchAgent(evaluator, SearchSettings(sims=8, considered=4, max_days=2), seed=1)
    results = play([s, s], (agent, RandomAgent(1)), max_days=2)
    assert len(results) == 2
    for res in results:
        assert res.actions > 0 and res.winner in (0, 1, DRAW)
    assert agent.calls > 0 and all(r.sims == 8 for r in agent.last)


def test_self_play_gives_samples_and_records(md):
    s = start(md)
    cfg = _core.SearchConfig(sims=4, considered=4, max_days=2, full_fraction=0.5, fast_sims=1, seed=7)
    samples, records, evals = [], [], 0
    for batch in self_play(NetEvaluator(_net()), iter([s] * 3), cfg, 16, parallel=2):
        samples.append(batch.samples)
        records += batch.records
        evals += batch.evals
    assert sorted(r.game for r in records) == [0, 1, 2]
    for r in records:
        assert r.winner in (0, 1, DRAW) and r.decisions > 0
        assert r.capped == (r.final_state.winner == _core.NO_OWNER)
    n = sum(len(b["game"]) for b in samples)
    assert 0 < n < sum(r.decisions for r in records)  # half the searches are fast, and not kept
    b = {k: np.concatenate([x[k] for x in samples]) for k in samples[0] if k != "offsets"}
    offsets = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(x["offsets"]) for x in samples]))])
    assert offsets[-1] == len(b["ids"]) and (b["sims"] == 4).all()
    for i in range(n):
        st = _core.GameState.from_bytes(b["states"][i].tobytes())
        sel = selection_of(st, b["picks"][i, : b["n_picks"][i]])
        ids = b["ids"][offsets[i] : offsets[i + 1]]
        np.testing.assert_array_equal(ids, _core.legal_ids(st, sel))
        assert b["chosen"][i] in ids and b["mover"][i] == st.active
        assert b["policy"][offsets[i] : offsets[i + 1]].sum() == pytest.approx(1, abs=1e-5)
    # The samples encode as training decisions.
    encode_decisions(b["states"], b["picks"], b["n_picks"], 16, targets=False)
