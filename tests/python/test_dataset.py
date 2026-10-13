import dataclasses
import json

import numpy as np
import pytest

from _factory import GAME_ID, build_replay_zip
from awbw import _core
from awbw.data.loader import batches, in_validation
from awbw.data.shards import Dataset, ShardWriter
from awbw.replay import stats
from awbw.replay.extract import extract_v0

MAP_ID = 77  # the map the factory's replay is on


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A cache with the factory replay and its 5x5 map, and no minimum game length."""
    monkeypatch.setenv("AWBW_CACHE", str(tmp_path))
    monkeypatch.setattr(stats, "MIN_DAYS", 1)
    terrain = [[34 if (x, y) == (1, 2) else 1 for y in range(5)] for x in range(5)]  # [x][y]
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / f"{MAP_ID}.json").write_text(
        json.dumps({"Name": "t", "Size X": 5, "Size Y": 5, "Terrain Map": terrain, "Predeployed Units": []})
    )
    (tmp_path / "replays").mkdir()
    path = tmp_path / "replays" / f"{GAME_ID}.zip"
    path.write_bytes(build_replay_zip())
    return path


def test_extract_labels_each_turn_for_the_mover(cache):
    g = extract_v0(cache)
    assert not isinstance(g, str), g
    assert (g.game_id, g.map_id, g.winner, g.end_action, g.cos) == (GAME_ID, MAP_ID, 0, "Resign", (22, 11))
    assert g.states.shape == (2, _core.STATE_BYTES)
    assert g.to_play.tolist() == [0, 1] and g.day.tolist() == [1, 1]
    assert g.outcome.tolist() == [1, -1]
    assert _core.GameState.from_bytes(g.states[1].tobytes()).active == 1


def test_extract_applies_the_filter(cache, monkeypatch):
    monkeypatch.setattr(stats, "MIN_DAYS", 6)
    assert extract_v0(cache) == "under 6 days"


def test_shards_round_trip_into_batches(cache, tmp_path):
    g = extract_v0(cache)
    assert not isinstance(g, str)
    writer = ShardWriter(tmp_path / "ds", games_per_shard=2)
    for i in range(3):
        writer.add(dataclasses.replace(g, game_id=GAME_ID + i), ratings=(900 + i, 1000 + i))
    meta = writer.close(note="test")
    assert (meta["games"], meta["samples"], meta["note"]) == (3, 6, "test")
    assert meta["shards"] == ["shard-00000.npz", "shard-00001.npz"]
    with pytest.raises(FileExistsError):
        ShardWriter(tmp_path / "ds")

    ds = Dataset(tmp_path / "ds")
    assert ds.map_size == {MAP_ID: (5, 5)}
    sh = ds.load(0)
    assert len(sh) == 4 and sh.game.tolist() == [0, 0, 1, 1] and sh.turn.tolist() == [0, 1, 0, 1]
    assert sh.g_start.tolist() == [0, 2] and sh.g_id.tolist() == [GAME_ID, GAME_ID + 1]
    assert sh.z.tolist() == [1, -1, 1, -1] and sh.map_id.tolist() == [MAP_ID] * 4

    out = list(batches(ds, 4, split="all", shuffle=False))
    assert [len(b.z) for b in out] == [4, 2]
    first = out[0]
    assert first.planes.shape == (4, _core.NUM_PLANES, 16, 16)
    assert first.rating.tolist() == [[900, 1000], [1000, 900], [901, 1001], [1001, 901]]  # mover first
    assert sum(len(b.z) for b in batches(ds, 8, keep=lambda sh: sh.turn == 0, split="all")) == 3


def test_validation_split_is_a_fixed_partition():
    ids = np.arange(1_000_000, 1_200_000)
    val = in_validation(ids)
    assert 0.045 < val.mean() < 0.055
    np.testing.assert_array_equal(val, in_validation(ids.copy()))
    assert in_validation(ids, 0.0).sum() == 0 and in_validation(ids, 1.0).all()
