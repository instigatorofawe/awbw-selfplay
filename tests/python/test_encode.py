import numpy as np
import pytest

from awbw import _core, rules
from awbw.data.loader import bucket_for, encode

R = rules.load()
U = {u.name: u.index for u in R.units}
KIND = {k: i for i, k in enumerate(R.terrain_kinds)}
PLANE = {name: i for i, (name, _, _) in enumerate(_core.PLANES)}
GLOBAL = {name: i for i, name in enumerate(_core.GLOBAL_NAMES)}
N_KINDS = len(R.terrain_kinds)

# Orange Star HQ, plains, a neutral city, mountains; sea; Blue Moon HQ.
TERRAIN = np.array([[42, 1, 1, 34, 2], [1, 1, 2, 1, 1], [28, 28, 1, 1, 47]], dtype=np.int16)


@pytest.fixture(scope="module")
def md():
    return _core.register_map(910002, TERRAIN)


def state(md, active=0):
    players = np.array([[1, 1, 3000, 0, 270000, 630000, 0, 0], [7, 2, 8000, 0, 270000, 540000, 0, 0]])
    buildings = np.array([[0, 0, 42, 20], [3, 0, 34, 12], [4, 2, 47, 20]])
    # id, type, owner, x, y, hp, fuel, ammo, flags, cargo1, cargo2
    units = np.array(
        [[1, U["Tank"], 0, 1, 1, 73, 50, 9, 0, 0, 0], [2, U["Infantry"], 1, 3, 0, 100, 99, 0, 0, 0, 0]],
        dtype=np.int64,
    )
    return _core.load_snapshot(md, _core.Settings(), 4, active, players, buildings, units)


def rows(*states):
    return np.stack([np.frombuffer(s.to_bytes(), np.uint8) for s in states])


def test_tables_line_up():
    assert len(_core.PLANES) == _core.NUM_PLANES
    assert len(_core.GLOBAL_NAMES) == _core.NUM_GLOBALS == len(set(_core.GLOBAL_NAMES))
    assert {"me.funds", "opp.hidden_value", "opp.co.Max", "weather.clear", "moved_first"} <= set(_core.GLOBAL_NAMES)


def test_planes_and_globals_from_each_side(md):
    planes, g = encode(rows(state(md, 0), state(md, 1)), 8)
    assert planes.shape == (2, _core.NUM_PLANES, 8, 8) and g.shape == (2, _core.NUM_GLOBALS)
    mine, theirs = planes
    unit = PLANE["unit"]
    assert mine[unit, 1, 1] == 1 + U["Tank"] and mine[unit, 0, 3] == 1 + len(U) + U["Infantry"]
    assert theirs[unit, 1, 1] == 1 + len(U) + U["Tank"] and theirs[unit, 0, 3] == 1 + U["Infantry"]
    assert mine[PLANE["hp"], 1, 1] == 73 and mine[PLANE["fuel"], 1, 1] == 50
    terrain = PLANE["terrain"]
    assert mine[terrain, 0, 0] == 1 + KIND["hq"] + N_KINDS  # my HQ
    assert theirs[terrain, 0, 0] == 1 + KIND["hq"] + 2 * N_KINDS
    assert mine[terrain, 0, 3] == 1 + KIND["city"]  # neutral
    assert mine[PLANE["points"], 0, 3] == 12
    assert not planes[:, :, 3:, :].any() and not planes[:, :, :, 5:].any()  # off the board
    assert g[0, GLOBAL["me.funds"]] == pytest.approx(0.3) and g[1, GLOBAL["me.funds"]] == pytest.approx(0.8)
    assert g[0, GLOBAL["opp.co.Max"]] == 1 and g[1, GLOBAL["me.co.Max"]] == 1
    assert g[0, GLOBAL["moved_first"]] == 1 and g[1, GLOBAL["moved_first"]] == 0


def test_bad_inputs_raise(md):
    good = rows(state(md))
    planes = np.zeros((1, _core.NUM_PLANES, 4, 4), np.uint8)
    globals_ = np.zeros((1, _core.NUM_GLOBALS), np.float32)
    with pytest.raises(ValueError, match="larger than the planes"):
        _core.encode_batch(good, planes, globals_)
    with pytest.raises(ValueError, match="STATE_BYTES"):
        _core.encode_batch(good[:, :-1], planes, globals_)
    with pytest.raises(ValueError, match="globals"):
        _core.encode_batch(good, np.zeros((1, _core.NUM_PLANES, 8, 8), np.uint8), globals_[:, :-1])
    unknown = good.copy()
    unknown[0, :4] = np.frombuffer(np.uint32(123456789).tobytes(), np.uint8)  # map_id leads the state
    with pytest.raises(ValueError, match="not registered"):
        encode(unknown, 8)


def test_buckets():
    assert bucket_for(19, 19) == 24 and bucket_for(16, 16) == 16 and bucket_for(17, 23) == 24
    assert bucket_for(40, 12) == 40
    with pytest.raises(ValueError):
        bucket_for(41, 10)
