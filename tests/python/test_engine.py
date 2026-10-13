import pickle

import numpy as np
import pytest

from awbw import _core, rules

R = rules.load()
U = {u.name: u.index for u in R.units}
KIND = {k: i for i, k in enumerate(R.terrain_kinds)}

# Orange Star HQ, plains, a neutral city, mountains, sea; Blue Moon HQ.
TERRAIN = np.array(
    [
        [42, 1, 1, 34, 2],
        [1, 1, 2, 1, 1],
        [28, 28, 1, 1, 47],
    ],
    dtype=np.int16,
)


@pytest.fixture(scope="module")
def md():
    return _core.register_map(910001, TERRAIN)


def players(funds=(1000, 2000)):
    # co, country, funds, power, cop_cost, scop_cost, power_on, eliminated
    return np.array([[1, 1, funds[0], 0, 270000, 630000, 0, 0], [7, 2, funds[1], 0, 270000, 540000, 0, 0]])


def units(*rows):
    # id, type, owner, x, y, hp, fuel, ammo, flags, cargo1, cargo2
    return np.array(rows, dtype=np.int64).reshape(-1, 11)


def snapshot(md, unit_rows=(), buildings=None, active=0):
    if buildings is None:
        buildings = np.array([[0, 0, 42, 20], [3, 0, 34, 20], [4, 2, 47, 20]])
    return _core.load_snapshot(md, _core.Settings(), 4, active, players(), buildings, units(*unit_rows))


def test_map_registry(md):
    assert (md.width, md.height, md.n_props) == (5, 3, 3)
    assert md.props() == [(0, 0, KIND["hq"], 1), (3, 0, KIND["city"], 0), (4, 2, KIND["hq"], 2)]
    assert _core.find_map(910001) is not None and _core.find_map(910001).id == 910001
    assert _core.register_map(910001, TERRAIN).width == 5  # idempotent
    other = TERRAIN.copy()
    other[1, 1] = 3
    with pytest.raises(ValueError, match="different terrain"):
        _core.register_map(910001, other)
    with pytest.raises(ValueError, match="unknown terrain"):
        _core.register_map(910002, np.full((2, 2), 9999, dtype=np.int16))


def test_load_snapshot(md):
    s = snapshot(md, [(70, U["Tank"], 1, 3, 1, 64, 50, 7, 0, 0, 0), (60, U["Infantry"], 0, 1, 1, 100, 99, 0, 0, 0, 0)])
    assert s.check() == ""
    assert s.day == 4 and s.active == 0 and s.map_id == 910001
    assert [s.players[i].funds for i in range(2)] == [1000, 2000]
    assert [s.unit(i).id for i in s.unit_slots()] == [60, 70]  # slots in id order
    assert s.unit_at(3, 1) == 1 and s.unit_at(0, 1) == -1
    tank = s.unit(1)
    assert (tank.type, tank.owner, tank.hp, tank.fuel, tank.ammo) == (U["Tank"], 1, 64, 50, 7)
    assert [s.prop_owner(p) for p in range(3)] == [0, _core.NO_OWNER, 1]
    with pytest.raises(ValueError, match="two units"):
        snapshot(md, [(1, 0, 0, 1, 1, 100, 99, 0, 0, 0, 0), (2, 0, 1, 1, 1, 100, 99, 0, 0, 0, 0)])


def test_state_is_bytes(md):
    s = snapshot(md, [(60, U["Infantry"], 0, 1, 1, 100, 99, 0, 0, 0, 0)])
    raw = s.to_bytes()
    assert len(raw) == _core.STATE_BYTES
    assert _core.GameState.from_bytes(raw).to_bytes() == raw
    assert pickle.loads(pickle.dumps(s)).to_bytes() == raw
    assert s.copy().to_bytes() == raw


def test_movement(md):
    s = snapshot(md, [(60, U["Infantry"], 0, 1, 1, 100, 99, 0, 0, 0, 0), (70, U["Tank"], 1, 3, 1, 100, 70, 9, 0, 0, 0)])
    reach = {(x, y): c for x, y, c in _core.reachable(s, 0)}
    assert reach[(1, 1)] == 0
    assert reach[(2, 1)] == 2  # mountain
    assert (3, 1) not in reach  # enemy tank blocks
    assert (0, 2) not in reach  # sea
    assert _core.path_cost(s, 0, [(1, 1), (1, 0), (2, 0)]) == 2
    assert _core.path_cost(s, 0, [(1, 1), (3, 1)]) == -1
    assert _core.move_points(s, 1) == 6
    with pytest.raises(IndexError):
        _core.reachable(s, 5)
