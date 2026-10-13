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


A = _core.ACTIONS


def test_apply_capture_then_end_turn(md):
    s = snapshot(
        md, [(60, U["Infantry"], 0, 1, 1, 100, 99, 0, 0, 0, 0), (70, U["Infantry"], 1, 4, 1, 100, 99, 0, 0, 0, 0)]
    )
    inf = s.find_unit(60)
    assert _core.apply(s, A["Capture"], inf, to=(3, 0)) == _core.OK
    city = next(i for i, p in enumerate(md.props()) if p[:2] == (3, 0))
    assert s.prop_points(city) == 10
    assert s.unit(inf).fuel == 96
    before = s.to_bytes()
    status = _core.apply(s, A["Wait"], inf, to=(2, 0))
    assert _core.status_name(status) == "cannot act"
    assert s.to_bytes() == before  # errors change nothing
    funds = s.players[1].funds
    assert _core.apply(s, A["EndTurn"]) == _core.OK
    assert s.active == 1 and s.day == 4
    assert s.players[1].funds == funds + 1000  # Blue Moon's HQ pays
    assert not s.unit(inf).flags & _core.UNIT_MOVED


def test_apply_fire_with_a_forced_result(md):
    s = snapshot(md, [(60, U["Tank"], 0, 1, 1, 100, 70, 9, 0, 0, 0), (70, U["Tank"], 1, 2, 0, 100, 70, 9, 0, 0, 0)])
    tank = s.find_unit(60)
    outs = _core.fight_outcomes(s, tank, (1, 0), (2, 0))
    assert outs and abs(sum(p for _, _, p in outs) - 1) < 1e-9
    att_hp, def_hp, _ = outs[0]
    assert _core.apply(s, A["Fire"], tank, to=(1, 0), target=(2, 0), att_hp=att_hp, def_hp=def_hp) == _core.OK
    rows = {r[0]: r for r in s.unit_rows()}
    assert (rows[60][5], rows[70][5]) == (att_hp, def_hp)
    assert rows[60][7] == 8 and rows[60][3:5] == (1, 0)
    assert _core.apply(s, A["Fire"], s.find_unit(60), target=(2, 0)) != _core.OK  # already acted


def test_apply_build_and_explicit_path(md):
    s = snapshot(md, [(60, U["Recon"], 0, 1, 1, 100, 80, 0, 0, 0, 0)])
    recon = s.find_unit(60)
    assert _core.apply(s, A["Wait"], recon, to=(2, 0), path=[(1, 1), (1, 0), (2, 0)]) == _core.OK
    assert s.unit(recon).fuel == 80 - 4  # tires pay 2 on plains
    with pytest.raises(ValueError):
        _core.apply(s, 99)
    assert _core.status_name(_core.apply(s, A["Build"], target=(0, 0), arg=U["Infantry"])) == "cannot build"  # HQ


def test_cargo_slots(md):
    s = snapshot(
        md,
        [
            (50, U["Lander"], 0, 1, 2, 100, 99, 0, 0, 0, 51),
            (51, U["Infantry"], 0, 1, 2, 100, 99, 0, 0, 0, 0),
        ],
    )
    lander = s.find_unit(50)
    assert s.unit(lander).cargo == (-1, s.find_unit(51))  # AWBW's slot 2, kept
    assert _core.apply(s, A["Unload"], lander, target=(1, 1), arg=1) == _core.OK
    assert s.unit(lander).cargo == (-1, -1)
    assert not s.unit(lander).flags & _core.UNIT_MOVED  # unloading in place keeps its action


def test_snapshot_weather_and_its_owner(md):
    s = _core.load_snapshot(
        md, _core.Settings(), 4, 0, players(), np.zeros((0, 4), dtype=np.int64), units(), weather=2, weather_owner=0
    )
    assert _core.WEATHERS[s.weather] == "snow" and s.weather_owner == 0
    assert _core.apply(s, A["EndTurn"]) == _core.OK
    assert _core.WEATHERS[s.weather] == "snow"  # lasts until its owner's next turn
    assert _core.apply(s, A["EndTurn"]) == _core.OK
    assert _core.WEATHERS[s.weather] == "clear"


def test_missiles_aimed_or_given(md):
    sturm = next(c.id for c in R.cos.values() if c.name == "Sturm")
    p = players()
    p[0, 0] = sturm
    p[0, 3] = p[0, 4] = p[0, 5] = 10 * _core.STAR_VALUE
    rows = [(60, U["Infantry"], 0, 0, 1, 100, 99, 0, 0, 0, 0), (70, U["Tank"], 1, 4, 1, 100, 70, 9, 0, 0, 0)]
    s = _core.load_snapshot(md, _core.Settings(), 4, 0, p, np.zeros((0, 4), dtype=np.int64), units(*rows))
    assert _core.missile_targets(s, 1) == [(4, 1)]  # Meteor Strike centres on an enemy unit
    assert _core.missile_targets(s, 2) == [(4, 1)]
    given = s.copy()
    assert _core.apply(given, A["Power"], arg=1, missiles=[(0, 1)]) == _core.OK  # the logged centre
    assert given.unit(given.find_unit(60)).hp == 60  # missiles hit their user's units too
    assert given.unit(given.find_unit(70)).hp == 100
    assert _core.apply(s, A["Power"], arg=1) == _core.OK
    assert s.unit(s.find_unit(70)).hp == 60
