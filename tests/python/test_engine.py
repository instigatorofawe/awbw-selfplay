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


def test_move_path_is_the_one_apply_takes(md):
    s = snapshot(md, [(60, U["Infantry"], 0, 1, 1, 100, 99, 0, 0, 0, 0)])
    assert _core.move_path(s, 0, (2, 0)) == ([(1, 1), (1, 0), (2, 0)], 2, False, 2)  # around the mountain
    assert _core.move_path(s, 0, (1, 1)) == ([(1, 1)], 0, False, 0)
    assert _core.move_path(s, 0, (0, 2)) is None  # sea
    # A hidden Stealth no unit of Orange Star's is next to blocks the only way
    # round the mountain: the Recon is planned through it and stops on the city.
    hidden = (70, U["Stealth"], 1, 2, 0, 100, 60, 6, _core.UNIT_HIDDEN, 0, 0)
    s = snapshot(md, [(60, U["Recon"], 0, 3, 2, 100, 80, 0, 0, 0, 0), hidden])
    recon = s.find_unit(60)
    path = [(3, 2), (3, 1), (3, 0), (2, 0), (1, 0)]
    assert _core.move_path(s, recon, (1, 0)) == (path, 2, True, 3)
    assert _core.apply(s, A["Wait"], recon, to=(1, 0)) == _core.TRAPPED
    assert s.unit_at(3, 0) == recon and s.unit(recon).fuel == 77


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
    # Sea, a shoal and plains.
    shore = _core.register_map(910003, np.array([[28, 29, 1], [28, 28, 1]], dtype=np.int16))
    no_props = np.zeros((0, 4), dtype=np.int64)

    def lander_at(x, y):
        rows = [(50, U["Lander"], 0, x, y, 100, 99, 0, 0, 0, 51), (51, U["Infantry"], 0, x, y, 100, 99, 0, 0, 0, 0)]
        return _core.load_snapshot(shore, _core.Settings(), 4, 0, players(), no_props, units(*rows))

    at_sea = lander_at(1, 1)
    assert _core.status_name(_core.apply(at_sea, A["Unload"], at_sea.find_unit(50), target=(2, 1), arg=1)) == (
        "cannot unload"  # infantry can't stand where the Lander is
    )
    s = lander_at(1, 0)
    lander = s.find_unit(50)
    assert s.unit(lander).cargo == (-1, s.find_unit(51))  # AWBW's slot 2, kept
    assert _core.apply(s, A["Unload"], lander, target=(2, 0), arg=1) == _core.OK
    assert s.unit(lander).cargo == (-1, -1)
    assert not s.unit(lander).flags & _core.UNIT_MOVED  # unloading in place keeps its action


def test_cargo_of_cargo_moves_with_it(md):
    # AWBW lets a loaded APC board a Lander (52 games of the archive); the
    # APC's infantry goes wherever the APC does.
    shore = _core.register_map(910003, np.array([[28, 29, 1], [28, 28, 1]], dtype=np.int16))
    rows = [
        (50, U["Lander"], 0, 0, 1, 100, 99, 0, 0, 51, 0),
        (51, U["APC"], 0, 0, 1, 100, 70, 0, 0, 52, 0),
        (52, U["Infantry"], 0, 0, 1, 100, 99, 0, 0, 0, 0),
    ]
    s = _core.load_snapshot(shore, _core.Settings(), 4, 0, players(), np.zeros((0, 4), dtype=np.int64), units(*rows))
    lander, apc, inf = (s.find_unit(i) for i in (50, 51, 52))
    assert s.check() == "" and s.unit(inf).carrier == apc
    assert _core.apply(s, A["Unload"], lander, to=(1, 0), target=(2, 0)) == _core.OK
    assert s.check() == ""
    assert s.unit(apc).tile == s.unit(inf).tile == shore.tile(2, 0)
    assert s.unit(inf).carrier == apc and s.unit(lander).tile == shore.tile(1, 0)


@pytest.mark.parametrize(
    ("outer", "middle", "y"),
    [("Lander", "APC", 0), ("Cruiser", "T-Copter", 1), ("Carrier", "T-Copter", 1)],
)
def test_loaded_transport_boards_another(md, outer, middle, y):
    # A Lander on the shoal or a ship at sea, the loaded transport on the plains next to it.
    shore = _core.register_map(910003, np.array([[28, 29, 1], [28, 28, 1]], dtype=np.int16))
    rows = [
        (50, U[outer], 0, 1, y, 100, 99, 9, 0, 0, 0),
        (51, U[middle], 0, 2, y, 100, 70, 0, 0, 52, 0),
        (52, U["Infantry"], 0, 2, y, 100, 99, 0, 0, 0, 0),
    ]
    s = _core.load_snapshot(shore, _core.Settings(), 4, 0, players(), np.zeros((0, 4), dtype=np.int64), units(*rows))
    ship, mid, inf = (s.find_unit(i) for i in (50, 51, 52))
    assert any(a[:2] == (A["Load"], mid) for a in _core.legal_actions(s))
    assert _core.apply(s, A["Load"], mid, to=(1, y)) == _core.OK
    assert s.check() == ""
    assert s.unit(ship).cargo == (mid, -1) and s.unit(inf).carrier == mid
    assert s.unit(inf).tile == shore.tile(1, y)
    hurt = s.copy()
    hurt.set_hp(inf, 50)
    assert _core.state_hash(hurt) != _core.state_hash(s)  # the cargo's cargo counts


def test_loaded_transport_cannot_join(md):
    rows = [
        (50, U["APC"], 0, 1, 1, 50, 70, 0, 0, 0, 0),
        (51, U["APC"], 0, 2, 1, 100, 70, 0, 0, 52, 0),
        (52, U["Infantry"], 0, 2, 1, 100, 99, 0, 0, 0, 0),
    ]
    s = snapshot(md, rows)
    loaded = s.find_unit(51)
    assert not any(a[:2] == (A["Join"], loaded) for a in _core.legal_actions(s))
    assert _core.status_name(_core.apply(s, A["Join"], loaded, to=(1, 1))) == "cannot join"


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


CH = {name: i for i, name in enumerate(_core.CHANNELS)}


def step_id(channel, md, x, y):
    return CH[channel] * _core.MAX_TILES + md.tile(x, y)


def fighting(md):
    return snapshot(
        md, [(60, U["Infantry"], 0, 1, 1, 100, 99, 0, 0, 0, 0), (70, U["Tank"], 1, 3, 1, 64, 50, 7, 0, 0, 0)]
    )


def test_legal_actions_are_what_apply_takes(md):
    s = fighting(md)
    acts = _core.legal_actions(s)
    assert len(set(acts)) == len(acts)
    assert (A["EndTurn"], -1, None, None, 0) in acts
    assert (A["Wait"], 0, (1, 1), None, 0) in acts
    assert (A["Capture"], 0, (3, 0), None, 0) in acts
    assert (A["Fire"], 0, (3, 0), (3, 1), 0) in acts
    assert (A["Delete"], 0, None, None, 0) in acts
    assert not any(a[0] == A["Wait"] and a[2] == (3, 1) for a in acts)  # the tank's tile
    for a in acts:
        assert _core.apply(s.copy(), *a) == _core.OK, a


def test_micro_steps_pick_a_unit_its_destination_then_its_action(md):
    s = fighting(md)
    sel = _core.Selection()
    ids = _core.legal_ids(s, sel)
    mask = _core.legal_mask(s, sel)
    assert mask.shape == (_core.NUM_ACTION_IDS,) and mask.sum() == len(ids) and mask[ids].all()
    assert list(ids) == sorted(ids)
    assert _core.ID_END_TURN in ids and _core.ID_COP not in ids
    pick = step_id("Select", md, 1, 1)
    assert pick in ids and step_id("Select", md, 3, 1) not in ids  # the enemy's tank
    assert _core.step_action(s, sel, pick) is None
    before = s.to_bytes()
    assert _core.step(s, sel, pick) == _core.OK
    assert (_core.PHASES[sel.phase], sel.unit, sel.tile) == ("selected", 0, md.tile(1, 1))
    assert _core.step(s, sel, _core.ID_END_TURN) != _core.OK  # a unit is picked
    dest = step_id("Move", md, 3, 0)
    assert dest in _core.legal_ids(s, sel) and step_id("Move", md, 3, 1) not in _core.legal_ids(s, sel)
    assert _core.step(s, sel, dest) == _core.OK
    assert s.to_bytes() == before  # picks change only the selection
    ids = set(_core.legal_ids(s, sel).tolist())
    assert ids == {step_id("Wait", md, 3, 0), step_id("Capture", md, 3, 0), step_id("Fire", md, 3, 1)}
    capture = step_id("Capture", md, 3, 0)
    assert _core.step_action(s, sel, capture) == (A["Capture"], 0, (3, 0), None, 0)
    assert _core.step(s, sel, capture) == _core.OK
    assert sel == _core.Selection()
    assert s.unit(0).tile == md.tile(3, 0) and s.unit(0).flags & _core.UNIT_CAPTURING
    assert step_id("Select", md, 3, 0) not in _core.legal_ids(s, sel)  # it has acted


def test_each_actions_steps_play_it(md):
    s = fighting(md)
    for a in _core.legal_actions(s):
        ids = _core.action_steps(s, *a)
        t, u, sel = s.copy(), s.copy(), _core.Selection()
        for i in ids[:-1]:
            assert i in _core.legal_ids(t, sel)
            assert _core.step(t, sel, i) == _core.OK
        assert _core.step_action(t, sel, ids[-1]) == a
        assert _core.step(t, sel, ids[-1]) == _core.apply(u, *a) == _core.OK
        assert t.to_bytes() == u.to_bytes()
    assert _core.action_steps(s, A["Resign"]) == []


def test_state_hash_ignores_slots_ids_and_rng(md):
    inf, tank = (60, U["Infantry"], 0, 1, 1, 100, 99, 0, 0, 0, 0), (70, U["Tank"], 1, 3, 1, 64, 50, 7, 0, 0, 0)
    s = snapshot(md, [inf, tank])
    t = snapshot(md, [inf, (55, *tank[1:])])  # other ids, so other slots (snapshots go by id)
    t.seed(7)
    assert s.to_bytes() != t.to_bytes()
    assert _core.state_hash(s) == _core.state_hash(t)
    moved = s.copy()
    assert _core.apply(moved, A["Wait"], 0, (0, 1), None, 0) == _core.OK
    assert _core.state_hash(moved) != _core.state_hash(s)
    rows = np.stack([np.frombuffer(x.to_bytes(), np.uint8) for x in (s, t, moved)])
    keys = _core.hash_batch(rows)
    assert keys.dtype == np.uint64 and keys.tolist() == [_core.state_hash(x) for x in (s, t, moved)]

    # A decision in progress: its unit by tile, not slot.
    sel, other = _core.Selection(), _core.Selection()
    assert _core.state_hash(s, sel) == _core.state_hash(s)
    assert _core.step(s, sel, step_id("Select", md, 1, 1)) == _core.OK
    assert _core.step(t, other, step_id("Select", md, 1, 1)) == _core.OK
    assert sel.unit != other.unit
    assert _core.state_hash(s, sel) == _core.state_hash(t, other) != _core.state_hash(s)
