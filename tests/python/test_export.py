import numpy as np

from awbw import _core, engine, rules
from awbw.games import replay
from awbw.replay import load_replay
from awbw.replay.export import FIRST_GAME_ID, ActionLog, GameMeta, action_lines, snapshots
from awbw.replay.step_check import check_turns
from awbw.replay.write import replay_zip
from test_games import _arena_logs
from test_search import md  # noqa: F401 (a fixture)

A = _core.ACTIONS
KEPT = _core.UNIT_HIDDEN | _core.UNIT_CAPTURING | _core.UNIT_CARRIED  # what a turn start's flags say


def _units(s: _core.GameState) -> list[tuple]:
    return sorted((*u[:8], u[8] & KEPT, *u[9:]) for u in s.unit_rows())


def _players(s: _core.GameState) -> list[tuple]:
    return [(p.co, p.country, p.funds, p.power, p.cop_cost, p.scop_cost, p.power_on) for p in s.players]


def test_exported_snapshots_read_back_as_the_game(md):  # noqa: F811
    log = _arena_logs(md, 1)[0]
    meta = GameMeta(id=FIRST_GAME_ID, name="search vs random", players=(11, 12))
    snaps = snapshots(log, meta)
    starts = [log.start] + [m.state for m in replay(log).moves if m.action[0] == _core.ACTIONS["EndTurn"]]
    assert len(snaps) == len(starts) > 2
    r = load_replay(replay_zip(meta.id, snaps, []))
    assert (r.game.id, r.game.name, r.game.map_id, r.player_ids) == (meta.id, meta.name, log.map_id, [11, 12])
    for turn, s in zip(r.turns, starts, strict=True):
        back = engine.state_from_turn(r, turn, md, s.settings)
        assert (back.day, back.active, back.weather) == (s.day, s.active, s.weather)
        assert _units(back) == _units(s)
        assert _players(back) == _players(s)
        assert [(back.prop_owner(p), back.prop_points(p)) for p in range(md.n_props)] == [
            (s.prop_owner(p), s.prop_points(p)) for p in range(md.n_props)
        ]


def test_exported_actions_step_as_the_game(md):  # noqa: F811
    """The log steps through every turn to the next snapshot, as a real replay's does."""
    for log in _arena_logs(md, 2):
        meta = GameMeta(id=FIRST_GAME_ID, name="search vs random", players=(11, 12))
        lines = action_lines(log, meta)
        snaps = snapshots(log, meta)
        # A game stopped at the day limit ends on a snapshot with nothing logged yet.
        assert len(lines) == len(snaps) - (log.decided == "day limit")
        r = load_replay(replay_zip(meta.id, snaps, lines))
        results = check_turns(r, md)
        assert len(results) == len(r.turns)
        bad = [(t.day, t.category, t.detail) for t, _ in results[:-1] if not t.ok]
        assert not bad, bad[:3]


def test_launch_and_explode_log_as_the_replay_player_reads_them():
    """No std replay launches a missile or explodes a Black Bomb (banned), so the
    shapes come from the Replay Player's parser and the one old Explode logged."""
    r = rules.load()
    u = {t.name: t.index for t in r.units}
    terrain = np.array([[42, 1, 1, 1, 47], [1, 1, 111, 1, 1], [1, 1, 1, 1, 1]], dtype=np.int16)
    silo_map = _core.register_map(910101, terrain)
    players = np.array([[1, 1, 0, 0, 270000, 630000, 0, 0], [7, 2, 0, 0, 270000, 540000, 0, 0]])
    buildings = np.array([[0, 0, 42, 20], [4, 0, 47, 20], [2, 1, 111, 20]])
    rows = np.array(
        [
            (10, u["Infantry"], 0, 1, 1, 100, 99, 0, 0, 0, 0),
            (11, u["Black Bomb"], 0, 0, 2, 100, 45, 0, 0, 0, 0),
            (20, u["Tank"], 1, 4, 1, 100, 70, 9, 0, 0, 0),
        ],
        dtype=np.int64,
    )
    start = _core.load_snapshot(silo_map, _core.Settings(), 2, 0, players, buildings, rows)
    meta = GameMeta(id=FIRST_GAME_ID, name="silo")
    log = ActionLog(silo_map, meta, {(0, 0): 1, (2, 1): 2, (4, 0): 3})

    launched = start.copy()
    assert _core.apply(launched, A["Launch"], unit=launched.find_unit(10), to=(2, 1), target=(4, 1)) == _core.OK
    (entry,) = log.entries(start, launched, (A["Launch"], start.find_unit(10), (2, 1), (4, 1), 0), _core.OK)
    assert entry["Move"]["paths"]["global"] == [{"unit_visible": True, "x": x, "y": 1} for x in (1, 2)]
    assert entry["Launch"] == {"action": "Launch", "siloX": 2, "siloY": 1, "targetX": 4, "targetY": 1, "hp": -3}

    exploded = start.copy()
    assert _core.apply(exploded, A["Explode"], unit=exploded.find_unit(11), to=(1, 2)) == _core.OK
    (entry,) = log.entries(start, exploded, (A["Explode"], start.find_unit(11), (1, 2), None, 0), _core.OK)
    assert entry["Explode"] == {"action": "Explode", "hp": -5, "vision": {"global": {"x": 1, "y": 2}}, "unitId": 11}
    assert entry["Move"]["dist"] == 1
