import numpy as np

from awbw import _core, engine, games, rules
from awbw.agents import RandomAgent, play
from awbw.games import replay
from awbw.replay import load_replay
from awbw.replay.export import FIRST_GAME_ID, ActionLog, GameMeta, action_lines, snapshots
from awbw.replay.step_check import check_turns
from awbw.replay.write import replay_zip
from test_games import _arena_logs
from test_policy import start
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


class _PowerFirst(RandomAgent):
    """Random, but uses a power whenever it can (``level`` 1 COP, 2 SCOP)."""

    def __init__(self, level: int, seed: int = 0):
        super().__init__(seed)
        self.power = _core.ID_COP if level == 1 else _core.ID_SCOP

    def choose(self, decisions):
        return [
            self.power if self.power in d.legal else c
            for d, c in zip(decisions, super().choose(decisions), strict=True)
        ]


# CO pairs whose powers change units, HP, meters or the weather. Not Colin: a
# snapshot doesn't hold Power of Money's attack bonus, so the step check can't
# rebuild the turns after it.
POWER_PAIRS = [
    ("Rachel", "Von Bolt"),
    ("Sturm", "Kindle"),
    ("Jess", "Sami"),
    ("Max", "Jake"),
    ("Grit", "Eagle"),
    ("Hawke", "Drake"),
    ("Olaf", "Sasha"),
    ("Andy", "Adder"),
    ("Koal", "Javier"),
    ("Sensei", "Lash"),
]


def _with_cos(md, s: _core.GameState, cos: tuple[str, str]) -> _core.GameState:  # noqa: F811
    """``s`` with these COs, their meters full."""
    by_name = {c.name: c for c in rules.load().cos.values()}
    players = []
    for p, name in zip(s.players[:2], cos, strict=True):
        c = by_name[name]
        cop, scop = (c.cop_stars or 0) * _core.STAR_VALUE, (c.scop_stars or 0) * _core.STAR_VALUE
        players.append([c.id, p.country, p.funds, scop, cop, scop, 0, 0])
    terrain = md.terrain()  # a fresh start's owners are the map's
    props = [(x, y, int(terrain[y, x]), s.prop_points(i)) for i, (x, y, _, _) in enumerate(md.props())]
    rows = np.array(s.unit_rows(), dtype=np.int64).reshape(-1, 11)
    return _core.load_snapshot(md, s.settings, int(s.day), int(s.active), np.array(players), np.array(props), rows)


def test_exported_powers_step_as_the_game(md):  # noqa: F811
    """Games where every CO above uses a power, one each level: the exported log
    steps through every turn to the next snapshot."""
    used = set()
    for i, pair in enumerate(POWER_PAIRS):
        for level in (1, 2):
            s = _with_cos(md, start(md), pair)
            (result,) = play([s], (_PowerFirst(level, seed=i), _PowerFirst(3 - level, seed=i)), max_days=3)
            log = games.from_result(result, ("power", "power"), seed=i)
            meta = GameMeta(id=FIRST_GAME_ID + i, name=" vs ".join(pair), players=(11, 12))
            r = load_replay(replay_zip(meta.id, snapshots(log, meta), action_lines(log, meta)))
            used |= {(e["coName"], e["coPower"]) for t in r.turns for e in t.actions or [] if e["action"] == "Power"}
            results = check_turns(r, md)
            bad = [(pair, t.day, t.category, t.detail) for t, _ in results[:-1] if not t.ok]
            assert not bad, bad[:3]
    has_cop = {c.name for c in rules.load().cos.values() if c.cop_stars}
    assert used == {(name, lv) for pair in POWER_PAIRS for name in pair for lv in "YS" if lv == "S" or name in has_cop}


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


def test_colin_and_sensei_powers_log_as_the_replay_player_reads_them():
    """No cached replay has Colin's or Sensei's powers: the shapes are the Replay Player's."""
    r = rules.load()
    co = {c.name: c.id for c in r.cos.values()}
    terrain = np.array([[42, 38, 38, 1, 47], [1, 1, 1, 1, 1]], dtype=np.int16)
    city_map = _core.register_map(910102, terrain)
    buildings = np.array([[0, 0, 42, 20], [1, 0, 38, 20], [2, 0, 38, 20], [4, 0, 47, 20]])
    rows = np.array(
        [
            (10, r.unit("Infantry").index, 0, 1, 0, 100, 99, 0, 0, 0, 0),
            (20, r.unit("Tank").index, 1, 4, 1, 100, 70, 9, 0, 0, 0),
        ],
        dtype=np.int64,
    )
    meta = GameMeta(id=FIRST_GAME_ID, name="powers", players=(11, 12))
    log = ActionLog(city_map, meta, {(0, 0): 1, (1, 0): 2, (2, 0): 3, (4, 0): 4})
    entries = {}
    for name in ("Colin", "Sensei"):
        players = np.array([[co[name], 1, 10000, 60000, 18000, 54000, 0, 0], [co["Andy"], 2, 0, 0, 27000, 63000, 0, 0]])
        start = _core.load_snapshot(city_map, _core.Settings(), 2, 0, players, buildings, rows)
        after = start.copy()
        assert _core.apply(after, A["Power"], arg=1) == _core.OK
        (entries[name],) = log.entries(start, after, (A["Power"], -1, None, None, 1), _core.OK)
        assert entries[name]["playersCOP"] == 42000
    assert entries["Colin"]["playerReplace"] == {"global": {"11": {"players_funds": 15000}}}
    (new,) = entries["Sensei"]["unitAdd"]["global"]["units"]  # on the empty city only
    assert entries["Sensei"]["unitAdd"]["global"] == {
        "playerId": 11,
        "unitName": "Infantry",
        "units": [{"units_id": new["units_id"], "units_x": 2, "units_y": 0}],
    }
