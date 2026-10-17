import json
from pathlib import Path

import numpy as np

from awbw import _core, engine, games, rules
from awbw.agents import RandomAgent, play
from awbw.cli import main
from awbw.games import replay
from awbw.nn.train_selfplay import Games
from awbw.replay import load_replay
from awbw.replay.export import (
    FIRST_GAME_ID,
    NO_USER,
    ActionLog,
    GameMeta,
    action_lines,
    game_meta,
    made_up_user,
    player_storage,
    replay_file,
    snapshots,
)
from awbw.replay.step_check import _TurnStepper, check_turns
from awbw.replay.write import replay_zip
from awbw.search import NetEvaluator
from test_games import _arena_logs
from test_policy import start
from test_search import md  # noqa: F401 (a fixture)
from test_selfplay import _net

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


def _rebuilt(md, s: _core.GameState, players=None, settings=None) -> _core.GameState:  # noqa: F811
    """A fresh start ``s`` again, with these players' rows or settings."""
    if players is None:
        players = [[p.co, p.country, p.funds, p.power, p.cop_cost, p.scop_cost, 0, 0] for p in s.players[:2]]
    terrain = md.terrain()  # a fresh start's owners are the map's
    props = [(x, y, int(terrain[y, x]), s.prop_points(i)) for i, (x, y, _, _) in enumerate(md.props())]
    rows = np.array(s.unit_rows(), dtype=np.int64).reshape(-1, 11)
    settings = settings or s.settings
    return _core.load_snapshot(md, settings, int(s.day), int(s.active), np.array(players), np.array(props), rows)


def _with_cos(md, s: _core.GameState, cos: tuple[str, str]) -> _core.GameState:  # noqa: F811
    """``s`` with these COs, their meters full."""
    by_name = {c.name: c for c in rules.load().cos.values()}
    players = []
    for p, name in zip(s.players[:2], cos, strict=True):
        c = by_name[name]
        cop, scop = (c.cop_stars or 0) * _core.STAR_VALUE, (c.scop_stars or 0) * _core.STAR_VALUE
        players.append([c.id, p.country, p.funds, scop, cop, scop, 0, 0])
    return _rebuilt(md, s, players)


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


def _bad_turns(r, md) -> list[tuple]:  # noqa: F811
    results = check_turns(r, md)
    assert len(results) == len(r.turns)
    return [(t.day, t.category, t.detail) for t, _ in results[:-1] if not t.ok]


def test_self_play_games_export_and_step_as_played(md):  # noqa: F811
    """Self-play records, stopped at the day cap or resigned, export under a
    stable id and name, read back and step exactly every turn."""
    s = start(md)
    starts = np.frombuffer(s.to_bytes(), np.uint8)[None]
    capped = _core.SearchConfig(sims=4, considered=4, max_days=3, seed=1)
    resigning = _core.SearchConfig(sims=4, considered=4, max_days=6, resign_turns=2, resign_off=0, seed=1)

    def lost(planes, globals_, legal, movers):  # player 0 sees every position as lost
        return np.zeros(legal.shape, np.float32), np.where(movers == 0, -1.0, 1.0).astype(np.float32)

    logs = []
    for evaluate, cfg in ((NetEvaluator(_net()), capped), (lost, resigning)):
        played = Games(evaluate, starts, np.random.default_rng(0), cfg, 2)  # type: ignore[arg-type]
        done = []
        while len(done) < 2:
            done += played.round()
        logs += [games.from_record(g.record, ("self-play step 7", "self-play step 7"), step=7) for g in done]
    assert {log.decided for log in logs} == {"day limit", "resigned"}
    ids = set()
    for log in logs:
        meta = game_meta(log)
        assert FIRST_GAME_ID <= meta.id < 2 * FIRST_GAME_ID and game_meta(log).id == meta.id
        ids.add(meta.id)
        assert meta.name == "self-play step 7: Andy vs Max"
        r = load_replay(replay_file(log, meta))
        assert (r.game.id, r.game.name, r.game.map_id) == (meta.id, meta.name, log.map_id)
        assert not _bad_turns(r, md)
        last = (r.turns[-1].actions or [{}])[-1]
        if log.decided == "resigned":
            assert last["action"] == "Resign" and last["Resign"]["message"] == "Player 1 (Andy) has resigned!"
        else:  # stopped after an End Turn: the last snapshot has nothing logged
            assert r.turns[-1].actions is None or not r.turns[-1].actions
    assert len(ids) == len(logs)


def test_a_game_its_day_limit_decides_ends_at_the_last_end_turn(md):  # noqa: F811
    """AWBW's day limit decides the game at End Turn: the End logs a GameOver,
    and no snapshot follows it."""
    settings = _core.Settings()
    settings.day_limit = 2
    s = _rebuilt(md, start(md), settings=settings)
    (result,) = play([s], (RandomAgent(1), RandomAgent(2)), max_days=30)
    assert result.decided == "game over" and result.days == 2
    log = games.from_result(result, ("random", "random"))
    meta = game_meta(log)
    snaps, lines = snapshots(log, meta), action_lines(log, meta)
    assert len(snaps) == len(lines) == 4  # two days of two turns
    r = load_replay(replay_zip(meta.id, snaps, lines))
    end = (r.turns[-1].actions or [{}])[-1]
    elim = {str(meta.players[p]): "Y" if result.winner == 1 - p else "N" for p in (0, 1)}
    assert end["action"] == "End" and end["updatedInfo"]["event"] == "GameOver"
    assert end["updatedInfo"]["playersElim"] == elim
    assert not _bad_turns(r, md)
    # A snapshot has no day limit, so the last turn steps to the result only
    # from a state that has it.
    _TurnStepper(r, md, r.turns[-1], None).run(engine.state_from_turn(r, r.turns[-1], md, settings))


def test_export_replay_writes_a_zip_per_game(md, tmp_path, capsys):  # noqa: F811
    record = tmp_path / "games.jsonl"
    logs = _arena_logs(md, 2)
    assert games.append(record, logs) == 2
    assert main(["export-replay", str(record)]) == 0
    one = tmp_path / "games-1.zip"
    assert load_replay(one.read_bytes()).game.name == "search (Andy) vs random (Max)"
    out = tmp_path / "zips"
    assert main(["export-replay", str(record), "--all", "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["games-0.zip", "games-1.zip"]
    assert (out / "games-1.zip").read_bytes() == one.read_bytes()
    assert main(["export-replay", str(record), "--game", "0", "--out", str(tmp_path / "first.zip")]) == 0
    assert (tmp_path / "first.zip").read_bytes() == (out / "games-0.zip").read_bytes()
    assert "games-0.zip: game 1" in capsys.readouterr().out


def test_export_replay_names_the_players(md, tmp_path, capsys):  # noqa: F811
    """An AWBW user goes in by their user id, which the Replay Player looks up;
    anyone else by an id made up from their name, which goes into the Player's
    list of usernames."""
    record = tmp_path / "games.jsonl"
    games.append(record, _arena_logs(md, 1))
    data = tmp_path / "player"
    zip_ = tmp_path / "named.zip"
    args = ["export-replay", str(record), "--out", str(zip_), "--player-data", str(data)]
    assert main([*args, "--players", "Absox=361968", "tf128-s1"]) == 0
    bot = made_up_user("tf128-s1")
    assert bot < -1 and made_up_user("tf128-s1") == bot != made_up_user("tf128-s2")
    assert json.loads((data / "UsernameStorage.json").read_text()) == {str(bot): "tf128-s1"}
    r = load_replay(zip_.read_bytes())
    assert r.game.name == "Absox (Andy) vs tf128-s1 (Max)"
    assert [p.users_id for p in r.turns[0].players] == [361968, bot]
    assert not _bad_turns(r, md)
    (data / "UsernameStorage.json").write_text(json.dumps({"7": "someone"}))
    assert main([*args, "--players", "-", "tf128-s1"]) == 0  # adds to the list, keeps the record's label
    assert json.loads((data / "UsernameStorage.json").read_text()) == {"7": "someone", str(bot): "tf128-s1"}
    r = load_replay(zip_.read_bytes())
    assert r.game.name == "search (Andy) vs tf128-s1 (Max)"
    assert [p.users_id for p in r.turns[0].players] == [NO_USER, bot]
    assert main([*args, "--players", "Absox=x", "-"]) == 1
    assert "NAME=ID" in capsys.readouterr().out


def test_the_replay_players_data_is_where_its_storage_ini_says(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    data = tmp_path / "AWBWReplayPlayer"
    assert player_storage() == data / "ReplayData" / "Replays"
    data.mkdir()
    (data / "storage.ini").write_text("FullPath = \n")  # as it writes it by default
    assert player_storage() == data / "ReplayData" / "Replays"
    (data / "storage.ini").write_text("FullPath = /elsewhere/awbw data\n")
    assert player_storage() == Path("/elsewhere/awbw data/ReplayData/Replays")
