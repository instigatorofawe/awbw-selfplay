from awbw import _core, engine
from awbw.games import replay
from awbw.replay import load_replay
from awbw.replay.export import FIRST_GAME_ID, GameMeta, action_lines, snapshots
from awbw.replay.step_check import check_turns
from awbw.replay.write import replay_zip
from test_games import _arena_logs
from test_search import md  # noqa: F401 (a fixture)

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
