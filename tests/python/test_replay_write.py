from _factory import GAME_ID, P1, P2, _snapshot, _unit
from awbw.replay import load_replay
from awbw.replay.write import action_line, php_json, replay_zip


def test_php_json_is_json_encode():
    assert (
        php_json({"a": "x/y", "b": [1, 2.5], "c": "é", "d": None}) == '{"a":"x\\/y","b":[1,2.5],"c":"\\u00e9","d":null}'
    )


def test_a_written_replay_reads_back():
    snaps = [
        _snapshot(1, P1, [_unit(1, P2, 3, 3, 10.0)], (1000, 0)),
        _snapshot(1, P2, [_unit(1, P2, 3, 3, 7.3)], (0, 1000)),  # its comment holds a newline
    ]
    actions = [[{"action": "End", "updatedInfo": {"nextPId": P2}}], ["Array", {"action": "Resign", "m": "a/b é"}]]
    r = load_replay(replay_zip(GAME_ID, snaps, [action_line(P1, 1, actions[0]), action_line(P2, 1, actions[1])]))
    assert r.game.id == GAME_ID and [(t.day, t.player_id) for t in r.turns] == [(1, P1), (1, P2)]
    assert r.turns[1].units[0].hp == 73
    assert r.turns[0].actions == actions[0] and r.turns[1].actions == actions[1][1:]  # the parser drops "Array"
