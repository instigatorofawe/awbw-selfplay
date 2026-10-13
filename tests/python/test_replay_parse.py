import gzip
import io
import zipfile

import pytest

from _factory import GAME_ID, P1, P2, build_replay_zip
from awbw.replay import ReplayFormatError, load_replay


@pytest.fixture(scope="module")
def replay():
    return load_replay(build_replay_zip())


def test_game_settings(replay):
    g = replay.game
    assert (g.id, g.map_id, g.funds_per_property, g.capture_win) == (GAME_ID, 77, 1000, 21)
    assert not g.fog and g.use_powers and g.official and not g.team
    assert g.extra["comment"] == "line one\nline two"


def test_turn_snapshots(replay):
    assert [(t.day, t.player_id) for t in replay.turns] == [(1, P1), (1, P2)]
    assert replay.player_ids == [P1, P2]
    assert replay.days == 1
    second = replay.turns[1]
    assert [p.funds for p in second.players] == [0, 1000]
    assert [u.hp for u in second.units] == [73, 100]  # hit_points 7.3 -> internal 73
    assert second.buildings[0].terrain_id == 34


def test_actions_attach_to_turns(replay):
    first, second = replay.turns
    assert [a["action"] for a in first.actions] == ["Build", "End"]  # "Array" placeholder dropped
    assert [a["action"] for a in second.actions] == ["Resign"]
    assert replay.orphan_action_turns == 1


def test_game_over(replay):
    over = replay.game_over()
    assert (over.winners, over.losers, over.end_action) == ([P1], [P2], "Resign")


def test_replay_without_action_log():
    r = load_replay(build_replay_zip(with_actions=False))
    assert len(r.turns) == 2
    assert all(t.actions is None for t in r.turns)
    assert r.game_over() is None


def test_rejects_non_zip():
    with pytest.raises(ReplayFormatError, match="not a zip"):
        load_replay(b'{"err":true,"message":"Must be logged in to download replay"}')


def test_rejects_zip_without_snapshots():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x", gzip.compress(b"<xml/>"))
    with pytest.raises(ReplayFormatError, match="no awbwGame snapshot"):
        load_replay(buf.getvalue())
