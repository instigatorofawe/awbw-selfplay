import pytest

from awbw.replay.stats import ReplaySummary, v1_rejection


def standard(**overrides) -> ReplaySummary:
    s = ReplaySummary(
        path="x.zip",
        players=2,
        weather_type="Clear",
        funds_per_property=1000,
        capture_win=26,
        width=20,
        height=18,
        end_action="Resign",
        days=14,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_standard_game_is_in_scope():
    assert v1_rejection(standard()) is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"error": "ReplayFormatError: x"}, "parse error"),
        ({"players": 4}, "not 2 players"),
        ({"tag": True}, "tag"),
        ({"team": True}, "team game"),
        ({"fog": True}, "fog"),
        ({"weather_type": "Random"}, "weather Random"),
        ({"use_powers": False}, "powers off"),
        ({"funds_per_property": 2000}, "funds/property 2000"),
        ({"starting_funds": 5000}, "starting funds"),
        ({"width": 0, "height": 0}, "map not cached"),
        ({"width": 41}, "map too large"),
        ({"end_action": None}, "no game-over in log"),
        ({"turns_without_actions": 1}, "turns missing actions"),
        ({"days": 5}, "under 6 days"),
    ],
)
def test_out_of_scope(overrides, reason):
    assert v1_rejection(standard(**overrides)) == reason


def test_minimum_length_is_inclusive():
    assert v1_rejection(standard(days=6)) is None
