from awbw import _core


def test_extension_limits():
    assert _core.MAX_TILES == _core.MAX_WIDTH * _core.MAX_HEIGHT
    assert _core.MAX_PLAYERS == 2
