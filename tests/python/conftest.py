import json

import pytest

from _factory import GAME_ID, MAP_ID, build_replay_zip
from awbw.replay import stats


@pytest.fixture
def replay_cache(tmp_path, monkeypatch):
    """A cache with the factory replay and its 5x5 map, and no minimum game length.
    Returns the replay's path."""
    monkeypatch.setenv("AWBW_CACHE", str(tmp_path))
    monkeypatch.setattr(stats, "MIN_DAYS", 1)
    terrain = [[34 if (x, y) == (1, 2) else 1 for y in range(5)] for x in range(5)]  # [x][y]
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / f"{MAP_ID}.json").write_text(
        json.dumps({"Name": "t", "Size X": 5, "Size Y": 5, "Terrain Map": terrain, "Predeployed Units": []})
    )
    (tmp_path / "replays").mkdir()
    path = tmp_path / "replays" / f"{GAME_ID}.zip"
    path.write_bytes(build_replay_zip())
    return path
