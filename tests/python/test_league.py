import json

import pytest

from awbw.league import parse_rotation

# Shape of AWBW's api/league/maps.php `listMaps` response, trimmed.
RAW = json.dumps(
    {
        "maps": [
            {
                "id": 174402,
                "name": "Water Pressure",
                "type": "std",
                "unitLimit": 50,
                "capLimit": 30,
                "tiers": [
                    {"enabled": False, "name": "T0", "prefs_id": 1, "cos": [{"id": 15}]},
                    {"enabled": True, "name": "T1", "prefs_id": 2, "cos": [{"id": 12}, {"id": 19}]},
                ],
                "unitBans": ["Black Bomb", "Stealth"],
                "unitLabs": [],
            },
            {"id": 9, "name": "Foggy", "type": "fog", "unitLimit": 50, "capLimit": 25, "tiers": [], "unitBans": []},
        ],
        "units": {"1": {"name": "Infantry", "image": "x.gif"}, "968731": {"name": "Black Bomb", "image": "y.gif"}},
        "cos": {"1": {"name": "Andy", "image": "andy.png"}},
    }
).encode()


def test_parse_rotation():
    rot = parse_rotation(RAW, "2026-10-12")
    [m] = rot.mode("std")
    assert (m.id, m.name, m.unit_limit, m.cap_limit) == (174402, "Water Pressure", 50, 30)
    assert m.unit_bans == ("Black Bomb", "Stealth")
    assert [(t.name, t.enabled, t.co_ids) for t in m.tiers] == [("T0", False, (15,)), ("T1", True, (12, 19))]
    assert [x.name for x in rot.mode("fog")] == ["Foggy"]
    assert rot.units == {1: "Infantry", 968731: "Black Bomb"}
    assert rot.cos == {1: "Andy"}


def test_rejects_error_response():
    with pytest.raises(ValueError, match="unexpected league maps response"):
        parse_rotation(b'{"err": true}', "2026-10-12")
