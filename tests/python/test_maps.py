import json

import numpy as np

from awbw.maps import parse_map_info


def test_terrain_is_converted_from_column_major():
    raw = json.dumps(
        {
            "Name": "tiny",
            "Author": "me",
            "Player Count": 2,
            "Size X": 3,
            "Size Y": 2,
            "Terrain Map": [[1, 2], [3, None], [28, 34]],  # [x][y]
            "Predeployed Units": [{"Unit ID": 1, "Unit X": 2, "Unit Y": 1, "Unit HP": 10, "Country Code": "os"}],
        }
    ).encode()
    m = parse_map_info(5, raw)
    assert (m.width, m.height) == (3, 2)
    np.testing.assert_array_equal(m.terrain, [[1, 3, 28], [2, 0, 34]])
    assert m.predeployed[0].x == 2 and m.predeployed[0].country_code == "os"
