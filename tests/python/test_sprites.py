import json
import struct
import zlib

import pytest

from awbw import engine, sprites, viewer
from test_search import md  # noqa: F401 (a fixture: a cache without a copy)

MAP_ID = 81
# An Orange Star city on the coast, sea, a reef and a shoal, a Blue Moon city and a pipe seam.
TERRAIN = [
    [38, 28, 28, 28, 1],
    [1, 29, 28, 33, 1],
    [1, 1, 43, 113, 1],
    [1, 1, 1, 1, 1],
]


def _png(w: int, h: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + b"\x10\x20\x30\xff" * w for _ in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _textures(names: dict[str, tuple[int, int]], base) -> None:
    for name, (w, h) in names.items():
        path = base / "Textures" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png(w, h))


def _tile(awbw_id: int, kind: str, clear: str, rain: str | None = None) -> dict:
    textures = {"Clear": clear} | ({"Rain": rain} if rain else {})
    return {"AWBWID": awbw_id, "TerrainType": kind, "Textures": textures}


@pytest.fixture
def copy(tmp_path, monkeypatch):
    """A cache with the map and a small copy of the Player's art: the index
    files (with the Player's comments and trailing commas) and stand-in PNGs."""
    monkeypatch.setenv("AWBW_CACHE", str(tmp_path))
    (tmp_path / "maps").mkdir()
    info = {"Name": "coast", "Size X": 5, "Size Y": 4, "Predeployed Units": []}
    info["Terrain Map"] = [list(col) for col in zip(*TERRAIN, strict=True)]  # [x][y]
    (tmp_path / "maps" / f"{MAP_ID}.json").write_text(json.dumps(info))
    base = tmp_path / "sprites"
    tiles = {
        "Plain": _tile(1, "Plain", "Plain", "Plain_Rain"),
        "Sea": _tile(28, "Sea", "Sea/Sea"),
        "Reef": _tile(33, "Sea", "Reef"),
        "HPipe Rubble": _tile(115, "Plain", "Neutral/HRubble-0"),
        "Sea-W": _tile(1030, "Sea", "Sea/W"),
        "Sea-E": _tile(1022, "Sea", "Sea/E"),
        "Sea-S": _tile(1026, "Sea", "Sea/S"),
        "Shoal-AN-AE-S-W": _tile(1100, "Shoal", "Shoal/AN-AE-S-W"),
    }
    buildings = {
        "Neutral City": {"AWBWID": 34, "Textures": {"Clear": "Neutral/City"}},
        "Orange Star City": {"AWBWID": 38, "frames": [300, 1200, 100], "Textures": {"Clear": "OrangeStar/City"}},
        "Blue Moon City": {"AWBWID": 43, "frames": [300, 1200, 100], "Textures": {"Clear": "BlueMoon/City"}},
        "Horizontal Pipe Seam": {"AWBWID": 113, "Textures": {"Clear": "Neutral/HSeam"}},
    }
    walk = {"Texture": "Infantry_MSide", "Frames": [120, 120, 120]}
    units = {"Infantry": {"AWBWId": 1, "IdleAnimation": {"Texture": "Infantry", "Frames": [160, 160, 320, 160]}}}
    units["Infantry"]["MoveSideAnimation"] = walk
    countries = {
        "Orange Star": {"AWBWID": 1, "unitPath": "Units/OrangeStar", "faceDirection": "Right"},
        "Blue Moon": {"AWBWID": 2, "unitPath": "Units/BlueMoon", "faceDirection": "Left"},
    }
    (base / "Json").mkdir(parents=True)
    for name, data in (("Tiles", tiles), ("Buildings", buildings), ("Units", units), ("Countries", countries)):
        text = json.dumps(data, indent=1)
        text = text.replace("{\n", '{\n // a "comment" with // inside\n /* and a block */\n', 1)
        (base / "Json" / f"{name}.json").write_text(text[:-2] + ",\n}")  # a trailing comma
    names = {f"Map/Classic/{t['Textures']['Clear']}": (16, 16) for t in tiles.values()}
    names |= {
        "Map/Classic/Plain_Rain": (16, 16),
        "Map/AW2/Neutral/City-0": (16, 22),
        "Map/AW2/Neutral/HSeam-0": (16, 16),
    }
    names |= {f"Map/AW2/{c}/City-{i}": (16, 22) for c in ("OrangeStar", "BlueMoon") for i in range(3)}
    names |= {f"Units/{c}/Infantry-{i}": (16, 16) for c in ("OrangeStar", "BlueMoon") for i in range(4)}
    names |= {f"Units/OrangeStar/Infantry_MSide-{i}": (15, 17) for i in range(3)}
    names |= {f"UI/Healthv2/{n}": (8, 7) for n in range(10)} | {"UI/Capturing": (8, 8)}
    _textures(names, base)
    return engine.map_data(MAP_ID)


def test_coast_pieces_follow_the_neighbours():
    sea = dict.fromkeys(("N", "E", "S", "W", "NE", "SE", "SW", "NW"), "sea")
    assert sprites.coast_tokens("sea", sea) == frozenset()
    # land on a corner shows only where both sides next to it are water, shoal or reef or a bridge too
    assert sprites.coast_tokens("sea", sea | {"NE": "plain", "N": "shoal", "E": "bridge"}) == {"NE"}
    assert sprites.coast_tokens("sea", sea | {"NE": "plain", "N": "plain"}) == {"N"}
    assert sprites.coast_tokens("sea", sea | {"SW": "wood", "S": "reef", "W": None}) == frozenset()  # off the map
    assert sprites.coast_tokens("sea", sea | {"E": "city", "W": "road", "NW": "plain", "SE": "plain"}) == {"E", "W"}
    shoal = {"N": "plain", "E": "bridge", "S": "sea", "W": None} | dict.fromkeys(("NE", "SE", "SW", "NW"), "plain")
    assert sprites.coast_tokens("shoal", shoal) == {"N", "E", "AS", "AW"}
    assert sprites.coast_tokens("shoal", shoal | {"N": "shoal", "S": "reef"}) == {"E", "AS", "AW"}


def test_no_copy_no_sprites(md):  # noqa: F811
    assert sprites.for_board(md, (1, 2)) is None


def test_a_board_takes_its_sprites_from_the_copy(copy):
    coast = copy
    s = sprites.for_board(coast, (1, 2), weathers=(0,))
    assert s is not None
    key = lambda x, y: s["ground"][y * coast.width + x][0][0]  # noqa: E731 (the clear texture)
    assert [key(x, 0) for x in range(4)] == [f"Map/Classic/{t}" for t in ("Plain", "Sea/W", "Sea/Sea", "Sea/E")]
    assert [key(x, 1) for x in range(1, 4)] == [f"Map/Classic/{t}" for t in ("Shoal/AN-AE-S-W", "Sea/S", "Reef")]
    assert s["ground"][0] == s["plain"] and s["plain"][1] == -1  # buildings stand on plain
    assert s["plain"][0] == ["Map/Classic/Plain", "Map/Classic/Plain_Rain", "Map/Classic/Plain"]  # no snow: clear
    city, bm_city, seam = s["props"]  # in the map's order
    assert city["n"] == [["Map/AW2/Neutral/City"] * 3, s["anim"].index([1000])]
    assert city["0"][0][0] == "Map/AW2/OrangeStar/City" and s["anim"][city["0"][1]] == [300, 1200, 100]
    assert city["1"][0][0] == bm_city["1"][0][0] == "Map/AW2/BlueMoon/City"
    assert seam["n"] == [["Map/AW2/Neutral/HSeam"] * 3, s["anim"].index([1000])]
    assert seam["x"] == [["Map/Classic/Neutral/HRubble-0"] * 3, -1]  # broken
    infantry = s["units"][0][0]
    assert infantry[0] == ["Units/OrangeStar/Infantry"] and s["anim"][infantry[1]] == [160, 160, 320, 160]
    assert s["units"][1][0][0] == ["Units/BlueMoon/Infantry"] and all(u is None for u in s["units"][0][1:])
    assert s["moves"] is None and s["ui"]["capture"] == "UI/Capturing" and s["ui"]["dive"] is None
    assert s["pad"] == 6  # the city on the top row is 22 high
    assert "Map/Classic/Plain_Rain" not in s["img"] and "Map/AW2/OrangeStar/City-2" in s["img"]
    assert all(v.startswith("data:image/png;base64,") for v in s["img"].values())

    rain = sprites.for_board(coast, (1, 2), weathers=(0, 1), moves=True)
    assert rain is not None and "Map/Classic/Plain_Rain" in rain["img"]
    assert rain["moves"][0][0]["S"] == [["Units/OrangeStar/Infantry_MSide"], rain["anim"].index([120, 120, 120])]
    assert rain["moves"][1][0]["S"] is None and rain["moves"][0][0]["U"] is None


def test_pages_carry_the_sprites(copy):
    s = engine.new_game(MAP_ID, ("Andy", "Max"))
    data = viewer.board_data(copy, s, ("a", "b"))
    assert data["sprites"] is not None and data["sprites"]["units"][1][0][0] == ["Units/BlueMoon/Infantry"]
    page = viewer.page("viewer.html", data)
    assert "data:image/png;base64," in page and "drawUnitSprite" in page
