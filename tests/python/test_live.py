import json
from typing import Any

import numpy as np
import pytest

from awbw import _core, engine, live, rules
from awbw.data.loader import encode
from awbw.maps import MapInfo
from awbw.net import Http

R = rules.load()
GLOBAL = {name: i for i, name in enumerate(_core.GLOBAL_NAMES)}
UNIT_PLANE = [name for name, _, _ in _core.PLANES].index("unit")
OLAF = next(c.id for c in R.cos.values() if c.name == "Olaf")
SONJA = next(c.id for c in R.cos.values() if c.name == "Sonja")
MAP_ID = 910301
OS, BM = 101, 102  # player ids; Orange Star moves first

# Orange Star HQ top left, Blue Moon HQ bottom right, sea along the bottom row.
TERRAIN = np.array(
    [[42, 1, 1, 1, 1, 1], [1, 1, 1, 34, 1, 1], [1, 1, 1, 1, 1, 1], [28, 28, 28, 28, 28, 47]], dtype=np.int16
)


@pytest.fixture(scope="module")
def info():
    m = MapInfo(MAP_ID, "t", "", 2, 6, 4, TERRAIN, [])
    engine.register(m)
    return m


def page(**over) -> str:
    js = {
        "gameId": 555,
        "gameCop": "Y",
        "gameFog": "N",
        "gameDay": 3,
        "gameEndDate": "",
        "mapId": MAP_ID,
        "playersInfo": {
            str(pid): {"players_id": pid, "users_username": name, "players_order": order}
            for pid, name, order in ((BM, "blue", 19), (OS, "orange", 4))
        },
        "unitsInfo": {"9": {"units_players_id": BM, "units_moved": 1}, "8": {"units_players_id": OS, "units_moved": 1}},
        "banUnits": {"Black Bomb": 1},
        "gameCaptureLimit": 1000,
        "currentTurn": BM,
    } | over
    lines = [f"  {'let' if i % 2 else 'const'} {k} = {json.dumps(v)};" for i, (k, v) in enumerate(js.items())]
    rows = [("Weather", "Clear"), ("Fog", "Off"), ("Funds", "1000"), ("Tags", "Off"), ("Day Limit", "Off")]
    rows.append(("Unit Limit", "50"))
    table = "".join(f"<tr><td class=small_text>{k}:</td><td class=small_text>{v}</td></tr>\n" for k, v in rows)
    return "<title>Game - test game - AWBW</title>\n" + table + "<script>\n" + "\n".join(lines) + "\n</script>\n"


def player(pid, country, co, order, power_on="N"):
    return {
        "players_id": pid,
        "users_id": pid + 1000,
        "players_countries_id": country,
        "players_co_id": co,
        "players_order": order,
        "players_team": str(pid),
        "players_funds": 5000,
        "players_eliminated": "N",
        "co_max_power": 1,  # the CO's base; the player's own meter is below
        "players_co_power": 10000,
        "players_co_power_on": power_on,
        "players_co_max_power": 270000,
        "players_co_max_spower": 630000,
    }


def unit(uid, pid, name, x, y, hp, **over):
    t = R.unit(name)
    return {
        "units_id": uid,
        "units_players_id": pid,
        "units_name": name,
        "units_movement_points": t.move,
        "units_vision": t.vision,
        "units_fuel": t.fuel,
        "units_fuel_per_turn": t.fuel_per_day,
        "units_sub_dive": "N",
        "units_ammo": t.ammo,
        "units_short_range": t.min_range,
        "units_long_range": t.max_range,
        "units_second_weapon": "N",
        "units_cost": t.cost,
        "units_movement_type": t.move_type,
        "units_x": x,
        "units_y": y,
        "units_moved": 0,
        "units_capture": 0,
        "units_fired": 0,
        "units_hit_points": hp,
        "units_cargo1_units_id": 0,
        "units_cargo2_units_id": 0,
        "units_carried": "N",
    } | over


def turn_data(units, counts, *, os_co=OLAF, power_on="Y", weather="S", to_move=BM) -> dict[str, Any]:
    def building(bid, x, y, terrain, owner):
        b = {"buildings_id": bid, "buildings_x": x, "buildings_y": y, "terrain_id": str(terrain)}
        return b | {"buildings_capture": 20, "buildings_last_capture": 20, "buildings_players_id": owner}

    return {
        "day": 3,
        "gameState": {
            "players": {str(OS): player(OS, 1, os_co, 4, power_on), str(BM): player(BM, 2, 1, 19)},
            "buildings": {
                "0": {"0": building(1, 0, 0, 42, OS)},
                "3": {"1": building(2, 3, 1, 34, "")},
                "5": {"3": building(3, 5, 3, 47, BM)},
            },
            "units": {str(u["units_id"]): u for u in units},
            "players_units_count": counts,
            "gameWeather": {"code": weather, "name": "clear"},
            "currentTurnPId": to_move,
        },
    }


def test_game_id_from():
    assert live.game_id_from("1741633") == 1741633
    assert live.game_id_from("https://awbw.amarriner.com/game.php?games_id=1741633&ndx=3") == 1741633
    with pytest.raises(ValueError):
        live.game_id_from("game")


def test_parse_page():
    g = live.parse_page(page())
    assert (g.game.id, g.game.name, g.game.map_id, g.day, g.to_move) == (555, "test game", MAP_ID, 3, BM)
    assert g.order == (OS, BM) and g.names == {OS: "orange", BM: "blue"}
    assert g.acted == 1  # Blue Moon's unit; Orange Star's moved on its own turn
    assert (g.game.funds_per_property, g.game.fog, g.game.use_powers, g.tags) == (1000, False, True, False)
    assert (g.day_limit, g.unit_limit, g.bans, g.ended) == (0, 50, ("Black Bomb",), False)
    assert g.settings().banned_units == 1 << R.unit("Black Bomb").index and g.settings().unit_limit == 50
    g.check_supported()
    with pytest.raises(live.Unsupported, match="fog"):
        live.parse_page(page(gameFog="Y")).check_supported()
    with pytest.raises(live.LiveUnavailable):
        live.parse_page("<html>no game here</html>")


def test_hp_from_display():
    assert [live.hp_from_display(d) for d in (1, 7, 10, "10")] == [6, 66, 100, 100]
    assert live.hp_from_display(7.3) == 73  # an exact value, when the site gives one
    assert live.hp_from_display("?") is None


def test_live_turn_puts_back_hidden_units(info):
    units = [unit(11, OS, "Infantry", 1, 0, 7), unit(12, BM, "Tank", 4, 2, 10)]
    stealth = R.unit("Stealth").cost
    counts: dict[str, Any] = {
        str(OS): {"total": 2, "value": 700 + stealth * 8 // 10, "count": {"Infantry": 1, "Stealth": 1}},
        str(BM): {"total": 1, "value": 7000, "count": {"Tank": 1}},
    }
    lt = live.live_turn(4, turn_data(units, counts), info)
    assert lt.estimated_hp == 1 and lt.unknown_hp == 0 and not lt.unplaced
    (back,) = lt.put_back
    assert (back.name, back.players_id, back.hp, back.sub_dive) == ("Stealth", OS, 76, "Y")
    assert abs(back.x - 4) + abs(back.y - 2) > 1  # out of the Tank's sight
    assert lt.turn.weather_start == OS  # Olaf's power is on and it snows

    g = live.parse_page(page())
    s = live.state(g, lt)
    assert s.active == 1 and s.weather_owner == 0 and s.unit(s.unit_at(1, 0)).hp == 66
    assert s.hp_belief(s.unit_at(1, 0)) == [255] * 10  # anything behind the 7 shown
    assert s.hp_belief(s.unit_at(4, 2)) == [0] * 9 + [255]  # a 10 is taken as full
    assert s.hp_belief(s.unit_at(back.x, back.y)) == [255] * 10
    planes, globals_ = encode(np.frombuffer(s.to_bytes(), np.uint8)[None], 8)
    assert planes[0, UNIT_PLANE, back.y, back.x] == 0  # Blue Moon can't see it
    assert globals_[0, GLOBAL["opp.hidden_stealths"]] == pytest.approx(0.1)
    assert globals_[0, GLOBAL["opp.army_value"]] == pytest.approx(counts[str(OS)]["value"] / 100000)
    assert globals_[0, GLOBAL["opp.units"]] == pytest.approx(2 / 50)


def test_live_turn_hidden_hp_and_unexplained_gaps(info):
    units = [unit(11, OS, "Infantry", 1, 0, "?"), unit(12, BM, "Tank", 4, 2, 10)]
    counts = {str(OS): {"total": 2, "value": 1000, "count": {"Infantry": 1, "Recon": 1}}}
    lt = live.live_turn(0, turn_data(units, counts, os_co=SONJA, power_on="N", weather="C"), info)
    assert lt.unknown_hp == 1 and lt.turn.units[0].hp == 100 and lt.hidden_hp == {11}
    assert lt.unplaced == {(OS, "Recon"): 1} and not lt.put_back  # only Subs and Stealths hide
    assert lt.turn.weather_start is None
    s = live.state(live.parse_page(page()), lt)
    assert s.hidden_belief(s.unit_at(1, 0)) == [255] * 10  # any display, to Blue Moon


class FakeSite(Http):
    """The game page and the replay API for a game whose current turn is ``count - 1``."""

    def __init__(self, count: int):
        super().__init__(min_interval=0)
        self.count = count
        self.posts: list[tuple[int, bool]] = []

    def get(self, url: str) -> bytes:
        return page().encode()

    def post_json(self, url: str, payload: object) -> bytes:
        assert isinstance(payload, dict)
        turn, initial = payload["turn"], payload["initial"]
        self.posts.append((turn, initial))
        if turn >= self.count:
            return json.dumps({"err": True, "message": "Requested turn has no replay stored"}).encode()
        data = turn_data([], {}, weather="C", to_move=(OS, BM)[turn % 2])
        data["day"] = 1 + turn // 2
        if initial:
            data["daySelector"] = [f"{1 + i // 2}_{(OS, BM)[i % 2]}" for i in range(self.count)]
        return json.dumps(data).encode()


def test_fetch_turn_data_guesses_the_turn_and_caches_ended_ones(tmp_path, monkeypatch):
    monkeypatch.setenv("AWBW_CACHE", str(tmp_path))
    g = live.parse_page(page())  # day 3, Blue Moon to move: turn 5
    site = FakeSite(6)
    assert [i for i, _ in live.fetch_turn_data(g, site)] == [5]
    assert site.posts == [(5, True)]
    site.posts.clear()
    got = live.fetch_turn_data(g, site, history=True)
    assert [i for i, _ in got] == [0, 1, 2, 3, 4, 5] and got[3][1]["day"] == 2
    assert len(list((tmp_path / "live" / "555").iterdir())) == 5  # not the turn in progress
    site.posts.clear()
    live.fetch_turn_data(g, site, history=True)
    assert site.posts == [(5, True)]

    late = FakeSite(8)  # the page is behind: two more turns were played
    assert [i for i, _ in live.fetch_turn_data(g, late)] == [7]
    assert late.posts == [(5, True), (7, False)]
    gone = FakeSite(3)  # the guess is past the last turn
    assert [i for i, _ in live.fetch_turn_data(g, gone)] == [2]
    assert gone.posts == [(5, True), (0, True), (2, False)]
