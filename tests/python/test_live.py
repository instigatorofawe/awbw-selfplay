import json
from typing import Any

import numpy as np
import pytest

from awbw import _core, engine, live, rules
from awbw.data.loader import encode
from awbw.maps import MapInfo, PredeployedUnit
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


def test_replay_actions_nests_moves_as_the_replay_does():
    tank = {"units_id": 21, "units_name": "Tank", "units_x": 2, "units_y": 1, "units_hit_points": 10}
    inf = {"units_id": 22, "units_name": "Infantry", "units_x": 3, "units_y": 1, "units_hit_points": 9}
    apc = {"units_id": 23, "units_name": "APC", "units_x": 4, "units_y": 2, "units_hit_points": 10}
    path = [{"unit_visible": True, "x": 1, "y": 1}, {"unit_visible": True, "x": 2, "y": 1}]
    fight = {"units_id": 21, "units_hit_points": 9, "units_ammo": 8, "units_x": 2, "units_y": 1}
    target = {"units_id": 31, "units_hit_points": 4, "units_ammo": 0, "units_x": 2, "units_y": 2}
    cop = {"attacker": {"playerId": OS, "copValue": 100}, "defender": {"playerId": BM, "copValue": 200}}
    building = {"buildings_capture": 11, "buildings_x": 3, "buildings_y": 1}
    got = live.replay_actions(
        [
            {"action": "Move", "unit": tank, "path": path, "dist": 1, "trapped": False},
            {"action": "Fire", "attacker": fight, "defender": target, "copValues": cop, "gainedFunds": None},
            {"action": "Move", "unit": inf, "path": path, "dist": 1, "trapped": False},
            {"action": "Capt", "buildingInfo": building},  # by the infantry, which just moved there
            {"action": "Capt", "buildingInfo": building | {"buildings_capture": 2}},  # in place
            {"action": "Move", "unit": apc, "path": path, "dist": 1, "trapped": False},
            {"action": "Supply", "suppliedIds": ["22"]},
            {"action": "Power", "playerId": OS, "copValue": 0, "coName": "Olaf", "coPower": "Y", "powerName": "x"},
            {"action": "NextTurn", "day": 4, "nextFunds": 9000, "nextPId": BM, "repaired": [], "supplied": []},
        ]
    )
    assert [a["action"] for a in got] == ["Fire", "Capt", "Capt", "Supply", "Power", "End"]
    fire, capt, in_place, supply, power, end = got
    assert fire["Move"]["unit"] == {"global": tank} and fire["Move"]["paths"] == {"global": path}
    assert fire["Fire"]["combatInfoVision"]["global"]["combatInfo"] == {"attacker": fight, "defender": target}
    assert fire["Fire"]["copValues"] == cop
    assert capt["Move"]["unit"]["global"] is inf and capt["Capt"]["buildingInfo"] is building
    assert in_place["Move"] == [] and in_place["Capt"]["buildingInfo"]["buildings_capture"] == 2
    assert supply["Supply"]["unit"] == {"global": 23} and supply["Supply"]["rows"] == ["22"]
    assert power["playerID"] == OS and power["playersCOP"] == 0
    assert end["updatedInfo"]["event"] == "NextTurn" and end["updatedInfo"]["nextFunds"] == {"global": 9000}

    # A move with nothing after it is a move (wait), even when the next action is another unit's.
    other = {"units_id": 24, "units_hit_points": 10, "units_ammo": 9, "units_x": 5, "units_y": 1}
    got = live.replay_actions(
        [
            {"action": "Move", "unit": inf, "path": path, "dist": 1, "trapped": False},
            {"action": "Fire", "attacker": other, "defender": target, "copValues": cop, "gainedFunds": None},
        ]
    )
    assert [a["action"] for a in got] == ["Move", "Fire"] and got[1]["Move"] == []


A = _core.ACTIONS
STEP_MAP = 910302  # the test terrain again, with predeployed units


def _shown(hp: int) -> int:
    return (hp + 9) // 10


def _fight_turns(info, *, sonja: bool, corrupt: bool = False):
    """A Tank of Orange Star's (Sonja's, with ``sonja``) attacks a Blue Moon Infantry
    in turn 0; Blue Moon's Recon (predeployed at 7 HP) stays out of it. Returns the
    live turns, the game, and the true state after the turn (the engine's)."""
    units = [unit(11, OS, "Tank", 1, 1, "?" if sonja else 10), unit(12, BM, "Infantry", 2, 1, 10)]
    units.append(unit(13, BM, "Recon", 4, 0, 7))
    first = turn_data(units, {}, os_co=SONJA if sonja else OLAF, power_on="N", weather="C", to_move=OS)
    first["gameState"]["players"][str(OS)]["players_co_power"] = 0
    first["gameState"]["players"][str(BM)]["players_co_power"] = 0
    g = live.parse_page(page(mapId=STEP_MAP))
    g.predeployed = [PredeployedUnit(1, 1, 1, 10, "os"), PredeployedUnit(1, 2, 1, 10, "bm")]
    g.predeployed.append(PredeployedUnit(1, 4, 0, 7, "bm"))

    truth = live.state(g, live.live_turn(0, first, info))
    tank, _, recon = (truth.find_unit(i) for i in (11, 12, 13))
    truth.set_hp_unknown(recon)
    truth.set_hp(recon, 70)
    truth.set_hp_belief(recon, [0] * 9 + [255])  # predeployed: known
    if sonja:
        truth.set_hidden_belief(tank, [0] * 9 + [255])
    outcomes = _core.fight_outcomes(truth, tank, (1, 1), (2, 1))
    att, dfn, _ = max(outcomes, key=lambda o: (_shown(o[1]) < 10, o[2]))
    assert _core.apply(truth, A["Fire"], unit=tank, target=(2, 1), att_hp=att, def_hp=dfn) == _core.OK
    players = truth.players
    shown_att = "?" if sonja else _shown(att)
    fire: dict[str, Any] = {
        "action": "Fire",
        "attacker": {"units_id": 11, "units_hit_points": shown_att, "units_ammo": truth.unit(tank).ammo},
        "defender": {"units_id": 12, "units_hit_points": 1 if corrupt else _shown(dfn), "units_ammo": 0},
        "copValues": {
            "attacker": {"playerId": OS, "copValue": players[0].power},
            "defender": {"playerId": BM, "copValue": players[1].power},
        },
        "gainedFunds": None,
    }
    for side, (x, y) in (("attacker", (1, 1)), ("defender", (2, 1))):
        fire[side] |= {"units_x": x, "units_y": y}
    assert _core.apply(truth, A["EndTurn"]) == _core.OK
    end = {"action": "NextTurn", "day": 3, "nextFunds": truth.players[1].funds, "nextPId": BM}
    first["actions"] = [fire, end | {"nextWeather": "C", "repaired": [], "supplied": []}]

    second = json.loads(json.dumps(first))
    second["actions"] = []
    gs = second["gameState"]
    gs["currentTurnPId"] = BM
    for pid, p in zip((OS, BM), truth.players, strict=False):
        gs["players"][str(pid)] |= {"players_funds": p.funds, "players_co_power": p.power}
    for uid, row in list(gs["units"].items()):
        slot = truth.find_unit(int(uid))
        if slot < 0:
            del gs["units"][uid]
            continue
        u = truth.unit(slot)
        hidden = sonja and u.owner == 0
        row |= {"units_hit_points": "?" if hidden else _shown(u.hp), "units_fuel": u.fuel, "units_ammo": u.ammo}
    return g, [live.live_turn(0, first, info), live.live_turn(1, second, info)], truth


@pytest.fixture(scope="module")
def step_info():
    m = MapInfo(STEP_MAP, "t", "", 2, 6, 4, TERRAIN, [])
    engine.register(m)
    return m


@pytest.mark.parametrize("sonja", [False, True], ids=["plain", "sonja"])
def test_states_step_a_fight_into_hp_beliefs(step_info, sonja):
    g, turns, truth = _fight_turns(step_info, sonja=sonja)
    states, report = live.states(g, turns)
    assert report.stepped == 1 and not report.failed
    s = states[1]
    page_only = live.state(g, turns[1])
    for uid in (11, 12, 13):
        got, want = s.find_unit(uid), truth.find_unit(uid)
        if want < 0:
            assert got < 0
            continue
        u = s.unit(got)
        assert _shown(u.hp) == _shown(truth.unit(want).hp)
        if not (sonja and uid == 11):  # her own view of her Tank isn't the spectator's to know
            assert s.hp_belief(got) == truth.hp_belief(want)
    inf = s.find_unit(12)
    if inf >= 0 and _shown(s.unit(inf).hp) < 10:
        assert s.hp_belief(inf) != page_only.hp_belief(page_only.find_unit(12))  # watching tells more
    assert s.hp_belief(s.find_unit(13)) == [0] * 9 + [255]  # the predeployed Recon: 70 exactly
    if sonja:
        tank = s.find_unit(11)
        assert s.hidden_belief(tank) == truth.hidden_belief(truth.find_unit(11))
        assert page_only.hidden_belief(page_only.find_unit(11)) == [255] * 10


def test_states_restart_the_units_a_turn_it_cannot_step_touched(step_info):
    g, turns, _ = _fight_turns(step_info, sonja=False, corrupt=True)
    states, report = live.states(g, turns)
    assert report.stepped == 0 and sum(report.failed.values()) == 1
    s = states[1]
    page_only = live.state(g, turns[1])
    inf = s.find_unit(12)
    if inf >= 0:
        assert s.hp_belief(inf) == page_only.hp_belief(page_only.find_unit(12))  # back to the page's
    assert s.hp_belief(s.find_unit(13)) == [0] * 9 + [255]  # untouched: kept
