import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from awbw import _core, engine, play, rules
from awbw.agents import RandomAgent
from awbw.games import replay
from awbw.play import Game, attack_range, serve, step_id
from test_policy import U, start
from test_search import md  # noqa: F401 (a fixture)

HUMAN, AGENT = 0, 1


def _game(md, logs=None, agent=None) -> Game:  # noqa: F811
    on_over = logs.append if logs is not None else None
    return Game(start(md), HUMAN, agent or RandomAgent(1), ("human", "random"), on_over=on_over)


def _option_ids(o: dict) -> list[int]:
    """Every pick the options offer in their phase (the selected phase's Delete is an idle pick)."""
    ids = [i for _, _, i in o.get("select", [])]
    if isinstance(o.get("delete"), list):
        ids += [i for _, _, i in o["delete"]]
    ids += [o[k] for k in ("cop", "scop", "end_turn") if o.get(k) is not None]
    ids += [m["id"] for m in o.get("moves", [])]
    for entry in o.get("menu", []):
        ids += [t["id"] for t in entry["targets"]] if "targets" in entry else [entry["id"]]
    ids += [b["id"] for b in o.get("builds", []) if b["id"] is not None]
    return ids


def _pick_randomly(g: Game, rng) -> None:
    o = g.view()["options"]
    ids = _option_ids(o)
    assert sorted(ids) == sorted(int(i) for i in _core.legal_ids(g.state, g.sel)), g.view()["phase"]
    keep = [i for i in ids if i >= _core.ID_END_TURN or i // _core.MAX_TILES != _core.CHANNELS.index("Delete")]
    if g.view()["phase"] == "idle" and len(keep) > 1 and rng.random() < 0.8:
        keep = [i for i in keep if i != _core.ID_END_TURN]
    g.pick(int(rng.choice(keep)))


def test_a_game_through_the_api_offers_every_legal_pick_and_records_what_was_played(md):  # noqa: F811
    logs = []
    g = _game(md, logs)
    rng = np.random.default_rng(3)
    phases, undos = set(), 0
    for _ in range(3000):
        if g.over:
            break
        if not g.human_to_move:
            assert g.think() is not None
            continue
        phases.add(g.view()["phase"])
        _pick_randomly(g, rng)
        if rng.random() < 0.05 and g.undo():
            undos += 1
    assert g.over and phases == {"idle", "selected", "moved", "factory"} and undos > 3
    (log,) = logs
    assert (log.source, log.players, log.winner, log.decided) == ("local", ("human", "random"), g.winner, "game over")
    r = replay(log)
    assert r.matches and len(r.moves) == len(g.log)


def test_undo_steps_back_an_action_and_an_attack_retried_rolls_the_same(md):  # noqa: F811
    g = _game(md)
    before = g.state.to_bytes()
    tank, target = (2, 2), (4, 3)

    def attack():
        g.pick(step_id(md, "Select", *tank))
        g.pick(step_id(md, "Move", 3, 3))
        menu = {e["channel"]: e for e in g.view()["options"]["menu"]}
        (t,) = [t for t in menu["Fire"]["targets"] if tuple(t["at"]) == target]
        preview = t["preview"]
        assert abs(sum(p for _, p in preview["after"]) - 1) < 1e-3 and preview["hp"] == preview["own"] == 10
        assert 0 < preview["damage"][0] < preview["damage"][1] and preview["kill"] == 0
        done = g.pick(t["id"])
        assert done is not None and done["path"][0] == [2, 2] and done["path"][-1] == [3, 3]
        return g.state.to_bytes()

    after = attack()
    assert after != before and g.view()["can_undo"]
    assert g.undo() and g.state.to_bytes() == before and not g.view()["can_undo"]
    assert attack() == after


def test_undo_from_the_next_turn_undoes_the_agents_turn_and_the_end_turn(md):  # noqa: F811
    g = _game(md)
    g.pick(step_id(md, "Select", 1, 1))
    g.pick(step_id(md, "Move", 2, 1))
    g.pick(step_id(md, "Capture", 2, 1))
    captured = g.state.to_bytes()
    g.pick(_core.ID_END_TURN)
    while not g.human_to_move:
        g.think()
    assert int(g.state.day) == 2 and len(g.log) > 2
    assert g.undo() and g.state.to_bytes() == captured and len(g.log) == 1
    assert g.undo() and len(g.log) == 0 and not g.undo()


def test_options_decode_paths_ranges_menus_and_builds(md):  # noqa: F811
    g = _game(md)
    s = g.state
    g.pick(step_id(md, "Select", 2, 2))  # the Tank
    o = g.view()["options"]
    assert o["unit"] == [2, 2] and o["delete"] == step_id(md, "Delete", 2, 2)
    for m in o["moves"]:
        assert m["path"][0] == [2, 2] and m["path"][-1] == m["to"]
    assert [4, 3] in o["attack"]  # next to a destination
    g.cancel()
    art = s.unit_at(0, 2)
    assert sorted(attack_range(s, art, [(1, 2)])) == sorted(
        (x, y) for y in range(md.height) for x in range(md.width) if 2 <= abs(x) + abs(y - 2) <= 3
    )  # from where it stands, not from where it could move
    g.pick(step_id(md, "Select", 1, 0))  # the base
    builds = {b["name"]: b for b in g.view()["options"]["builds"]}
    assert builds["Infantry"]["cost"] == 1000 and builds["Infantry"]["id"] == _core.ID_BUILD + U["Infantry"]
    assert builds["Neotank"]["cost"] == 22000 and builds["Neotank"]["id"] is None  # 8,000 funds
    assert "Fighter" not in builds and "Lander" not in builds
    g.cancel()
    assert g.view()["phase"] == "idle"
    enemy = g.unit_info(4, 3)
    assert enemy is not None and enemy["owner"] == AGENT and [4, 3] in enemy["moves"] and [2, 2] in enemy["attack"]


def test_resigning_records_the_game_for_the_agent(md):  # noqa: F811
    logs = []
    g = _game(md, logs)
    g.resign()
    assert g.over and g.winner == AGENT and g.view()["decided"] == "resigned" and not g.view()["can_undo"]
    (log,) = logs
    assert (log.decided, log.winner, len(log.steps)) == ("resigned", AGENT, 0)


def _post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_server_serves_the_page_and_plays_through_the_api(md):  # noqa: F811
    server = serve(_game(md), new_game=lambda params: _game(md))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = server.url
        with urllib.request.urlopen(url) as r:
            html = r.read().decode()
        assert "makeBoard" in html and '"human":0' in html
        with urllib.request.urlopen(url + "api/view") as r:
            view = json.loads(r.read())
        assert view["phase"] == "idle" and view["options"]["end_turn"] == _core.ID_END_TURN
        assert _post(url + "api/pick", {"id": step_id(md, "Move", 0, 0)})[0] == 409
        assert _post(url + "api/think", {}) == (200, {"done": None, "view": view})
        code, out = _post(url + "api/pick", {"id": _core.ID_END_TURN})
        assert code == 200 and out["done"]["t"] == "ends the turn" and out["view"]["options"] is None
        while True:
            code, out = _post(url + "api/think", {})
            if out["done"] is None:
                break
        assert out["view"]["options"] is not None and out["view"]["frame"]["d"] == 2
        code, out = _post(url + "api/new", {})
        assert code == 200 and out["view"]["frame"]["d"] == 1 and out["view"]["actions"] == 0
        with urllib.request.urlopen(url + "api/unit?x=4&y=3") as r:
            assert json.loads(r.read())["owner"] == AGENT
    finally:
        server.shutdown()
        server.server_close()


def _predeployed_map(tmp_path) -> int:
    """test_policy's map with a predeployed unit for each side and a 9 HP one."""
    from test_policy import TERRAIN, H, W

    map_id = 79
    units = [
        {"Unit ID": rules.load().unit("Infantry").id, "Unit X": 3, "Unit Y": 1, "Unit HP": 10, "Country Code": "os"},
        {"Unit ID": rules.load().unit("Tank").id, "Unit X": 4, "Unit Y": 4, "Unit HP": 9, "Country Code": "bm"},
    ]
    info = {"Name": "p", "Size X": W, "Size Y": H, "Terrain Map": TERRAIN.T.tolist(), "Predeployed Units": units}
    (tmp_path / "maps" / f"{map_id}.json").write_text(json.dumps(info))
    return map_id


def test_new_game_is_awbws_first_state(md, tmp_path):  # noqa: F811
    s = engine.new_game(md.id, ("Andy", "Max"))
    assert (s.day, s.active, s.weather, s.winner) == (1, 0, 0, _core.NO_OWNER)
    andy, mx = rules.load().cos[1], next(c for c in rules.load().cos.values() if c.name == "Max")
    assert [(p.co, p.country, p.funds, p.power) for p in s.players[:2]] == [(andy.id, 1, 2000, 0), (mx.id, 2, 0, 0)]
    assert (s.players[0].cop_cost, s.players[1].scop_cost) == (_core.cop_cost(andy.id, 0), _core.scop_cost(mx.id, 0))
    owners = {(x, y): s.prop_owner(i) for i, (x, y, _, _) in enumerate(md.props())}
    assert owners[(0, 0)] == owners[(1, 0)] == 0 and owners[(5, 5)] == owners[(6, 5)] == 1 and owners[(2, 1)] == 255
    assert all(s.prop_points(i) == 20 for i in range(md.n_props)) and not s.unit_slots()
    other = engine.new_game(md.id, (1, "max"), ("Black Hole", "ge"), starting_funds=500)
    assert [(p.country, p.funds) for p in other.players[:2]] == [(5, 2500), (3, 500)]
    assert [other.prop_owner(i) for i in range(md.n_props)] == [s.prop_owner(i) for i in range(md.n_props)]
    pre = engine.new_game(_predeployed_map(tmp_path), ("Andy", "Andy"), seed=3)
    tank = rules.load().unit("Tank")
    assert sorted(u[1:8] for u in pre.unit_rows()) == [
        (U["Infantry"], 0, 3, 1, 100, 99, 0),
        (U["Tank"], 1, 4, 4, 90, tank.fuel, tank.ammo),
    ]
    with pytest.raises(ValueError, match="different countries"):
        engine.new_game(md.id, ("Andy", "Max"), ("os", "os"))
    with pytest.raises(ValueError, match="no CO"):
        engine.new_game(md.id, ("Andy", "Nobody"))


def test_starts_take_a_real_games_setup_with_what_is_given_instead(md, tmp_path):  # noqa: F811
    settings = _core.Settings()
    settings.capture_win = 9
    base = engine.new_game(md.id, ("Sami", "Kindle"), ("bh", "ge"), settings)
    rng = np.random.default_rng(0)
    same = play.start_for(base, rng)
    assert _core.state_hash(same) == _core.state_hash(base) and same.settings.capture_win == 9
    cos = play.start_for(base, rng, cos=("Andy", "Max"))
    assert [rules.load().cos[p.co].name for p in cos.players[:2]] == ["Andy", "Max"]
    assert [p.country for p in cos.players[:2]] == [5, 3]
    other = play.start_for(base, rng, map_id=_predeployed_map(tmp_path))
    assert other.map_id == 79 and [p.country for p in other.players[:2]] == [1, 2]
    assert "Sonja" not in [rules.load().cos[p.co].name for p in other.players[:2]] and other.settings.capture_win != 9
