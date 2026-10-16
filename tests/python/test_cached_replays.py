"""Invariants over whatever real replays are in the local cache (skipped if none).

Fetch some with e.g. ``awbw index "2p global league std" --max-pages 1`` and
``awbw fetch --index cache/index/2p_global_league_std.jsonl --limit 50``.
"""

import json
import random
import zipfile
import zlib

import pytest

from awbw import _core, engine, phpser, rules
from awbw.maps import map_path
from awbw.net import cache_dir
from awbw.replay import load_replay
from awbw.replay.combat_check import check_file
from awbw.replay.export import ActionLog, GameMeta, Snapshots
from awbw.replay.extract import extract_v0, extract_v1
from awbw.replay.move_check import check_replay as check_moves
from awbw.replay.step_check import check_file as check_turns
from awbw.replay.step_check import step_last_turn, step_turn
from awbw.replay.write import action_line

# The 200 newest games: the cache grows back in time, so these stay put.
PATHS = sorted((cache_dir() / "replays").glob("*.zip"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)[-200:]


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_replay_invariants(path):
    r = load_replay(path)
    order = r.player_ids
    assert len(set(order)) == len(order)

    days = [t.day for t in r.turns]
    assert days == sorted(days) and days[0] == 1

    if len(order) == 2:
        assert [t.player_id for t in r.turns] == [order[i % 2] for i in range(len(r.turns))]

    for turn in r.turns:
        ids = {u.id for u in turn.units}
        assert len(ids) == len(turn.units)
        for u in turn.units:
            assert 1 <= u.hp <= 100
            assert all(c in ids for c in u.cargo if c)
        carried = {u.id for u in turn.units if u.carried}
        assert carried == {c for u in turn.units for c in u.cargo if c}

    # Games cancelled early (day 1-2, no loser on mooo) log no GameOver at all.
    over = r.game_over()
    if over is not None:
        assert set(over.winners) | set(over.losers) <= set(order)
        assert not set(over.winners) & set(over.losers)
    # Multiplayer logs have gaps (timed-out turns, mid-game resignations). In
    # 2-player games a timeout loss leaves the final turn unlogged, and rarely
    # (2 of 2,000 cached games) a mid-game turn has no entry either; the v1
    # filter drops those games.
    if len(order) == 2:
        assert sum(turn.actions is None for turn in r.turns[:-1]) <= 1


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_logged_attacks_match_engine(path):
    checks, _ = check_file(path)
    replay = load_replay(path)
    bad = [c for c in checks if c.ok is False and not known_oddity(replay, c.day)]
    assert not bad, bad[:3]


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_snapshots_load_and_logged_moves_are_legal(path):
    rep = check_moves(path)
    assert not rep.snapshot_errors, rep.snapshot_errors[:3]
    assert not rep.failures, rep.failures[:3]


def _gzip_members(data: bytes) -> list[bytes]:
    out = []
    while data:
        d = zlib.decompressobj(wbits=31)
        out.append(d.decompress(data))
        data = d.unused_data
    return out


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_replay_files_rewrite_byte_for_byte(path):
    """What the export writes with: every snapshot and log line, parsed and written
    back, is the same bytes."""
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            for member in _gzip_members(z.read(name)):
                body = member.rstrip(b"\n")
                if body.startswith(b"p:"):
                    for line in body.split(b"\n"):
                        pid, day, actions = phpser.loads(line.split(b";a:", 1)[1])
                        assert isinstance(actions, list)
                        decoded = [a if a == "Array" else json.loads(a) for a in actions]
                        assert action_line(pid, day, decoded) == line
                elif body:
                    assert phpser.dumps(phpser.loads(body)) == body


# Snapshot fields an export can't match: ids, names, dates, users, timers, and
# settings of real games (league, official, ratings).
GAME_UNCHECKED = {"id", "name", "creator", "start_date", "activity_date", "aet_date", "type", "official", "league"}
GAME_UNCHECKED |= {"min_rating", "max_rating", "password", "comment", "boot_interval", "aet_interval"}
GAME_UNCHECKED |= {"timers_initial", "timers_increment", "timers_max_turn", "players", "buildings", "units"}
PLAYER_UNCHECKED = {"id", "users_id", "games_id", "last_read", "turn_start", "turn_clock", "order", "team"}
PLAYER_UNCHECKED |= {"accept_draw", "aet_count"}
UNIT_IDS = {"id", "games_id", "players_id", "cargo1_units_id", "cargo2_units_id"}


def _raw_snapshots(path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            members = _gzip_members(z.read(name))
            if members and members[0].startswith(b'O:8:"awbwGame"'):
                return [phpser.loads(m.rstrip(b"\n")).fields for m in members]
    return []


def _snapshot_diffs(ours: dict, theirs: dict) -> list[str]:
    """Where an exported snapshot differs from AWBW's, matching units by tile, and
    leaving out AWBW's pipes and the waiting player's ``fired`` (see export)."""
    diffs = []
    if list(ours) != list(theirs):
        diffs.append("game fields")
    diffs += [
        f"game {k}: {ours.get(k)!r} vs {v!r}" for k, v in theirs.items() if k not in GAME_UNCHECKED and ours.get(k) != v
    ]
    for a, b in zip(ours["players"], sorted(theirs["players"], key=lambda p: p.fields["order"]), strict=True):
        a, b = a.fields, b.fields
        if list(a) != list(b):
            diffs.append("player fields")
        diffs += [
            f"player {k}: {a.get(k)!r} vs {v!r}" for k, v in b.items() if k not in PLAYER_UNCHECKED and a.get(k) != v
        ]
    built = {(b.fields["x"], b.fields["y"]): b.fields for b in ours["buildings"]}
    for b in (x.fields for x in theirs["buildings"]):
        a = built.pop((b["x"], b["y"]), None)
        if a is None:
            if not 101 <= b["terrain_id"] <= 110:  # only some maps list their pipes
                diffs.append(f"building at {b['x'], b['y']} missing")
            continue
        broken = b["terrain_id"] in (115, 116)  # a broken seam keeps a capture value of its own
        for k in ("terrain_id", "capture", "last_capture"):
            if a[k] != b[k] and not (k == "capture" and broken):
                diffs.append(f"building {b['x'], b['y']} {k}: {a[k]} vs {b[k]}")
        if list(a) != list(b):
            diffs.append("building fields")
    diffs += [f"extra building at {xy}" for xy in built]

    def key(u: dict) -> tuple:
        return (u["x"], u["y"], u["name"], u["players_id"], u["carried"])

    pool: dict[tuple, list[dict]] = {}
    for u in theirs["units"]:
        pool.setdefault(key(u.fields), []).append(u.fields)
    matched = {}
    for a in (u.fields for u in ours["units"]):
        cands = pool.get(key(a), [])
        if not cands:
            diffs.append(f"unit {a['name']} at {a['x'], a['y']} unmatched")
            continue
        b = min(cands, key=lambda c: abs(c["hit_points"] - a["hit_points"]))
        cands.remove(b)
        matched[a["id"]] = (a, b)
    diffs += [f"AWBW's {u['name']} at {u['x'], u['y']} unmatched" for us in pool.values() for u in us]
    ids = {k: b["id"] for k, (_, b) in matched.items()}
    for a, b in matched.values():
        if list(a) != list(b):
            diffs.append("unit fields")
        for k, v in b.items():
            mine = a[k]
            if k in ("cargo1_units_id", "cargo2_units_id"):
                mine = ids.get(mine, mine)
            elif k in UNIT_IDS or (k == "fired" and b["players_id"] != theirs["turn"]):
                continue
            elif k == "sub_dive":  # the Replay Player reads D as Y and R as N
                mine, v = mine in "YD", v in "YD"
            if mine != v:
                diffs.append(f"{b['name']} at {b['x'], b['y']} {k}: {mine!r} vs {v!r}")
    return diffs


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_snapshots_export_as_awbw_writes_them(path):
    """Each turn start the engine reaches, exported, is AWBW's snapshot."""
    r = load_replay(path)
    if r.game.fog or len(r.player_ids) != 2 or not map_path(r.game.map_id).exists():
        pytest.skip("not 2 players without fog, or map not cached")
    raw = _raw_snapshots(path)
    md = engine.map_data(r.game.map_id)
    snaps = Snapshots(md, GameMeta(id=r.game.id, name=r.game.name, players=(r.player_ids[0], r.player_ids[1])))
    diffs = _snapshot_diffs(snaps.take(engine.state_from_turn(r, r.turns[0], md)).fields, raw[0])
    end = _core.ACTIONS["EndTurn"]
    for k in range(len(r.turns) - 1):
        trace = {}
        try:
            why = step_turn(r, md, k, engine.state_from_turn(r, r.turns[k], md), trace)[2]
        except ValueError:
            why = "unloadable"
        if why or not trace or trace[max(trace)].kind != end:
            # Go on from AWBW's own snapshot.
            snaps.take(engine.state_from_turn(r, r.turns[k + 1], md))
            real = {(b.fields["x"], b.fields["y"]): b.fields["last_capture"] for b in raw[k + 1]["buildings"]}
            for b in snaps.buildings:
                b.last_capture = real.get((b.x, b.y), b.last_capture)
            continue
        states = [_core.GameState.from_bytes(trace[i].state) for i in sorted(trace)]
        s = states[-1].copy()
        assert _core.apply(s, end) == _core.OK
        diffs += [f"turn {k + 1}: {d}" for d in _snapshot_diffs(snaps.take(s, states).fields, raw[k + 1])]
    assert not diffs, diffs[:5]


A = _core.ACTIONS
EXPORTED = {A[k] for k in ("Wait", "Fire", "Capture", "Load", "Join", "Unload", "Build", "Delete", "EndTurn", "Resign")}
# Dates, AWBW's timers, and messages naming its users can't match.
LOG_UNCHECKED = {"nextTurnStart", "gameEndDate", "nextTimer", "message"}
LOG_IDS = {"0", "units_id", "units_cargo1_units_id", "units_cargo2_units_id", "transportID", "global"}
SONJA = next(c.id for c in rules.load().cos.values() if c.name == "Sonja")
# Log entries with no rule behind them that we can see.
LOG_ODDITIES = {
    "1741903 day 8 Capt",  # an enemy lab with one left logs onElimination (6 of 567 such captures)
}


def _forced_after(before: _core.GameState, steps: list[int], logged: dict) -> _core.GameState | None:
    """A turn's last traced action played again, with its logged fight result:
    no later trace state holds it. None if the engine refuses that result."""
    s = before.copy()
    sel = _core.Selection()
    att_hp = def_hp = -1
    if logged.get("action") == "Fire":
        info = logged["Fire"]["combatInfoVision"]["global"]["combatInfo"]

        def hp(side: dict) -> int:
            now = before.unit(before.find_unit(side["units_id"])).hp
            shown = side["units_hit_points"]
            return now if (now + 9) // 10 == shown else min(now, shown * 10)

        att_hp, def_hp = hp(info["attacker"]), hp(info["defender"])
    status = _core.OK
    for step in steps:
        status = _core.step(s, sel, step, att_hp=att_hp, def_hp=def_hp)
    return s if status in (_core.OK, _core.TRAPPED) else None


def _logged_actions(r, md, k: int) -> list[tuple[list[dict], _core.GameState, _core.GameState, tuple, int]]:
    """Each action of turn ``k`` the engine steps: (its log entries, the states
    before and after it, the action as step_action gives it, its status)."""
    turn = r.turns[k]
    last = k == len(r.turns) - 1
    start = engine.state_from_turn(r, turn, md)
    trace = {}
    try:
        why = step_last_turn(r, md, start, trace) if last else step_turn(r, md, k, start, trace)[2]
    except ValueError:
        why = "unloadable"
    if why:
        return []
    out = []
    order = sorted(trace)
    prev, state = -1, start
    for j, i in enumerate(order):
        before = _core.GameState.from_bytes(trace[i].state)
        s, sel = before.copy(), _core.Selection()
        action, status = None, _core.OK
        for step in trace[i].steps:
            action = _core.step_action(s, sel, step)
            status = _core.step(s, sel, step)
        assert action is not None
        theirs = turn.actions[prev + 1 : i + 1]
        if j + 1 < len(order):
            after = _core.GameState.from_bytes(trace[order[j + 1]].state)
        elif (after := _forced_after(before, trace[i].steps, theirs[-1])) is None:
            break
        out.append((theirs, before, after, tuple(action), status))
        prev, state = i, after
    rest = turn.actions[prev + 1 :]
    # Resigning in the opponent's turn is something the engine doesn't do.
    if last and rest and rest[0]["action"] == "Resign" and rest[0]["Resign"]["playerId"] == r.player_ids[state.active]:
        out.append((rest[:1], state, state, (A["Resign"], -1, None, None, 0), _core.OK))
    return out


def _log_diffs(ours, theirs, where: str, ids: dict[int, int], key: str = "") -> list[str]:
    if key in LOG_IDS and isinstance(ours, int):
        ours = ids.get(ours, ours)
    if isinstance(theirs, dict) and isinstance(ours, dict):
        diffs = [f"{where}: keys {list(ours)} vs {list(theirs)}"] if list(ours) != list(theirs) else []
        for k, v in theirs.items():
            if k in ours and k not in LOG_UNCHECKED:
                diffs += _log_diffs(ours[k], v, f"{where}.{k}", ids, k)
        return diffs
    if isinstance(theirs, list) and isinstance(ours, list) and len(ours) == len(theirs):
        return [d for a, b in zip(ours, theirs, strict=True) for d in _log_diffs(a, b, f"{where}[]", ids, key)]
    if key in ("supplied", "units_id") and isinstance(ours, str) and ours.isdigit():
        ours = str(ids.get(int(ours), int(ours)))
    if ours != theirs or type(ours) is not type(theirs):
        return [f"{where}: {ours!r} vs {theirs!r}"]
    return []


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_actions_export_as_awbw_logs_them(path):
    """Each action the engine steps, exported, is AWBW's log entry, given the
    path the player took (the engine's own export takes its cheapest)."""
    r = load_replay(path)
    if r.game.fog or len(r.player_ids) != 2 or not map_path(r.game.map_id).exists():
        pytest.skip("not 2 players without fog, or map not cached")
    if SONJA in {p.co_id for p in r.turns[0].players}:
        pytest.skip("Sonja hides HP in the log")
    md = engine.map_data(r.game.map_id)
    meta = GameMeta(id=r.game.id, name=r.game.name, players=(r.player_ids[0], r.player_ids[1]))
    diffs = []
    for k, turn in enumerate(r.turns):
        log = ActionLog(md, meta, {(b.x, b.y): b.id for b in turn.buildings})
        ids: dict[int, int] = {}  # engine id -> AWBW id of the units built this turn
        for theirs, before, after, action, status in _logged_actions(r, md, k):
            if action[0] not in EXPORTED:
                continue
            move = theirs[0] if theirs[0]["action"] == "Move" else theirs[0].get("Move")
            path_taken = None
            if isinstance(move, dict):  # the mover's own view: a hidden unit's has no global one
                paths = move["paths"]
                path_taken = [(p["x"], p["y"]) for p in paths.get(str(meta.players[before.active])) or paths["global"]]
            ours = log.entries(before, after, action, status, path_taken)
            where = f"day {turn.day} {theirs[-1]['action']}"
            if f"{r.game.id} {where}" in LOG_ODDITIES:
                continue
            if action[0] == A["Build"]:
                ids[ours[0]["newUnit"]["global"]["units_id"]] = theirs[0]["newUnit"]["global"]["units_id"]
            if len(ours) != len(theirs):
                diffs.append(f"{where}: {len(ours)} entries vs {len(theirs)}")
                continue
            for o, t in zip(ours, theirs, strict=True):
                diffs += _log_diffs(o, t, where, ids)
    assert not diffs, diffs[:5]


# Turns where AWBW's result has no rule behind it that we can see.
KNOWN_ODDITIES = {
    # A Black Boat repair leaves a 9.1-9.9 HP unit as it was; 315 others top up to 10.
    "1643462 day 12",
    "1644362 day 16",
    "1645952 day 10",
    "1650385 day 12",
    "1651661 day 19",
    "1653448 day 12",
    "1654751 day 16",
    "1671326 day 9",
    "1674937 day 11",
    "1724707 day 15",
    "1631097 day 17",
    "1636466 day 13",  # a city's capture points drop to 0 with no capture logged; it is taken next turn
    "1695038 day 12",  # too many fight-result choices for the search budget
    "1728785 day 15",  # needs 508k fight-result tries, over the search budget
    "1628095 day 16",  # too many fight-result choices for the search budget
}

# Lash's terrain-star attack bonus is missing in every mismatched fight of a
# Lash game (127 of 1.7M attacks), all in turns overlapping 2026-07-27 16:47 to
# 07-28 04:57 (AWBW server time); it holds in her 90,277 other attacks from
# starred terrain.
LASH = 16
LASH_GAP = ("2026-07-27 16", "2026-07-28 06")


def known_oddity(replay, day: int) -> bool:
    if f"{replay.game.id} day {day}" in KNOWN_ODDITIES:
        return True
    if LASH not in {p.co_id for p in replay.turns[0].players}:
        return False
    turns = replay.turns
    for k, t in enumerate(turns):
        if t.day == day:
            ended = turns[k + 1].started if k + 1 < len(turns) else "9999"
            if t.started < LASH_GAP[1] and ended >= LASH_GAP[0]:
                return True
    return False


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_turns_replay_exactly(path):
    rep = check_turns(path)
    replay = load_replay(path)
    bad = [e for kind in rep.diverged for e in rep.examples[kind] if not known_oddity(replay, int(e.split()[2]))]
    assert not bad, bad[:3]


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_hp_beliefs_carry_through_every_turn(path):
    """Extraction steps every turn, so each snapshot's HPs are ones the beliefs
    carried into it allow (extract_v0 rejects a state that breaks that)."""
    g = extract_v0(path)
    if isinstance(g, str):
        pytest.skip(g)
    if g.unstepped:
        replay = load_replay(path)
        assert any(known_oddity(replay, t.day) for t in replay.turns), f"{g.unstepped} turns not stepped"
    else:
        assert g.hp_unknown == 0


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS[-20:], ids=[p.stem for p in PATHS[-20:]])
def test_random_micro_steps_from_logged_positions(path):
    """From a few logged positions, random legal micro-steps keep the state sound
    and always leave a move until the game ends."""
    replay = load_replay(path)
    if len(replay.player_ids) != 2 or replay.game.fog or not map_path(replay.game.map_id).exists():
        pytest.skip("not 2 players, fog, or map not cached")
    md = engine.map_data(replay.game.map_id)
    rng = random.Random(int(path.stem))
    for turn in replay.turns[:: max(1, len(replay.turns) // 4)]:
        s = engine.state_from_turn(replay, turn, md)
        sel = _core.Selection()
        for _ in range(400):
            ids = _core.legal_ids(s, sel)
            if s.winner != _core.NO_OWNER:
                assert len(ids) == 0
                break
            assert len(ids), f"no legal step on day {s.day}"
            assert _core.step(s, sel, int(rng.choice(ids))) in (_core.OK, _core.TRAPPED)
            assert s.check() == ""


@pytest.mark.skipif(not PATHS, reason="no cached replays")
@pytest.mark.parametrize("path", PATHS[-20:], ids=[p.stem for p in PATHS[-20:]])
def test_v1_labels_every_action_with_legal_steps(path):
    """v1's turn starts are v0's samples, and each action's micro-steps are legal in
    turn from the state it was taken in, with the counts recorded."""
    g = extract_v1(path)
    if isinstance(g, str):
        pytest.skip(g)
    v0 = extract_v0(path)
    assert not isinstance(v0, str)
    a = g.actions
    assert a is not None
    assert (g.states[a.action == 0] == v0.states).all()
    assert len(set(a.turn[a.action == 0].tolist())) == len(v0.states)
    assert (a.n_steps > 0).mean() > 0.95
    for i in range(0, len(g.states), 7):
        s, sel = _core.GameState.from_bytes(g.states[i].tobytes()), _core.Selection()
        for k in range(int(a.n_steps[i])):
            legal = _core.legal_ids(s, sel)
            assert a.steps[i, k] in legal and len(legal) == a.n_legal[i, k]
            if k + 1 < a.n_steps[i]:
                assert _core.step(s, sel, int(a.steps[i, k])) == _core.OK
        if a.n_steps[i]:
            assert _core.step(s, sel, int(a.steps[i, a.n_steps[i] - 1])) in (_core.OK, _core.TRAPPED)
