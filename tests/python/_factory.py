"""Builds small synthetic AWBW replay zips in the real on-disk format."""

from __future__ import annotations

import gzip
import io
import json
import zipfile

from awbw.phpser import PhpObject


def php_dumps(value: object) -> bytes:
    if value is None:
        return b"N;"
    if isinstance(value, bool):
        return b"b:1;" if value else b"b:0;"
    if isinstance(value, int):
        return b"i:%d;" % value
    if isinstance(value, float):
        return b"d:" + repr(value).encode() + b";"
    if isinstance(value, str):
        raw = value.encode()
        return b's:%d:"' % len(raw) + raw + b'";'
    if isinstance(value, list):
        value = dict(enumerate(value))
    if isinstance(value, dict):
        body = b"".join(php_dumps(k) + php_dumps(v) for k, v in value.items())
        return b"a:%d:{" % len(value) + body + b"}"
    if isinstance(value, PhpObject):
        name = value.class_name.encode()
        body = b"".join(php_dumps(k) + php_dumps(v) for k, v in value.fields.items())
        return b'O:%d:"' % len(name) + name + b'":%d:{' % len(value.fields) + body + b"}"
    raise TypeError(type(value))


GAME_ID = 1000
P1, P2 = 501, 502


def _player(pid: int, order: int, co_id: int, funds: int) -> PhpObject:
    return PhpObject(
        "awbwPlayer",
        {
            "id": pid,
            "users_id": pid + 10_000,
            "games_id": GAME_ID,
            "countries_id": order,
            "co_id": co_id,
            "funds": funds,
            "eliminated": "N",
            "co_power": 0,
            "co_power_on": "N",
            "order": order,
            "co_max_power": 270000,
            "co_max_spower": 540000,
            "team": str(pid),
            "tags_co_id": None,
        },
    )


def _unit(uid: int, owner: int, x: int, y: int, hp: float) -> PhpObject:
    return PhpObject(
        "awbwUnit",
        {
            "id": uid,
            "games_id": GAME_ID,
            "players_id": owner,
            "name": "Infantry",
            "movement_points": 3,
            "vision": 2,
            "fuel": 99,
            "fuel_per_turn": 0,
            "sub_dive": "N",
            "ammo": 0,
            "short_range": 0,
            "long_range": 0,
            "second_weapon": "N",
            "symbol": "G",
            "cost": 1000,
            "movement_type": "F",
            "x": x,
            "y": y,
            "moved": 0,
            "capture": 0,
            "fired": 0,
            "hit_points": hp,
            "cargo1_units_id": 0,
            "cargo2_units_id": 0,
            "carried": "N",
        },
    )


def _snapshot(day: int, active: int, units: list[PhpObject], funds: tuple[int, int]) -> bytes:
    game = PhpObject(
        "awbwGame",
        {
            "id": GAME_ID,
            "name": "test game",
            "maps_id": 77,
            "weather_type": "Clear",
            "weather_code": "C",
            "turn": active,
            "day": day,
            "funds": 1000,
            "capture_win": 21,
            "fog": "N",
            "comment": "line one\nline two",  # newlines inside strings must not split turns
            "type": "L",
            "starting_funds": 0,
            "official": "Y",
            "use_powers": "Y",
            "team": "N",
            "league": None,
            "players": [_player(P1, 1, 22, funds[0]), _player(P2, 2, 11, funds[1])],
            "buildings": [
                PhpObject(
                    "awbwBuilding",
                    {"id": 9, "games_id": GAME_ID, "terrain_id": 34, "x": 1, "y": 2, "capture": 20, "last_capture": 20},
                )
            ],
            "units": units,
        },
    )
    return php_dumps(game)


def _action_line(pid: int, day: int, actions: list[object]) -> bytes:
    encoded = [a if isinstance(a, str) else json.dumps(a) for a in actions]
    return b"p:%d;d:%d;a:" % (pid, day) + php_dumps([pid, day, encoded])


GAME_OVER = {"day": 1, "gameEndDate": "2026-01-01", "losers": [P2], "winners": [P1]}


def build_replay_zip(*, with_actions: bool = True) -> bytes:
    snapshots = [
        _snapshot(1, P1, [_unit(1, P2, 3, 3, 10.0)], (1000, 0)),
        _snapshot(1, P2, [_unit(1, P2, 3, 3, 7.3), _unit(2, P1, 4, 4, 10.0)], (0, 1000)),
    ]
    # AWBW appends one gzip member per turn to the snapshot file.
    snapshot_file = b"".join(
        gzip.compress(s + (b"\n" if i < len(snapshots) - 1 else b"")) for i, s in enumerate(snapshots)
    )
    lines = [
        _action_line(P1, 1, [{"action": "Build", "newUnit": {}}, "Array", {"action": "End", "updatedInfo": {}}]),
        _action_line(P2, 1, [{"action": "Resign", "Resign": {"playerId": P2}, "GameOver": GAME_OVER}]),
        _action_line(P1, 2, [{"action": "End"}]),  # no snapshot for day 2: an orphan entry
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(str(GAME_ID), snapshot_file)
        if with_actions:
            z.writestr(f"a{GAME_ID}", gzip.compress(b"\n".join(lines)))
    return buf.getvalue()
