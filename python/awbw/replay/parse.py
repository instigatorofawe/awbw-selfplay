"""Parse AWBW replay zips.

A replay zip holds two gzipped entries, told apart by content:

* the snapshot file: one PHP-serialized ``awbwGame`` per turn (the state at the
  start of that turn), separated by newlines;
* the action log (absent in old replays): one line per turn,
  ``p:<player>;d:<day>;a:<php array>``, where the array is
  ``[player, turn number, [json action, ...]]``.

Values are parsed sequentially rather than split on newlines, because strings
inside them (comments, messages) may contain newlines.
"""

from __future__ import annotations

import io
import json
import zipfile
import zlib
from pathlib import Path
from typing import Any

from .. import phpser
from .model import BuildingState, GameInfo, PlayerState, Replay, Turn, UnitState

MAX_ENTRY_BYTES = 512 * 1024 * 1024
_SNAPSHOT_PREFIX = b'O:8:"awbwGame":'
_ACTIONS_PREFIX = b"p:"


class ReplayFormatError(ValueError):
    pass


def load_replay(source: str | Path | bytes) -> Replay:
    """Parse a replay zip from a path or from its bytes."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    snapshots, actions = _read_entries(data)
    replay = parse_snapshots(snapshots)
    if actions is not None:
        _attach_actions(replay, actions)
    return replay


def parse_snapshots(text: bytes) -> Replay:
    turns: list[Turn] = []
    game: GameInfo | None = None
    for obj in _sequence(text, lambda pos: phpser.load_at(text, pos)):
        if not isinstance(obj, phpser.PhpObject) or obj.class_name != "awbwGame":
            raise ReplayFormatError(f"expected awbwGame snapshot, got {type(obj).__name__}")
        f = obj.fields
        if game is None:
            game = GameInfo.from_php(f)
        turns.append(
            Turn(
                day=int(f["day"]),
                player_id=int(f["turn"]),
                weather=str(f.get("weather_code") or "C"),
                players=[PlayerState.from_php(p.fields) for p in _members(f["players"])],
                buildings=[BuildingState.from_php(b.fields) for b in _members(f["buildings"])],
                units=[UnitState.from_php(u.fields) for u in _members(f["units"])],
                weather_start=int(f["weather_start"]) if f.get("weather_start") else None,
                started=str(f.get("activity_date") or ""),
            )
        )
    if game is None:
        raise ReplayFormatError("replay has no snapshots")
    return Replay(game=game, turns=turns)


def _attach_actions(replay: Replay, text: bytes) -> None:
    by_key = {(t.player_id, t.day): t for t in replay.turns}
    for player_id, day, actions in _sequence(text, lambda pos: _action_line(text, pos)):
        turn = by_key.get((player_id, day))
        if turn is None:
            # Multiplayer games occasionally log turns that never got a snapshot.
            replay.orphan_action_turns += 1
            continue
        turn.actions = actions


def _action_line(text: bytes, pos: int) -> tuple[tuple[int, int, list[dict]], int]:
    if text[pos : pos + 2] != _ACTIONS_PREFIX:
        raise ReplayFormatError(f"bad action line at offset {pos}")
    end = text.index(b";", pos)
    player_id = int(text[pos + 2 : end])
    if text[end + 1 : end + 3] != b"d:":
        raise ReplayFormatError(f"bad action line day at offset {end + 1}")
    pos = text.index(b";", end + 1)
    day = int(text[end + 3 : pos])
    if text[pos + 1 : pos + 3] != b"a:":
        raise ReplayFormatError(f"bad action line array at offset {pos + 1}")
    body, pos = phpser.load_at(text, pos + 3)
    if not isinstance(body, list) or len(body) != 3 or int(body[0]) != player_id:
        raise ReplayFormatError(f"unexpected action header for player {player_id}, day {day}")
    # AWBW sometimes logs the literal string "Array" in place of an action.
    actions = [json.loads(s) for s in _members(body[2]) if s != "Array"]
    return (player_id, day, actions), pos


def _sequence(text: bytes, parse_one):
    """Yield newline-separated values parsed by ``parse_one(pos) -> (value, end)``."""
    pos, n = 0, len(text)
    while pos < n:
        value, pos = parse_one(pos)
        yield value
        if pos < n:
            if text[pos] != 0x0A:
                raise ReplayFormatError(f"expected newline at offset {pos}")
            pos += 1


def _members(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    return list(value)


def _read_entries(data: bytes) -> tuple[bytes, bytes | None]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ReplayFormatError(f"not a zip file: {data[:80]!r}") from e
    snapshots = actions = None
    with archive:
        for info in archive.infolist():
            if info.is_dir() or "__MACOSX" in info.filename or info.compress_size == 0:
                continue
            text = _gunzip(archive.read(info))
            if text.startswith(_SNAPSHOT_PREFIX):
                if snapshots is not None:
                    raise ReplayFormatError("zip holds more than one snapshot file")
                snapshots = text
            elif text.startswith(_ACTIONS_PREFIX):
                if actions is not None:
                    raise ReplayFormatError("zip holds more than one action log")
                actions = text
    if snapshots is None:
        raise ReplayFormatError("no awbwGame snapshot file in zip (old XML-format replay?)")
    return snapshots, actions


def _gunzip(data: bytes) -> bytes:
    """Decompress a gzip stream of one or more members (AWBW appends one per turn)."""
    parts: list[bytes] = []
    budget = MAX_ENTRY_BYTES
    while data:
        d = zlib.decompressobj(wbits=31)
        try:
            part = d.decompress(data, budget)
        except zlib.error as e:
            raise ReplayFormatError(f"corrupt gzip entry: {e}") from e
        if d.unconsumed_tail:
            raise ReplayFormatError(f"zip entry decompresses to more than {MAX_ENTRY_BYTES} bytes")
        if not d.eof:
            raise ReplayFormatError("truncated gzip entry")
        parts.append(part)
        budget -= len(part)
        data = d.unused_data
    return b"".join(parts)
