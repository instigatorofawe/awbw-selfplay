"""Write AWBW replay zips as AWBW does (``parse`` reads them back).

A zip holds two entries, ``<id>`` (the snapshots) and ``a<id>`` (the action
log), each a gzip member per turn ending in a newline. Action JSON is PHP's
``json_encode``: compact, ASCII only, slashes escaped. ``php_json(json.loads(s))
== s`` for every logged action of 2,000 replays.
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from collections.abc import Sequence
from typing import Any

from .. import phpser


def php_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).replace("/", "\\/")


def action_line(player_id: int, day: int, actions: Sequence[dict[str, Any] | str]) -> bytes:
    """One turn's log line: ``p:<player>;d:<day>;a:`` and ``[player, day, [json, ...]]``.
    A str action is written as it is (AWBW sometimes logs "Array")."""
    encoded = [a if isinstance(a, str) else php_json(a) for a in actions]
    return b"p:%d;d:%d;a:" % (player_id, day) + phpser.dumps([player_id, day, encoded])


def replay_zip(game_id: int, snapshots: Sequence[phpser.PhpObject | bytes], lines: Sequence[bytes]) -> bytes:
    """The zip for one game: a snapshot per turn (``awbwGame`` objects, or their
    serialized bytes) and the log's lines, one per turn."""

    def entry(parts: Sequence[bytes]) -> bytes:
        return b"".join(gzip.compress(p + b"\n", 9, mtime=0) for p in parts)

    def fixed(name: str) -> zipfile.ZipInfo:  # dated 1980 like the gzip members' 0, so a game's zip is always the same
        info = zipfile.ZipInfo(name)
        info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o600 << 16
        return info

    snaps = [s if isinstance(s, bytes) else phpser.dumps(s) for s in snapshots]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(fixed(str(game_id)), entry(snaps))
        if lines:
            z.writestr(fixed(f"a{game_id}"), entry(lines))
    return buf.getvalue()
