"""Minimal PHP ``unserialize`` and ``serialize`` for AWBW replay files.

Parses bytes, not str: PHP string lengths count bytes, so slicing by length is
only exact on the raw encoding. Arrays whose keys are exactly 0..n-1 become
lists; other arrays become dicts. Objects become :class:`PhpObject`.

``dumps`` writes what AWBW writes, byte for byte: ``dumps(loads(x)) == x`` for
every snapshot and log line of 2,000 replays. Floats have 17 significant
digits, as AWBW's PHP gives them (``d:5.7000000000000002;``, ``d:10;``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PhpParseError(ValueError):
    pass


@dataclass(slots=True)
class PhpObject:
    class_name: str
    fields: dict[str, Any] = field(default_factory=dict)


def loads(data: bytes) -> Any:
    """Parse exactly one serialized value spanning all of ``data``."""
    value, end = load_at(data, 0)
    if end != len(data):
        raise PhpParseError(f"trailing data at offset {end}")
    return value


def load_at(data: bytes, pos: int) -> tuple[Any, int]:
    """Parse one value starting at ``pos``; return it and the offset after it."""
    try:
        return _value(data, pos)
    except PhpParseError:
        raise
    except (IndexError, ValueError) as e:
        raise PhpParseError(f"malformed value near offset {pos}: {e}") from e


def _value(data: bytes, pos: int) -> tuple[Any, int]:
    tag = data[pos]
    if data[pos + 1] != 0x3A and tag != 0x4E:  # every tag but N is followed by ':'
        raise PhpParseError(f"expected ':' after tag at offset {pos}")

    if tag == 0x69:  # i:<int>;
        end = data.index(b";", pos + 2)
        return int(data[pos + 2 : end]), end + 1

    if tag == 0x73:  # s:<len>:"<bytes>";
        colon = data.index(b":", pos + 2)
        start = colon + 2
        stop = start + int(data[pos + 2 : colon])
        if data[colon + 1] != 0x22 or data[stop : stop + 2] != b'";':
            raise PhpParseError(f"bad string at offset {pos}")
        return data[start:stop].decode("utf-8", "replace"), stop + 2

    if tag == 0x4E:  # N;
        if data[pos + 1] != 0x3B:
            raise PhpParseError(f"bad null at offset {pos}")
        return None, pos + 2

    if tag == 0x64:  # d:<float>;  (float() accepts PHP's INF / -INF / NAN)
        end = data.index(b";", pos + 2)
        return float(data[pos + 2 : end]), end + 1

    if tag == 0x62:  # b:0; or b:1;
        if data[pos + 3] != 0x3B:
            raise PhpParseError(f"bad bool at offset {pos}")
        return data[pos + 2] == 0x31, pos + 4

    if tag == 0x61:  # a:<n>:{<key><value>...}
        colon = data.index(b":", pos + 2)
        count = int(data[pos + 2 : colon])
        if data[colon + 1] != 0x7B:
            raise PhpParseError(f"bad array at offset {pos}")
        return _members(data, colon + 2, count, as_list=True)

    if tag == 0x4F:  # O:<len>:"<class>":<n>:{<name><value>...}
        colon = data.index(b":", pos + 2)
        start = colon + 2
        stop = start + int(data[pos + 2 : colon])
        if data[colon + 1] != 0x22 or data[stop : stop + 2] != b'":':
            raise PhpParseError(f"bad object header at offset {pos}")
        name = data[start:stop].decode("utf-8", "replace")
        colon = data.index(b":", stop + 2)
        count = int(data[stop + 2 : colon])
        if data[colon + 1] != 0x7B:
            raise PhpParseError(f"bad object at offset {pos}")
        fields, end = _members(data, colon + 2, count, as_list=False)
        return PhpObject(name, fields), end

    raise PhpParseError(f"unknown tag {chr(tag)!r} at offset {pos}")


def dumps(value: Any) -> bytes:
    """Serialize as PHP does: lists become arrays keyed 0..n-1, dict keys keep
    their type (int or str), and :class:`PhpObject` fields keep their order."""
    out: list[bytes] = []
    _dump(value, out)
    return b"".join(out)


def _dump(value: Any, out: list[bytes]) -> None:
    if value is None:
        out.append(b"N;")
    elif isinstance(value, bool):
        out.append(b"b:1;" if value else b"b:0;")
    elif isinstance(value, int):
        out.append(b"i:%d;" % value)
    elif isinstance(value, float):
        out.append(b"d:" + (b"%.17G" % value) + b";")
    elif isinstance(value, str):
        raw = value.encode()
        out.append(b's:%d:"' % len(raw) + raw + b'";')
    elif isinstance(value, PhpObject):
        name = value.class_name.encode()
        out.append(b'O:%d:"' % len(name) + name + b'":%d:{' % len(value.fields))
        for k, v in value.fields.items():
            _dump(k, out)
            _dump(v, out)
        out.append(b"}")
    elif isinstance(value, (list, tuple, dict)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        out.append(b"a:%d:{" % len(value))
        for k, v in items:
            if not isinstance(k, (int, str)) or isinstance(k, bool):
                raise TypeError(f"PHP array keys are int or str, not {type(k).__name__}")
            _dump(k, out)
            _dump(v, out)
        out.append(b"}")
    else:
        raise TypeError(f"can't serialize {type(value).__name__}")


def _members(data: bytes, pos: int, count: int, *, as_list: bool) -> tuple[Any, int]:
    keys: list[Any] = []
    values: list[Any] = []
    for _ in range(count):
        key, pos = _value(data, pos)
        value, pos = _value(data, pos)
        keys.append(key)
        values.append(value)
    if data[pos] != 0x7D:
        raise PhpParseError(f"expected '}}' at offset {pos}")
    if as_list and keys == list(range(count)):
        return values, pos + 1
    return dict(zip(keys, values, strict=True)), pos + 1
