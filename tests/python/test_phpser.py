import pytest

from awbw.phpser import PhpObject, PhpParseError, load_at, loads


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"i:42;", 42),
        (b"i:-7;", -7),
        (b"d:7.3;", 7.3),
        (b"b:1;", True),
        (b"b:0;", False),
        (b"N;", None),
        (b's:0:"";', ""),
        (b's:5:"a;b"c";', 'a;b"c'),
    ],
)
def test_scalars(raw, expected):
    assert loads(raw) == expected


def test_string_length_counts_bytes():
    assert loads('s:8:"Pokémon";'.encode()) == "Pokémon"  # é is 2 bytes in UTF-8


def test_sequential_array_is_list_and_sparse_array_is_dict():
    assert loads(b'a:2:{i:0;s:1:"x";i:1;i:5;}') == ["x", 5]
    assert loads(b'a:2:{i:1;i:5;s:1:"k";N;}') == {1: 5, "k": None}
    assert loads(b"a:0:{}") == []


def test_object():
    obj = loads(b'O:8:"awbwUnit":2:{s:2:"id";i:9;s:10:"hit_points";d:7.3;}')
    assert obj == PhpObject("awbwUnit", {"id": 9, "hit_points": 7.3})


def test_load_at_returns_end_offset():
    data = b"i:1;\ni:2;"
    value, end = load_at(data, 0)
    assert (value, data[end]) == (1, ord("\n"))
    assert load_at(data, end + 1) == (2, len(data))


@pytest.mark.parametrize(
    "raw",
    [b"i:1", b's:9:"short";', b"a:2:{i:0;i:1;}", b"x:1;", b"i:1;junk", b'O:3:"abc":1:{s:1:"k";i:1;'],
)
def test_malformed_input_raises(raw):
    with pytest.raises(PhpParseError):
        loads(raw)
