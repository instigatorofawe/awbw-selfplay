from awbw.replay.sources import IndexEntry, parse_search_page

# One result row as served by awbw.mooo.com (including its stray </a> in rating cells).
ROW = (
    '<tr><td class="dC"><a href="replay/1745964.zip"></a></td>'
    '<td class="nC"><a href="https://awbw.amarriner.com/game.php?games_id=1745964">GL STD [T4]: a vs b</a></td>'
    '<td class="mapIC"><img loading="lazy" src="https://awbw.amarriner.com/smallmaps/176376.png"></td>'
    '<td class="mC"><a href="https://awbw.amarriner.com/prevmaps.php?maps_id=176376">Indochina</a></td>'
    '<td class="coC l" data-sort="jake"></td>'
    '<td class="pC l"><a href="https://awbw.amarriner.com/profile.php?username=a">a</a></td>'
    '<td class="eC">886</a></td>'
    '<td class="coC" data-sort="adder"></td>'
    '<td class="pC"><a href="https://awbw.amarriner.com/profile.php?username=b">b</a></td>'
    '<td class="eC">958</a></td>'
    '<td class="daC">15</td><td class="dtC" data-sort="500">2026-10-12</td></tr>'
)
PAGE = f'<table class="sortable"><thead><tr><th>Game</th></tr></thead><tbody>{ROW}</tbody></table>'


def test_parse_search_row():
    [e] = parse_search_page(PAGE)
    assert (e.game_id, e.map_id, e.map_name, e.days, e.date) == (1745964, 176376, "Indochina", 15, "2026-10-12")
    assert e.name == "GL STD [T4]: a vs b"
    assert [(p.name, p.co, p.rating, p.lost) for p in e.players] == [
        ("a", "jake", 886, True),
        ("b", "adder", 958, False),
    ]


def test_header_and_empty_pages():
    assert parse_search_page("<html><body>no results</body></html>") == []


def test_index_entry_json_roundtrip():
    [e] = parse_search_page(PAGE)
    assert IndexEntry.from_json(e.to_json()) == e
