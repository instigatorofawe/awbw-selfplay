from awbw.net import Http
from awbw.replay import sources
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


def _entry(game_id: int, date: str) -> IndexEntry:
    return IndexEntry(game_id=game_id, name="", map_id=1, map_name="", days=10, date=date)


def test_search_stops_when_mooo_repeats_its_last_page(monkeypatch):
    monkeypatch.setattr(sources, "MOOO_PAGE_SIZE", 2)
    pages = [[1, 2], [3, 4]]  # every later offset serves the last page again

    class FakeHttp(Http):
        calls = 0

        def get(self, url: str) -> bytes:
            page = pages[min(self.calls, len(pages) - 1)]
            self.calls += 1
            return f"<tbody>{''.join(ROW.replace('1745964', str(g)) for g in page)}</tbody>".encode()

    http = FakeHttp()
    assert [e.game_id for e in sources.search_mooo("2p", http)] == [1, 2, 3, 4]
    assert http.calls == 3


def test_next_slice_repeats_the_oldest_day_of_a_full_slice():
    full = [_entry(i, "2026-04-25" if i < 10 else "2026-05-01") for i in range(sources.MOOO_MAX_RESULTS)]
    assert sources.next_slice_before(full, "2026-07-17") == "2026-04-26"
    one_day = [_entry(i, "2026-04-25") for i in range(sources.MOOO_MAX_RESULTS)]
    assert sources.next_slice_before(one_day, "2026-04-26") == "2026-04-25"
    assert sources.next_slice_before(full[:9000], "2026-07-17") is None
