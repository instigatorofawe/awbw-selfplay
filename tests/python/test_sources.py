import email.utils
import io
import threading
import time
import urllib.error
from email.message import Message

from awbw import net
from awbw.net import Http, HttpError, retry_after
from awbw.replay import sources
from awbw.replay.sources import IndexEntry, IndexPlayer, RatingHistory, parse_search_page

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


def test_ratings_come_from_games_that_ended_on_an_earlier_day():
    def game(game_id: int, date: str, a: int | None, b: int) -> IndexEntry:
        players = [IndexPlayer("a", "jake", a, False), IndexPlayer("b", "adder", b, True)]
        return IndexEntry(game_id=game_id, name="", map_id=1, map_name="", days=10, date=date, players=players)

    h = RatingHistory(
        [game(3, "2026-10-05", 910, 890), game(1, "2026-10-01", 900, 900), game(2, "2026-10-05", None, 880)]
    )
    assert h.before("a", "2026-10-01 09:00:00") is None  # nothing ended before that day
    assert h.before("a", "2026-10-02") == 900
    assert h.before("a", "2026-10-05 23:59:59") == 900  # games ending that day may have ended after it started
    assert h.before("a", "2026-10-06") == 910
    assert h.before("b", "2026-10-06") == 890  # same day: the later game id
    assert h.before("c", "2026-10-06") is None


def _headers(**fields: str) -> Message:
    m = Message()
    for k, v in fields.items():
        m[k.replace("_", "-")] = v
    return m


def test_retry_after_reads_seconds_or_a_date():
    assert retry_after(_headers(Retry_After="30"), 2.0) == 30.0
    assert retry_after(_headers(Retry_After="1"), 2.0) == 2.0  # never sooner than our own backoff
    assert retry_after(_headers(Retry_After="86400"), 2.0) == net.MAX_PAUSE
    assert retry_after(_headers(Retry_After="soon"), 2.0) == 2.0
    assert retry_after(_headers(), 4.0) == 4.0
    when = email.utils.formatdate(time.time() + 120, usegmt=True)
    assert 100 < retry_after(_headers(Retry_After=when), 2.0) <= 120


def test_http_slows_down_when_the_server_asks(monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = Message()

    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", _headers(Retry_After="5"), None)
        return Response(b"ok")

    slept = []
    monkeypatch.setattr(net.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(net.time, "sleep", slept.append)
    http = Http(min_interval=0.25)
    assert http.get("http://example.test/a") == b"ok"
    assert len(calls) == 2
    assert slept and 4.5 < slept[0] <= 5.0  # waited out Retry-After, not the usual backoff
    assert http.min_interval == 1.0  # and spaces requests further apart from now on
    assert http.retried == 1


def test_http_retries_a_dropped_connection(monkeypatch, capsys):
    class Response(io.BytesIO):
        status = 200
        headers = Message()

    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise ConnectionResetError(54, "Connection reset by peer")
        return Response(b"ok")

    slept = []
    monkeypatch.setattr(net.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(net.time, "sleep", slept.append)
    http = Http(min_interval=0.25)
    assert http.get("http://example.test/a") == b"ok"
    assert len(calls) == 2 and http.retried == 1 and http.min_interval == 0.25
    assert "Connection reset by peer" in capsys.readouterr().err


def test_http_gives_up_on_a_response_that_trickles_in(monkeypatch, capsys):
    clock = [0.0]

    class Trickle(io.BytesIO):
        status = 200
        headers = Message()

        def read1(self, size: int | None = -1, /) -> bytes:
            clock[0] += 50.0  # each byte arrives inside the socket timeout
            return super().read1(1)

    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        return Trickle(b"stalled" if len(calls) == 1 else b"ok")

    monkeypatch.setattr(net.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(net.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(net.time, "sleep", lambda s: None)
    http = Http(min_interval=0.25, deadline=300.0)
    assert http.get("http://example.test/a") == b"ok"  # 2 bytes in 100 s; 7 would take 350
    assert len(calls) == 2 and http.retried == 1
    assert "not complete after 300s" in capsys.readouterr().err


def test_fetch_replays_keeps_n_in_flight(monkeypatch, tmp_path):
    monkeypatch.setenv("AWBW_CACHE", str(tmp_path))
    both = threading.Barrier(2, timeout=10)
    lock = threading.Lock()
    active, peak = 0, 0

    class FakeHttp(Http):
        def get(self, url: str) -> bytes:
            nonlocal active, peak
            game_id = int(url.rsplit("/", 1)[1].removesuffix(".zip"))
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if game_id in (1, 2):
                    both.wait()  # the first two are downloading at once
                if game_id == 4:
                    raise HttpError(url, 404)
                return b"PK" + bytes([game_id])
            finally:
                with lock:
                    active -= 1

    got = dict(sources.fetch_replays([1, 2, 3, 4, 5], FakeHttp(min_interval=0.0), parallel=2))
    assert sorted(got) == [1, 2, 3, 4, 5] and peak == 2
    assert isinstance(got[4], sources.ReplayUnavailable)
    assert not isinstance(got[5], sources.ReplayUnavailable) and got[5].read_bytes() == b"PK\x05"

    seen = []
    for game_id, _ in sources.fetch_replays([6, 7, 8, 9], FakeHttp(min_interval=0.0), parallel=2):
        seen.append(game_id)
        break
    started = {int(p.stem) for p in (tmp_path / "replays").glob("*.zip")} - set(got)
    assert len(seen) == 1 and seen[0] in started and started <= {6, 7}  # stopping waits only for those in flight
