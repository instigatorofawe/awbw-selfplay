"""Where replays come from.

* awbw.mooo.com: a public replay archive with a search index (no login).
  Search results carry ratings (after the game; see RatingHistory) and the
  loser of each game.
* awbw.amarriner.com: the official site. Replay downloads need a logged-in
  session; credentials come from the environment or a local file, never code.
"""

from __future__ import annotations

import bisect
import datetime
import itertools
import json
import os
import re
import urllib.parse
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from ..net import Http, HttpError, cache_dir, write_atomic

MOOO = "http://awbw.mooo.com"
MOOO_PAGE_SIZE = 500
MOOO_MAX_RESULTS = 10_000  # per query; slice by date to go further back
AWBW = "https://awbw.amarriner.com"
CREDENTIALS_FILE = Path.home() / ".config" / "awbw-selfplay" / "credentials"


class ReplayUnavailable(RuntimeError):
    pass


def replay_path(game_id: int, root: Path | None = None) -> Path:
    """Where a replay is cached: in ``root``, by default the cache's ``replays``."""
    return (root or cache_dir() / "replays") / f"{game_id}.zip"


def fetch_replay(
    game_id: int, http: Http, source: str = "mooo", session: AwbwSession | None = None, root: Path | None = None
) -> Path:
    """Download a replay zip into the cache (``root``) unless it is already there."""
    path = replay_path(game_id, root)
    if path.exists():
        return path
    if source == "mooo":
        try:
            data = http.get(f"{MOOO}/replay/{game_id}.zip")
        except HttpError as e:
            raise ReplayUnavailable(f"game {game_id}: mooo returned HTTP {e.status}") from e
    elif source == "awbw":
        if session is None:
            raise ValueError("source='awbw' needs a logged-in AwbwSession")
        data = session.download_replay(game_id)
    else:
        raise ValueError(f"unknown replay source {source!r}")
    if not data.startswith(b"PK"):
        raise ReplayUnavailable(f"game {game_id}: not a zip: {data[:200]!r}")
    write_atomic(path, data)
    return path


def fetch_replays(
    game_ids: Iterable[int], http: Http, parallel: int = 1, root: Path | None = None
) -> Iterator[tuple[int, Path | ReplayUnavailable]]:
    """Download mooo replays with up to ``parallel`` in flight (``http``'s rate
    limit still spaces their starts), yielding each game's path, or why it is
    unavailable, as it finishes. Stopping early waits for those in flight."""

    def one(game_id: int) -> tuple[int, Path | ReplayUnavailable]:
        try:
            return game_id, fetch_replay(game_id, http, root=root)
        except ReplayUnavailable as e:
            return game_id, e

    ids = iter(game_ids)
    with ThreadPoolExecutor(parallel) as pool:
        pending = {pool.submit(one, game_id) for game_id in itertools.islice(ids, parallel)}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                game_id = next(ids, None)
                if game_id is not None:
                    pending.add(pool.submit(one, game_id))


# --- awbw.mooo.com search index ---------------------------------------------


@dataclass(slots=True)
class IndexPlayer:
    name: str
    co: str
    rating: int | None
    lost: bool


@dataclass(slots=True)
class IndexEntry:
    game_id: int
    name: str
    map_id: int
    map_name: str
    days: int
    date: str
    players: list[IndexPlayer] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> IndexEntry:
        d = json.loads(line)
        d["players"] = [IndexPlayer(**p) for p in d["players"]]
        return cls(**d)


def search_mooo(query: str, http: Http, *, max_pages: int | None = None) -> list[IndexEntry]:
    """Run a mooo search and follow its pagination."""
    entries: dict[int, IndexEntry] = {}
    page = 0
    while max_pages is None or page < max_pages:
        params = {"q": query}
        if page:
            params["offset"] = str(1 + page * MOOO_PAGE_SIZE)
        html = http.get(f"{MOOO}/search?{urllib.parse.urlencode(params)}").decode("utf-8", "replace")
        rows = parse_search_page(html)
        before = len(entries)
        for row in rows:
            entries.setdefault(row.game_id, row)
        # Past MOOO_MAX_RESULTS, mooo repeats its last page instead of ending.
        if len(rows) < MOOO_PAGE_SIZE or len(entries) == before:
            break
        page += 1
    return list(entries.values())


def next_slice_before(entries: list[IndexEntry], before: str) -> str | None:
    """The ``before`` date for the slice after one whose results are ``entries``.

    mooo's ``before YYYY-MM-DD`` is strict and results run newest first, so a
    slice cut off by MOOO_MAX_RESULTS may hold only part of its oldest day. The
    next slice starts the day after it and repeats that day; fetching skips the
    replays already cached. Returns None when the slice reached the archive's
    start, which shows as a short last page.
    """
    if len(entries) <= MOOO_MAX_RESULTS - MOOO_PAGE_SIZE:
        return None
    oldest = min(e.date for e in entries)
    after = (datetime.date.fromisoformat(oldest) + datetime.timedelta(days=1)).isoformat()
    # A whole slice inside one day would repeat forever; skip past that day instead.
    return after if after < before else oldest


def parse_search_page(html: str) -> list[IndexEntry]:
    parser = _SearchPageParser()
    parser.feed(html)
    parser.close()
    return [entry for cells in parser.rows if (entry := _row_to_entry(cells)) is not None]


@dataclass(slots=True)
class _Cell:
    classes: list[str]
    attrs: dict[str, str]
    hrefs: list[str] = field(default_factory=list)
    text: str = ""


class _SearchPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_Cell]] = []
        self._in_body = False
        self._row: list[_Cell] | None = None
        self._cell: _Cell | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if tag == "tbody":
            self._in_body = True
        elif tag == "tr" and self._in_body:
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = _Cell(classes=a.get("class", "").split(), attrs=a)
            self._row.append(self._cell)
        elif tag == "a" and self._cell is not None and "href" in a:
            self._cell.hrefs.append(a["href"])

    def handle_endtag(self, tag: str) -> None:
        if tag == "tbody":
            self._in_body = False
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = self._cell = None
        elif tag == "td":
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.text += data


def _first_int(pattern: str, texts: list[str]) -> int | None:
    for t in texts:
        if m := re.search(pattern, t):
            return int(m.group(1))
    return None


def _row_to_entry(cells: list[_Cell]) -> IndexEntry | None:
    by_class: dict[str, list[_Cell]] = {}
    for c in cells:
        for cls in c.classes:
            by_class.setdefault(cls, []).append(c)
    game_id = _first_int(r"replay/(\d+)\.zip", [h for c in by_class.get("dC", []) for h in c.hrefs])
    if game_id is None:
        return None
    name_cell = (by_class.get("nC") or [None])[0]
    map_cell = (by_class.get("mC") or [None])[0]
    # Each player contributes a CO cell, a player cell and a rating cell, in order.
    cos, names, ratings = by_class.get("coC", []), by_class.get("pC", []), by_class.get("eC", [])
    if not len(cos) == len(names) == len(ratings):
        return None
    players = []
    for co, player, rating in zip(cos, names, ratings, strict=True):
        text = rating.text.strip()
        players.append(
            IndexPlayer(
                name=player.text.strip(),
                co=co.attrs.get("data-sort", ""),
                rating=int(text) if text.lstrip("-").isdigit() else None,
                lost="l" in player.classes,
            )
        )
    days = (by_class.get("daC") or [None])[0]
    date = (by_class.get("dtC") or [None])[0]
    return IndexEntry(
        game_id=game_id,
        name=name_cell.text.strip() if name_cell else "",
        map_id=_first_int(r"maps_id=(\d+)", map_cell.hrefs if map_cell else []) or 0,
        map_name=map_cell.text.strip() if map_cell else "",
        days=int(days.text.strip()) if days and days.text.strip().isdigit() else 0,
        date=date.text.strip() if date else "",
        players=players,
    )


def write_index(path: Path, entries: list[IndexEntry]) -> None:
    write_atomic(path, "".join(e.to_json() + "\n" for e in entries).encode())


def read_index(path: Path) -> list[IndexEntry]:
    return [IndexEntry.from_json(line) for line in path.read_text().splitlines() if line.strip()]


class RatingHistory:
    """Each player's rating over time, from mooo index rows.

    A row shows ratings after its game: in 61,700 rows of players with 20+
    games, winners sat 9.6 points above their neighbouring games and losers
    9.8 below. Players in a game are usually close, so the higher shown
    rating won 83% of 39,000 games; of v0's games, the higher rating from
    before the game started won 61%. ``before`` reads only earlier games.
    """

    def __init__(self, entries: Iterable[IndexEntry]):
        seen: dict[str, dict[int, tuple[str, int]]] = {}
        for e in entries:
            for p in e.players:
                if p.rating is not None and e.date:
                    seen.setdefault(p.name, {})[e.game_id] = (e.date, p.rating)
        # By end date, then game id: the order games on the same day ended in is unknown.
        self._games = {name: sorted((d, g, r) for g, (d, r) in games.items()) for name, games in seen.items()}

    def before(self, name: str, date: str) -> int | None:
        """``name``'s rating after their latest game that ended on a day before ``date``
        (YYYY-MM-DD, anything after it ignored), or None."""
        games = self._games.get(name, [])
        i = bisect.bisect_left(games, (date[:10],))
        return games[i - 1][2] if i else None


# --- awbw.amarriner.com ------------------------------------------------------


class AwbwSession:
    """Logged-in session for replay downloads from the official site.

    Credentials, in order of precedence:

    * ``AWBW_SESSION``: an existing ``PHPSESSID`` cookie value;
    * ``AWBW_USERNAME`` and ``AWBW_PASSWORD``;
    * ``~/.config/awbw-selfplay/credentials`` with ``username=...`` and
      ``password=...`` lines.
    """

    def __init__(self, http: Http):
        self.http = http

    def login(self) -> None:
        if session_id := os.environ.get("AWBW_SESSION"):
            self.http.cookies["PHPSESSID"] = session_id
            return
        username, password = _credentials()
        body, _ = self.http.post(f"{AWBW}/logincheck.php", {"username": username, "password": password})
        if body.strip() != b"1" or "PHPSESSID" not in self.http.cookies:
            raise PermissionError("AWBW login failed: check username and password")

    def download_replay(self, game_id: int) -> bytes:
        data = self.http.get(f"{AWBW}/replay_download.php?games_id={game_id}")
        if data.startswith(b"{"):
            message = json.loads(data).get("message", data[:200])
            raise ReplayUnavailable(f"game {game_id}: {message}")
        return data


def _credentials() -> tuple[str, str]:
    username, password = os.environ.get("AWBW_USERNAME"), os.environ.get("AWBW_PASSWORD")
    if not (username and password) and CREDENTIALS_FILE.exists():
        values = dict(line.split("=", 1) for line in CREDENTIALS_FILE.read_text().splitlines() if "=" in line)
        username = username or values.get("username", "").strip()
        password = password or values.get("password", "").strip()
    if not (username and password):
        raise PermissionError(
            f"no AWBW credentials: set AWBW_USERNAME/AWBW_PASSWORD or AWBW_SESSION, or write {CREDENTIALS_FILE}"
        )
    return username, password
