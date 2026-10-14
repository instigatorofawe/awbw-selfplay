"""Polite HTTP client: one shared rate limit, retries with backoff, size caps.

A 429 or 503 is the server asking us to slow down: every thread waits out its
Retry-After, and requests are spaced twice as far apart from then on. Other
5xx answers, timeouts and dropped connections are retried after a backoff.
A request also times out if its whole response takes longer than
``deadline``: a server that sends a few bytes now and then never trips the
socket's own timeout. Each retry is reported on stderr and counted in
``Http.retried``.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path

USER_AGENT = "awbw-selfplay/0.1 (research; replay ingestion)"
MAX_RESPONSE_BYTES = 256 * 1024 * 1024
MAX_INTERVAL = 60.0  # the widest spacing slowing down goes to
MAX_PAUSE = 600.0  # the longest Retry-After honoured


def cache_dir() -> Path:
    """Root of the local data cache (``$AWBW_CACHE``, default ``./cache``)."""
    return Path(os.environ.get("AWBW_CACHE", "cache"))


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


class HttpError(RuntimeError):
    def __init__(self, url: str, status: int, body: bytes = b""):
        super().__init__(f"HTTP {status} for {url}")
        self.url = url
        self.status = status
        self.body = body


class Http:
    def __init__(self, min_interval: float = 1.0, timeout: float = 60.0, deadline: float = 300.0, retries: int = 3):
        self.min_interval = min_interval
        self.timeout = timeout  # for each socket operation
        self.deadline = deadline  # for the whole response
        self.retries = retries
        self.cookies: dict[str, str] = {}
        self.retried = 0
        self._lock = threading.Lock()
        self._next_request = 0.0

    def get(self, url: str) -> bytes:
        return self._request(urllib.request.Request(url))[0]

    def post(self, url: str, form: dict[str, str]) -> tuple[bytes, Message]:
        body = urllib.parse.urlencode(form).encode()
        return self._request(urllib.request.Request(url, data=body, method="POST"))

    def post_json(self, url: str, payload: object) -> bytes:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        return self._request(req)[0]

    def _wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_request - now
            self._next_request = max(now, self._next_request) + self.min_interval
        if delay > 0:
            time.sleep(delay)

    def _slow_down(self, pause: float) -> None:
        """Every thread waits ``pause`` seconds; requests are spaced twice as far apart."""
        with self._lock:
            self.min_interval = min(max(2 * self.min_interval, 1.0), MAX_INTERVAL)
            self._next_request = max(self._next_request, time.monotonic() + pause)

    def _request(self, req: urllib.request.Request) -> tuple[bytes, Message]:
        req.add_header("User-Agent", USER_AGENT)
        if self.cookies:
            req.add_header("Cookie", "; ".join(f"{k}={v}" for k, v in self.cookies.items()))
        for attempt in range(self.retries + 1):
            self._wait_turn()
            deadline = time.monotonic() + self.deadline
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = self._read_body(req, resp, deadline)
                    self._remember_cookies(resp.headers)
                    return data, resp.headers
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == self.retries:
                    raise HttpError(req.full_url, e.code, e.read(4096)) from e
                e.close()
                self._note_retry(req, f"HTTP {e.code}", attempt)
                if e.code in (429, 503):
                    self._slow_down(retry_after(e.headers, 2.0 * 2**attempt))
                    continue
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
                if attempt == self.retries:
                    raise
                self._note_retry(req, repr(e), attempt)
            time.sleep(2.0 * 2**attempt)
        raise AssertionError("unreachable")

    def _read_body(self, req: urllib.request.Request, resp, deadline: float) -> bytes:
        chunks: list[bytes] = []
        size = 0
        while chunk := resp.read1(1 << 16):
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise HttpError(req.full_url, resp.status, b"response too large")
            if time.monotonic() > deadline:
                raise TimeoutError(f"response not complete after {self.deadline:g}s")
            chunks.append(chunk)
        return b"".join(chunks)

    def _note_retry(self, req: urllib.request.Request, why: str, attempt: int) -> None:
        with self._lock:
            self.retried += 1
        print(f"{req.full_url}: {why}; retry {attempt + 1} of {self.retries}", file=sys.stderr)

    def _remember_cookies(self, headers: Message) -> None:
        for header in headers.get_all("Set-Cookie") or []:
            name, _, rest = header.partition("=")
            self.cookies[name.strip()] = rest.split(";", 1)[0]


def retry_after(headers: Message | None, default: float) -> float:
    """Seconds a response's Retry-After asks for (seconds or an HTTP date), at
    least ``default`` and at most MAX_PAUSE."""
    value = headers.get("Retry-After") if headers is not None else None
    seconds = default
    if value:
        try:
            seconds = float(value)
        except ValueError:
            with contextlib.suppress(TypeError, ValueError):
                seconds = parsedate_to_datetime(value).timestamp() - time.time()
    return min(max(seconds, default), MAX_PAUSE)
