"""Polite HTTP client: one shared rate limit, retries with backoff, size caps."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from email.message import Message
from pathlib import Path

USER_AGENT = "awbw-selfplay/0.1 (research; replay ingestion)"
MAX_RESPONSE_BYTES = 256 * 1024 * 1024


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
    def __init__(self, min_interval: float = 1.0, timeout: float = 60.0, retries: int = 3):
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self.cookies: dict[str, str] = {}
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

    def _request(self, req: urllib.request.Request) -> tuple[bytes, Message]:
        req.add_header("User-Agent", USER_AGENT)
        if self.cookies:
            req.add_header("Cookie", "; ".join(f"{k}={v}" for k, v in self.cookies.items()))
        for attempt in range(self.retries + 1):
            self._wait_turn()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read(MAX_RESPONSE_BYTES + 1)
                    if len(data) > MAX_RESPONSE_BYTES:
                        raise HttpError(req.full_url, resp.status, b"response too large")
                    self._remember_cookies(resp.headers)
                    return data, resp.headers
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == self.retries:
                    raise HttpError(req.full_url, e.code, e.read(4096)) from e
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries:
                    raise
            time.sleep(2.0 * 2**attempt)
        raise AssertionError("unreachable")

    def _remember_cookies(self, headers: Message) -> None:
        for header in headers.get_all("Set-Cookie") or []:
            name, _, rest = header.partition("=")
            self.cookies[name.strip()] = rest.split(";", 1)[0]
