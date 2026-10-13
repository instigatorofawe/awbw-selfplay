# awbw-selfplay

An exact, fast [Advance Wars By Web](https://awbw.amarriner.com) engine in
C++, bound to Python with nanobind. It is for:

1. parsing real AWBW replays into training data (a critic first),
2. deep MCTS self-play.

See [PLAN.md](PLAN.md) for the design and milestones. Current status: **M0,
replay ingestion**.

## Setup

Requires [uv](https://docs.astral.sh/uv/), CMake ≥ 3.27 and a C++20 compiler.

```sh
uv sync                      # builds the C++ extension; rebuilds when src/ changes
uv run pre-commit install    # ruff, clang-format and hygiene hooks
uv run pytest
```

## Replay data

Replays come from the public archive at [awbw.mooo.com](http://awbw.mooo.com).
Everything downloaded lives under `cache/` (or `$AWBW_CACHE`), which is
gitignored.

```sh
# index standard 2p league games (mooo search syntax), then download some
uv run awbw index "2p global league std nodraw" --max-pages 1
uv run awbw fetch --index cache/index/2p_global_league_std_nodraw.jsonl --limit 100

uv run awbw inspect 1745964   # summarize one cached replay
uv run awbw stats             # settings, endings and v1-filter counts over the cache
uv run awbw fetch-map 176376  # map terrain from AWBW's public map API
```

Requests are rate-limited (`--rate`, default one per second). Please keep it
that way: both sites are community-run.

`awbw fetch --source awbw` downloads from the official site instead, which
needs a login. It reads `AWBW_SESSION` (a `PHPSESSID` value), or
`AWBW_USERNAME` / `AWBW_PASSWORD`, or `~/.config/awbw-selfplay/credentials`
with `username=` and `password=` lines.

## Layout

```
src/core/        C++ engine (no Python dependency)
src/bindings/    nanobind module awbw._core
python/awbw/     Python package: replay parsing, sources, maps, CLI
tests/python/    pytest; tests over real replays run on whatever is cached
```
