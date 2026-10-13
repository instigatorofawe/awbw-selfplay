# awbw-selfplay

An exact, fast [Advance Wars By Web](https://awbw.amarriner.com) engine in
C++, bound to Python with nanobind. It is for:

1. parsing real AWBW replays into training data (a critic first),
2. deep MCTS self-play.

See [PLAN.md](PLAN.md) for the design and milestones. Current status: **M1,
the C++ rules engine**. Replay ingestion (M0) is done. The engine applies
every action kind, every CO power's immediate effects, rain and snow, and the
start of each turn. Replaying the cached games turn by turn from their
snapshots, it reproduces 197,481 of 197,503 checked turns exactly (3.53M
actions); the 22 others are AWBW oddities listed in PLAN.md.

## Setup

Requires [uv](https://docs.astral.sh/uv/), CMake ≥ 3.27 and a C++20 compiler.

```sh
uv sync                      # builds the C++ extension; rebuilds when src/ changes
uv run pre-commit install    # ruff, clang-format and hygiene hooks
uv run pytest
```

C++ unit tests (googletest) and micro-benchmarks (Google Benchmark) build
with plain CMake; both libraries are fetched at pinned versions:

```sh
cmake -S . -B build/cpp -DAWBW_BUILD_TESTS=ON -DAWBW_BUILD_BENCHMARKS=ON
cmake --build build/cpp -j
ctest --test-dir build/cpp       # or build/cpp/awbw_tests
build/cpp/awbw_bench
# -DAWBW_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug for ASan/UBSan
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
uv run awbw league --fetch-maps   # snapshot the Global League std rotation + settings
uv run awbw check-combat      # every logged attack vs the engine's damage formula
uv run awbw check-moves       # load every snapshot into the engine; check logged moves
uv run awbw check-turns       # replay every turn with the engine; compare with the next snapshot
```

Requests are rate-limited (`--rate`, default one per second). Please keep it
that way: both sites are community-run.

`awbw fetch --source awbw` downloads from the official site instead, which
needs a login. It reads `AWBW_SESSION` (a `PHPSESSID` value), or
`AWBW_USERNAME` / `AWBW_PASSWORD`, or `~/.config/awbw-selfplay/credentials`
with `username=` and `password=` lines.

## Layout

```
src/core/        C++ engine (no Python dependency); tables.h is generated
src/bindings/    nanobind module awbw._core
python/awbw/     Python package: replay parsing, sources, maps, engine glue, CLI
python/awbw/rules/  rules tables (JSON): units, terrain, damage, COs
tools/           gen_tables.py: rules JSON -> src/core/tables.h
tests/cpp/       googletest unit tests for the engine core
bench/           Google Benchmark micro-benchmarks
tests/python/    pytest; tests over real replays run on whatever is cached
```
