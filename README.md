# awbw-selfplay

An exact, fast [Advance Wars By Web](https://awbw.amarriner.com) engine in
C++, bound to Python with nanobind. It is for:

1. parsing real AWBW replays into training data (a critic first),
2. deep MCTS self-play.

See [PLAN.md](PLAN.md) for the design and milestones. Current status: M0
(replay ingestion) and M1 (the C++ rules engine, network input encoding, the
v0 dataset and a first critic) are finished, and M3 is under way: a
micro-step action space with its legal mask, the v1 dataset of every action
taken, and a policy-value network pre-trained on it. The engine applies
every action kind, every CO power's immediate effects, rain and snow, and
the start of each turn. Replaying 38,936 cached games turn by turn from
their snapshots, it reproduces 1,044,228 of 1,044,350 checked turns exactly
(18.5M actions, every one in the legal mask); the others are AWBW oddities
listed in PLAN.md.

## Setup

Requires [uv](https://docs.astral.sh/uv/), CMake ≥ 3.28 and a C++20 compiler.

```sh
uv sync                      # builds the C++ extension (rebuilt when src/ changes); installs torch
uv run pre-commit install    # ruff, clang-format and hygiene hooks
uv run pytest
```

On Linux, torch comes from PyTorch's own index through an extra: `uv sync
--extra cuda` (CUDA 13) or `uv sync --extra cpu`. Later `uv run`s keep it,
but a plain `uv sync` there removes it.

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
# or walk back through the archive, 10,000 games per date slice, up to a size cap
uv run awbw crawl "2p global league std" --before 2026-04-26 --max-gb 30

uv run awbw inspect 1745964   # summarize one cached replay
uv run awbw stats             # settings, endings and v1-filter counts over the cache
uv run awbw fetch-map 176376  # map terrain from AWBW's public map API
uv run awbw league --fetch-maps   # snapshot the Global League std rotation + settings
uv run awbw check-combat      # every logged attack vs the engine's damage formula
uv run awbw check-moves       # load every snapshot into the engine; check logged moves
uv run awbw check-turns       # replay every turn with the engine; compare with the next snapshot
uv run awbw dataset           # v0 critic dataset (turn-start states + results) -> cache/datasets/v0
uv run awbw dataset --v1      # every action taken, with its micro-steps -> cache/datasets/v1
```

Datasets store raw engine states; `awbw.data.loader.batches` encodes them
for the network (`awbw._core.encode_batch`) as it reads, and
`decision_batches` gives a v1 dataset's micro-step decisions
(`encode_policy_batch`).

## Critic

```sh
uv run awbw train-critic --out cache/critics/run1 --compile   # on CUDA, MPS or CPU
uv run awbw eval-critic cache/critics/run1/best.pt            # validation metrics, calibration
uv run awbw value cache/critics/run1/best.pt 1745964          # win chances through a replay
```

```python
from awbw.nn.critic import Critic

critic = Critic.load("cache/critics/run1/best.pt")
critic.value(states)  # raw states [N, STATE_BYTES] -> [-1, 1] for each player to move
```

## Policy

A policy-value network (the critic's trunk, plus a head that scores each
micro-step of a decision) pre-trained by behaviour cloning on the v1 dataset:

```sh
uv run awbw train-policy --out cache/policies/run1 --compile   # --fraction: share of decisions per epoch
uv run awbw eval-policy cache/policies/run1/best.pt            # by phase, against a uniform choice
uv run awbw arena cache/policies/run1/best.pt --games 32       # whole games against a random mover
uv run awbw arena cache/policies/run1/best.pt --record games.jsonl   # ...and keep them
uv run awbw view-game games.jsonl --game 0 --open              # step through one in the browser
uv run awbw export-replay games.jsonl --game 0                 # as a zip for the AWBW Replay Player
uv run awbw play cache/policies/run1/best.pt                   # play it yourself, in the browser
```

`awbw play` starts a local server and opens the page: a validation game's
map, COs and countries (`--map`, `--cos`, `--countries` change them), you
as player 1 (`--side 2` for the other), the network searching 32
simulations a decision (`--sims`, 0 for its prior). The page shows move
and attack ranges, paths, damage previews and any unit's range
(right-click), and allows undo. Finished games go to
`cache/games/local.jsonl`, for `view-game` and `export-replay`.

The art is Nintendo's, so none is in the repo. With a copy of the AWBW
Replay Player's resources in `cache/sprites/` (its `Textures/` and `Json/`
folders, extracted from `AWBWApp.Resources.dll`), the play page and the
viewer draw the board with the Player's sprites, animated; without one,
they draw coloured squares.

```python
from awbw.nn.policy import Policy

policy = Policy.load("cache/policies/run1/best.pt")
policy.score(states, picks, n_picks)  # each decision's legal ids, their priors and the value
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
python/awbw/     Python package: replay parsing, sources, maps, engine glue, datasets, CLI
python/awbw/rules/  rules tables (JSON): units, terrain, damage, COs
tools/           gen_tables.py: rules JSON -> src/core/tables.h
tests/cpp/       googletest unit tests for the engine core
bench/           Google Benchmark micro-benchmarks
tests/python/    pytest; tests over real replays run on whatever is cached
```
