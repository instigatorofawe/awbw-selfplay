# awbw-selfplay — plan

A fast, exact Advance Wars By Web (AWBW) engine in C++, bound to Python with
nanobind, for:

1. replaying real AWBW games into training data (critic / value model first),
2. deep MCTS self-play (AlphaZero-style) using that critic.

## Scope decisions

| Decision | Choice |
|---|---|
| Rules | Exact AWBW match, verified against real replays |
| Players | 2 (no team or tag games) |
| Settings | Standard only: 1000 funds per property (no high income), no starting funds, powers on; any capture limit |
| Fog | Off (games with fog are filtered out of datasets) |
| Weather | Clear only to start; random weather comes later |
| Search | Gumbel AlphaZero-style MCTS, exact simulator, explicit chance nodes |
| First model | Critic trained on positions parsed from human replays |

## 1. Repository layout

```
awbw-selfplay/
  pyproject.toml              # scikit-build-core + nanobind
  CMakeLists.txt
  data/rules/*.json           # single source of truth: units, terrain, damage chart, COs
  src/core/                   # C++20, no Python dependency
    tables.h                  # generated constexpr from data/rules
    map.h  state.h  movegen.cpp  apply.cpp  combat.cpp  turn.cpp  cos.cpp
    hash.cpp  encode.cpp  snapshot.cpp
    search/                   # batched Gumbel MCTS
  src/bindings/module.cpp     # thin nanobind layer
  python/awbw/
    rules.py  maps.py
    replay/   download.py  parse.py  validate.py  extract.py
    data/     shards.py  loader.py
    nn/       model.py  train_critic.py  selfplay.py
    render.py  agents/
  tests/{cpp,python}/   bench/
  cache/                      # gitignored: raw replays, maps, shards
```

The core rule: **Python never runs once per action in the hot loop.**
Python makes one call per batch (`collect` / `submit` / `encode_batch`), and
C++ releases the GIL for all heavy work.

## 2. State representation

### Static: `MapData`, immutable and shared through a process-wide registry

- dimensions, terrain id per tile
- move-cost tables precomputed per `[move_type][weather][tile]`
- defense stars per tile
- tile → property index; property list
- neighbor table

### Dynamic: `GameState`, POD with no pointers

```cpp
struct Unit {                 // 16 bytes
  uint32_t seq;               // creation order; AWBW tie-breaks follow unit-id order
  uint16_t tile;              // or kCarried
  uint8_t  type;              // 25 unit types; kNone = free slot
  uint8_t  owner;
  uint8_t  hp;                // internal 1..100; displayed = (hp + 9) / 10
  uint8_t  fuel, ammo;
  uint8_t  flags;             // moved | hidden (sub/stealth) | stunned | ...
  uint8_t  cargo[2];          // unit slots (Lander/Cruiser/Carrier hold 2)
  uint8_t  _pad[2];
};

struct Player {
  int32_t funds, power_charge;
  uint8_t co, power_active, power_uses, alive;
  uint8_t co_scratch[8];      // CO-specific state
};

struct GameSettings {         // varies per AWBW game; needed for exact match
  int32_t  starting_funds;
  uint16_t funds_per_property, day_limit;
  uint8_t  capture_limit, unit_limit;
  uint32_t banned_units;      // bitmask
};

struct GameState {
  uint32_t     map_id;        // resolved via MapRegistry; keeps state bytes position-independent
  GameSettings settings;
  Unit         units[MAX_UNITS];
  uint8_t      unit_at[MAX_TILES];             // occupancy grid, debug-asserted in sync
  uint8_t      prop_owner[MAX_PROPS], prop_capture[MAX_PROPS];
  Player       players[2];
  uint16_t     day;
  uint8_t      active, weather;
  uint64_t     zobrist;       // maintained incrementally
  uint64_t     rng[2];        // luck RNG lives in the state
  Pending      pending;       // partially chosen unit action (see §3)
};
static_assert(std::is_trivially_copyable_v<GameState>);
```

- **Size:** about 4–5 KB at `MAX_W = MAX_H = 40`. Check this against the map
  pool; maps that don't fit are filtered out.
- **Cloning:** a memcpy.
- **Serialization:** the raw bytes. The same bytes are what the datasets store.
- **Derived data stays out of the state.** Reachable tiles, threat maps and
  similar data are computed on demand into per-thread scratch memory.
- **Exact AWBW match needs AWBW's ordering.** Repair order, CO-power
  targeting, etc. follow unit creation order, so `seq` is kept and the order
  of units within a turn is never "normalized".

## 3. Action space: micro-steps

| Phase | Choice | Encoding |
|---|---|---|
| Idle | select a unit or factory tile, or End Turn / COP / SCOP | tile + 3 globals |
| Unit selected | destination tile | tile |
| Unit moved | Wait, Capture, Fire@tile, Load, Join, Unload@tile, Supply, Dive/Hide, Launch@tile, Repair@tile, Explode, Delete | channel × tile |
| Factory selected | unit type | 25 |

- **Action ID:** `channel * MAX_TILES + tile`, plus a few non-spatial IDs.
- **No cancel/undo.** Several unloads before Wait are allowed by keeping
  `pending` open.
- **Forced steps auto-advance.** A phase with exactly one legal action is
  applied immediately. It produces no search and no training sample.
- **Full-action API.** `enumerate_full_actions()` returns complete
  `(unit, dest, cmd, target)` tuples, used by replay decomposition and
  heuristic bots.
- **Paths:** without fog, the destination is enough. Fuel is charged at the
  shortest-path cost.

## 4. Rules engine

- **Tables:** generated from `data/rules/*.json`, which Python also reads.
  The values come from AWBW's own pages and are verified against replays,
  because AWBW differs from the cartridges in places.
- **Movement:** Dijkstra with a bucket queue (costs are small integers).
  Enemy units block; friendly units can be passed through. The fuel cap
  applies.
- **Combat:** integer math, AWBW formula.
  - Attacker uses displayed HP.
  - Defender terrain stars × defender displayed HP.
  - Air units get no terrain stars.
  - CO attack/defense modifiers; luck (good and bad ranges, depending on CO).
  - The counterattack uses post-damage HP.
- **Chance API.** One interface serves search and replays alike:

  ```cpp
  // Exact distribution over results, merged by resulting (attacker_hp, defender_hp).
  int  outcomes(const MapData&, const GameState&, Action, Outcome* out);
  void apply(const MapData&, GameState&, Action, const Outcome* forced = nullptr);
  ```

  - Search uses `outcomes` for its chance nodes.
  - Replays force the logged result. If the logged result isn't in
    `outcomes`, that is a damage-formula bug.
  - Random play samples with the in-state RNG.
- **Turn start:**
  - income
  - repairs, in AWBW order, only if affordable, and only on the right
    property type
  - resupply
  - daily fuel burn; crashing/sinking
  - power reset
- **COs:**
  - A `CoSpec` data struct covers the day-to-day modifiers.
  - A `switch` on CO id handles power effects (Sturm, Rachel, Von Bolt stun,
    Olaf snow, ...).
  - No virtual dispatch in the hot path.
- **Easy-to-miss details:**
  - Join HP cap and funds refund.
  - Capture progress resets when the capturer leaves the tile.
  - Pipe seams have HP and can be attacked.
  - Silos, Black Bomb explosions, Black Boat repairs.
  - Power-meter charging is based on funds value, and the cost per star rises
    with each use.
- **Game start:** maps carry predeployed units. In standard play the second
  player usually gets an extra infantry to offset the first-turn advantage.
  - Replay-derived states get these from the turn-0 snapshot.
  - Self-play games started from a map must place them, and must assign map
    countries to turn order exactly as AWBW does, or the compensation goes to
    the wrong player. Verify this against turn-0 snapshots.
- **Game end:** HQ capture, rout, capture limit, resign, day limit.
  - Self-play also has a training-only day cap; results there are decided by
    army value plus properties.

## 5. Replays → training data

### Sources

- **[awbw.mooo.com](http://awbw.mooo.com)** (primary, no login): a public
  replay archive. `search?q=...` returns 500 results per page (`&offset=501`,
  ...), with filters like `2p`, `global league std`, `rating>1500`, `nodraw`.
  Each row gives game id, map id, both COs, both players' ratings at match
  time, days, date, and which player lost; players are listed in turn order.
  `replay/<id>.zip` serves the replay itself.
- **awbw.amarriner.com** (fallback): `replay_download.php?games_id=N` needs a
  logged-in session (`logincheck.php`, `PHPSESSID` cookie). The map API
  `api/map/map_info.php?maps_id=N` is public JSON; its terrain grid is
  column-major (`[x][y]`).
- **Global League rotation** (`awbw league`): `api/league/maps.php` with
  `{"method": "listMaps"}` (the read-only call the public GL maps page makes).
  - Per map: mode (std/fog/hf), unit limit, capture limit, enabled CO tiers
    with their COs, and unit bans.
  - Also returns AWBW's unit-id and CO-id name tables, which the M1 rules
    tables should key on.
  - **Replays don't record unit bans or the unit limit**, so for league games
    these settings come from here. Snapshots are dated (`cache/league/`)
    because the rotation changes; the current snapshot only covers current
    maps. As of 2026-10-12, std has 14 maps, all banning Black Bomb and
    Stealth, with a unit limit of 50. Capture limits matched `capture_win` in
    all 172 cached replays on those maps.
- **Reference implementation:**
  [DeamonHunter/AWBW-Replay-Player](https://github.com/DeamonHunter/AWBW-Replay-Player)
  (C#, MIT). It also has an XML parser for an older replay format, which we
  don't support.

### Replay format (verified in M0)

- The zip holds two gzipped entries, `<id>` (snapshots) and `a<id>` (action
  log), told apart by content. The snapshot file is **one gzip member per
  turn**: a single-member decompressor silently stops after the first turn.
- Snapshots are PHP-serialized `awbwGame` objects, one per turn start, holding
  players, buildings and units. Unit `hit_points` has one decimal, so
  **snapshots give exact internal HP**.
- Action-log lines are `p:<player>;d:<day>;a:<php array>`, where the array is
  `[player, turn, [json action, ...]]`. Combat results in the log are
  **displayed HP only**, so within a turn the internal HP is only known up to
  the set of matching luck rolls. The next snapshot settles it, which means
  validation carries a candidate set until then. Logs also give paths, fuel
  after moves, power-meter values after combat (`copValues`), and the
  turn-start repairs, funds and weather (`End.updatedInfo`).
- The result is in a `GameOver` object (`winners`, `losers`) nested in the
  game-ending action.
- **Timeouts in 2p games:**
  - **Max turn length:** a turn cut off by the server ends with no `End`
    action (the player's clock drops by exactly `timers_max_turn`, 7 days),
    and the game continues. The engine treats the next snapshot as an
    implicit end of turn. Policy extraction must not treat it as a decision.
  - **Timeout loss:** the final turn has no log entry and there is no
    `GameOver`. mooo marks the player whose turn it was as the loser (30 of 30
    checked). The result doesn't reflect the position, so these are excluded.
  - Games cancelled on day 1-2 also have no `GameOver`, and no loser on mooo.
- Multiplayer logs have gaps (timed-out turns get no entry) and mid-game
  resignations; tag games end turns with `Tag`. Both are out of scope.

### Pipeline (Python, offline, multiprocess)

1. **Collect** finished games from mooo searches (`awbw index`) as JSONL in
   `cache/index/`.
2. **Download** to `cache/replays/<id>.zip` (`awbw fetch`), never re-fetched.
   Rate-limited (1 request/s by default) and polite.
3. **Parse** into a `Replay` object: settings, players, map id, per-turn
   snapshots, per-turn actions, result. Uses our own bytes-based PHP
   unserializer, since PHP string lengths count bytes.
4. **Filter** (`awbw.replay.stats.v1_rejection`):
   - keep: 2 players, no tag/team, no fog, clear weather, powers on,
     1000 funds per property, no starting funds, map ≤ `MAX_W × MAX_H`
   - keep: a `GameOver` in the log (this drops timeout losses and cancelled
     games), and every turn logged
   - still to decide: a minimum game length
5. **Convert** each snapshot to a `GameState` (C++ `from_snapshot`) and run
   invariant checks.
6. **Validate** (once the engine runs): apply the logged actions with forced
   outcomes and diff against the next snapshot (units, HP, fuel, ammo, funds,
   ownership, capture points, power meter).
   - Record either "verified" or the first point where engine and replay
     diverge (CO, action type, day).
   - The divergence report *is* the rules bug tracker.
7. **Extract samples:**
   - **v0 (needs only `from_snapshot`, not the full engine):** turn-start
     states + game outcome. The critic can start training before the engine
     is finished.
   - **v1 (verified replays only):** every idle-phase state + outcome, plus
     micro-step action labels (from `enumerate_full_actions` decomposition)
     for a policy prior.
8. **Store** shards of fixed-size records:
   - fields: `{state_bytes, to_play, outcome, game_id, day, ply, ratings, flags}`
   - zstd-compressed; maps live in a separate table
   - *Raw state is stored, never encoded features.* The dataloader calls the
     C++ `encode_batch`, so changing features never requires re-extraction.

### Critic training notes

- **Labels:** outcome z ∈ {−1, 0, +1} from the to-play player's side.
- **Auxiliary heads** (cheap from replays, help the critic learn):
  - final property ownership per tile (like KataGo's ownership head)
  - army-value difference N days ahead
- **Splits:** by `game_id` (positions from one game are highly correlated).
  Also hold out whole maps to measure generalization to new maps.
- **Metrics:** log loss, calibration (ECE), accuracy by game day.
- **Augmentation:** dihedral board symmetries (4 for non-square maps).
  Player swap is already covered by the relative encoding.
- **Known biases:**
  - Human resignation thresholds vary.
  - Results correlate with player skill: weight or filter by rating.
- **Distribution shift:** the critic will see off-distribution positions once
  search uses it, so mix self-play data in over time.
- **Snapshot timing (checked in M0):** snapshots are taken *after*
  turn-start processing. Across 20 games, all 501 turn-start funds matched the
  previous `End` action's `nextFunds`, and all 2,447 logged repairs were
  already reflected in snapshot HP.

## 6. Observations / network input

Shipping C×H×W float32 one-hots per leaf is too heavy: about 600 KB per leaf
at 40×40, around 600 MB per batch of 1024. Instead:

- **Compact `uint8` index planes `[K, H, W]`**, about 12–16 planes, ~25 KB per
  leaf:
  - terrain class
  - property type, property owner (relative: me / enemy / neutral)
  - unit type, unit owner (relative)
  - HP, fuel bucket, ammo, flags, capture points
  - cargo types
  - pending-selection and reachable/attackable masks
  - on-map mask
- **Global vector:** funds, income, day, power meters and thresholds, power
  active, CO ids, settings, and whether the to-play player moved first.
  Starting positions are deliberately unbalanced to compensate for moving
  first, so the critic needs to know which side it is evaluating.
- **Network input:** the first layer is embedding lookups summed per tile, on
  the GPU.
- **Map sizes:** pad to size buckets (e.g. 24/32/40) and batch by bucket.

## 7. Networks

- **Trunk:** a ResNet or U-Net over tiles, or a transformer over tiles.
  Input is the embedded planes plus the broadcast global vector.
- **Heads:**
  - value (tanh or W/D/L)
  - policy logits per `(channel, tile)`
  - auxiliary heads (§5)
- **Policy gather on device.** `collect()` provides padded legal-action
  indices `[B, L_max]` (L_max ≈ 256, with an overflow fallback). The net
  gathers logits on the GPU, so only `[B, L_max]` crosses back to C++.

## 8. Search: batched Gumbel MCTS (C++)

- **Algorithm:** Gumbel AlphaZero (Danihelka et al., ICLR 2022).
  - Gumbel-top-k plus sequential halving at the root.
  - Completed-Q improved policy at interior nodes.
  - It improves the policy reliably even with few simulations. That matters
    because one game has thousands of decisions.
- **Tree over micro-steps:**
  - Every node stores `to_play`.
  - **Backup flips the value's sign only when `to_play` changes, not every
    ply.** A player makes many consecutive decisions within one turn.
- **Chance nodes after Fire** (and any other stochastic step):
  - exact merged outcome distribution from `outcomes()`
  - children chosen by sampling ∝ probability, or by the lowest
    visits-to-probability ratio
- **Depth:** a turn has dozens of decisions, so with hundreds of simulations
  the search rarely reaches the opponent's reply. The critic carries most of
  the weight, which is why §5 comes first.
  - Later: a transposition table keyed by Zobrist hash. Different unit orders
    within a turn often reach the same position.
- **Batching across games:**
  - One `BatchedSearch` owns M concurrent games.
  - Each round, each tree descends to one leaf. With M ≥ 512 that is enough
    parallelism without virtual loss.
- **Node storage:** a per-game arena; states are stored at expanded nodes
  (~4 KB each). Re-simulating from the root along the path is the fallback if
  memory becomes a problem.
- **Self-play extras:**
  - Playout-cap randomization (KataGo): full search on ~25% of decisions,
    which are the ones recorded as policy targets; cheap search on the rest.
  - Resignation threshold, with a fraction of games played out to check for
    false resignations.

```python
search = awbw.BatchedSearch(maps, n_games=512, sims=64, cfg=cfg)
while training:
    planes, globals_, legal_idx = search.collect()  # GIL released inside
    value, legal_logits = net(planes, globals_, legal_idx)  # on GPU
    search.submit(legal_logits, value)  # expand + backup + advance
    buffer.add(search.drain_samples())  # state bytes, sparse pi, z
```

- **Later optimization:** export the net to ONNX / Core ML / TensorRT and run
  inference inside C++ for actor-only processes.

## 9. Bindings (nanobind)

- **Build:** scikit-build-core with `nanobind_add_module(_core NB_STATIC ...)`
  and `nanobind_add_stub` for `.pyi` type stubs.
  - Release builds use `-O3`; a `-march=native` option for local builds.
- **`GameState`:**
  - `copy()`; pickling via raw bytes
  - `legal_actions()`, `outcomes()`, `apply()`
  - `from_snapshot()`, `zobrist`
  - zero-copy numpy views for debugging
- **Batch functions:** `encode_batch(state_bytes[N]) -> planes, globals`,
  writing into caller-provided numpy buffers.
- **`BatchedSearch`:** as in §8.
- **macOS:** Apple clang ships no OpenMP, so parallelism uses a `std::thread`
  pool.

## 10. Testing & benchmarks

- **C++ (doctest):** movement, damage, turn start, every CO power.
- **Damage formula across all logged attacks:** every logged combat result
  must be in `outcomes()`. This checks the formula over the whole replay
  archive.
- **Full replay verification rate:** tracked per CO and per action type.
  A local regression corpus of verified replays runs in CI. Replays stay
  gitignored; only game IDs and hashes are committed.
- **Random-playout invariant tests under ASan/UBSan:**
  - occupancy grid stays in sync
  - HP and funds stay in range
  - the legal-action mask is non-empty until the game ends
  - the incremental Zobrist hash equals a hash recomputed from scratch
- **Benchmarks:**
  - random games/sec
  - clone (ns)
  - `outcomes` (ns)
  - `encode_batch` (µs per state)
  - MCTS simulations/sec, with a dummy net and with the real net

## 11. Milestones

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Replay ingestion (Python): download + cache, parser, rules JSON | Thousands of replays parsed; settings/filter stats known |
| M1 | C++ core + bindings: state, `from_snapshot`, movegen, all actions except CO powers, `outcomes`, turn start, `encode_batch` | Every logged attack in `outcomes()`; v0 dataset → **critic v0** |
| M2 | Exact match: full replay stepping, all COs/powers, edge cases | ≥ 99% of filtered replays fully verified; benchmarks in CI |
| M3 | v1 datasets: all idle-phase states + action labels | **Critic v1** + policy prior (behavior cloning); baselines: random, greedy, critic-greedy 1-ply |
| M4 | Batched Gumbel MCTS with chance nodes, `collect`/`submit` | Search + critic beats critic-greedy; sims/sec measured |
| M5 | Self-play loop: actors / learner / evaluator, mixed with human data | Elo rises across checkpoints vs. fixed baselines |
