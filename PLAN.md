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
| Hidden units | Dived Subs and hidden Stealths stay hidden from the opponent even without fog (AWBW's rule), so the game has hidden information. A move that runs into an unseen one stops short ("trapped") |
| Weather | Clear, plus the rain and snow CO powers bring (Drake, Olaf); random weather comes later |
| Search | Gumbel AlphaZero-style MCTS, exact simulator, explicit chance nodes |
| First model | Critic trained on positions parsed from human replays |

## 1. Repository layout

```
awbw-selfplay/
  pyproject.toml              # scikit-build-core + nanobind
  CMakeLists.txt
  python/awbw/rules/*.json    # single source of truth: units, terrain, damage chart, COs
  tools/gen_tables.py         # rules JSON -> src/core/tables.h (pre-commit checks it is current)
  src/core/                   # C++20, no Python dependency
    tables.h                  # generated constexpr tables
    map  state  snapshot  movegen  combat  actions   # done (.h/.cpp each); rng.h
    cos  hash  encode                               # to come
    search/                   # batched Gumbel MCTS
  src/bindings/module.cpp     # thin nanobind layer
  python/awbw/
    rules/    maps.py
    engine.py                 # replay snapshot -> GameState glue
    replay/   sources.py  parse.py  combat_check.py  move_check.py  step_check.py  extract.py
    data/     shards.py  loader.py
    nn/       model.py  train_critic.py  selfplay.py
    render.py  agents/
  tests/cpp/                  # googletest, fetched by CMake (AWBW_BUILD_TESTS)
  tests/python/               # pytest
  bench/                      # Google Benchmark, fetched by CMake (AWBW_BUILD_BENCHMARKS)
  cache/                      # gitignored: raw replays, maps, shards
```

The core rule: **Python never runs once per action in the hot loop.**
Python makes one call per batch (`collect` / `submit` / `encode_batch`), and
C++ releases the GIL for all heavy work.

## 2. State representation

### Static: `MapData`, immutable and shared through a process-wide registry

`src/core/map.h`. Tiles are numbered `y * width + x`.

- dimensions, AWBW terrain id, terrain kind and defense stars per tile
- move costs precomputed per `[weather][move_type][tile]`
- neighbor table
- the property list: every tile with per-game state (capturable properties,
  pipe seams, loaded silos), with its kind and starting country; tile →
  property index

### Dynamic: `GameState`, POD with no pointers

`src/core/state.h`; 3,996 bytes.

```cpp
struct Unit {                 // 16 bytes
  uint32_t id;                // creation order (AWBW unit id from replays); 0 = free slot
  uint16_t tile;              // carried units: their transport's tile
  uint8_t  type, owner;
  uint8_t  hp;                // internal 1..100; displayed = (hp + 9) / 10
  uint8_t  fuel, ammo;
  uint8_t  flags;             // moved | hidden | carried | stunned | captured this turn
  uint8_t  cargo[2];          // slots of loaded units; AWBW's slots, which don't shift
  uint8_t  carrier;           // slot of the transport carrying it
  uint8_t  _pad;
};

struct Player {               // 24 bytes
  int32_t funds, power;       // power meter in AWBW units (co_power)
  int32_t cop_cost, scop_cost;  // co_max_power / co_max_spower
  uint8_t co, country, power_on, power_uses, eliminated, _pad;
  int16_t power_atk;          // attack % fixed when a power starts (Colin's SCOP)
};

struct Settings {             // varies per AWBW game
  uint16_t funds_per_property, capture_win, day_limit;
  uint8_t  unit_limit, use_powers;
  uint32_t banned_units;      // bit per unit type
};

struct GameState {
  uint32_t map_id;            // resolved via the map registry
  Settings settings;
  uint32_t next_id;           // id of the next unit built
  uint16_t day;
  uint8_t  active, weather, weather_owner, winner;  // weather_owner: whose power set it
  Player   players[2];
  Unit     units[kMaxUnits];               // 128
  uint8_t  unit_at[kMaxTiles];             // occupancy grid; check_state() verifies it
  uint8_t  prop_owner[kMaxProps];          // 128; player index or none
  uint8_t  prop_points[kMaxProps];         // capture points / seam HP / silo loaded
  uint64_t rng[2];            // luck RNG lives in the state
};
static_assert(std::is_trivially_copyable_v<GameState>);
```

- **Limits:** 40×40 tiles, 128 units (Global League caps each player at 50;
  the most seen in a snapshot is 93), 128 properties (the largest cached map
  has 72).
- **Dynamic terrain:** a seam at 0 HP is pipe rubble and a fired silo is
  empty; `terrain_kind` / `terrain_stars` / `move_cost` read the state.
- **Zobrist hash:** not yet; added for the transposition table (§8).

- **Cloning:** a memcpy (35 ns).
- **Serialization:** the raw bytes. The same bytes are what the datasets store.
- **Derived data stays out of the state.** Reachable tiles, threat maps and
  similar data are computed on demand into per-thread scratch memory.
- **Exact AWBW match may need AWBW's ordering** (repair order when funds run
  short, CO-power targeting), so units keep their creation `id`.
  `load_snapshot` puts units in slots in id order; new units take the lowest
  free slot but still get the next id.
- **Snapshots** (`src/core/snapshot.h`, Python glue in `awbw/engine.py`):
  owners come from each building's terrain country, carried units from the
  transports' cargo lists. A unit marked moved at turn start was stunned by
  Von Bolt's SCOP (all 198 such units in 800 replays).

## 3. Action space: micro-steps

| Phase | Choice | Encoding |
|---|---|---|
| Idle | select a unit or factory tile, or End Turn / COP / SCOP | tile + 3 globals |
| Unit selected | destination tile | tile |
| Unit moved | Wait, Capture, Fire@tile, Load, Join, Unload@tile, Supply, Dive/Hide, Launch@tile, Repair@tile, Explode, Delete | channel × tile |
| Factory selected | unit type | 25 |

- **Action ID:** `channel * MAX_TILES + tile`, plus a few non-spatial IDs.
- **No cancel/undo.** A transport may unload where it stands at any time in
  its turn (AWBW allows it even after the transport has acted), so a second
  unload is just another action.
- **Forced steps auto-advance.** A phase with exactly one legal action is
  applied immediately. It produces no search and no training sample.
- **Full-action API.** `enumerate_full_actions()` returns complete
  `(unit, dest, cmd, target)` tuples, used by replay decomposition and
  heuristic bots.
- **Paths:** the destination is enough: the engine takes the cheapest path the
  mover's owner can see (`apply` also takes an explicit path; replays pass the
  logged one, which costs more than the cheapest in 0.7% of moves). The path
  only matters for fuel and for hidden enemies: a player may route around a
  suspected Stealth. Choosing paths is left out of the action space for now.

## 4. Rules engine

- **Tables:** generated from `python/awbw/rules/*.json`, which Python also
  reads (`awbw.rules.load()`), keyed on AWBW's unit, terrain, country and CO
  ids. The values come from AWBW's own pages and are verified against replays,
  because AWBW differs from the cartridges in places.
  - Units: stats equal the `awbwUnit` fields in snapshots. Terrain ids,
    countries and move costs: from the MIT-licensed AWBW Replay Player's
    tables. Damage: AWBW's `damage.php` chart, split into ammo / no-ammo
    weapons by which targets consumed ammo in replays (e.g. Tank cannon vs
    vehicles and ships, machine gun vs foot and copters; Cruiser missiles vs
    ships, free anti-air gun vs air). The out-of-ammo machine-gun values are
    AWDS numbers, still unverified.
  - COs: modifiers from AWBW's `co.php`; star costs from snapshot
    `co_max_power` / `co_max_spower`.
- **Movement** (`src/core/movegen.cpp`): Dijkstra over small integer costs
  (about 0.4 µs per unit on a 30×30 map). Movement points = unit move + CO
  bonus for the current power state, capped by fuel. Enemy units block;
  friendly units can be passed through. Sturm (always) and Lash (powers) pay 1
  per passable tile. A broken seam moves like pipe rubble.
  - Moves are planned from the mover's view: enemies its owner can see block,
    unseen hidden ones don't. Carrying out the move on the real board, the
    unit stops on the last tile before an unseen enemy (`kTrapped`), pays
    only for the tiles it crossed, and loses its action. A hidden unit is seen
    by enemies next to it.
  - **Weather:** rain and snow use their own cost tables (AW2's, checked
    against 881 logged moves in them first). Olaf moves in snow as in clear
    and in rain as in snow; Drake ignores rain; Sturm's and Lash's cost of 1
    doesn't apply in snow. Power weather lasts until the start of its user's
    next turn (the opponent's turn is the one in it).
  - **Verified** (`awbw check-moves`, 7,359 replays): 201,628 snapshots load
    with no errors, and all 653,924 checkable logged moves pass, those in rain
    and snow included. The path is legal, its cost equals the fuel spent, and
    the search finds no cheaper route. (Earlier, on 2,000 replays: for each
    of the 50 (unit, CO, power) combinations with ≥ 300 moves, the longest
    logged move reaches the engine's movement points exactly.)
  - `check-moves` checks moves against the turn-start state up to the first
    action that can change movement (Fire, Power, AttackSeam, Supply, Repair);
    `check-turns` (below) checks every move with the full engine.
- **Combat** (`src/core/combat.cpp`; [wiki formula](https://awbw.fandom.com/wiki/Damage_Formula)),
  in exact integer math. **Verified: 627,074 of the 627,101 attacks logged
  in 7,338 cached replays are in `combat_outcomes()`** (`awbw check-combat`).
  The 27 others are Lash attacks in turns played between 2026-07-26 and
  07-28 02:11 that get no terrain-star bonus; it holds in her attacks at any
  other time, so AWBW seems to have dropped it for those days.
  - raw = (B·ATK/100 + luck − bad luck) · HP_att/10 · (200 − (DEF + stars·HP_def))/100,
    with displayed HPs.
  - Damage = floor(raw + 0.05). The wiki says "round up to a multiple of 0.05,
    then down", but replays show an exact x.95 also rounding up (floating
    point in AWBW); three logged attacks need this.
  - ATK = 100 + CO + 10 per owned comtower (+10 on any power) + terrain bonus.
    DEF = 100 + CO (+10 on any power). Air units get no terrain stars.
  - Counterattacks only from direct units against adjacent attackers, using
    post-damage HP. Sonja's and Kanbei's "×1.5 counters" add 50 to the
    counter's ATK; luck isn't scaled. (Scaling the whole raw damage passes
    every displayed result too, but internal HPs in later snapshots rule it
    out.) Sonja's Counter Break strike comes first and gets no bonus.
  - A hidden Stealth can only be hit by Fighters and Stealths, a dived Sub
    only by Subs and Cruisers; that also rules out counters against them.
  - Checked quirks: Koal's road bonus excludes bridges; Jake's plains bonus
    includes pipe rubble; Kindle's SCOP bonus counts her urban properties at
    the time of the attack (it persists into the opponent's turn).
- **Actions** (`src/core/actions.h`): `Action` is 8 bytes `{kind, unit, to,
  target, arg}`; `Status apply(map, state, action, Detail = {})` validates and
  applies, leaving the state unchanged on an error. Kinds: Wait, Fire,
  Capture, Load, Join, Unload, Supply, Repair, Hide, Unhide, AttackSeam,
  Explode, Launch (unit actions: move to `to`, then act), Build, Power,
  Delete, EndTurn, Resign. `Detail` carries an optional path, a forced
  fight result and a power's missile centres.
- **Chance API.** One interface serves search and replays alike:
  `combat_outcomes()` gives the exact distribution over (attacker HP,
  defender HP); `apply` takes a forced result (`Detail::att_hp / def_hp`,
  rejected if impossible) or rolls one from the in-state RNG.
  - Search uses `combat_outcomes` for its chance nodes.
  - Replays force the logged result.
- **Rules found in replays** (all now enforced by `check-turns`):
  - **Power meter** (35,408 fights with known HP): each side gains the full
    cost of the displayed HP it loses plus half the cost of what it deals
    (AWBW counts 10 meter per funds), nothing while its own power is on,
    capped at the SCOP cost. A star is 90,000. Using a power subtracts its
    cost; each use raises both costs by a fifth of the base, up to 10 uses
    (3,975 power uses). A CO without a COP reports the SCOP cost for both.
  - **Capture:** points = displayed HP × CO capture % (Sami's SCOP takes it at
    once). Progress lives on the property and resets when the unit on it
    leaves or dies. AWBW's unit `capture` flag only means "captured this
    turn". Taking the HQ, or the last lab of a player without an HQ, wins.
    The capture limit counts properties **except comtowers and labs**: all
    30 capture-limit wins ended exactly when that count reached the limit,
    and 335 captures across 142 games passed the limit counting them
    without ending the game.
  - **Join:** displayed HPs add up (the result is a whole HP), the excess
    over 10 is refunded at a tenth of the unit cost per HP, fuel and ammo add
    up to the max, and the joined unit keeps the stationary unit's id.
  - **Transports:** unloading without moving keeps the transport's action
    (players hop infantry through APCs this way); a transport may unload in
    place even after acting (an APC supplies, then unloads); moving then
    unloading ends its turn; cargo may land on the tile the transport just
    left. Cargo slots never shift: a load takes the first empty slot.
  - **Supply:** an APC also refills the cargo of the transports next to it,
    as an action and at turn start.
  - **Black Boat repair:** +1 HP priced per displayed HP gained, so topping up
    9.6 to 10 is free (26 of 27 cases; one 9.5 stays put).
  - **Pipe seams** (1,483 attacks): hit as an undefended full-HP Neotank on
    0-star terrain with no luck: base × ATK/100 × HP/10. Broken at 0.
  - **Hidden attackers:** a unit that can't see a hidden Stealth deals it
    nothing back but still spends ammo on the counter.
  - **CO powers' immediate effects:**
    - Andy +2 / +5 HP to all his units, units in transports included; Hawke
      +1 / +2 to his and −1 / −2 to the enemy's, neither reaching cargo.
    - Drake −1 / −2 HP and half the fuel (rounded up) of enemy units, plus
      rain on his SCOP; Olaf snow, plus −2 HP on his SCOP; Kindle's COP −3 HP
      to enemies on urban terrain. Power damage never kills (0.1 HP left) and
      never reaches cargo.
    - Eagle's SCOP lets every non-footsoldier unit act again; Jess resupplies
      every unit; Sensei places 9 HP unmoved infantry (mechs on SCOP) on every
      empty owned city.
    - Sasha's COP drains the enemy meter by scop_cost × funds / 50,000; her
      SCOP (War Bonds) pays half the value of the damage her units deal while
      it is on, in her attacks and in their counterattacks in the opponent's
      turn.
    - Missiles hit every uncarried unit within 2 tiles, the user's own too:
      Rachel's three × 3 HP, Sturm's 4 / 8 HP, Von Bolt's 3 HP plus a stun.
      A stunned unit keeps its AWBW capture flag through the turn it sits
      out.
    - Missile aim follows the AWBW wiki's rules (Rachel, Von Bolt and Sturm
      pages): each candidate centre scores the units in reach, the user's
      own negative; ties go to a second score and then the top-left tile.
      Replays give the logged centres; otherwise the engine aims them. It
      matches AWBW for 1,521 of 1,578 Rachel missiles, 347 of 358 Von Bolt
      missiles and all 59 Sturm missiles.
  - Builds logged with no unit didn't happen. Game-ending captures log the
    remaining points without resetting them.
- **Turn start** (for the player whose turn begins):
  - their power ends, and the weather their power set; stunned units sit the
    turn out; AWBW's per-turn flags reset (the previous player's moved flags
    clear at the end of their turn)
  - daily fuel for air units, ships and hidden units (Eagle's air units burn
    2 less); none on day 1
  - income from cities, bases, airports, ports and HQs (not comtowers or
    labs), plus Sasha's 100 per property
  - repairs on own properties of the right type (cities, HQs and bases for
    ground units, airports for air, ports for ships): +2 HP (+3 Rachel)
    priced per displayed HP gained, in slot (creation) order, falling back to
    +1 HP when only that is affordable; resupply there in any case
  - APC resupply of neighbours and their cargo; air units and ships at 0 fuel
    are destroyed
- **COs:**
  - A `CoSpec` data struct covers the day-to-day modifiers.
  - A `switch` on CO id handles power effects (heals, damage, weather,
    missiles, stuns, ...); `weather_as` in the CO table covers Olaf and Drake.
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
   - Cached so far: the 2,000 most recent 2p GL std games (to 2026-10-12),
     all with clear weather, no fog, powers on, 1000 funds per property.
     1,653 pass the v1 filter; 299 have no `GameOver` (timeouts and cancelled
     games); 46 end before day 6 (all resignations, 14 on day 1); 2 miss a
     mid-game turn in the log. Of the games with a `GameOver`, 1,589 end by
     resignation, 105 by capture, 7 by rout and 1 by deletion; median length
     is 14 days. 41 have turns cut off at the max turn length.
     Extending to 10,000 games (`cache/index/2p_global_league_std_10k.jsonl`).
3. **Parse** into a `Replay` object: settings, players, map id, per-turn
   snapshots, per-turn actions, result. Uses our own bytes-based PHP
   unserializer, since PHP string lengths count bytes.
4. **Filter** (`awbw.replay.stats.v1_rejection`):
   - keep: 2 players, no tag/team, no fog, clear weather, powers on,
     1000 funds per property, no starting funds, map ≤ `MAX_W × MAX_H`
   - keep: a `GameOver` in the log (this drops timeout losses and cancelled
     games), and every turn logged
   - keep: games that reach day 6 (`MIN_DAYS`). Earlier resignations say
     little about who was winning. Provisional; a later cutoff may suit better.
5. **Convert** each snapshot to a `GameState` (C++ `from_snapshot`) and run
   invariant checks.
6. **Validate** (`awbw check-turns`, `awbw.replay.step_check`): every turn
   starts from its snapshot, the logged actions are applied with the engine,
   and the result is compared after each action with what the log shows
   (position, fuel, fight results, ammo, meters, capture points, funds) and
   at the end with the next snapshot (every unit, both players, every
   property).
   - Fights are forced to the logged displayed result; when that leaves
     several internal HPs, the checker searches them, guided by the next
     snapshot, with conflict-directed backjumping (each unit's HP records the
     fight choices it depends on).
   - The divergence report *is* the rules bug tracker.
   - **Result on the cache (7,245 replays, 198,438 turns): 197,481 of
     197,503 checked turns match exactly (3.53M actions).** 935 turns are
     skipped: 934 with no log, and one whose next snapshot is missing (the
     opponent's turn was never recorded). The 22 mismatches are
     known oddities in `tests/python/test_cached_replays.py`:
     - 19 turns of Lash games in that window (her missing terrain bonus);
     - a Black Boat repair that leaves 9.5 HP (two others top up to 10);
     - the only turn whose repairs ran short of funds: AWBW skipped a hidden
       Stealth (repaired in other turns) for a Tank built after it, and left
       funds that would have paid for 1 HP of the Stealth;
     - one turn whose fight results take 508k tries, over the search budget.
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

- **C++ (googletest, `tests/cpp/`):** maps, state bookkeeping, snapshots,
  movement, damage, every action kind and the turn start (68 tests); every
  CO power to come. CI runs them on
  Linux and macOS, in Release and under ASan/UBSan.
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
- **Benchmarks** (Google Benchmark, `bench/`; current numbers on an Apple
  M-series laptop in parentheses):
  - random games/sec
  - clone (35 ns), `check_state` (0.6 µs), `reachable` (0.4 µs per unit)
  - `damage` (3.5 ns), `combat_outcomes` (0.7 µs for tank vs tank)
  - copy + `apply`: a move (0.5 µs), a fight with rolled luck (0.2 µs), end
    of turn on a 30×30 board with ~50 units (0.3 µs)
  - `encode_batch` (µs per state)
  - MCTS simulations/sec, with a dummy net and with the real net

## 11. Milestones

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Replay ingestion (Python): download + cache, parser, rules JSON | Thousands of replays parsed; settings/filter stats known |
| M1 | C++ core + bindings: state, `from_snapshot`, movegen, all actions except CO powers, `outcomes`, turn start, `encode_batch` | Every logged attack in `outcomes()` (**done**); state, snapshots, movegen (**done**); apply and turn start, every checkable turn replayed exactly (**done**); `encode_batch`; v0 dataset → **critic v0** |
| M2 | Exact match: full replay stepping, all COs/powers, edge cases | CO powers' effects, missile aim, rain and snow (**done**); ≥ 99% of filtered replays fully verified; benchmarks in CI |
| M3 | v1 datasets: all idle-phase states + action labels | **Critic v1** + policy prior (behavior cloning); baselines: random, greedy, critic-greedy 1-ply |
| M4 | Batched Gumbel MCTS with chance nodes, `collect`/`submit` | Search + critic beats critic-greedy; sims/sec measured |
| M5 | Self-play loop: actors / learner / evaluator, mixed with human data | Elo rises across checkpoints vs. fixed baselines |
