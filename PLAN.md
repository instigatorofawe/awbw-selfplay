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
| Hidden units | Dived Subs and hidden Stealths stay hidden from the opponent even without fog (AWBW's rule), so the game has hidden information. They are seen when next to an enemy unit or on an enemy property (wiki). A move that runs into an unseen one stops short ("trapped") |
| Weather | Clear, plus the rain and snow CO powers bring (Drake, Olaf); random weather comes later |
| Search | Gumbel AlphaZero-style MCTS, exact simulator, explicit chance nodes |
| First model | Critic trained on positions parsed from human replays |
| Compute | One GPU (an RTX 5070, 12 GB) for pre-training, search and self-play; no rented compute |

## 1. Repository layout

```
awbw-selfplay/
  pyproject.toml              # scikit-build-core + nanobind
  CMakeLists.txt
  python/awbw/rules/*.json    # single source of truth: units, terrain, damage chart, COs
  tools/gen_tables.py         # rules JSON -> src/core/tables.h (pre-commit checks it is current)
  src/core/                   # C++20, no Python dependency
    tables.h                  # generated constexpr tables
    map  state  snapshot  movegen  combat  actions  encode   # done (.h/.cpp each); rng.h
    hash  legal  belief  movement                            # done
    search                    # batched Gumbel MCTS (done)
  src/bindings/module.cpp     # thin nanobind layer
  python/awbw/
    rules/    maps.py
    engine.py                 # replay snapshot -> GameState glue
    replay/   sources.py  parse.py  combat_check.py  move_check.py  step_check.py  extract.py
    data/     shards.py  loader.py  build.py
    nn/       model.py  critic.py  train_critic.py  policy.py  train_policy.py  train_selfplay.py
    agents.py  search.py      # arena agents; the network side of search, self-play
    ladder.py                 # matches from fixed starts; checkpoints rated against a baseline
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

`src/core/state.h`; 6,560 bytes.

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
  uint8_t  hp_belief[kMaxUnits][10];       // what players know of each internal HP (below)
  uint8_t  hp_hidden[kMaxUnits][10];       // Sonja's opponent's belief over each display
  uint64_t rng[2];            // luck RNG lives in the state
};
static_assert(std::is_trivially_copyable_v<GameState>);
```

- **Limits:** 40×40 tiles, 128 units (Global League caps each player at 50;
  the most seen in a snapshot is 93), 128 properties (the largest cached map
  has 72).
- **Dynamic terrain:** a seam at 0 HP is pipe rubble and a fired silo is
  empty; `terrain_kind` / `terrain_stars` / `move_cost` read the state.
- **HP beliefs** (`src/core/belief.h`): players see only displayed HP, so
  each unit also carries the posterior over the 10 internal HPs its display
  allows, as weights 0–255 (the likeliest 255, never 0 at the true HP;
  `check_state` enforces it). The engine updates it wherever HP changes,
  so self-play and search keep it for free; see §6.
- **Hidden displays:** Sonja's opponent doesn't see her units' displayed HP
  either, so `hp_hidden` holds their belief over each display 1–10 (same
  weights; never 0 at the true display), used only for units of a player
  who `hides_hp`.
- **Transposition key** (`core/hash.h`, **done**): `state_hash(state, sel)`,
  64 bits, the same for states that play the same whatever order reached
  them. Unit slots and ids, the RNG and free slots don't count (no rule reads
  them); units are hashed one by one and summed. Computed from scratch, not
  incrementally: 0.10 µs on the §10 board (59 units). A selection's unit is
  keyed by its tile.

- **Cloning:** a memcpy (65 ns).
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

Done (`src/core/legal.h`). A decision is one micro-step the policy scores
tile by tile:

| Phase | Choice | Ids |
|---|---|---|
| Idle | a unit or factory, scrap a unit, COP / SCOP, End Turn | Select@tile, Delete@tile, 3 non-spatial |
| Unit selected | its destination (its own tile to stay) | Move@tile |
| Unit moved | its action | Wait, Capture, Load, Join, Supply, Hide, Unhide, Explode @its tile; Fire, AttackSeam, Repair, Launch, Unload1, Unload2 @the tile acted on |
| Factory selected | unit type | 25 non-spatial |

- **Action ID:** `channel * MAX_TILES + tile` over 17 channels (map tiles,
  y × width + x), then End Turn, COP, SCOP and Build + unit type: 27,228 ids.
- **`Selection`** (4 bytes: phase, unit, tile) holds a decision in
  progress, next to the `GameState` rather than in it. Picks don't touch the
  state; the last step builds the `Action` and applies it.
- **Legal means `apply` takes it,** as the mover sees the board. Destinations
  come from `reachable`, so moving into or through an enemy the player can't
  see is legal and traps (`kTrapped`). Fire and seam targets are what the
  unit sees from its destination: a hidden unit is seen once the attacker
  is next to it. Each id leads on to at least one legal action, so the list
  is empty only when the game is over.
- **API:** `legal_actions()` gives complete actions (heuristic bots, replay
  checks); `legal_ids(sel)`, `step(sel, id)`, `step_action(sel, id)` (the
  action a last step completes) and `action_steps(action)` (its ids, for
  replay labels) give the micro-steps.
- **Cost** (the §10 board, player to move ~25 units): `legal_ids` 0.3 µs
  idle, 0.8 µs for a Tank's destinations, 0.7 µs for its options there;
  `legal_actions` 47 µs for 1,237 actions.
- **No cancel/undo.** A transport may unload where it stands at any time in
  its turn (AWBW allows it even after the transport has acted), so a second
  unload is just another action. A moved transport with cargo stays
  selectable for that alone.
- **Forced steps** (a single legal id, 17% of logged steps, mostly Wait
  after a move) carry no choice: the policy dataset leaves them out, and
  search is to take them without expanding a node.
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
  - **Verified** (`awbw check-moves`, 19,926 replays): 547,143 snapshots load
    with no errors, and all 1,775,167 checkable logged moves pass, those in
    rain and snow included (a hidden unit's path is read from its owner's
    view; the global copy leaves out unseen steps). The path is legal, its cost equals the fuel spent, and
    the search finds no cheaper route. (Earlier, on 2,000 replays: for each
    of the 50 (unit, CO, power) combinations with ≥ 300 moves, the longest
    logged move reaches the engine's movement points exactly.)
  - `check-moves` checks moves against the turn-start state up to the first
    action that can change movement (Fire, Power, AttackSeam, Supply, Repair);
    `check-turns` (below) checks every move with the full engine.
- **Combat** (`src/core/combat.cpp`; [wiki formula](https://awbw.fandom.com/wiki/Damage_Formula)),
  in exact integer math. **Verified: 1,709,016 of the 1,709,143 attacks
  logged in 19,925 cached replays are in `combat_outcomes()`** (`awbw
  check-combat`). The 127 others are fights of Lash games in turns
  overlapping 2026-07-27 16:47 to 07-28 04:57 (AWBW server time) where her
  terrain-star bonus is missing; it holds in her 90,277 other attacks from
  starred terrain, so AWBW seems to have dropped it for those hours.
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
    out.) Sonja's Counter Break strike comes first and gets no bonus; the
    attacker's reply is then the counterattack and does get one (Sonja
    mirrors: 310 attacks in 3 games).
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
    The capture limit counts properties **except comtowers and labs**, as
    the AWBW wiki says: all 30 capture-limit wins ended exactly when that
    count reached the limit, and 335 captures across 142 games passed the
    limit counting them without ending the game.
  - **Join:** displayed HPs add up (the result is a whole HP), the excess
    over 10 is refunded at a tenth of the unit cost per HP, fuel and ammo add
    up to the max, and the joined unit keeps the stationary unit's id.
  - **Transports:** unloading without moving keeps the transport's action
    (players hop infantry through APCs this way); a transport may unload in
    place even after acting (an APC supplies, then unloads); moving then
    unloading ends its turn; cargo may land on the tile the transport just
    left. Cargo slots never shift: a load takes the first empty slot. A unit
    loaded this turn can be dropped further on the same turn (AWBW's
    boosting).
    - Cargo comes off only onto terrain it could move on, from a transport
      standing on such terrain, so a Lander or Black Boat unloads from a
      shoal or a port, never open sea or reef.
    - A loaded transport may board another, its cargo riding along: an
      APC onto a Lander (57 logged loads in 52 of the 216,598 cached
      replays), a T-Copter onto a Cruiser (2) or a Carrier. A loaded
      transport can't join another (none logged).
  - **Delete:** only a unit that hasn't moved this turn.
  - These three hold for every logged action in the 38,936 cached replays
    (`check-turns`).
  - **Supply:** an APC also refills the cargo of the transports next to it,
    as an action and at turn start.
  - **Black Boat repair:** +1 HP priced per displayed HP gained, so topping up
    9.6 to 10 is free (315 of 325 repairs of a 9.1-9.9 HP unit; the other
    10 leave it as it was, with no pattern found).
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
      matches AWBW for 4,520 of 4,701 Rachel missiles, 1,016 of 1,049 Von
      Bolt missiles and all 165 Sturm missiles.
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
    priced per displayed HP gained, property by property column by column
    (x, then y), as AWBW numbers its buildings; a unit whose whole repair is
    unaffordable gets none, and the next is tried (the only two such turns
    in 545k: column order and no partial repair leave exactly AWBW's funds,
    creation order doesn't); resupply there in any case
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
  - Day limit (AWBW wiki): after the last day is played, the player with
    more properties wins, comtowers and labs included; a tie is a draw.
    No cached replay has a day limit, so this is unverified.
  - Self-play also has a training-only day cap; results there are decided by
    army value plus properties.

## 5. Replays → training data

### Sources

- **[awbw.mooo.com](http://awbw.mooo.com)** (primary, no login): a public
  replay archive. `search?q=...` returns 500 results per page (`&offset=501`,
  ...), with filters like `2p`, `global league std`, `rating>1500`, `nodraw`.
  Each row gives game id, map id, both COs, both players' ratings, days,
  the end date, and which player lost; players are listed in turn order.
  `replay/<id>.zip` serves the replay itself.
  - **A row's ratings are from after its game, so they give the result
    away.** In 61,700 rows of players with 20+ games, winners sat 9.6
    points above the player's neighbouring games and losers 9.8 below.
    Players in a game are usually close, so on 39,000 games the higher
    shown rating won 83% (Elo log loss 0.552, about as good as the whole
    critic). `sources.RatingHistory` instead takes each player's rating
    after their latest game that ended on a day before this one started:
    on v0, the higher of those won 61% (log loss 0.653), and 88% of games
    have both. (Taking the latest game ending before this one *ended* does
    better, 0.640, but uses results from during the game, like a player
    who stopped playing and lost several games at once.)
- **awbw.amarriner.com** (fallback): `replay_download.php?games_id=N` needs a
  logged-in session (`logincheck.php`, `PHPSESSID` cookie). The map API
  `api/map/map_info.php?maps_id=N` is public JSON; its terrain grid is
  column-major (`[x][y]`).
- **Games in progress** (`awbw live CKPT GAME`, `awbw.live`), no login:
  - `game.php?games_id=N` embeds the settings (funds, fog, powers, tags, capture,
    day and unit limits, bans) and the live state as one-line JS values
    (`playersInfo`, `unitsInfo`, `currentTurn`, ...).
  - The page's replay viewer, `api/game/load_replay.php` (POST JSON `gameId`,
    `turn`, `initial`), returns the state at the start of any turn so far, in
    the same format; `initial` adds `daySelector`, one entry per turn. Fields
    match the replay snapshots' once their `units_`, `players_` and
    `buildings_` prefixes go, so live turns go through `state_from_turn`.
  - A spectator sees less than a replay: HP is the displayed 1–10 (Sonja's
    is "?"), and hidden Subs and Stealths are left out unless the other player
    can see them, though `players_units_count` still counts them and their
    displayed-HP value. `awbw.live` estimates HP as 10d − 4 (the average exact
    HP behind a display of d in replays; 100 for a 10) and puts each left-out
    unit back, hidden and out of the other player's sight, with the HP its
    army value implies. On 1,342 validation games that reproduced the
    replay's encoding in all 45 positions with an unseen enemy unit. Ended
    turns are cached in `cache/live/<game>/`, with their actions.
  - Each turn's response also has its `actions`, flattened from the replay
    log as the viewer plays them: a move and what the unit did next are two
    entries (a unit acting where it stands has no move), the end of a turn is
    `NextTurn`, fields are renamed (`copValue` for `playersCOP`, ...), and
    every per-viewer field holds one copy. `live.replay_actions` puts them
    back in the replay format: on 6 finished games without Sonja (136
    turns, every action kind replays log but Unhide), stepping the replays
    with their converted viewer actions gave the same result as with their
    own in every turn, and filled in two turns the replay files have no log
    for.
  - Whose view it is: a spectator of a game in progress gets "?" for Sonja's
    HP everywhere, as her opponent does. A finished game's viewer shows its
    turn starts in full (hidden units and Sonja's HP) and each turn's log as
    the player whose turn it was saw it.
  - The log of a game in progress leaves out supplies: an APC's Supply has no
    entry, and only the fuel logged next shows the refill (2 games in
    progress checked; finished games and recent replays log them).
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
   - **Crawling back in time** (`awbw crawl QUERY --before DATE --max-gb N`):
     a mooo search returns at most 10,000 games (later pages repeat the
     last one), so the crawl walks back in date slices (`before YYYY-MM-DD`,
     strict), saving each slice's index, fetching its maps and replays, and
     stopping at the size cap or the start of the archive. Reruns resume.
   - **Size:** replays average 111 KB (about 9 KB per day of play; game
     lengths are the same back to 2022). The archive holds about 215,000
     2p GL std games back to 2022-08-30, so all of them would take about
     24 GB. Replays from 2022, 2024 and 2025 parse and replay exactly.
   - **Cached:** the 20,000 most recent games (2026-04-25 to 2026-10-13),
     all with clear weather, no fog, powers on, 1000 funds per property;
     crawling the rest of the archive (all ~215,000 games, about 24 GB,
     with a 30 GB safety cap; about 2.5 days at one replay per second). Of the 20,000, 2,642 have no `GameOver` (timeouts and
     cancelled games) and 810 end before day 6; the median game is 14 days.
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
   - leave out: games on maps edited since (`engine.map_edit`, reason
     "map edited since the game"). AWBW keeps only a map's latest version;
     even its own game pages show older games on the new terrain. Snapshots
     list a game's buildings but no other terrain, so the version a game was
     played on can't be rebuilt. The first snapshot's buildings must match
     the map's tiles, and every property on the map must be listed. In the
     70,170 replays behind v1-70k, the one edited map is "A Sailor's Moon"
     (164657): its 963 games from before the edit have 34 of 50 buildings
     elsewhere. (Before this check, 766 of them failed later as "snapshot
     rejected"; the rest failed the filter above.)
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
   - **Result on the cache (20,945 replays, 574,141 turns): 571,454 of
     571,552 checked turns match exactly (10.16M actions).** 2,589 turns
     are skipped, nearly all with no log. The 98 mismatches are known
     oddities in `tests/python/test_cached_replays.py`:
     - 83 turns of Lash games in that window (her missing terrain bonus);
     - 11 Black Boat repairs that leave a 9.x HP unit as it was;
     - a city whose capture points drop to 0 with no capture logged;
     - three turns whose fight results take more tries than the search budget.
7. **Extract samples** (`awbw.replay.extract`):
   - **v0 (`awbw dataset`):** the state at the start of every turn,
     labelled with the result for the player to move. On the 20,576 replays
     cached on 2026-10-13: 17,108 games pass the filter, giving 520,297
     samples. No game is a draw; the first mover wins 49%.
     - **HP beliefs:** each state is the turn's snapshot, but its units'
       HP beliefs (§6) come from the engine stepping through every turn
       before, as `check-turns` does (`step_check.step_turn`), carried to
       the next snapshot by unit id. A turn the engine can't step (76 of
       520k, the known oddities below) leaves each unit whose HP changed
       knowing only its display (625 units). Built in 109 s on 12 cores
       (70 s from snapshots alone).
   - Ratings come from the mooo index, which lists players in turn order:
     its loser matched the replay's `GameOver` in all 17,108 games.
   - **v1 (`awbw dataset --v1`, `extract.extract_v1`):** a sample for every
     action taken, in the state it was taken in, labelled with its
     micro-steps (`action_steps`) and how many ids were legal at each, plus
     the result. The engine steps each turn as `check-turns` does and
     records each applied action with the state before it
     (`step_check.Labelled`; a failed fight-result branch is overwritten by
     the one that passes), so only turns that replay exactly get labels; the
     start of every turn is a sample either way, so v1's turn starts are
     v0's samples (tested on cached replays). The last turn has no snapshot
     after it and is checked against the log only. A server-ended turn has
     no End Turn label, a resignation none at all.
     - On the 43,060 replays cached on 2026-10-13: 35,684 games pass the
       filter, giving 19,225,064 samples (539 per game), 19,213,293 of them
       labelled: 52,339,186 micro-steps, 9,060,294 forced. 104 turns couldn't
       be stepped. 1.74 GB; built in about 25 minutes on 11 workers, which
       write whole shards themselves (`data/build.py`, 100 replays per
       shard), since moving and compressing 125 GB of raw states in one
       process would be the bottleneck.
     - Consecutive states differ in a few bytes, so zlib takes a state to
       83 bytes here (XOR against the previous row gave 44, not worth a
       format change).
     - Two replays from the older archive have a building without a terrain
       id; parse errors like that now raise `ReplayFormatError`.
8. **Store** (`awbw.data.shards`): a directory of `numpy` `.npz` shards of
   whole games (1,000 per shard; v1 about 83), plus `maps.npz` (terrain of
   every map used, so a dataset needs no replay cache) and `meta.json`.
   - Per sample: `state` (raw `GameState` bytes), `game`, `turn`, `day`,
     `to_play`, `z`; v1 (`meta["rows"] == "actions"`) adds `action` (its
     place in the turn), `kind`, `steps` [3], `n_steps` and `n_legal` [3].
     Per game: id, map, winner, how it ended, days, both COs and both
     players' ratings from before the game (shard format v2; v1 stored the
     leaky ones).
   - zlib (`savez_compressed`) shrinks a state from 4,000 bytes to about
     240 and decompresses at about 3 GB/s; zstd would save 15% more, not
     worth a dependency.
   - *Raw state is stored, never encoded features.* The loader
     (`awbw.data.loader.batches`) calls the C++ `encode_batch`, so changing
     features never requires re-extraction. It pads boards to square
     buckets (16/24/32/40) so a batch has one shape, splits off validation
     games by a fixed hash of the game id (5%), and takes a `keep` mask for
     rating or map filters. About 66k samples/s in one process.
   - `decision_batches` turns a v1 dataset's actions into decisions, one
     per non-forced micro-step, each encoded with the picks before it
     (`encode_policy_batch`, §6) and the policy index of the step taken. It
     takes a random `fraction` of the decisions per pass (or a fixed
     `row_fraction` of the actions, for validation) and keeps only the
     states those need, so many shards can be shuffled together. About
     160k decisions/s in one thread.

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
  - Results depend on player skill and on future misplays, which no
    position shows: even a perfect critic has a ceiling (the Bayes error).
    The critic gets both players' ratings as an input (§7) so the skill
    gap isn't noise to it, and self-play can ask for the value between
    two equal players.
- **Distribution shift:** the critic will see off-distribution positions once
  search uses it, so mix self-play data in over time.
- **Snapshot timing (checked in M0):** snapshots are taken *after*
  turn-start processing. Across 20 games, all 501 turn-start funds matched the
  previous `End` action's `nextFunds`, and all 2,447 logged repairs were
  already reflected in snapshot HP.

## 6. Observations / network input

Shipping C×H×W float32 one-hots per leaf is too heavy: about 600 KB per leaf
at 40×40, around 600 MB per batch of 1024. Instead:

`src/core/encode.h` (version 4), `encode_batch` in Python. Everything is
from the side of the player to move ("me"), with owners relative, and `me`
knows what a player who watched every turn knows:

- Enemy units `me` can't see (hidden Subs and Stealths that are neither
  next to one of my units nor on my property) are left off the board, since
  where they went is unknown. The unit count and army value include them,
  as AWBW's game page does: both players saw them built.
- Cargo shows with its transport. AWBW doesn't show what's in an enemy
  transport, but every load happened in plain sight.
- **HP is what players know (encoder v3):** players see only displayed HP
  (1–10), their own units' too. Power meters, Sasha's War Bonds and the
  army value count displayed HP, so nothing gives the rest away. Damage
  depends only on the displayed HPs and luck, so someone who watched every
  fight knows each unit's internal HP as a posterior over the 10 values
  its display allows, and different units' posteriors are independent.
  `GameState::hp_belief` keeps it (`src/core/belief.h`):
  - builds, Sensei's footsoldiers and joins have a known HP (a join sets
    10 × the summed display);
  - heals, repairs and the damage of powers, bombs and silos shift it by a
    seen amount, clamped to 1–100 (a clamp piles it onto 1 or 100);
  - a strike conditions it on the display seen afterwards, by Bayes' rule
    over the attacker's luck rolls; the counterattack is a second strike
    at the defender's new display. Weights are rescaled to a top of 255,
    and anything possible keeps at least 1.
  - Tested against brute force (every pair of internal HPs through
    `combat_outcomes`, including Sonja's Counter Break and Nell's and
    Flak's luck), and on replays: over 322,740 (unit, internal HP) pairs
    with a spread-out belief, the probability given to the true HP
    (from the next snapshot) is calibrated, e.g. 0.137 predicted vs 0.137
    seen, 0.334 vs 0.341, 0.512 vs 0.506. In 60 recent games 24% of units
    are not known to be at full HP; for those, the posterior mean is off
    by 0.98 internal HP on average, against 2.79 for the middle of the
    display.
  - Left out: AWBW aims Rachel's and Von Bolt's missiles by internal HP,
    so where they land hints at the HPs around them.
  - **Sonja hides her units' HP from her opponent (encoder v4).** They
    show "?", and so does her army value: the live game API reports it as
    0 for Sonja (her unit count, funds and power meter are shown).
    `GameState::hp_hidden` is the opponent's belief over each Sonja unit's
    display, and the engine narrows it with everything public:
    - a fight: the opponent's own unit's display afterwards, who survived,
      both power meters (a fight charges each meter by the displayed HP
      lost × cost unless that player's power is on or the meter is full)
      and Sasha's War Bonds. `hidden_fight` enumerates display × internal
      HP within it × luck; within a display the weights are `hp_belief`
      while the display is known, uniform otherwise;
    - repairs, seen only through her funds: a joint enumeration over the
      repaired units' price classes with funds spent in order;
    - joins (the joined unit must have been at most 9; the refund is
      public), capture points taken, and the shifts of powers and silos.
    - Tested against brute force for 10 fight setups (meters moving, one
      meter, full meters, Counter Break, Sonja mirrors, War Bonds) and for
      repairs with binding funds. On 413 Sonja games in v0: the true
      display was never ruled out; at her opponent's turn starts 98.0% of
      her units' displays are known (107,487 pairs); the other 2,203
      average 2.2 possible displays with P(true) 0.72 (argmax right
      81%), slightly over-spread: predicted 0.09 → seen 0.06, 0.23 →
      0.19, 0.40 → 0.41, 0.77 → 0.80, 0.90 → 0.94.
    - The encoder shows a display the player to move doesn't know as HP
      plane 0, with `hp_p0`..`hp_p9` holding P(display = k + 1); the
      army value uses the expected display.
  - **Live games** (`live.states`) get the same beliefs by stepping every
    turn from the start as a spectator (`step_check.step_spectator_turn`),
    from the viewer's log (§5):
    - Internal HPs are unknown, so a fight is tried from every pair of
      internal HPs the two units' beliefs allow (for a Sonja unit whose
      display is unknown, every display her hidden belief allows), keeping
      the outcomes the logged displays, power meters and funds agree with,
      likeliest first. Beliefs only read what both players see, so any
      such choice updates them the same. A "?" unit that is still around
      later survived.
    - Captures and the repairs at a turn's start show a Sonja unit's
      display through capture points and funds, so they choose it too.
    - The turn's end is checked by display (not fuel and ammo, which come
      from the log since supplies go unlogged); the units put back for
      hidden ones are left out of both.
    - At the start every unit is at 10 × its map HP (predeployed units,
      Sonja's too), which both players know.
    - A turn the engine can't step leaves each unit whose HP its log may
      have changed knowing only what the page shows; the others keep theirs.
    - Validation (`tests/python/test_cached_live.py`): on the 3 finished
      games without Sonja that the datasets keep, each turn's encoding
      equals the dataset's (79 turns), and 3 more step every turn. In 2
      finished Sonja games, 58 of 64 do; the rest come from a fight with a
      Sonja unit whose display
      her opponent didn't know, which moves the other unit's belief by her
      true display in the replay (the watcher's view) and by the likeliest
      in live. In progress: 35 of 35 turns stepped in one game, 19 of 20 in
      a Sonja game (one fight the engine can't reproduce: the meters say her
      Tank was at 9 when everything logged before leaves it at 10). Stepping
      takes 0.2 s for 36 turns.
    - On that Sonja game, P1's win chance at the current turn went from
      96.8% with page-only HP (any display of hers equally likely) to 59.2%.
    - With Sonja to move, her own units show the displays stepping chose;
      `awbw live --no-beliefs` skips the history (page-only HP).

- **21 `uint8` planes `[21, H, W]`** (21 KB per state at 32×32). `PLANES`
  gives each one's name, number of values and whether it is an index to
  embed or a quantity:
  - `terrain` (index): 0 off the board, else the terrain kind now (broken
    seams, fired silos) joined with the property's owner (none, me, enemy),
    67 values
  - `points` (quantity): capture points left (20 untouched), pipe seam HP
  - `unit` (index): type joined with owner, 51 values
  - `hp` (displayed 1..10, or 0 if `me` doesn't know it), `hp_p0`..`hp_p9`
    (255 × the probability that the internal HP is 10 × (displayed − 1) +
    1 + k; for an unknown display, that it is k + 1), `fuel`, `ammo`
    (quantities); `flags` (index: moved, hidden, stunned, capturing)
  - `cargo1`, `cargo2` (index: type), `cargo1_hp`, `cargo2_hp` (expected
    internal HP, rounded)
- **The decision in progress** (`encode_selection`, selection encoder v1)
  adds 18 planes and 29 globals for the policy network:
  - `pick` (index): 1 on the picked unit's tile, 2 where it moves, 3 both,
    4 on a picked factory;
  - `legal.select` .. `legal.unload2` (one per channel, 0/1): where that
    channel's step is legal. These are the spatial part of the policy's
    mask, and show the net the reachable tiles, targets and drops directly;
  - globals: the phase (one-hot) and the picked unit's type (one-hot).
  - The non-spatial ids' legality (End Turn, COP, SCOP, Build + type) comes
    back separately, `[28]` per decision, for the mask.
  - `encode_policy_batch(states, picks, n_picks, ...)` reaches each decision
    by taking its first `n_picks` picks from idle (picks change no state),
    and with `targets` gives the next pick's `policy_index`: channel × S² +
    y × S + x over the S × S bucket, then the 28 non-spatial ids, so
    `policy_size(40, 40) == NUM_ACTION_IDS`.
- **Movement planes** (optional: `PolicyConfig.movement`; `encode_movement`,
  movement encoder v1) add 10 index planes after the decision planes. They
  count turns of moving on an empty board, where no unit blocks and fuel
  never runs out. The network could work these out from terrain and units,
  but only over many layers.
  - **Counting turns** (`movement.h`'s `turns_from`): a Dijkstra over
    (turn, points spent) labels with Dial's buckets. A step that doesn't fit
    in the turn's remaining points waits for the next turn, so points don't
    carry over, and a tile dearer than a whole turn is never entered. Tested
    against a turn-by-turn expansion on 300 random boards.
  - **`home`, `away`:** on each unit's tile, the turns its type needs from
    the nearest factory its owner holds that builds it, and from the nearest
    of the other side's, capped at 15. They show how far reinforcements are
    and how overextended the unit is.
    - `build_map` stores the turns from every factory to every tile, for
      each of the 16 mover classes (movement type × points) it builds. It
      uses the starting terrain in clear weather without CO bonuses
      (`MapData::factory_turns`; 172 rows, 0.8 ms, for a 30×30 map with 24
      factories).
  - **`reach.{me,opp}.{foot,vehicle,air,naval}`:** the fewest turns (up to
    3) that any of the side's units in the group needs to get to each tile.
    - Mine count from now: a unit that has moved starts next turn, a stunned
      one the turn after. The opponent's count from their next turn.
    - CO movement bonuses, the weather and Sturm's and Lash's cost-1 moves
      apply as they will then. The opponent's power and power weather end
      before their turn.
    - Carried units and enemies `me` can't see are left out.
  - **Cost:** 24 µs per state on the benchmark board (12 searches), about 10×
    the rest of a decision's encoding. On the box the loader still gives
    15,900 decisions/s from one thread with them, against 38,400 without.
- **104 globals (`float32`)**, named in `GLOBAL_NAMES`: for me, then the
  opponent, funds, income, power meter as a share of each power's cost, the
  costs, which power is on, power uses, properties, comtowers, unit count
  and army value (all units), the Subs, Stealths and army value the other
  player can't see, and the CO one-hot; then the day,
  weather and whose power set it, the capture limit, funds per property, and
  whether `me` moves first. Starting positions are deliberately unbalanced
  to make up for moving first, so the critic needs to know which side it is
  evaluating.
  - Left out for now: unit bans, the unit limit and the day limit. Replays
    don't record the first two, and no cached game has a day limit, so the
    critic would only ever see them at 0.
- **Cost:** about 2 µs per state, including `check_state` on the raw bytes.
- **Network input:** the first layer is embedding lookups summed per tile
  (index planes) plus scaled quantities, on the GPU.
- **Map sizes:** boards sit at the top-left of a square bucket (16/24/32/40);
  batches hold one bucket.

## 7. Networks

### Critic v0 (`awbw.nn`, done)

- **Model** (`nn/model.py`, `CriticNet`):
  - **Input:** the categorical planes' embeddings summed per tile (one
    table with offsets), plus the quantity planes scaled to [0, 1] through a
    1×1 conv, plus the globals through a linear layer added to every tile.
  - **Trunk:** pre-activation residual blocks (two 3×3 convs, SiLU); every
    second block adds a KataGo-style board summary (masked mean,
    size-scaled mean and max of 32 channels, through a linear layer) as a
    channel bias.
  - **Pre-norm:** each block reads its input through an RMS norm over each
    tile's channels, and its last conv starts at zero, so every block
    starts as the identity (as in SkipInit). Compiled, the RMS norm costs
    nothing (138 ms per step of 256 against 137 without it); a per-tile
    LayerNorm more than doubled the step time on the M4 Pro's GPU, and
    BatchNorm would make a value depend on its batch and count padding in
    its statistics. One LayerNorm sits before the head. The first critics
    had no norm in the trunk (`prenorm=False`, still loadable) and ignored
    the board (below).
  - **Masking:** off-board tiles are zero at every layer and pooling reads
    only the board, so a board's value doesn't depend on its bucket,
    where it sits in it, or the rest of the batch (tested).
  - **Head:** a 1×1 conv to 32 channels, masked pooling, concatenated with
    the raw globals, then an MLP to loss/draw/win logits. Value =
    P(win) − P(loss).
  - **Ratings** (`ratings=True`, the default): the globals get three more
    features: 1 if both players' pre-game ratings are known, their mean
    and the mover's lead (centred on 900, over 200), all 0 when unknown.
    Training hides a sample's ratings with probability 0.25
    (`--rating-dropout`), so one network values positions with ratings and
    without. `Critic.wdl(states, ratings)` takes (mover's, opponent's) or
    nothing; self-play can pass the same rating for both sides; `awbw value`
    and `awbw live` take `--ratings P1 P2`. Critics from before load with
    `ratings=False`.
- **Training** (`awbw train-critic`, `nn/train_critic.py`): cross-entropy
  on the result, AdamW (no decay on biases or embeddings), linear warmup
  then cosine decay over the planned samples, gradient clipping at 1.0,
  one random flip/transpose per batch. Validation each epoch; checkpoints
  per epoch, `best.pt`, and `last.pt` + `optimizer.pt` for `--resume`.
  `--min-rating` keeps games whose players were both rated at least that
  before the game. Evaluations report metrics with and without ratings.
- **Checkpoints** record the encoder signature (version, planes, global
  names); loading refuses a mismatch. `Critic.load(path).value(states)`
  values raw states for MCTS and analysis; `awbw value CKPT GAME` prints a
  replay's win chances turn by turn; `awbw eval-critic` prints validation
  metrics by day and a calibration table.
- **Speed on an M4 Pro (MPS):** channels-last convolutions train about 3×
  slower there, so the trunk stays NCHW. `--compile` (torch.compile)
  gives another 1.5×. The default net (64 channels, 6 blocks, 0.51M
  parameters) trains at about 1,600 samples/s, 5 minutes per epoch of v0.
- **Results on v0 (17,108 games, 26,737 validation positions), log loss:**

  | | all | days 1–3 | 4–6 | 7–9 | 10–12 | 13–15 | 16–20 | 21+ |
  |---|---|---|---|---|---|---|---|---|
  | coin flip | 0.693 | | | | | | | |
  | logistic regression on the 104 globals | 0.550 | 0.690 | 0.682 | 0.632 | 0.500 | 0.344 | 0.269 | 0.212 |
  | critic without pre-norm, 2 epochs | 0.542 | 0.691 | 0.683 | 0.628 | 0.482 | 0.321 | 0.255 | 0.203 |
  | critic, 2 epochs (default; encoder v2, exact HP) | **0.532** | 0.689 | 0.678 | 0.616 | 0.463 | 0.300 | 0.248 | 0.199 |
  | critic, 2 epochs, encoder v3 (HP beliefs) | 0.533 | 0.691 | 0.682 | 0.619 | 0.462 | 0.298 | 0.243 | 0.201 |
  | critic, 2 epochs, encoder v4 (Sonja's displays hidden) | 0.531 | 0.689 | 0.675 | 0.615 | 0.460 | 0.299 | 0.247 | 0.208 |
  | critic with ratings (default), given none | 0.531 | 0.688 | 0.677 | 0.616 | 0.460 | 0.297 | 0.243 | 0.204 |
  | critic with ratings (default), given both players' | **0.492** | 0.629 | 0.619 | 0.570 | 0.438 | 0.285 | 0.231 | 0.199 |

  - **Encoder v3** gives the critic what players know of HP (§6) instead of
    the exact HP: 0.533 against 0.532 on the same games (accuracy 68.8%,
    Brier 0.184, calibration error 0.016), about seed noise. Exact HP was
    worth almost nothing to it, so the v3 critic loses little by not
    seeing what players can't.
  - **Encoder v4** also hides Sonja's displays from her opponent: 0.531
    (accuracy 69.1%, Brier 0.183, calibration error 0.013), within seed
    noise of v3. The rebuilt v0 had 76 turns the engine couldn't step and
    625 belief restarts after them.
  - **Ratings** (pre-game, known for both players in 88% of games): 0.492
    given them (accuracy 73.7%, Brier 0.166, calibration error 0.022) and
    0.531 without, the same as a critic that never saw them, so hiding
    them in a quarter of training costs nothing. The gain is mostly early:
    on validation positions with both rated, days 1–3 score 0.620 with
    ratings, 0.626 for a logistic model of the rating gap alone, and 0.687
    without; by days 7–9 the board adds more (0.573 vs 0.625 and 0.625).
    The fitted gap model is steeper than Elo's 400 scale: 1.68 logits per
    200 points (a 200-point favourite wins 84%, where Elo says 76%), and
    at equal ratings the first player wins 47.3% (logit −0.108).

  - Accuracy 68.9%, Brier 0.184, calibration error 0.012 (without
    pre-norm: 68.3%, 0.187, 0.022, slightly under-confident at the
    extremes).
  - Encoder v2 (hidden units in the totals, 104 globals) changed little:
    0.542 against 0.543 on v1. Only 0.1% of positions have a hidden enemy
    unit.
  - **Overfits with more epochs:** on an 8-epoch schedule, validation loss
    was 0.550 after epoch 1 and 0.554 after epoch 2, with days 1–6 worse
    than a coin flip (0.705). Every position of a game shares its label, so
    17k games are too few; the full archive (~200k games) is the fix.
  - Positions near the end are easy because the loser resigns soon after.
    The early game is close to a coin flip for any model: who wins is
    decided by play still to come.
  - **Without pre-norm it ignored the board** (found and fixed
    2026-10-13). Change in P(win) for the player to move over 4,096
    validation positions, editing the encoded input:

    | edit | no pre-norm | pre-norm |
    |---|---|---|
    | remove my units from the planes | −0.0001 | −0.076 |
    | remove the opponent's units | +0.0001 | +0.045 |
    | halve my units' HP on the planes | −0.0000 | −0.004 |
    | halve the `me.army_value` global | −0.204 | −0.221 |
    | trunk RMS, embedding → block 6 | 1.2 → 4,900 | 0.8 → 8.6 |

    Without the norm, activations grew about tenfold per block from the
    third, burying the board, so the head read the globals and the net was
    in effect a model of them (0.542, barely past the logistic baseline).
    HP still moves little with the norm, because `army_value` already
    counts it and the edit contradicts that global.

### Policy-value network (`PolicyValueNet`, done)

- **Model** (`nn/model.py`): the critic's trunk and value head, reading the
  decision in progress too (`encode_policy_batch`, §6: 18 more planes, 29
  more globals). The policy head maps each tile's features through a 1×1
  conv (64 channels) to a logit per (channel, tile), and the board summary
  through an MLP to a logit per non-spatial id (End Turn, the powers,
  Build + type), laid out as `policy_index` says; illegal steps are masked
  before the softmax. 64 channels × 6 blocks: 0.57M parameters.
- **Training** (`awbw train-policy`, `nn/train_policy.py`): cross-entropy of
  the step taken over the legal ones, on every non-forced micro-step of the
  v1 dataset's actions; the result trains the value head on 5% of
  decisions only (a game's ~1,100 decisions share one result, and the
  critic already overfits v0). One flip/transpose per batch, with the
  spatial logits mapped back; ratings hidden at random as for the critic.
  On CUDA the forward pass runs in bfloat16, losses and validation in
  float32. `awbw eval-policy` scores validation decisions by phase;
  `awbw arena` plays a checkpoint against a random mover or another
  checkpoint from validation games' starts, each start twice with seats
  swapped (`agents.py`).
- **Choosing the training games** (for experiments on how methods scale
  with data): `--games-fraction F` trains on a share of the training games
  by a hash of the id unrelated to the validation split's, so a quarter is
  part of the half (`loader.in_subset`); `--hold-out-maps` leaves maps out;
  `--game-ids LO HI` keeps an id range (ids grow with time). Validation is
  unchanged by them. `eval-policy --maps` and `--game-ids` score a part of
  the validation games.
- **Speed** (decisions/s, batch 256): RTX 5070 bf16 compiled 9,060, fp32
  compiled 5,602, bf16 eager 4,965, fp32 eager 3,902; M4 Pro (MPS) 1,260.
  The 5070 needs `uv sync --extra cuda` (PyTorch's CUDA 13 wheels).
- **Results on v1** (35,684 games; 4 epochs of 12.5% of the 41M training
  decisions, 10 minutes each on the 5070): validation over 221,654
  decisions of the validation games, and the value over their 53,871 turn
  starts.

  | epoch | policy loss | top-1 | idle | selected | moved | factory | whole action | value, ratings | without |
  |---|---|---|---|---|---|---|---|---|---|
  | uniform over legal steps | 2.464 | | 2.793 | 2.991 | 0.773 | 1.674 | | | |
  | 1 | 1.214 | 0.580 | 1.681 | 1.227 | 0.110 | 0.797 | 2.822 / 0.278 | 0.5485 | 0.5831 |
  | 2 | 1.170 | 0.593 | 1.641 | 1.157 | 0.104 | 0.756 | 2.718 / 0.292 | 0.5406 | 0.5726 |
  | 3 | 1.148 | 0.600 | 1.620 | 1.126 | 0.100 | 0.737 | 2.668 / 0.302 | 0.5194 | 0.5536 |
  | 4 | **1.138** | 0.603 | 1.611 | 1.110 | 0.098 | 0.725 | 2.643 / 0.305 | **0.5147** | **0.5487** |
  | critic v0 on the same turn starts | | | | | | | | 0.5194 | 0.5526 |

  Phase columns are log loss; "whole action" is the log loss / top-1 of
  the action's micro-steps together. Epoch 4's value is calibrated (ECE
  0.008; accuracy 72.5% with ratings, 68.3% without) and edges out
  critic v0 trained on v0 alone.
- **What the numbers mean:** many orders of a turn's moves transpose to the
  same position, and humans pick among them arbitrarily, so top-1 accuracy
  on the idle phase (which unit next) understates the policy, and no model
  can get its log loss near 0. The order entropy is the same floor for
  every model, so log loss still ranks models; the real test is play (the
  arena, and search). Order does matter where luck, powers or End Turn
  come in, which the network has to learn.

### Movement planes and auxiliary head (experiment)

- **Movement planes** (`PolicyConfig.movement`, `--movement`): the 10 planes
  of §6 after the decision planes.
- **Auxiliary head** (`PolicyConfig.aux`, `--aux`): a second head like the
  value head predicts what happens next (`loader.aux_targets`), as
  KataGo's auxiliary heads do. These are predictions, not rewards: a reward
  for material would punish trading units for position, or sacrificing
  B-Copters into Anti-Air to charge a decisive power.
  - Targets, for the player to move:
    - the change in my and the opponent's army value and unit count by
      the end of my turn, after their reply, three days on and at the
      game's last state: 16 numbers, scaled by about their spread (10k, 12k,
      20k and 45k funds; 2, 2, 5 and 9 units);
    - the power bars: my COP and SCOP charged at my next turn, theirs at
      theirs, whether I use a power before my turn ends, and whether they
      use one in their next turn. These 6 are masked where the game has no
      powers, the CO has no such power, or the game ends first.
  - Loss: Huber over the changes plus masked cross-entropy, on 25% of the
    decisions (`aux_fraction`), weight 1. Nothing at play time reads the
    head.
- **Cost on the 5070** (decisions/s, alone on the GPU): baseline 8,700,
  aux 8,400, movement 5,600, both 5,500. The movement planes are the cost,
  all of it on the GPU (the loader thread is at 20%). It isn't the
  embedding's lookups: one one-hot matmul instead changes nothing under
  `torch.compile`.
- **Results on v1** (same schedule as the baseline above, one run each;
  epoch 4):

  | run | policy loss | top-1 | idle | selected | moved | factory | whole action | value, ratings (ECE) | without (ECE) |
  |---|---|---|---|---|---|---|---|---|---|
  | baseline | 1.1375 | 0.603 | 1.611 | 1.110 | 0.098 | 0.725 | 2.643 / 0.305 | 0.5147 (0.008) | 0.5487 (0.015) |
  | movement | **1.1343** | **0.605** | **1.606** | 1.110 | 0.099 | 0.712 | **2.636 / 0.307** | 0.5118 (0.005) | 0.5448 (0.013) |
  | aux | 1.1383 | 0.603 | 1.613 | 1.111 | 0.100 | 0.723 | 2.645 / 0.304 | 0.5087 (0.011) | 0.5424 (0.017) |
  | both | 1.1389 | 0.603 | 1.610 | 1.117 | 0.102 | **0.708** | 2.646 / 0.305 | **0.5062** (0.008) | **0.5394** (0.014) |

  - The movement planes help the policy a little (0.003, most of it in
    the factory phase) and the value a little. With the aux head as well,
    the policy loses that gain: selected and moved get worse, as if the
    aux head takes trunk capacity the policy wants.
  - The aux head helps the value: 0.006 alone, 0.009 with the planes,
    about additive. It does nothing for the policy.
  - Mid-schedule, the aux runs' value was worse and miscalibrated (epoch
    2: ECE 0.08–0.10, against 0.03 at epoch 1); it settles as the learning
    rate decays.
  - The head's own predictions (both, epoch 4), squared error against
    "no change": @1 0.127 (1.260), @2 0.386 (1.400), @6 0.399 (1.461),
    @end 0.381 (1.127); power bars beat their base rate (my COP charged at
    my next turn: log loss 0.171 against 0.589).
  - One seed each, and the gaps are small. Run-to-run noise isn't
    measured, so 0.003 in policy loss may be noise. The aux head's value
    gain costs 3% of the speed; the movement planes' 0.003 costs 35%.

### Transformer trunk (`PolicyConfig.trunk`, experiment)

The default trunk is the ResNet above (pre-activation residual blocks,
global pooling every second block). `--trunk transformer` swaps the blocks
for transformer blocks over the board's tiles; the embedding and heads are
the same, so the two compare directly. `--channels` and `--blocks` are then
its width and layers; `--heads` (4) and `--mlp-ratio` (4) are its own.

- **Tokens:** one per tile of the bucket. Tiles off the board are masked
  out as keys and stay zero, as in the ResNet.
- **Positions:** 2-d rotary embeddings (`rope_angles`). Half of each
  head's channel pairs turn with the row and half with the column, at
  frequencies from π/2 to π/128 per tile. Attention then depends only on
  the offset between two tiles, so a board gets the same outputs in any
  bucket and wherever it sits in it (the ResNet's invariance, tested for
  both trunks).
- **Block:** pre-norm (RMS), queries and keys RMS-normed per head,
  attention, then a SiLU MLP. The output layers start at zero, so each
  block starts as the identity. There is no global pooling: attention
  reads the whole board.
- **The mask rides in an extra key channel** (`board_attention`): a key
  off the board scores −10⁴. That way the kernel needs no attention mask,
  so flash attention can run. Under torch.compile a boolean mask becomes a
  dense [B, 1, N, N] bias for the memory-efficient kernel. Per-head
  widths are padded to a multiple of 8: 16 pads to 24 and 32 to 40.
- **Speed and memory on the 5070** (bf16, compiled, real v1-70k batches:
  89% of decisions in the 24 bucket, 11% in 32, none in 40; measured while
  sp1 shared the GPU, so compare within the table only):

  | trunk | params | train, batch 128 | inference, batch 256 | train memory, batch 128 |
  |---|---|---|---|---|
  | ResNet 64×6 | 0.57M | 4,781/s | 62,566/s | 1.1 GiB |
  | transformer 64×6, 4 heads | 0.41M | 2,159/s (mask: 1,297) | 15,210/s (mask: 9,936) | 3.3 GiB |
  | transformer 96×6, 4 heads | 0.79M | 1,429/s (mask: 1,039) | 10,764/s (mask: 8,004) | 4.4 GiB |
  | transformer 128×6, 4 heads | 1.32M | 903/s (mask: 818) | 7,091/s (mask: 6,844) | 5.5 GiB |

  - At equal size the transformer trains at about 0.45× the ResNet's speed
    and searches at about 0.25×: attention over 576–1,024 tokens costs most
    of it.
  - Its activations take memory: a 64×6 transformer needs 5.5 GiB to train
    at batch 256, and a 128-wide one doesn't fit beside sp1.
    `--recompute` recomputes each block's activations in the backward pass.
    That brings 64×6 at batch 256 from 5.5 to 3.3 GiB and costs 18% of the
    speed. 128×6 at batch 256 trains in 3.4 GiB at 780/s with it, and runs
    out of memory without it. Eight heads of 16 channels are no faster than
    four of 32 (735/s).
- Training on v1-70k in the scaling grid (below) as `tf-s1` (64×6) and
  `tf128-s1` (128×6, `--recompute`).

### Scaling grid on v1-70k (experiment)

What to scale before the full archive is in: games, net size, training
length, trunk, and which games. v1-70k has 57,050 games, about 66M
training decisions. Every run trains on 20.5M decisions, the v1
baseline's 4 epochs × 5.13M, so the runs on a share of the games repeat
them. The runs share one validation set: 356,221 decisions and 86,515
turn starts. One seed each, on the RTX 5070 (`grid1.sh`, `grid2.sh` and
`grid3.sh` on the box, results in `cache/grid/NAME/metrics.jsonl`).

- **Results at the end of each run** (4 epochs unless the net column says
  otherwise; policy columns are log loss; value is log loss, with ECE in
  brackets):

  | run | games | net | params | policy | top-1 | idle | selected | moved | factory | whole action | value, ratings | without |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | s8 | 1/8 | 64×6 | 0.57M | 1.1411 | 0.602 | 1.615 | 1.120 | 0.097 | 0.723 | 2.656 / 0.303 | 0.5167 (0.014) | 0.5479 (0.024) |
  | s4 | 1/4 | 64×6 | 0.57M | 1.1412 | 0.602 | 1.613 | 1.122 | 0.097 | 0.726 | 2.657 / 0.303 | 0.5118 (0.007) | 0.5445 (0.014) |
  | s1 | all | 64×6 | 0.57M | 1.1389 | 0.603 | 1.613 | 1.117 | 0.097 | 0.720 | 2.651 / 0.303 | 0.5096 (0.010) | 0.5433 (0.012) |
  | s1-long | all | 64×6, 8 epochs | 0.57M | 1.1185 | 0.609 | 1.596 | 1.086 | 0.093 | 0.689 | 2.604 / 0.310 | 0.5039 (0.013) | 0.5371 (0.013) |
  | c96-s4 | 1/4 | 96×8 | 1.49M | 1.1224 | 0.608 | 1.595 | 1.095 | 0.094 | 0.705 | 2.613 / 0.310 | 0.5101 (0.010) | 0.5428 (0.019) |
  | c96-s1 | all | 96×8 | 1.49M | 1.1210 | 0.609 | 1.597 | 1.090 | 0.094 | 0.697 | 2.610 / 0.310 | 0.5091 (0.009) | 0.5425 (0.009) |
  | c128-s1 | all | 128×10 | 3.15M | 1.1182 | 0.609 | 1.591 | 1.082 | 0.093 | 0.743 | 2.603 / 0.312 | 0.5081 (0.009) | 0.5414 (0.009) |
  | tf-s1 | all | transformer 64×6 | 0.41M | 1.1022 | 0.615 | 1.577 | 1.070 | **0.090** | 0.657 | 2.566 / 0.319 | 0.4962 (0.009) | 0.5277 (0.008) |
  | tf128-s1 | all | transformer 128×6 | 1.32M | **1.0923** | **0.618** | **1.561** | **1.063** | **0.090** | **0.647** | **2.543 / 0.322** | **0.4953** (0.011) | **0.5262** (0.005) |

  - **Games barely matter at this length.** Eight times the games (s8 to
    s1) gives 0.002 in policy loss and 0.007 in value loss. A game's
    positions share one result, so the value gains more from new games
    than the policy does.
  - **Width and depth matter more.** 64×6 to 96×8 gives 0.018 in policy
    loss, against 0.001 for 4× the games at 96×8. 128×10 gives only
    0.003 more, though all of it in idle, selected and moved. Its factory
    phase is worse (0.743 against 0.697), which may be seed noise: factory
    is 6% of the decisions.
  - **The transformer beats every ResNet.** At 64×6 (0.41M parameters)
    it beats the ResNet of the same width and depth by 0.037 in policy
    loss and 0.013 in value loss. It beats 128×10 (3.15M) by 0.016 and
    0.012 (0.014 without ratings), in every phase. In value, that is eight
    times the step from 64×6 to 128×10.
    - The two cost about the same to train: 3,600 decisions/s against
      128×10's 3,790. In search it scores leaves at about a quarter of the
      speed of a 64×6 ResNet, near 128×10's speed.
    - Its curve by epoch: policy 1.1748, 1.1350, 1.1117, 1.1022; value
      0.5402, 0.5173, 0.5063, 0.4962. Mid-schedule its value was
      miscalibrated (ECE 0.045 after epoch 1), settling to 0.009 as the
      learning rate decayed, as the aux runs did.
    - Twice the width (`tf128-s1`, 1.32M parameters, `--recompute`) gives
      0.010 more in policy loss but only 0.001 in value. It trained for
      3 h 33 min against 1 h 36 min.
    - Attention reads the whole board at every layer. The 64×6 ResNet's
      convolutions reach 12 tiles out by its last block, and the rest
      reaches it only through a pooled summary. Long-range threats may be
      what the ResNet was missing.
  - **Length matters as much as size.** `s1-long` trains 64×6 on twice
    the decisions (41M, 8 epochs on the same schedule).
    - Policy loss 1.1185: 0.020 better than s1, and level with 128×10 at
      20.5M.
    - Value loss 0.5039: better than 128×10's 0.5081.
    - It trained for 81 minutes against 128×10's 94 (not counting
      validation).
    - So at 20.5M decisions every size is undertrained: training steps
      limit these nets more than games or parameters do. The
      full-archive run should train on far more than 20.5M decisions.
    - Its value was miscalibrated mid-run, as the transformer's was. ECE
      was 0.074 after epoch 1, 0.038 after epoch 4 and 0.041 after epoch
      6, settling to 0.013.
- **Which games** (`grid4.sh`): the 64×6 transformer on 20.5M decisions,
  as `tf-s1`, but on a subset of the games. Each is scored on the
  validation games of its subset, against `tf-s1` on the same games
  (`cache/grid/NAME/eval-TAG.txt`):

  | run | trained on | scored on | policy | idle | selected | moved | factory | value, ratings | without |
  |---|---|---|---|---|---|---|---|---|---|
  | tf-s1 | all 54,127 games | rated 850+ | 1.1118 | 1.596 | 1.064 | 0.088 | 0.633 | 0.5101 (0.009) | 0.5503 (0.012) |
  | tf-rated | 24,142 rated 850+ | rated 850+ | 1.1116 | 1.601 | 1.060 | 0.086 | 0.622 | 0.5132 (0.004) | 0.5542 (0.017) |
  | tf-s1 | all | all | 1.1022 | 1.577 | 1.070 | 0.090 | 0.657 | 0.4962 (0.009) | 0.5277 (0.008) |
  | tf-rated | rated 850+ | all | 1.1086 | 1.586 | 1.077 | 0.090 | 0.656 | 0.5025 (0.011) | 0.5343 (0.012) |
  | tf-s1 | all | 4 held-out maps | 1.0968 | 1.587 | 1.071 | 0.090 | 0.684 | 0.4930 (0.014) | 0.5216 (0.009) |
  | tf-hold | 48,764 on other maps | 4 held-out maps | 1.1125 | 1.593 | 1.090 | 0.091 | 0.786 | 0.4966 (0.010) | 0.5266 (0.012) |
  | tf-s1 | all | newest quarter | 1.1136 | 1.587 | 1.080 | 0.093 | 0.674 | 0.4744 (0.015) | 0.5127 (0.010) |
  | tf-q4 | 13,532 newest | newest quarter | 1.1095 | 1.580 | 1.078 | 0.093 | 0.671 | 0.4714 (0.009) | 0.5130 (0.015) |

  - **Rated games alone don't help, even on rated games.** `tf-rated`
    is level with `tf-s1` in policy loss on rated games (1.1116 against
    1.1118) and 0.003 worse in value; on all games it is 0.006 worse in
    both. The rating input may already let the net tell strong play
    from weak, so weaker players' games cost it little.
  - **The net carries over to maps it has never seen, except for
    builds.** On the 4 held-out maps (Cold Flame, Misery, Walls Are
    Closing In, Beads on a String), `tf-hold` is 0.016 worse in policy
    loss. Per decision the gap is largest in factory, 0.10 (0.786
    against 0.684), against 0.001–0.018 in the other phases. Value
    loses only 0.004 (0.005 without ratings). Build orders are what
    depends on the map most; a new map's self-play will have to learn
    them.
  - **The meta drifts only a little.** On the newest quarter of the
    games, `tf-q4`, trained on that quarter alone (1.23 passes over it),
    beats `tf-s1` by 0.004 in policy loss and 0.003 in value with
    ratings, level without. Old games cost almost nothing.
  - So the full-archive run trains on every game. Weighting the newest
    games more during the final decay is a possible refinement, worth
    at most a few thousandths.
- **Speed and memory by size** (`bench_sizes.py`, bf16, compiled, real
  v1-70k batches, alone on the GPU; training at batch 256, search-style
  inference at batch 1,024; the 24 and 32 buckets mixed as the data is):

  | net | params | train | inference | train memory | hours per 100M decisions |
  |---|---|---|---|---|---|
  | 64×6 | 0.57M | 9,081/s | 53,849/s | 1.5 GB | 3.1 |
  | 96×8 | 1.49M | 6,049/s | 20,734/s | 2.7 GB | 4.6 |
  | 128×10 | 3.15M | 3,790/s | 12,841/s | 4.1 GB | 7.3 |
  | 160×12 | 5.78M | 1,755/s | 6,120/s | 6.0 GB | 15.8 |
  | 192×12 | 8.25M | 1,574/s | 5,130/s | 7.1 GB | 17.6 |
  | 256×15 | 18.08M | 692/s | 2,537/s | 6.2 GB (`--recompute`) | 40 |

  - 256×15 runs out of memory at batch 256 without `--recompute`.
  - The 32 bucket runs at 52–60% of the 24 bucket's speed at every size.
  - Search pays for size more than training does: 128×10 trains at 0.42×
    the speed of 64×6 but scores leaves at 0.24×.
- **Lean for the full-archive run:** the transformer.
  - Follow-up runs use the transformer trunk too. It trains at 0.4× the
    speed of the 64×6 ResNet but wins at equal training time: `tf-s1`
    (96 min) against `s1-long` (81 min) is 0.016 better in policy loss and
    0.008 in value.
  - 128×6 trains at about 1,650 decisions/s alone on the GPU, 17 h per
    100M decisions; 64×6 at about 3,600/s, 8 h.
  - Since `s1-long`, time is better spent on length than on size for the
    ResNet. Whether that holds for the transformer is untested; a 64×6
    transformer on 8 epochs would show it.
  - If the transformer is too slow for search, the fallback is a ResNet
    of 96×8 or 128×10.

### The full-archive run (running)

The 128×6 transformer on v1-full (171,791 games, about 192M training
decisions per epoch) for 8 epochs, 1.54B decisions: 75 times the grid's
length. `bigrun.sh` on the box, started 2026-10-15 after grid4; the log
is `cache/logs/bigrun.log`, the runs `cache/runs/tf128-full` and
`cache/runs/tf128-full-mid`. Its value learned each group of shards'
results, so it stopped after epoch 1 and went on mixed, as `tf128-mix`
(`bigrun2.sh`, below). That run's new value head read the trunk's memory
of epoch 1, so the run restarted from scratch as `tf128-mix2`
(`bigrun3.sh`, below), two epochs at most before self-play.

- **Restart:** training began at 15:10 and stopped three times, at step
  13,400 each time, on a stored state that fails the engine's check: a
  Lander carrying a loaded APC, which unloaded and left its infantry on
  the Lander's tile. The scan of v1-full found 22 such states, in 12
  games: the grid's v1-70k has none. The engine now moves cargo's cargo
  with it, and training leaves out a stored state that fails the check
  (1552b22). Training restarted from scratch at 16:36.
- **Schedule:** flat-then-decay (`--schedule wsd`): the peak rate until
  the last 1.2 epochs, then linear to 0. A run on a flat rate can branch
  at any point into a finished net with a short decay.
- **Checkpoints** every 60 minutes between shard groups, so a crash or a
  reboot loses at most an hour; `--resume` picks up within the epoch.
- **Memory:** 128×6 fits at batch 256 without `--recompute`, but only
  just (11.1 of 12 GB, and the allocator sometimes frees its cache to
  make room). It trains at about 1,765 decisions/s on v1-full against
  about 1,600 with `--recompute`: 31 h per epoch with validation, about
  10.5 days for the 8 epochs and the midway check, ending around Oct 26.
  A later out-of-memory error switches to `--recompute` and resumes.
- **Midway check:** the run pauses after epoch 4. A copy decays over 0.2
  epochs (about 7 h) and is
  - scored on v1-full's validation games, as `tf128-s1` is;
  - played greedy (no search) against `tf128-s1` and the self-play prior
    `v1-64x6-5070`, 200 games each from validation starts;
  - played against `tf128-s1` with 64 simulations each side, 100 games.

  Results land in `cache/runs/eval-*.txt` and `arena-*.txt`.
- **Likely stop at epoch 4** (decided 2026-10-16; replaced on 2026-10-17
  by two epochs at most, `tf128-mix2` below). The grid's runs were
  80k steps of 5.1M decisions an epoch (`--fraction 0.0772`); this one
  uses every decision, 750k steps an epoch, and by step 200k its training
  batches (unseen games, peak rate) were as low as `tf128-s1`'s at its
  end (1.091 policy). Eight epochs may be more than the data rewards.
  Validation every 250k steps, three times an epoch on the flat rate,
  gives each epoch's gain. If epoch 4 improves validation policy loss on
  epoch 3 by less than about 0.003, or validation value loss rises, the
  run stops once the midway arenas are in: kill `bigrun.sh` and the
  training it resumed, by PID. Relaunching `bigrun.sh` would go on with
  epochs 5–8, as every finished step is skipped and `--pause-at 4`
  returns at once. The midway copy is then the pretrained net (a longer
  decay from the epoch-4 checkpoint if 0.2 epochs looks short), and the
  box installs the current code for the search tests (§8 "Cheaper
  search") and self-play from that copy.
- Otherwise the main run goes on through epochs 5–8, and the final net
  plays the midway copy and `tf128-s1` the same way.
- If the midway copy doesn't beat `tf128-s1` clearly, the rest of the
  run is in question: stop it and rethink, rather than spend 5 more days.
- **Epoch 1: the value learned the games in hand.** Validation every 250k
  steps:

  | step | policy | value | ECE |
  |---|---|---|---|
  | 250k | 1.0923 | 0.5595 | 0.052 |
  | 500k | 1.0898 | 0.5860 | 0.081 |
  | 750k | 1.0857 | 1.0299 | 0.230 |

  - At step 750k, 60% of validation positions were valued below 0.05 or
    above 0.95; those sides won 29% and 74%. That is worse than a coin
    flip (0.69) on every range of days.
  - The training value loss was a sawtooth: about 0.40 at the start of
    each group of shards, falling to 0.25 by its end, every 11k steps.
    v1-full has 2,166 shards of about 80 games. Read 32 at a time, a
    group holds about 2,500 games, and each game's ~1,100 decisions, ~55
    of them valued, come within its 11k steps. The net learns those
    games' results, then judges new games by them. The grid's runs took
    7.7% of the decisions, about 4 valued per game per epoch, and didn't
    show it.
  - Checkpoints were saved only at groups' ends, where the sawtooth is
    lowest, so none was earlier or healthier than epoch 1.
  - The policy was unharmed (training and validation level). It was
    stopped on 2026-10-16 at 23:34, at step 756k.
- **Mixed: `tf128-mix`** (`bigrun2.sh`, from 2026-10-16 23:55), the same
  schedule, pause and midway check, as `tf128-mix` and `tf128-mix-mid`:
  - It starts from `tf128-full`'s epoch-1 checkpoint with a new value
    head (`reset_value_head`: the head a new network starts with, its
    optimizer moments dropped). The trunk and policy are kept.
  - `--passes 16` reads each shard 16 times an epoch, a random 1/16 of its
    actions each time (split by action, so a state is held once), 512
    shards at a time. A group then holds as many decisions from about
    40,000 games. Loading a shard takes 0.2–0.3 s, so a group's 512 load
    in about 3 minutes in the read-ahead thread, against about 26 minutes
    of training.
  - `--value-fraction 0.01`: about 11 valued decisions per game per epoch,
    under one per game in a group (55 before).
  - Validation every 125k steps, each checkpoint kept
    (`policy-stepNNNNNNN.pt`), and `best-value.pt` beside `best.pt`.
  - The check: at the first validation (step 875k, about 5 h in), the
    value should be back under about 0.56 with its ECE under about 0.05.
    If not, restart from scratch with the same reading.
  - Speed: 1,790 decisions/s, as before, so the read-ahead keeps up. The
    box uses 26 of its 46 GB RAM and 11.2 GB of the GPU at 98%. Epoch 4
    should land late on 2026-10-20, with the midway results on 2026-10-21.
  - **It failed the check.** At step 875k (2026-10-17 05:15):

    | step | policy | value | ECE | tilt |
    |---|---|---|---|---|
    | 875k | 1.0794 | 0.7107 | 0.153 | +0.14 |

    The tilt is the side to move's mean predicted win chance at the
    validation turn starts (0.64) minus how often it won (0.50).
    `tf128-full` had -0.014 at 250k and +0.077 at its epoch-1
    checkpoint. A third of the turn starts (88k) were valued at 0.97, and
    those sides won 76%; `tf128-full` at 250k put 33k there, which won
    92%. Over the 125k steps the training value loss fell smoothly from
    0.64 to 0.556, with no sawtooth, so the mixing worked. The head
    instead read the trunk's features, which had learned every training
    game's result in epoch 1 and so keep fitting them. The policy was the
    best yet (1.0794). The run was stopped by hand on 2026-10-17 at 07:28,
    at step ~897k.
- **From scratch: `tf128-mix2`** (`bigrun3.sh`, from 2026-10-17 07:29):
  the same reading (`--passes 16 --value-fraction 0.01`, validation
  every 125k steps) from a new network. **Two epochs at most before
  self-play** (decided 2026-10-17).
  - `--epochs 2 --schedule wsd --decay-epochs 0.3`: the peak rate until
    epoch 1.7, then linear to 0.
  - It pauses at epoch 1. A copy (`tf128-mix2-e1`) decays over 0.2 epochs
    (about 6 h) and is checked as the midway copy would have been:
    - scored on the validation games;
    - played greedy against `tf128-s1` and `v1-64x6-5070`, 200 games each;
    - played against `tf128-s1` with 64 simulations each side, 100 games.
  - The run then goes on to epoch 2. The final net plays the epoch-1 copy
    and `tf128-s1` the same way.
  - Self-play starts from the better of the two. If the epoch-1 copy
    clearly beats `tf128-s1` and a second epoch looks worth little, self-play
    can start from it at once, and the run stops.
  - The value check: by step 250k, the value under about 0.60, ECE under
    about 0.06 and a tilt within about 0.03. The head learns slowly at 1%,
    so the 125k validation is only a first look.
  - Epoch 1 should land around 13:30 on 2026-10-18, its copy's results
    late that day, and epoch 2 around 2026-10-20.

### Later

- **Trunk:** a larger ResNet once the full archive is in. Possible
  transformer variants: attention in only some blocks of the ResNet (a
  hybrid), and a token per unit instead of per tile.
- **Heads:** final ownership as an auxiliary target.
- **Policy gather on device.** `collect()` provides padded legal-action
  indices `[B, L_max]` (L_max ≈ 256, with an overflow fallback). The net
  gathers logits on the GPU, so only `[B, L_max]` crosses back to C++.

## 8. Search: batched Gumbel MCTS (C++)

`src/core/search.h` (**done**), driven from Python by `awbw.search`.

- **Algorithm:** Gumbel AlphaZero (Danihelka et al., ICLR 2022), making
  the same choices as DeepMind's mctx:
  - At the root, Gumbel-top-k picks `considered` steps (16), and sequential
    halving spreads the simulations over them. The step played is the
    one left at the end.
  - Below the root, a decision visits the step whose share of the visits is
    furthest below its share of the improved policy, softmax(logits +
    σ(completed Q)).
  - An unvisited step's Q is the node's value mixed with its visited
    steps' Qs (mctx's `qtransform_completed_by_mix_value`). σ(q) = (50 +
    the most visits of a step) × 0.1 × q.
  - It improves the policy reliably even with few simulations. That matters
    because one game has about 1,100 decisions with a choice.
  - **Q is mapped to 0..1 by its bounds, (Q + 1) / 2, not by its spread
    among the node's steps,** which is mctx's default (`rescale`). Many
    micro-steps are worth the same: which unit to pick first, say. Rescaling
    stretches the value head's noise between them to the whole range, and
    that overrides the prior. In the arena below, the same search loses
    with it, 11–21, and wins without it, 24–8.
  - **Values at picks:** picks change no state, but the value head reads the
    selection too. On 800 human idle positions, the Qs of the root's visited
    steps spread 0.34 on average (on −1..1) after 16 simulations. With
    `pick_values` off, a decision with an action in progress takes the
    value of the decision it was picked from instead (the network still
    gives its prior), and the spread falls to 0.25. In the arena that made
    no measurable difference (below), so the network's value is the
    default.
  - The improved policy at the root is the training target.
- **Tree over micro-steps:**
  - There are three kinds of node:
    - a decision: a state and a `Selection`;
    - a chance node: a Fire, with one child per result;
    - the end of the game.
  - **Values are kept from player 0's side and read from the mover's, so
    the sign changes only where the mover does, not at every step.** A
    player makes many decisions in a row within a turn.
  - Forced steps (one legal id) are taken on the way, without a node.
  - Picks change no state, so a pick's node shares its parent's state. A
    node stores a new state (6.5 KB) only where an action was applied: at
    most one per simulation.
- **Chance nodes after Fire:**
  - **Results are what the players see:** `fire_outcomes` gives the
    displayed HPs a fight can leave, none if the move traps. Players know
    only displayed HP, so each result's probability averages
    `combat_outcomes` over the luck rolls and over the internal HPs the two
    units' beliefs (§6) allow. The tree goes on from the likeliest internal
    HPs that show the result, before and after the fight, and the beliefs
    update as in the game.
    - So nothing in the tree depends on internal HPs the players can't see,
      except Rachel's and Von Bolt's missile aim. Two states that differ
      only there search the same (tested).
    - Merging by display also leaves fewer results to visit, and the
      search is no slower (`BM_Search`: 246k simulations/s, from 230k).
    - Sonja's hidden displays (`hp_hidden`) are left out: they count as
      seen.
  - **Sampling:** each visit goes to the result whose share of the visits
    is furthest below its probability. That keeps every result within 1.5
    visits of p × N (tested).
  - **Other actions:** no other action has chance (missile aim, bombs and
    traps are deterministic).
- **Hidden units: the search sees them,** although the network's input
  holds only what the mover knows.
  - A move into an unseen unit traps in the tree as it would in the game,
    so the search effectively knows where hidden Subs and Stealths are.
    Concealing a unit or scouting for one is worth nothing in the tree.
  - Self-play is symmetric, so both sides get the same knowledge. It comes
    up rarely: in a sample of the v1 replays, 0.1% of decisions have an
    unseen enemy Sub or Stealth (about 1% of games).
  - Searching a real game would need roots sampled from a belief over where
    those units are (later).
- **Day cap:** past `max_days`, the tree and self-play end a game as a day
  limit decides it: properties, then army value (`day_limit_winner`, the
  same rule as `agents.adjudicate`).
- **Batching:** a `BatchedSearch` runs many slots. Each round, every slot
  walks to one leaf (no virtual loss is needed), and the leaves go to the
  network in one batch.
  - **One decision:** `add_root(state, selection)` searches it and keeps
    the result. `SearchAgent` uses this, so `agents.play` and the arena can
    search.
  - **Self-play:** `add_game(state)` plays a whole game, searching every
    decision with a choice and taking the forced steps and luck in the
    game itself.
  - **Playout-cap randomization** (KataGo): a share `full_fraction` of the
    searches are full (`sims`) and kept as samples. The rest use
    `fast_sims` and only choose the step.
  - **Output:** `take_samples()` gives each full search's state, picks,
    legal steps, prior, improved policy, visits, Qs and root value.
    `take_records()` gives the finished games.
- **Not yet:**
  - tree reuse between decisions;
  - a transposition table keyed by `state_hash` (§2), in an
    `absl::flat_hash_map`. Different unit orders within a turn often reach
    the same position. Abseil (LTS 20260817.0) is fetched for the
    benchmarks so far: against `std::unordered_map`, 64-bit-key lookups are
    1.8 vs 2.0 ns at 4K entries and 9.5 vs 35 ns at 1M, and inserts 3–11×
    faster (M4 Pro);
  - threads for collect, which is single-threaded;
  - resignation (with a fraction of games played out to check for false
    resignations).
- **Depth:** a turn has dozens of decisions, so with tens of simulations the
  search rarely reaches the opponent's reply. The value head carries most of
  the weight.

```python
cfg = _core.SearchConfig(sims=64, fast_sims=16, full_fraction=0.25, max_days=30)
search = _core.BatchedSearch(cfg, size=32)
evaluate = NetEvaluator(Policy.load(path))  # gathers each leaf's legal logits on the GPU
for start in starts:
    search.add_game(start)
while search.collect():  # the GIL is released in collect, batch and submit
    planes, globals_, legal, movers = search.batch()
    logits, values = evaluate(planes, globals_, legal, movers)
    search.submit(logits, values)
    buffer.add(search.take_samples())
```

- **Speed:**
  - **CPU** (`BM_Search`, M4 Pro, one core, a stand-in network): 230,000
    simulations/s, 4.3 µs each, 2.5 µs of which is encoding the leaf.
  - **With the network** (self-play, 64 simulations per decision, 256
    games at once, the v1 baseline network on the RTX 5070, bf16): 7,300
    evaluations/s eager and 22,300 under `torch.compile`
    (`NetEvaluator(compile=True)`), or 341 searched decisions/s. The network
    takes 74% of the time, encoding the batch 20% and the tree 5%. The arena
    batches only its ~32 games' decisions, at 5,300 evaluations/s compiled.
- **Strength** (`awbw arena --sims N`): the v1 baseline network (§7,
  epoch 4) searching against its own prior played greedily (its likeliest
  step), from validation starts, each played twice with seats swapped,
  with a 30-day cap. The 32-game runs use the first 16 of the 64-game
  runs' 32 starts.

  | simulations | root noise | Q to 0..1 | values at picks | games | result |
  |---|---|---|---|---|---|
  | 16 | Gumbel | rescaled | network | 8 | 1–7 |
  | 16 | none | rescaled | network | 32 | 11–21 |
  | 16 | none | bounds | network | 32 | 24–8 |
  | 64 | none | bounds | network | 32 | 17–15 |
  | 16 | none | bounds | network | 64 | 38–26 |
  | 16 | none | bounds | the state's | 64 | 39–25 |
  | 64 | none | bounds | network | 64 | **46–18** |
  | 64 | none | bounds | the state's | 64 | 43–21 |

  - **Search beats the prior:** 59–61% at 16 simulations and 67–72% at 64,
    over 64 games.
  - **Rescaling Q loses** to its own prior, 11–21, where bounds win 24–8
    on the same starts.
  - **Noise:** 64 games give a standard error of about 6 points. It is more
    in practice, since a start's two games are alike: with the same
    settings, a 64-simulation run on the first 16 starts went 17–15, and
    one on all 32 went 46–18.
  - **Noise in the arena:** where the Qs say nothing, Gumbel noise at the
    root plays a *sample* from the prior. The prior agent plays its argmax,
    which is much stronger for an imitation policy. The arena therefore
    searches without noise (`--gumbel 0`, the default), and self-play keeps
    the noise for exploration.
  - **Cost:** a decision costs `sims` + 1 network evaluations, fewer where
    simulations end the game.

- **Later optimization:** export the net to ONNX / Core ML / TensorRT and run
  inference inside C++ for actor-only processes.

### Self-play training (M5, `awbw train-selfplay`)

One process alternates rounds of search with steps of training
(`nn/train_selfplay.py`). There are no separate actors, learner or gating:
the search always uses the network being trained, the same module.

- **Starts:** the first states of the v1 dataset's training games (33,860),
  in random order. Validation games stay unseen for the arena and ladder.
  Each game's luck is seeded on its own, so two games from one start differ.
- **Games:** `parallel` (512) at once. A game that ends gives its slot to
  a new one at once.
  - **Fixed slots per bucket** (`bucket_slots`): the games are shared
    among the board buckets by their share of the starts, so each bucket
    always has the same number going. Over v1-full that is 456 on 24×24
    boards and 56 on 32×32. Every game has a leaf each round, so the
    network's batches keep one shape, and a compiled evaluator pads to
    the slots (`launch`'s `pad_to`) instead of the next power of two.
    Before this, 456 leaves ran as 512 rows and 56 as 64: 12–14% of the
    network's work was padding. Random refills also let the bucket counts
    drift, and 65 leaves on 32×32 boards would have run as 128. With a
    fast network each network's share of a group's leaves varies, so
    those batches still pad to a power of two.
  - **Groups, queued** (`--group-size`, 256): a bucket's games are
    searched in groups of about 256, one `BatchedSearch` each, the
    buckets' groups interleaved (228, 56, 228). A round goes once round
    them. Each group submits the answers to the leaves it queued the time
    before, gives up its finished games, refills its slots, collects and
    queues its new leaves (`NetEvaluator.launch`) without waiting. So the
    GPU scores the groups before it while the CPU walks the next one.
    Copies go through pinned memory on CUDA, which don't wait for the
    queued work. Calling a `NetEvaluator` still waits, as the arena and
    `SearchAgent` want.
  - **Measured on the M4 Pro** (2026-10-17, `bench/selfplay.py`: 512
    games from turn starts spread over v1-full, 64 / 16 simulations,
    `NetEvaluator` with `compile` on MPS, no training). With the
    64×6 ResNet, whose 124 µs a leaf on MPS is near what tf128 should
    cost on the 5070, it was 6,540 evaluations/s before, 7,240 with fixed
    slots alone (+11%) and 7,720 queued as well (+18%). Group size 128,
    256 or one group per bucket made no difference, nor did 1,024 games.
    The search then waits for the network 86–90% of its time, and 130 µs
    a leaf is what the network alone runs at: the GPU is the limit. tf128
    on MPS runs 2.4 ms a leaf (MPS attention is slow), so only the padding
    counts there: 347 → 389 (+12%).
  - **To measure on the box:** the same with tf128 on the 5070; the CUDA
    path (pinned copies, timed events) has not run yet. Then
    `parallel` 512 / 1024 / 2048 (more games in flight make each game
    span more network updates, so the smallest that keeps the GPU busy),
    and CUDA graphs, which the fixed shapes now allow.
  - Searches use playout caps: 25% are full (64 simulations) and become
    samples; the rest use 16 and only choose the step.
  - **Cheaper search** (both off by default; to be measured on the box
    after the full-archive run). A decision costs 29 evaluations on
    average (0.25 × 65 + 0.75 × 17); the fast searches are 44% of them.
    - **Confident roots** (`--skip-prior P`): a fast search whose root
      gives one step at least probability P ends at the root's evaluation
      and plays as a search of 0 simulations: a Gumbel-max draw from the
      prior. Full searches never skip. The arena's `--skip-prior` and
      `--opponent-skip-prior` measure what it costs in strength, and it
      prints evaluations per decision.
    - **A fast network** (`--fast-net PATH`) scores the fast searches'
      leaves (`BatchedSearch.fast_leaves`); the full searches keep the main
      network. A search uses one network throughout, so values on two
      networks' scales never mix in a tree. At 1/6–1/8 of tf128's cost, a
      64×6 ResNet would save about 38% of the GPU time.
    - The fast network trains on the main one's batches, so it learns the
      main network's search targets as self-play moves them; without that
      it would play an ever larger share of the moves below the main
      network's level. `--no-fast-train` keeps it fixed. It is kept as
      `fast-last.pt` + `fast-optimizer.pt` for `--resume`.
    - How often a root is confident, on 107k of v1-70k's validation
      decisions (ratings hidden, forced steps left out): tf128-s1's top
      step reaches 0.9 / 0.95 / 0.99 at 28% / 24% / 19% of them, and the
      human took it 98.4% / 99.3% / 99.8% of the time. They are mostly
      the action after a move (86% of those reach 0.95), rarely the choice
      of unit (4%). v1-64x6 is a point behind. Skipping at 0.95 alone
      saves 0.44 × 0.24 × 16/17 ≈ 10% of evaluations (24% in the arena,
      where every search may skip), but on top of a fast network only
      about 1.5 points more (38% → 39%): the full searches are then about
      90% of the GPU time, so they are the next thing to cut.
  - Gumbel noise at the root (scale 1) explores.
  - A 30-day cap ends the game, decided as a day limit would.
  - **Resignation** (AlphaGo Zero's rule, by turns): a player resigns at
    its first searched decision of a turn when the search's value there
    was at most −0.9 (P(win) − P(loss), about a 5% chance to win) at the
    start of each of its last 2 turns, this one included. 10% of the games
    play on to the end and record when the rule would first have resigned;
    the share of those players who went on to win is the rule's error,
    which AlphaGo Zero kept under 5% (`resign_wrong` / `resign_checked` in
    the metrics). Arena and ladder games don't resign.
    - **Measured** with the `v1-64x6` prior and no training (64 / 16
      simulations, 25% full; 450 games, all played to the end on the M4
      Pro). Mean length was 23.9 days, and the cap decided 21% of games.
    - The rule fires in 89% of games, a mean 8.2 days before the end.
      That saves 30% of the days and about 31% of the decisions.
    - The would-be resigner went on to win 13 of 399 (3.3%). Eight of
      those games were decided by the cap, not on the board; in games
      that ended on the board it was 5 of 311 (1.6%).
    - Both are under 5%, so −0.9 over 2 turns stays.
    - With 10% of the games playing on, about 28% of the decisions are
      saved, so about 1.4× the games an hour.
- **Samples:** when a game ends, each of its full searches is labelled with
  the result for its mover and goes into the window. That is the state,
  the picks so far, and the improved policy as up to 192 (policy index,
  probability) pairs, the likeliest kept and renormalized. The most legal
  steps in v1 is 160.
  - Before a game's samples go in, the network's losses on them are logged
    as **fresh**: a game it hasn't trained on yet, so fresh against train
    shows overfitting.
- **Window:** the latest 300,000 samples, oldest out first. That is
  2.3 GB, mostly states (6.5 KB each). Training batches take one bucket at a
  time, chosen by its share of the window.
- **Training:** a fixed ratio of 4 self-play samples trained on per sample
  added, from 20,000 in the window on.
  - **Self-play batches:** the cross-entropy of the improved policy, and
    the result's cross-entropy on a random 25% of the samples (a game's
    samples share their result).
  - **Human batches:** 25% of the steps, as in pre-training: the step
    taken, and the result on 5%. They anchor the policy to human play.
  - AdamW at 2e-4 after 200 warm-up steps, with no decay. Flips and
    transposes as in pre-training.
  - **Ratings:** self-play tells the network none, at the leaves and in
    training; human batches keep theirs, hidden at random.
- **Output:** `metrics.jsonl` every minute, with games/h, game length, the
  shares capped and resigned, the first player's wins, and train, fresh and
  human losses. Where the time went: the share of the network's rows that
  were padding, the shares of the time in search rounds, waiting there for
  the network, and in training, and on CUDA the GPU's share busy with
  search (timed events around each queued batch). The waiting share
  misleads on MPS, which blocks while queueing once the GPU is far behind.
  `net-NNNNNN.pt` every 2,000 steps; `last.pt` + `optimizer.pt` for
  `--resume`, after which the window starts empty.
- **Game records** (`games.py`): every finished game goes to the run's
  `games.jsonl` (`--no-record-games` turns it off), and `awbw arena
  --record FILE` does the same for arena games.
  - A record is the game's first state, its luck seeded, and every
    micro-step taken, forced ones included. The luck lives in the state,
    so the steps replay the game exactly. It also keeps the map's terrain,
    so one file is enough, and each searched decision's value, the
    network's value and the simulations run.
  - About 15–20 KB a game, so roughly 20 MB an hour of self-play at 1,000
    games an hour.
  - A record keeps a hash of the final state. A replay that doesn't reach
    it, or a step the engine refuses, means the game was recorded under
    other rules (such as before ae055b9's transport change).
  - `awbw view-game FILE [--game N | --list]` writes one self-contained
    HTML page that steps through a game action by action. It shows the
    board, funds, power bars, each action in words, the turn's actions,
    and a chart of the value for player 1 that jumps to any point;
    `page.html#N` opens at action N.
  - First look, an arena game between the two v1-64x6 networks on Amber
    Valley: Hawke's SCOP took P1's army from 122k to 69k on day 15, and
    both networks put P1 near −1 from then on. P1 won on day 18 with an
    infantry capture of the HQ that neither network saw coming, even
    while the capture was under way.
  - Next: export to AWBW's replay format (below).
- **Ladder** (`awbw ladder RUN --baseline PATH`): each checkpoint plays the
  baseline from the first 100 validation starts, each played twice with
  seats swapped. It plays the priors, or with `--sims` both sides search.
  The score gives an Elo difference with its standard error, appended to
  `RUN/ladder.jsonl`; `--watch` keeps rating new checkpoints.

### Replay export (done, `awbw export-replay`)

A recorded game (`games.py`) as an AWBW replay zip. The AWBW site can't
upload replays, so the viewer is the community
[AWBW Replay Player](https://github.com/DeamonHunter/AWBW-Replay-Player)
(v0.13.1, with a macOS app), which imports local zips. Our own
`load_replay` and step_check must accept the export too.

- **What the Replay Player needs** (from `AWBWJsonReplayParser.cs` and
  `ReplayManager.cs`):
  - A zip of two gzipped entries, snapshots `<id>` and the log `a<id>`,
    told apart by content.
  - Its snapshot parser is hand-written and strict: an unknown field
    throws, and so does a game, player or building field that changes
    between turns where it expects a constant. Snapshots therefore copy
    AWBW's field set and order exactly. In the newest 300 replays that is
    `awbwGame` 36 fields, `awbwPlayer` 30, `awbwBuilding` 8 and `awbwUnit`
    25, then `timers_initial`, `timers_increment` and `timers_max_turn`.
  - It files imported replays by game id, so exported ids must not collide
    with real games (about 1.75M now): use 1,000,000,000 and up.
  - It looks up each player's username on AWBW by `users_id`, except −1.
    Exported players get −1 (they show as an unknown user), and the game's
    name says who played, such as "self-play 120k: tf128 (Hawke) vs tf128
    (Grit)".
  - It first looks in its own list of usernames by id
    (`ReplayData/Replays/UsernameStorage.json`, read at start and rewritten
    from memory). `export-replay --players` (done) gives an AWBW user their
    id and anyone else a negative id made up from their name, added to that
    list: "Absox (Sami) vs tf128-s1 (Max)".
  - It downloads the map from AWBW by `maps_id`. Our maps are AWBW's, so
    this needs nothing; a map edited on AWBW since we cached it would show
    the new terrain.
- **Snapshots**, one per turn start: the record's start, then the state
  after each EndTurn (AWBW's snapshot already has the turn's income and
  repairs).
  - The inverse of `state_from_turn`. Players: CO, country, funds, power
    meter (already in AWBW's units), power costs, power on, eliminated.
    Buildings: terrain id from kind and owner's country, capture points.
    Units: type, owner, tile, internal HP with one decimal, fuel, ammo,
    moved/capture/fired, dive, cargo and carried.
  - Per-type unit constants (movement points, vision, fuel per turn,
    ranges, cost, movement type) match the rules tables in every snapshot
    of 3,000 replays. Ranges are 0/0 for direct units, `symbol` follows the
    class (G ground, M air, S naval), and `second_weapon` is Y for Tank,
    Md.Tank, Neotank, Mega Tank, Mech, B-Copter and Piperunner.
  - Units keep the engine's ids. Building, player and game ids are
    synthetic and stable. Dates are synthetic (a minute a turn from the
    record's time) and timers are AWBW's defaults.
  - Done (`replay/export.py`, `Snapshots` and `snapshots(log)`). What the
    replays showed:
    - Buildings are every property, pipe seam (rubble once broken) and
      silo, loaded or empty, sorted by x then y. Some maps also list their
      pipes; the export doesn't. Units are sorted by tile, and a carried
      unit sits on its transport's tile. CO images are the name in
      lowercase letters plus `.png`.
    - `last_capture` is 0 once a building changes hands, and 20 once its
      capture points change or the unit on it leaves or dies; otherwise it
      stays as it was (every change in 200 replays).
    - At a turn start the mover's units are unmoved, except units Von
      Bolt's SCOP stunned (moved and fired). The engine clears the waiting
      player's moved flags at End Turn, so the export takes "acted" from
      the state just before it. AWBW's `fired` there has quirks: a transport
      that only moved stays 0, and some units that didn't act show 1. The
      Replay Player doesn't show it, so the check leaves that one field out.
    - The check (`test_snapshots_export_as_awbw_writes_them`) steps every
      turn of the 200 newest cached replays, exports each turn start the
      engine reaches, and compares it with AWBW's snapshot field by field,
      including field order: no differences in 5,130 snapshots.
- **Action log**, a line per turn: `p:<pid>;d:<day>;a:` and the PHP array
  `[pid, day, [json, ...]]` (the middle entry is the day in all 17,412
  lines checked).
  - Each action `games.replay` yields becomes AWBW's JSON, built from the
    states before and after it: Move (path, dist, trapped, the unit after),
    Fire, Capt, Build, Load, Unload, Join, Supply, Repair, Hide, Unhide,
    Delete, AttackSeam, Power, End and Resign. Of 216k cached replays, one
    (an old team game) explodes a Black Bomb and none launches a missile, so
    those shapes come from it and from the Replay Player's parser.
  - No fog, so every view is "global", except around a hidden Sub or
    Stealth: AWBW logs those per player even without fog (below).
  - **Paths:** AWBW logs the path taken. The engine moves along the
    cheapest path its owner can see, and fuel follows it. `move_path`
    returns that path, where an unseen enemy stops the unit, and the fuel
    spent (the planner `apply` uses).
  - **Missiles** (Von Bolt, Rachel, Sturm): `missileCoords` are the
    centres the engine aimed at. Aiming is deterministic, so
    `missile_targets` on the state before the power returns them. The check
    passes a real replay's logged centres (`entries(missiles=...)`): the
    engine aims about 96% of them where AWBW did.
  - **Fights:** displayed HP and ammo after, power meters after
    (`copValues`) and Sasha's `gainedFunds`, all from the state after.
  - **End's `updatedInfo`:** the next player, their funds, weather, units
    repaired and supplied, and the day, from the states before and after
    EndTurn.
  - **Powers:** AWBW's payload for each CO, from the states before and
    after in the shapes AWBW uses for that CO: `global` (a bonus for all
    units), `hpChange`, `unitReplace` (changed units), `hiddenReplace`,
    `playerReplace` (a meter or funds), `unitAdd`, `missileCoords` and
    `weather`.
  - **Game end:** a `GameOver` inside the action that ended the game (Capt
    for an HQ or the capture limit, Fire for a rout, Resign). A game stopped
    at the day limit has none, like an unfinished AWBW game.
- **Action log, done** (`ActionLog` and `action_lines(log)` in
  `replay/export.py`). The check (`test_actions_export_as_awbw_logs_them`)
  steps every turn of the 200 newest cached replays, last turns included,
  exports each action from the states before and after it, given the path
  the player took, and compares it with AWBW's entry field by field and in
  key order: no differences. On 1,000 replays what's left is rare or old:
  two games whose per-player views follow another order, hidden Stealths
  and Subs (step 5), 0.2% of End entries (mostly older games on day 2) that
  log null lists, and an enemy lab capture that logs `onElimination` in 6
  of 567. Exported arena games read back and step exactly every turn
  (`test_exported_actions_step_as_the_game`). What the logs showed:
  - A moved unit is logged three times alike: per player in player-id order,
    with "global" just before the first player who isn't moving (309k
    moves). Its movement and range include its CO's, its vision is one less
    in rain, and HP is the shown HP.
  - `dist` is the movement cost. A unit acting where it stands has `Move`
    []; a Wait in place is still a Move. A unit destroyed by the
    counterattack moves with `discovered` [], and destroyed units show 0 HP
    and 0 ammo.
  - End lists repaired units (every unit on a property of its owner that
    services it) and supplied ones (once per APC next to it), each by id.
  - Sasha's `gainedFunds` appears only when War Bonds actually gains funds.
  - An HQ, or a lab, captured to end the game is logged as it was: the
    loser's terrain, its points below zero (Victory March counts 30), and
    only the loser's income; `onElimination` lists the loser's buildings.
    A capture-limit win flips normally. A neutral lab taken always logs an
    empty `onElimination` (2,217 of 2,217).
  - AWBW gives transports (unarmed units) none of Max's movement: under his
    powers they log their base movement and none of 1,807 transport moves
    went further, while armed direct units often did. The engine gave them
    +1/+2; `direct` now means armed (rules and tables.h).
- **Rarer actions and hidden units, done** (step 5's first half). The same
  check now covers Supply, Repair, Hide, Unhide, AttackSeam, Explode and
  Launch, and runs on the 40 newest games where a Sub or Stealth hides too
  (`HIDDEN_GAMES`): no differences left but AWBW one-offs (`LOG_ODDITIES`).
  A synthetic test covers Explode and Launch. What the logs showed:
  - Supply lists the units next to the supplier and their cargo, by id. The
    opponent's copy has only those on woods, mountains, pipe seams (broken or
    not) and its own properties (634 of 751 Supplies list fewer for it), no
    hidden ones. End's supplied list has the cargo too.
  - A Repair of a hidden unit is listed for its owner only. End doesn't list
    a unit whose repair its owner couldn't afford.
  - AttackSeam logs the seam's HP after, below 0 once broken (AWBW doesn't
    floor it), the rubble's terrain, and then the rubble's movement costs by
    weather. `seam_damage` is bound for it.
  - Hide: the unit and what each player sees of it, per player. Unhide: the
    unit with `sub_dive` "R". `sub_dive` is "D" for a unit that hid in its
    owner's last turn and "R" for one that surfaced, until its owner's next
    turn starts, in snapshots too (the snapshot check had read D as Y and R
    as N; it's exact now).
  - A hidden unit's move is logged per player in player-id order, with no
    "global": the owner sees all of it; the opponent "" if it never sees
    the unit, its id, name and dive only if it doesn't see where it stops,
    and the stretch of path around the tiles it sees it on. It sees it on
    its own properties, and mostly next to its units: the export's rule
    gets about 80% of those tiles, so the check compares the owner's view
    (the Replay Player reads that one).
  - A fight with a hidden unit: both players see it, the "global" view
    shows the hidden side as "?" unless it was destroyed. A unit that stops
    next to hidden enemies lists them in `discovered`, as they were before
    the action, with the buildings they stand on. A unit stopped by a hidden
    enemy is logged as a Move alone, trapped and spent.
  - At End, a player with a hidden unit gets its own repaired and supplied
    lists, and the "global" ones leave hidden units out and list every other
    unit twice.
  - Sasha's `gainedFunds` gives 0, not null, for an opponent whose War Bonds
    is on but gains nothing.
- **Powers, done** (step 5's second half). The check covers every power in
  the cached sample (620 in the 200 newest, about 3,000 in the 1,000 before
  them), and a power-only run over the 4,000 newest games matches 9,728 of
  9,729. Unit lists are compared in id order: AWBW lists units by position
  or by id, as it loaded them. Exported games where 20 COs use all their
  powers step exactly (`test_exported_powers_step_as_the_game`). No cached
  replay has Colin, Hachi, Sensei, Kanbei, Nell, Flak or Jugger use a power
  (none in 216k games), so Colin's funds and Sensei's new units follow the
  Replay Player's parser, with a synthetic test. What the logs showed:
  - Every power logs the meter it leaves (`playersCOP`) and its name.
    Andy, Adder, Koal and Sonja log a bonus for all units (`global`:
    movement, vision), Andy's COP as 0 and 0. Andy, Hawke, Drake and Olaf's
    SCOP log `hpChange` (shown HP gained or lost by all of a player's
    units; Drake's halves fuel). Olaf's powers and Typhoon log the weather.
    Market Crash logs the opponent's meter after.
  - The rest go unit by unit: the user's units whose range or movement the
    power changes, with the new value (Max, Sami, Jake, Jess, Grit), all of
    Jess's units resupplied, Lightning Strike's non-footsoldiers refreshed,
    the HP Urban Blight leaves the enemy's units on urban terrain, and the
    units missiles hit (each once, however many missiles hit it; Ex Machina
    adds `units_moved` -1). A carried footsoldier is listed as its transport,
    with the transport's base movement plus the bonus (Sami); other cargo,
    and a transport listed already, aren't.
  - A missile's area past the map's left or right edge runs into the next or
    previous row, as a row-by-row tile index does: AWBW lists the units
    there as they are, undamaged (4 of 465 missile powers).
  - With any hidden unit on the map, a power that lists units adds
    `hiddenReplace`, per player: the hidden units it changes, or null.
  - AWBW one-offs the check skips: a list for one player and null for the
    other, often with 0 HP (0.7% of Rachel's SCOPs, 0.5% of Urban Blights).
    A Lander carrying a Tank is left out under Max Blast but listed by id
    alone under Block Rock (one of each; the export leaves it out).
  - The step check can't rebuild Colin's Power of Money: a snapshot doesn't
    hold its attack bonus.
- **Validation:**
  1. **Serializer:** `phpser.dumps(loads(x)) == x` byte for byte on real
     snapshot and log files (floats as `d:10;` and `d:9.5;`, strings
     counted in bytes, lists against dicts).
  2. **Real-replay round trip** (the main test): for recent std replays,
     take each turn step_check matches. Rebuild every snapshot with
     `state_from_turn`, and every action's JSON from `step_turn`'s trace
     (the state before each action, and the next snapshot after End). Diff
     them against the originals once what can't match is normalized
     (dates, timers, row ids, users), and count mismatches by field and
     action kind. The target is none on the fields we model. This tests
     the exporter against AWBW's own output, not against itself.
  3. **Exported games:** export, then `load_replay` and step_check match
     every turn and end at the record's final hash, for arena and
     self-play games in the tests.
  4. **By hand:** open a few exports in the Replay Player and look for
     parse errors in its log, especially powers, a resignation and a
     day-limit game. Done on 16 self-play and arena games on real maps
     (v0.13.1): resignations, games stopped at the day cap, and both
     powers of Rachel, Von Bolt, Sturm, Kindle, Sami, Jess, Olaf and Drake
     (missiles, stuns, meteor, weather) load and play. AWBW's own day-limit
     GameOver has no sample, since no start uses the setting; a test
     covers it.
- **Command:** `awbw export-replay FILE [--game N | --all] [--out PATH]`
  writes `<stem>-N.zip` (done, `replay_file` and `game_meta`).
  - The game id is 1,000,000,000 plus a hash of the record's start and
    steps (`game_id`), so exporting a game again gives the same id and the
    Replay Player files it in the same place.
  - The name says who played which CO: "self-play step 120000: Hawke vs
    Grit" when one label played both sides, otherwise "a.pt (64 sims)
    (Hawke) vs random (Grit)". Players are named "Player 1 (Hawke)" and so
    on in AWBW's messages ("... has resigned!").
  - AWBW's day-limit setting (`Settings.day_limit`, none in any cached
    replay or start) decides a game at End Turn without a next turn. The
    End then logs `"event": "GameOver"` with `playersElim`, the shape the
    Replay Player's End parser reads, and no snapshot follows. A snapshot
    has no field for the limit, so the step check reaches that result only
    from a state that has it. Games stopped at `max_days` (arena,
    self-play) end on a snapshot with nothing logged, like an unfinished
    AWBW game.
  - Tests: self-play games stopped at the day cap and resigned export,
    read back and step exactly every turn; a day-limit game ends on its
    GameOver End; the command writes one zip or all of them.
- **Order:**
  1. `phpser.dumps` and the byte round trip. Done: `replay/write.py`
     writes the zip, log lines and AWBW's JSON, byte for byte on 2,000
     replays (the cached tests check the 200 newest).
  2. Snapshots, the unit-constant table, and the round trip on snapshots
     (done).
  3. The path and missile bindings (done: `move_path`; `missile_targets`
     already existed).
  4. The common actions (Move, Fire, Capt, Build, End, Load, Unload, Join,
     Resign) through the round trip (done, with Delete and the game-ending
     entries).
  5. The rarer actions, then powers CO by CO (done).
  6. The command, tests, and the check in the Replay Player (done).

### Local play against the network (`awbw play`, built; to try by hand)

A person plays a network on their own machine, in the browser, with AWBW's
rules exactly (two players, no fog, as everywhere in v1).

- **Shape:** `awbw play CKPT [--sims N] [--map ID] [--cos A B]
  [--countries A B] [--side 1|2]` starts a local server (the standard
  library's `ThreadingHTTPServer`, no new dependencies) and opens the page
  (done: `play.py`, `play.html`). The server holds the game and is the only
  side that applies rules; the page draws and sends clicks.
  - The API: `GET /api/view` (the board, players, and the person's legal
    picks decoded), `/api/unit?x&y` (any unit's ranges), and `POST`
    `/api/pick`, `cancel`, `undo`, `resign`, `think` (the network's next
    action, which the page animates) and `new`.
  - The network resigns by self-play's rule (done): at its first decision
    of a turn, when its value there was at most -0.9 in each of its last
    two turns (`--resign-value`, `--resign-turns`; an undo forgets the
    turns it undoes).
  - New game opens a form (done): the map (the cached two-player maps by
    name, any AWBW map id, fetched if new, or a real game's setup at
    random), each player's CO (Sonja aside, as in training) and army, and
    the person's side, filled in with the current game's setup
    (`GET /api/setup`; `POST /api/new` takes the choices, `play.Setup`).
    A CO left unchosen is the real game's or random; an army, the map's
    (a player whose slot's country the other took gets the other slot's).
  - The viewer and the play page share `board.js`.
- **Input follows the micro-steps (§3)**, which already match AWBW's
  clicks:
  - idle: click a unit or factory (Select), or Delete, COP, SCOP, End Turn;
  - selected: the move range is the legal Move ids; click a tile;
  - moved: the action menu is the legal action ids at the unit's tile
    (Wait, Capture, Load, Join, ...), and Fire, Unload and the like mark
    their target tiles;
  - factory: the build menu, each unit with its cost.

  For each pick the server sends the legal ids with `CHANNELS`, `PHASES`
  and the `ID_*` constants decoded into tiles and menu entries, so the page
  never needs rules of its own. Cancel resets the `Selection`; picks don't
  touch the state, so cancelling is free until the last step applies the
  action.
- **What the page shows** (done), beyond the viewer's board, funds and
  power bars:
  - the network's moves animated one action at a time;
  - optionally its value for the position (off by default, so as not to
    spoil a close game).
- **AWBW's conveniences** (done), all from the engine so they follow its
  rules:
  - **Movement range:** selecting a unit shades the tiles it can reach
    (its legal Move ids), and hovering one draws the path the unit will
    take (the cheapest path the engine moves along; the binding the export
    needs).
  - **Attack range:** with a unit selected, the tiles it could fire on this
    turn (from any reachable tile for a direct unit, from where it stands
    for an indirect one), and after moving, its actual targets.
  - **Any unit's range, enemies included:** clicking a unit that can't act
    shows where it could move and fire on its next turn. `reachable`
    already works for any unit.
  - **Damage preview:** each target's damage % (the formula at the luck
    rolls' bounds), its displayed HP after and the counter's, with their
    chances. Players see only displayed HP, as the network does, so the
    chances come from `fire_outcomes` over the HP beliefs (search's chance
    nodes; bound as `seen_fight_outcomes`), and the page's tooltips give
    displayed HP. The viewer keeps internal HP.
  - **Status icons on units:** HP, low fuel, low ammo, capturing, cargo
    aboard, dived or hidden; units that have acted are greyed out.
  - **Info panels:** a tile's terrain, defence stars, owner and capture
    points; a unit's HP, fuel and ammo; each player's funds, income, unit
    count, army value and power meter.
  - **Build menu:** costs with the CO's discount, unaffordable units
    greyed out, banned ones left out.
  - Confirmation before End Turn while units haven't acted.
- **Undo:** allowed locally (AWBW has none). The server keeps each
  action's state, so undo steps back one action at a time, the luck
  included: an attack retried the same way rolls the same.
- **Sprites (built):** AWBW's unit and terrain art is Nintendo's (the Replay
  Player's README says the same of its own), and this repo is public, so
  none of it is committed. The viewer and the play page draw sprites from
  a local copy if one is there, and otherwise coloured squares.
  - **The copy:** `cache/sprites/` (outside git) holds the Player's 9,947
    embedded images from `AWBWApp.Resources.dll` (the .NET manifest
    resources, extracted by name with `dnfile`) under its own folders, and
    its index files `Json/{Tiles,Buildings,Units,Countries,COs}.json`,
    which name each AWBW terrain, building and unit's textures and frame
    timings. Every texture the index files name is there, and they cover
    all 196 terrain ids the rules know and every unit type.
  - **What the boards draw** (`sprites.py` picks one game's textures and
    inlines them as data URIs: 280–360 KB of a viewer page and 620–760 KB
    of the play page, which adds rain, snow and move animations, on two
    maps tried): the
    Player's defaults, Classic terrain and AW2 buildings; buildings on
    plain, by owner and country; broken seams as rubble, used silos empty;
    weather variants; units in their country's sprites, idle animations and
    buildings' lights running, greyed once they have acted; the Player's HP
    digits and status icons (capturing, cargo, hidden, stunned, low fuel or
    ammo, one at a time); its arrow for the move path; and, in play, the
    agent's units walking in their move sprites. Sprites scale by whole
    device pixels where that shrinks the board by under a fifth, and the
    board leaves room above the top row for tall buildings.
  - **Coastlines:** AWBW has one sea id, so the Player picks sea and shoal
    pieces from the neighbours. A sea tile shows land on each side that is
    land and on each corner that is land with water on both sides next to
    it (shoal, reef and bridges count as water); a shoal shows land toward
    land and bridges, open water toward sea, reef and the map's edge, and
    nothing toward more shoal. The Player ships five test maps with their
    expected renders; this rule picks the piece shown on all 2,176 of their
    sea and shoal tiles.
  - Not done: the Player's other skins, and its power and explosion
    effects (`Textures/Effects`).
- **The network's side:** `PolicyAgent` (the prior, instant) or
  `SearchAgent` with N simulations and no root noise (as the arena), one
  decision at a time, through the same `Agent` protocol as the arena. The
  box's GPU stays with training until it finishes.
  - **Timing on the Mac** (M4 Pro, one game, so a network call per
    simulation): a whole turn of player 1, from 12 turn starts (days 3–18)
    of three prior-vs-prior games:

    | network | device | prior | 16 sims | 32 sims | 64 sims | 128 sims |
    |---|---|---|---|---|---|---|
    | tf128-s1 | MPS | 0.2 s (6 ms a decision) | 3.8 s | 7.9 s | 13.8 s | 26.9 s |
    | v1-64x6 | MPS | 0.2 s (4 ms) | 3.3 s | 6.1 s | 12.3 s | 22.6 s |
    | tf128-s1 | CPU | 0.6 s (14 ms) | 10.3 s | 20.0 s | | |

    Medians; the slowest turns took 1.5–1.7 times as long. MPS is 2.6 times
    the CPU. In a game played through the page, tf128-s1's turns at 64 sims
    took 41–54 s on days 9–12 with a large army (moves animated). The
    default is 32 simulations (search beat its prior 59–61% at 16 and
    67–72% at 64, above).
- **Starts** (done): a validation game's setup (map, COs, countries,
  settings), each replaced by `--map`, `--cos` or `--countries`, built by
  `engine.new_game` (AWBW's day-1 state; `play.start_for`):
  - player 1 is the map's first country slot (by AWBW country id); players
    may take any country, and in 62% of real games at least one does;
  - each slot's properties and predeployed units (full fuel and ammo) go to
    its player; capture points 20, pipe seams 99 HP; meters empty;
  - player 1 has a day's income (CO bonus included), player 2 none, both
    plus the game's starting funds;
  - checked: the same `state_hash` as all 33,860 first states of the v1
    dataset's training games (43 maps), and as AWBW's own first snapshot
    in all 2,941 of 3,000 sampled cached replays that start at the game's
    start on the map as cached (56 maps were edited since; one replay
    starts at player 2's turn). The cached tests check the 200 newest.
- **Records** (done): every finished game goes to
  `cache/games/local.jsonl` (source "local", players "human" and, for
  example, "tf128-s1 (32 sims)"), so `view-game` and `export-replay` work on
  it, and it opens in the Replay Player. Undone actions leave no trace.
- **Order:**
  1. Time the agents on the Mac (done).
  2. The server and its JSON API, with tests that play a game through the
     API, offering exactly the legal picks at every step (done).
  3. The page (done): played through by clicks in headless Chrome, with
     undos, a whole game without a page error.
  4. `new_game`, checked against real first snapshots (done).
  5. Recording, then the command (done).
  6. Playing it by hand.

## 9. Bindings (nanobind)

- **Build:** scikit-build-core with `nanobind_add_module(_core NB_STATIC ...)`
  and `nanobind_add_stub` for `.pyi` type stubs.
  - Release builds use `-O3`; a `-march=native` option for local builds.
- **`GameState`:**
  - `copy()`; pickling via raw bytes
  - `legal_actions()`, `outcomes()`, `apply()`; the micro-steps as
    module functions `legal_ids(state, sel)`, `legal_mask(state, sel)`,
    `step(state, sel, id)`, `step_action`, `action_steps` with a `Selection`
    (**done**)
  - `from_snapshot()`; `state_hash(state, sel)` and `hash_batch(state_bytes[N])`
    (**done**)
  - zero-copy numpy views for debugging
- **Batch functions:** `encode_batch(state_bytes[N]) -> planes, globals`
  and `encode_policy_batch(state_bytes[N], picks[N, 3], n_picks[N]) ->
  planes, globals, non-spatial legality, targets`, writing into
  caller-provided numpy buffers; `policy_index` / `policy_size` (**done**).
- **`BatchedSearch`** (**done**), `SearchConfig`, `SearchResult`,
  `GameRecord`: as in §8. `batch()` returns new numpy arrays each round;
  `submit` takes the logits of each leaf's legal steps. The object isn't
  thread-safe.
- **macOS:** Apple clang ships no OpenMP, so parallelism uses a `std::thread`
  pool.

## 10. Testing & benchmarks

- **C++ (googletest, `tests/cpp/`):** maps, state bookkeeping, snapshots,
  movement, damage, every action kind, the turn start, CO powers, the
  network encoding (the decision and movement planes included), turns of
  moving (against a turn-by-turn expansion), legal actions, the
  transposition key and search (164 tests). CI runs them on Linux and
  macOS, in Release and under ASan/UBSan.
- **Damage formula across all logged attacks:** every logged combat result
  must be in `outcomes()`. This checks the formula over the whole replay
  archive.
- **Full replay verification rate:** tracked per CO and per action type.
  A local regression corpus of verified replays runs in CI. Replays stay
  gitignored; only game IDs and hashes are committed.
- **Legal actions** (`tests/cpp/test_legal.cpp`): on positions from random
  games on a 12×10 map with every unit type, random COs, hidden Subs and
  Stealths, transports with cargo, seams and silos, `legal_actions` equals
  what a brute force over `apply` takes (every kind, unit, destination,
  target and cargo slot); the micro-steps from idle expand to exactly those
  actions; and each action's `action_steps` play it as `apply` does.
  `check-turns` also checks that every logged action's micro-steps are
  legal: all 18,456,503 actions of the 1,044,228 matching turns in 38,936
  cached replays are. The only logged actions outside the mask are 4
  attacks on a seam the attack before had already broken (AWBW logs a fresh
  HP for it); `apply` refuses those as well.
- **Random-playout invariant tests under ASan/UBSan:**
  - occupancy grid stays in sync (**done**, `check_state` after every step)
  - HP and funds stay in range (**done**)
  - the legal-action mask is non-empty until the game ends, and an id not
    in it changes nothing (**done**)
- **Search** (`test_search.cpp`, `tests/python/test_search.py`):
  - sequential halving spreads the visits as mctx's schedule does;
  - values are read from the mover's side, for either player;
  - self-play takes a win three steps down;
  - chance nodes visit results by their probability;
  - trees stay sound through random self-play games on the legal-actions
    test map, with forced steps getting no node and visits adding up;
  - every sample's steps are its state's legal ids, and the same seed plays
    the same games;
  - fast searches are played but not kept, and their leaves are marked
    as fast; a confident root ends a fast or one-off search at one
    evaluation, never a full one; self-play routes fast leaves to the
    fast network, which trains alongside and resumes;
  - leaves encode exactly as `encode_policy_batch` does, and the gathered
    logits give `Policy.score`'s prior.
- **Policy network** (`tests/python/test_policy.py`): each decision's
  planes and label match stepping the action; the policy ignores padding,
  where the board sits and the rest of the batch; flips map back; training
  resumes; arena games end or are adjudicated. The auxiliary targets match
  the game's later states. Networks with the movement planes and
  auxiliary head train, save and score, and older checkpoints still load.
- **Transposition key** (`test_hash.cpp`, **done**): moves and builds in
  either order give one key; slots, ids, the RNG and freed slots don't change
  it; every rule input does, cargo slots and Sonja's hidden HP included.
- **Benchmarks** (Google Benchmark, `bench/`; current numbers on an Apple
  M-series laptop in parentheses):
  - random games/sec
  - clone (65 ns), `check_state` (0.6 µs), `reachable` (0.4 µs per unit)
  - `damage` (3.5 ns), `combat_outcomes` (0.7 µs for tank vs tank)
  - copy + `apply`: a move (0.5 µs), a fight with rolled luck (0.2 µs), end
    of turn on a 30×30 board with ~50 units (0.3 µs); a fight with a Sonja
    unit whose display the opponent doesn't know (0.7 µs)
  - `encode` (2.1 µs for that board in a 32×32 window); a whole policy
    decision, with the legal steps and the decision planes (2.9 µs); the
    movement planes (24 µs); one uncapped `turns_from` over the board
    (3.7 µs); a 30×30 map's factory turns (0.8 ms, 172 rows)
  - `legal_ids` and `legal_actions` on that board (§3)
  - `state_hash` (0.10 µs; `absl::Hash` over the raw bytes, which tells
    transposed states apart, 0.08 µs); transposition-table lookups (§8)
  - search: 230,000 simulations/s with a stand-in network (`BM_Search`, §8)

## 11. Milestones

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Replay ingestion (Python): download + cache, parser, rules JSON | Thousands of replays parsed; settings/filter stats known |
| M1 | C++ core + bindings: state, `from_snapshot`, movegen, all actions except CO powers, `outcomes`, turn start, `encode_batch` | Every logged attack in `outcomes()` (**done**); state, snapshots, movegen (**done**); apply and turn start, every checkable turn replayed exactly (**done**); `encode_batch`, v0 dataset, **critic v0** (**done**) |
| M2 | Exact match: full replay stepping, all COs/powers, edge cases | CO powers' effects, missile aim, rain and snow (**done**); ≥ 99% of filtered replays fully verified; benchmarks in CI |
| M3 | v1 datasets: all idle-phase states + action labels | Micro-step action space and legal mask (**done**); policy prior (behavior cloning) with a value head that matches critic v0 (**done**); scaling grid on v1-70k (in progress, §7); baselines: random (**done**), greedy, critic-greedy 1-ply |
| M4 | Batched Gumbel MCTS with chance nodes, `collect`/`submit` | Search with chance nodes, self-play mode, sims/sec measured (**done**); search beats its own prior in the arena (**done**: 46–18 at 64 simulations); vs critic-greedy |
| M5 | Self-play loop: actors / learner / evaluator, mixed with human data | Loop, ladder and resignation (**done**: 3.3% wrong, 28% of decisions saved); Elo rises across checkpoints vs. fixed baselines |
