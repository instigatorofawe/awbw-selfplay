// Micro-benchmarks for the engine core:
//   cmake -S . -B build/cpp -DAWBW_BUILD_BENCHMARKS=ON && cmake --build build/cpp
//   build/cpp/awbw_bench

#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/actions.h"
#include "core/combat.h"
#include "core/encode.h"
#include "core/hash.h"
#include "core/legal.h"
#include "core/movegen.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;
using testing::make_state;
using testing::put;

// A 30x30 mixed-terrain map with a few dozen units, roughly midgame.
struct Board {
  MapData map;
  GameState state;
  std::vector<int> slots;
};

const Board& board() {
  static const Board* b = [] {
    auto* out = new Board;
    std::vector<std::string> rows;
    const char* pattern = "..F..=..M..#.._..R..F..=..~~..";
    for (int y = 0; y < 30; ++y) {
      std::string row;
      for (int x = 0; x < 30; ++x) row += pattern[(x * 7 + y * 3) % 30];
      rows.push_back(row);
    }
    out->map = make_map(rows);
    out->state = make_state(out->map);
    const int types[] = {kInfantry, kMech, kTank, kRecon, kArtillery, kMdTank, kAntiAir, kBCopter};
    int i = 0;
    for (int y = 1; y < 30; y += 4) {
      for (int x = 1; x < 30; x += 4) {
        const int type = types[i % 8];
        const int t = out->map.tile(x, y);
        if (move_cost(out->map, out->state, t, kUnits[type].move_type, kClear) == 0) continue;
        out->slots.push_back(put(out->state, out->map, type, i % 2, x, y));
        ++i;
      }
    }
    return out;
  }();
  return *b;
}

void BM_Damage(benchmark::State& st) {
  Combatant a{}, d{};
  a.type = d.type = kTank;
  a.hp = d.hp = 100;
  a.ammo = 9;
  a.co = d.co = 1;
  int luck = 0;
  for (auto _ : st) {
    benchmark::DoNotOptimize(damage(a, d, luck, false));
    luck = (luck + 1) % 10;
  }
}
BENCHMARK(BM_Damage);

void BM_CombatOutcomes(benchmark::State& st) {
  Combatant a{}, d{};
  a.type = d.type = kTank;
  a.hp = d.hp = 100;
  a.ammo = 9;
  a.co = d.co = 1;
  a.terrain = d.terrain = kPlain;
  a.stars = d.stars = 1;
  std::vector<Outcome> out;
  for (auto _ : st) {
    combat_outcomes(a, d, 1, out);
    benchmark::DoNotOptimize(out.data());
  }
}
BENCHMARK(BM_CombatOutcomes);

void BM_Reachable(benchmark::State& st) {
  const Board& b = board();
  auto r = std::make_unique<Reach>();
  size_t i = 0;
  for (auto _ : st) {
    reachable(b.map, b.state, b.slots[i], *r);
    benchmark::DoNotOptimize(r->n);
    i = (i + 1) % b.slots.size();
  }
}
BENCHMARK(BM_Reachable);

void BM_StateCopy(benchmark::State& st) {
  const Board& b = board();
  auto src = std::make_unique<GameState>(b.state);
  auto copy = std::make_unique<GameState>();
  for (auto _ : st) {
    GameState* from = src.get();
    benchmark::DoNotOptimize(from);
    std::memcpy(copy.get(), from, sizeof(GameState));
    benchmark::DoNotOptimize(copy.get());
    benchmark::ClobberMemory();
  }
  st.SetBytesProcessed(static_cast<int64_t>(st.iterations()) * sizeof(GameState));
}
BENCHMARK(BM_StateCopy);

void BM_CheckState(benchmark::State& st) {
  const Board& b = board();
  for (auto _ : st) benchmark::DoNotOptimize(check_state(b.map, b.state));
}
BENCHMARK(BM_CheckState);

// Copy the board and move one unit to the far edge of its reach (the copy is
// included: search applies actions to copies).
void BM_ApplyMove(benchmark::State& st) {
  const Board& b = board();
  auto r = std::make_unique<Reach>();
  std::vector<Action> moves;
  for (int slot : b.slots) {
    if (b.state.units[slot].owner != b.state.active) continue;
    reachable(b.map, b.state, slot, *r);
    for (int i = r->n - 1; i >= 0; --i) {
      if (b.state.unit_at[r->tiles[i]] == kNoUnit) {
        Action a;
        a.kind = kActWait;
        a.unit = static_cast<uint8_t>(slot);
        a.to = r->tiles[i];
        moves.push_back(a);
        break;
      }
    }
  }
  auto s = std::make_unique<GameState>();
  size_t i = 0;
  for (auto _ : st) {
    *s = b.state;
    benchmark::DoNotOptimize(apply(b.map, *s, moves[i]));
    i = (i + 1) % moves.size();
  }
}
BENCHMARK(BM_ApplyMove);

// Copy a small state and resolve a tank duel with luck drawn from the RNG.
void BM_ApplyFire(benchmark::State& st) {
  const MapData m = make_map({"......", "......"});
  GameState base = make_state(m);
  const int tank = put(base, m, kTank, 0, 0, 0);
  put(base, m, kTank, 1, 2, 0);
  Action a;
  a.kind = kActFire;
  a.unit = static_cast<uint8_t>(tank);
  a.to = static_cast<uint16_t>(m.tile(1, 0));
  a.target = static_cast<uint16_t>(m.tile(2, 0));
  auto s = std::make_unique<GameState>();
  for (auto _ : st) {
    *s = base;
    rng_next(base.rng);
    benchmark::DoNotOptimize(apply(m, *s, a));
  }
}
BENCHMARK(BM_ApplyFire);

// The same duel against Sonja, whose opponent doesn't see her tank's HP, with
// both meters full so they tell nothing and the hidden belief spreads.
void BM_ApplyFireSonja(benchmark::State& st) {
  const MapData m = make_map({"......", "......"});
  GameState base = make_state(m, 1, 18);
  for (Player& p : base.players) p.power = p.cop_cost = p.scop_cost = 90000;
  const int tank = put(base, m, kTank, 0, 0, 0);
  const int theirs = put(base, m, kTank, 1, 2, 0, 57);
  std::fill(base.hp_hidden[theirs] + 3, base.hp_hidden[theirs] + 7, 255);  // 4 to 7
  Action a;
  a.kind = kActFire;
  a.unit = static_cast<uint8_t>(tank);
  a.to = static_cast<uint16_t>(m.tile(1, 0));
  a.target = static_cast<uint16_t>(m.tile(2, 0));
  auto s = std::make_unique<GameState>();
  for (auto _ : st) {
    *s = base;
    rng_next(base.rng);
    benchmark::DoNotOptimize(apply(m, *s, a));
  }
}
BENCHMARK(BM_ApplyFireSonja);

// Copy the board and end the turn: the next player's income, repairs and fuel.
void BM_EndTurn(benchmark::State& st) {
  const Board& b = board();
  Action end;
  end.kind = kActEndTurn;
  auto s = std::make_unique<GameState>();
  for (auto _ : st) {
    *s = b.state;
    benchmark::DoNotOptimize(apply(b.map, *s, end));
  }
}
BENCHMARK(BM_EndTurn);

// Encode the board into a 32x32 window: network planes and globals.
void BM_Encode(benchmark::State& st) {
  const Board& b = board();
  std::vector<uint8_t> planes(size_t{kNumPlanes} * 32 * 32);
  std::vector<float> globals(kNumGlobals);
  for (auto _ : st) {
    encode(b.map, b.state, planes.data(), 32, 32, globals.data());
    benchmark::DoNotOptimize(planes.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Encode);

// Every complete action of the player to move (about 25 units).
void BM_LegalActions(benchmark::State& st) {
  const Board& b = board();
  std::vector<Action> acts;
  for (auto _ : st) {
    legal_actions(b.map, b.state, acts);
    benchmark::DoNotOptimize(acts.data());
  }
  st.counters["actions"] = static_cast<double>(acts.size());
}
BENCHMARK(BM_LegalActions);

// Legal micro-steps: picking a unit, then a picked Tank's destinations, then
// what it can do on one of them.
void BM_LegalIds(benchmark::State& st) {
  const Board& b = board();
  Selection sel;
  if (st.range(0) > 0) {
    const int tank = *std::find_if(b.slots.begin(), b.slots.end(), [&](int i) {
      return b.state.units[i].type == kTank && b.state.units[i].owner == 0;
    });
    sel = Selection{kPhaseSelected, static_cast<uint8_t>(tank), b.state.units[tank].tile};
    if (st.range(0) > 1) {
      std::vector<uint16_t> dests;
      legal_ids(b.map, b.state, sel, dests);
      sel = Selection{kPhaseMoved, sel.unit, static_cast<uint16_t>(dests.back() % kMaxTiles)};
    }
  }
  std::vector<uint16_t> ids;
  for (auto _ : st) {
    legal_ids(b.map, b.state, sel, ids);
    benchmark::DoNotOptimize(ids.data());
  }
  st.counters["ids"] = static_cast<double>(ids.size());
}
BENCHMARK(BM_LegalIds)->Arg(0)->Arg(1)->Arg(2);

// A policy decision's whole input in a 32x32 window: the board, then the legal
// steps from idle and the decision planes.
void BM_EncodeDecision(benchmark::State& st) {
  const Board& b = board();
  std::vector<uint8_t> planes((size_t{kNumPlanes} + kNumSelectionPlanes) * 32 * 32);
  std::vector<float> globals(int{kNumGlobals} + kNumSelectionGlobals);
  uint8_t nonspatial[kNumNonSpatialIds];
  std::vector<uint16_t> ids;
  const Selection sel;
  for (auto _ : st) {
    encode(b.map, b.state, planes.data(), 32, 32, globals.data());
    legal_ids(b.map, b.state, sel, ids);
    encode_selection(b.map, b.state, sel, ids.data(), static_cast<int>(ids.size()),
                     planes.data() + size_t{kNumPlanes} * 32 * 32, 32, 32,
                     globals.data() + kNumGlobals, nonspatial);
    benchmark::DoNotOptimize(planes.data());
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_EncodeDecision);

// The transposition-table key, from scratch.
void BM_StateHash(benchmark::State& st) {
  const Board& b = board();
  for (auto _ : st) {
    const GameState* s = &b.state;
    benchmark::DoNotOptimize(s);
    benchmark::DoNotOptimize(state_hash(*s));
  }
  st.counters["units"] = static_cast<double>(b.slots.size());
}
BENCHMARK(BM_StateHash);

// For comparison: absl::Hash over the state's raw bytes, which would tell
// transposed states apart (slots, ids, the RNG).
void BM_StateBytesAbslHash(benchmark::State& st) {
  const Board& b = board();
  for (auto _ : st) {
    const GameState* s = &b.state;
    benchmark::DoNotOptimize(s);
    benchmark::DoNotOptimize(
        absl::HashOf(std::string_view(reinterpret_cast<const char*>(s), sizeof(GameState))));
  }
}
BENCHMARK(BM_StateBytesAbslHash);

// A transposition table's lookups: 64-bit keys to node indices, half the
// probes found. Arg: entries.
template <class Table>
void BM_TableFind(benchmark::State& st) {
  const int n = static_cast<int>(st.range(0));
  uint64_t seed = 1;
  auto next = [&] {  // splitmix64
    uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  };
  Table table;
  table.reserve(n);
  std::vector<uint64_t> probes;
  for (int i = 0; i < n; ++i) {
    const uint64_t key = next();
    table.emplace(key, static_cast<uint32_t>(i));
    probes.push_back(i % 2 ? key : next());
  }
  std::shuffle(probes.begin(), probes.end(), std::mt19937_64(0));
  size_t i = 0;
  for (auto _ : st) {
    const auto it = table.find(probes[i]);
    benchmark::DoNotOptimize(it == table.end() ? 0u : it->second);
    if (++i == probes.size()) i = 0;
  }
}
BENCHMARK(BM_TableFind<absl::flat_hash_map<uint64_t, uint32_t>>)->Arg(4096)->Arg(1 << 20);
BENCHMARK(BM_TableFind<std::unordered_map<uint64_t, uint32_t>>)->Arg(4096)->Arg(1 << 20);

// Filling one: try_emplace of new keys, as expanding nodes does. Arg: entries.
template <class Table>
void BM_TableInsert(benchmark::State& st) {
  const int n = static_cast<int>(st.range(0));
  std::vector<uint64_t> keys(n);
  uint64_t x = 0x1234;
  for (uint64_t& k : keys) k = x = x * 6364136223846793005ull + 1442695040888963407ull;
  for (auto _ : st) {
    Table table;
    for (int i = 0; i < n; ++i) table.try_emplace(keys[i], static_cast<uint32_t>(i));
    benchmark::DoNotOptimize(table.size());
  }
  st.SetItemsProcessed(st.iterations() * n);
}
BENCHMARK(BM_TableInsert<absl::flat_hash_map<uint64_t, uint32_t>>)->Arg(4096)->Arg(1 << 20);
BENCHMARK(BM_TableInsert<std::unordered_map<uint64_t, uint32_t>>)->Arg(4096)->Arg(1 << 20);

}  // namespace
}  // namespace awbw
