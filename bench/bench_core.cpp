// Micro-benchmarks for the engine core:
//   cmake -S . -B build/cpp -DAWBW_BUILD_BENCHMARKS=ON && cmake --build build/cpp
//   build/cpp/awbw_bench

#include <benchmark/benchmark.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/actions.h"
#include "core/combat.h"
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
        if (move_cost(out->map, out->state, t, kUnits[type].move_type) == 0) continue;
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

}  // namespace
}  // namespace awbw
