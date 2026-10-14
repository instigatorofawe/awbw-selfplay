#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "core/encode.h"
#include "core/movegen.h"
#include "core/movement.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;
using testing::make_state;
using testing::put;

constexpr int kAndy = 1, kAdder = 11;

std::vector<int> turns_row(const MapData& m, const std::vector<uint8_t>& cost, int mp,
                           std::vector<uint16_t> sources, std::vector<uint8_t> delay,
                           int max_turns = kMaxTurns) {
  std::vector<uint8_t> turns(m.n_tiles, 0xAB);
  turns_from(m, cost.data(), mp, sources.data(), delay.data(), static_cast<int>(sources.size()),
             max_turns, turns.data());
  return {turns.begin(), turns.end()};
}

constexpr int U = kUnreachable;

TEST(TurnsFrom, CountsTurnsOfMoving) {
  const MapData m = make_map({"........"});
  const std::vector<uint8_t> plain(m.n_tiles, 1);
  EXPECT_EQ(turns_row(m, plain, 3, {0}, {0}), (std::vector<int>{0, 1, 1, 1, 2, 2, 2, 3}));
  // A unit that has moved starts a turn later.
  EXPECT_EQ(turns_row(m, plain, 3, {0}, {1}), (std::vector<int>{0, 2, 2, 2, 3, 3, 3, 4}));
  // The nearest of several sources, each with its own start.
  EXPECT_EQ(turns_row(m, plain, 3, {0, 7}, {0, 1}), (std::vector<int>{0, 1, 1, 1, 2, 2, 2, 0}));
  EXPECT_EQ(turns_row(m, plain, 3, {0}, {0}, 2), (std::vector<int>{0, 1, 1, 1, 2, 2, 2, U}));
}

TEST(TurnsFrom, PointsDontCarryOver) {
  const MapData m = make_map({"......"});
  // Entering tile 3 costs 2: after 1 + 1 it doesn't fit in 3 points, so the
  // second turn starts there.
  EXPECT_EQ(turns_row(m, {1, 1, 1, 2, 1, 1}, 3, {0}, {0}), (std::vector<int>{0, 1, 1, 2, 2, 3}));
  // A tile dearer than a whole turn is never entered; 0 is impassable.
  EXPECT_EQ(turns_row(m, {1, 1, 4, 1, 1, 1}, 3, {0}, {0}), (std::vector<int>{0, 1, U, U, U, U}));
  EXPECT_EQ(turns_row(m, {1, 1, 0, 1, 1, 1}, 3, {0}, {0}), (std::vector<int>{0, 1, U, U, U, U}));
}

// The definition, turn by turn: where the unit can be after each turn is
// wherever it can get to within its points from where it could be after the
// one before.
std::vector<int> brute_force(const MapData& m, const std::vector<uint8_t>& cost, int mp,
                             const std::vector<uint16_t>& sources,
                             const std::vector<uint8_t>& delay, int max_turns) {
  std::vector<int> turns(m.n_tiles, U);
  std::vector<bool> at(m.n_tiles, false);
  for (uint16_t t : sources) turns[t] = 0;
  for (int k = 1; k <= max_turns; ++k) {
    for (size_t i = 0; i < sources.size(); ++i) {
      if (delay[i] == k - 1) at[sources[i]] = true;
    }
    std::vector<int> dist(m.n_tiles, 1 << 20);
    for (int t = 0; t < m.n_tiles; ++t) {
      if (at[t]) dist[t] = 0;
    }
    for (bool changed = true; changed;) {
      changed = false;
      for (int t = 0; t < m.n_tiles; ++t) {
        for (int d = 0; d < kNumDirs; ++d) {
          const int nb = m.neighbor[t][d];
          if (nb == kOffMap || cost[nb] == 0 || dist[t] + cost[nb] > mp) continue;
          if (dist[t] + cost[nb] < dist[nb]) {
            dist[nb] = dist[t] + cost[nb];
            changed = true;
          }
        }
      }
    }
    for (int t = 0; t < m.n_tiles; ++t) {
      if (dist[t] <= mp) {
        at[t] = true;
        if (turns[t] == U) turns[t] = k;
      }
    }
  }
  return turns;
}

TEST(TurnsFrom, MatchesTurnByTurnExpansion) {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 300; ++trial) {
    const int w = 3 + static_cast<int>(rng() % 10), h = 3 + static_cast<int>(rng() % 10);
    const MapData m = make_map(std::vector<std::string>(h, std::string(w, '.')));
    std::vector<uint8_t> cost(m.n_tiles);
    for (uint8_t& c : cost) c = static_cast<uint8_t>(rng() % 5);  // 0 impassable
    const int mp = 1 + static_cast<int>(rng() % 9);
    std::vector<uint16_t> sources;
    std::vector<uint8_t> delay;
    for (int i = 0, n = 1 + static_cast<int>(rng() % 4); i < n; ++i) {
      const uint16_t t = static_cast<uint16_t>(rng() % m.n_tiles);
      if (std::find(sources.begin(), sources.end(), t) != sources.end()) continue;
      sources.push_back(t);
      delay.push_back(static_cast<uint8_t>(rng() % 3));
    }
    const int max_turns = 1 + static_cast<int>(rng() % 6);
    ASSERT_EQ(turns_row(m, cost, mp, sources, delay, max_turns),
              brute_force(m, cost, mp, sources, delay, max_turns))
        << "trial " << trial;
  }
}

TEST(FactoryTurns, PerFactoryAndMoverClass) {
  // Orange Star base, then Blue Moon's airport and base.
  const MapData m = make_map({"B..M...ab"});
  EXPECT_EQ(kMovers.of_type[kTank], kMovers.of_type[kAntiAir]);
  EXPECT_NE(kMovers.of_type[kTank], kMovers.of_type[kMdTank]);
  const int base = m.prop[m.tile(0, 0)], airport = m.prop[m.tile(7, 0)];
  const int inf = kMovers.of_type[kInfantry], tank = kMovers.of_type[kTank];
  const int copter = kMovers.of_type[kBCopter];
  ASSERT_NE(m.factory_row[base][inf], kNoRow);
  EXPECT_EQ(m.factory_row[base][copter], kNoRow);
  EXPECT_EQ(m.factory_row[airport][inf], kNoRow);
  auto row = [&](int prop, int cls) {
    const uint8_t* r = m.turns_row(m.factory_row[prop][cls]);
    return std::vector<int>(r, r + m.n_tiles);
  };
  // Infantry: mountains cost 2. Tanks can't climb them.
  EXPECT_EQ(row(base, inf), (std::vector<int>{0, 1, 1, 2, 2, 3, 3, 3, 4}));
  EXPECT_EQ(row(base, tank), (std::vector<int>{0, 1, 1, U, U, U, U, U, U}));
  EXPECT_EQ(row(airport, copter), (std::vector<int>{2, 1, 1, 1, 1, 1, 1, 0, 1}));
}

struct Movement {
  int h, w;
  std::vector<uint8_t> planes;

  Movement(const MapData& m, const GameState& s, int h_ = 4, int w_ = 16)
      : h(h_), w(w_), planes(size_t{kNumMovementPlanes} * h_ * w_, 0xAB) {
    encode_movement(m, s, planes.data(), h, w);
  }
  int at(int plane, int x, int y = 0) const {
    return planes[(static_cast<size_t>(plane) * h + y) * w + x];
  }
  std::vector<int> row(int plane, int n, int y = 0) const {
    std::vector<int> out;
    for (int x = 0; x < n; ++x) out.push_back(at(plane, x, y));
    return out;
  }
};

int reach(int side, int group) { return kMovReach + side * kNumMoveGroups + group; }

TEST(EncodeMovement, FactoryPlanesOnUnitTiles) {
  const MapData m = make_map({"B......b"});
  GameState s = make_state(m, kAndy, kAndy);
  put(s, m, kInfantry, 0, 2, 0);
  put(s, m, kTank, 1, 6, 0);
  {
    const Movement e(m, s);
    EXPECT_EQ(e.at(kMovHome, 2), 1 + 1);  // two tiles from my base
    EXPECT_EQ(e.at(kMovAway, 2), 1 + 2);  // five from theirs
    EXPECT_EQ(e.at(kMovHome, 6), 1 + 1);
    EXPECT_EQ(e.at(kMovAway, 6), 1 + 1);  // six tiles at 6 a turn
    EXPECT_EQ(e.at(kMovHome, 3), 0);      // no unit
  }
  s.prop_owner[m.prop[m.tile(7, 0)]] = 0;  // I take their base
  const Movement e(m, s);
  EXPECT_EQ(e.at(kMovAway, 2), 0);
  EXPECT_EQ(e.at(kMovHome, 6), 0);
  EXPECT_EQ(e.at(kMovAway, 6), 1 + 1);
}

TEST(EncodeMovement, ReachFromEachSidesNextMove) {
  const MapData m = make_map({"..............", ".............."});
  GameState s = make_state(m, kAndy, kAndy);
  put(s, m, kInfantry, 0, 0, 0);
  const int moved = put(s, m, kInfantry, 0, 0, 1);
  s.units[moved].flags |= kMoved;
  const int tank = put(s, m, kTank, 1, 13, 1);
  {
    const Movement e(m, s);
    EXPECT_EQ(e.row(reach(0, kGroupFoot), 12),
              (std::vector<int>{1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0}));
    // The moved one starts next turn, so most of its row is nearer the other.
    EXPECT_EQ(e.row(reach(0, kGroupFoot), 12, 1),
              (std::vector<int>{1, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0}));
    EXPECT_EQ(e.at(reach(0, kGroupVehicle), 5), 0);
    // The opponent's Tank: 6 a turn, from their next turn.
    EXPECT_EQ(e.at(reach(1, kGroupVehicle), 13, 1), 1);
    EXPECT_EQ(e.at(reach(1, kGroupVehicle), 7, 1), 2);
    EXPECT_EQ(e.at(reach(1, kGroupVehicle), 6, 1), 3);
    EXPECT_EQ(e.at(reach(1, kGroupVehicle), 0, 0), 4);
  }
  s.units[tank].flags |= kStunned;  // sits out their next turn
  const Movement e(m, s);
  EXPECT_EQ(e.at(reach(1, kGroupVehicle), 7, 1), 3);
  EXPECT_EQ(e.at(reach(1, kGroupVehicle), 1, 1), 4);
  EXPECT_EQ(e.at(reach(1, kGroupVehicle), 0, 0), 0);
}

TEST(EncodeMovement, PowersCountForMeOnly) {
  const MapData m = make_map({"..............", ".............."});
  GameState s = make_state(m, kAdder, kAdder);
  put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 0, 1);
  s.players[0].power_on = 1;  // +1 move for my units now
  s.players[1].power_on = 1;  // over by the opponent's next turn
  const Movement e(m, s);
  EXPECT_EQ(e.at(reach(0, kGroupFoot), 4, 0), 2);
  EXPECT_EQ(e.at(reach(0, kGroupFoot), 5, 0), 3);
  EXPECT_EQ(e.at(reach(1, kGroupFoot), 3, 1), 2);
  EXPECT_EQ(e.at(reach(1, kGroupFoot), 4, 1), 3);
}

TEST(EncodeMovement, LeavesOutCargoAndOffBoardCells) {
  const MapData m = make_map({"....."});
  GameState s = make_state(m, kAndy, kAndy);
  const int apc = put(s, m, kApc, 0, 0, 0);
  const int inf = put(s, m, kInfantry, 0, 1, 0);
  load_unit(s, inf, apc);
  const Movement e(m, s, 2, 8);
  EXPECT_EQ(e.at(reach(0, kGroupFoot), 0), 0);
  EXPECT_EQ(e.at(reach(0, kGroupVehicle), 4), 2);
  for (int p = 0; p < kNumMovementPlanes; ++p) {
    for (int x = 5; x < 8; ++x) EXPECT_EQ(e.at(p, x), 0);
    for (int x = 0; x < 8; ++x) EXPECT_EQ(e.at(p, x, 1), 0);
  }
}

}  // namespace
}  // namespace awbw
