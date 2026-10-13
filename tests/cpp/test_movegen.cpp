#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "core/movegen.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;
using testing::make_state;
using testing::put;

constexpr int kAndy = 1, kSturm = 29, kLash = 16, kAdder = 11;

// Tiles reached within `max_cost` (or any), as a sorted list.
std::vector<int> reached(const Reach& r, int max_cost = 255) {
  std::vector<int> out;
  for (int i = 0; i < r.n; ++i) {
    if (r.cost[r.tiles[i]] <= max_cost) out.push_back(r.tiles[i]);
  }
  std::sort(out.begin(), out.end());
  return out;
}

class MovegenTest : public ::testing::Test {
 protected:
  std::unique_ptr<Reach> r = std::make_unique<Reach>();
};

TEST_F(MovegenTest, InfantryDiamondOnPlains) {
  const MapData m = make_map(std::vector<std::string>(9, "........."));
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 4, 4);
  reachable(m, s, inf, *r);
  EXPECT_EQ(r->n, 25);  // 1 + 4 + 8 + 12
  EXPECT_EQ(r->tiles[0], m.tile(4, 4));
  EXPECT_EQ(r->cost[m.tile(4, 1)], 3);
  EXPECT_EQ(r->cost[m.tile(5, 2)], 3);
  EXPECT_EQ(r->cost[m.tile(4, 0)], kUnreachable);
  for (int i = 1; i < r->n; ++i) EXPECT_LE(r->cost[r->tiles[i - 1]], r->cost[r->tiles[i]]);
}

TEST_F(MovegenTest, TerrainCostsByMoveType) {
  const MapData m = make_map({
      ".MM.",
  });
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  reachable(m, s, inf, *r);
  EXPECT_EQ(r->cost[m.tile(1, 0)], 2);
  EXPECT_EQ(r->cost[m.tile(2, 0)], kUnreachable);  // 2 + 2 > 3
  s.units[inf].type = kMech;
  reachable(m, s, inf, *r);
  EXPECT_EQ(r->cost[m.tile(2, 0)], 2);             // boots cross mountains at 1
  EXPECT_EQ(r->cost[m.tile(3, 0)], kUnreachable);  // mechs move 2
  s.units[inf].type = kTank;
  reachable(m, s, inf, *r);
  EXPECT_EQ(r->n, 1);  // treads can't climb
}

TEST_F(MovegenTest, EnemiesBlockFriendsDoNot) {
  const MapData m = make_map({
      ".....",
      "MMMMM",
  });
  GameState s = make_state(m);
  const int tank = put(s, m, kTank, 0, 0, 0);
  put(s, m, kInfantry, 0, 1, 0);
  reachable(m, s, tank, *r);
  EXPECT_EQ(r->cost[m.tile(4, 0)], 4);  // through the friendly infantry
  put(s, m, kInfantry, 1, 2, 0);
  reachable(m, s, tank, *r);
  EXPECT_EQ(r->cost[m.tile(2, 0)], kUnreachable);
  EXPECT_EQ(r->cost[m.tile(3, 0)], kUnreachable);
  EXPECT_EQ(r->cost[m.tile(1, 0)], 1);
}

TEST_F(MovegenTest, FuelCapsMovement) {
  const MapData m = make_map({"........"});
  GameState s = make_state(m);
  const int tank = put(s, m, kTank, 0, 0, 0);
  s.units[tank].fuel = 2;
  reachable(m, s, tank, *r);
  EXPECT_EQ(reached(*r).size(), 3u);
  s.units[tank].fuel = 0;
  reachable(m, s, tank, *r);
  EXPECT_EQ(r->n, 1);
}

TEST_F(MovegenTest, MovedStunnedAndCarriedUnitsStay) {
  const MapData m = make_map({"...."});
  GameState s = make_state(m);
  const int tank = put(s, m, kTank, 0, 0, 0);
  s.units[tank].flags |= kMoved;
  reachable(m, s, tank, *r);
  EXPECT_EQ(r->n, 1);
  s.units[tank].flags = kStunned;
  EXPECT_EQ(move_budget(s, s.units[tank]), 0);
}

TEST_F(MovegenTest, CoMovement) {
  const MapData m = make_map({
      "............",
      "FFFFFFFFFFFF",
  });
  GameState s = make_state(m, kAndy);
  const int tank = put(s, m, kTank, 0, 0, 0);
  EXPECT_EQ(move_points(s, s.units[tank]), 6);
  s.players[0].power_on = 2;  // Hyper Upgrade: +1
  EXPECT_EQ(move_points(s, s.units[tank]), 7);
  s.players[0].co = kAdder;
  EXPECT_EQ(move_points(s, s.units[tank]), 8);  // Sidewinder: +2
  // Sturm: every passable tile costs 1, so the tank crosses woods at 1.
  s.players[0].co = kSturm;
  s.players[0].power_on = 0;
  s.units[tank].tile = static_cast<uint16_t>(m.tile(0, 1));
  s.unit_at[m.tile(0, 0)] = kNoUnit;
  s.unit_at[m.tile(0, 1)] = static_cast<uint8_t>(tank);
  reachable(m, s, tank, *r);
  EXPECT_EQ(r->cost[m.tile(6, 1)], 6);
  // Lash only on her powers.
  s.players[0].co = kLash;
  reachable(m, s, tank, *r);
  EXPECT_EQ(r->cost[m.tile(3, 1)], 6);
  s.players[0].power_on = 1;
  reachable(m, s, tank, *r);
  EXPECT_EQ(r->cost[m.tile(6, 1)], 6);
}

TEST_F(MovegenTest, SeamsBlockUntilBroken) {
  const MapData m = make_map({".S."});
  GameState s = make_state(m);
  const int tank = put(s, m, kTank, 0, 0, 0);
  reachable(m, s, tank, *r);
  EXPECT_EQ(r->cost[m.tile(2, 0)], kUnreachable);
  s.prop_points[m.prop[m.tile(1, 0)]] = 0;
  reachable(m, s, tank, *r);
  EXPECT_EQ(r->cost[m.tile(2, 0)], 2);
}

TEST_F(MovegenTest, PrevGivesACheapestPath) {
  const MapData m = make_map({
      ".FF.",
      "====",
  });
  GameState s = make_state(m);
  const int recon = put(s, m, kRecon, 0, 0, 0);
  reachable(m, s, recon, *r);
  int t = m.tile(3, 0);
  std::vector<uint16_t> path;
  while (t != -1) {
    path.insert(path.begin(), static_cast<uint16_t>(t));
    t = r->prev[t];
  }
  EXPECT_EQ(path_cost(m, s, recon, path.data(), static_cast<int>(path.size())),
            r->cost[m.tile(3, 0)]);
  // Tires pay 3 in woods and 2 on plains: the road detour (1+1+1+1+2) beats
  // going straight (3+3+2).
  EXPECT_EQ(r->cost[m.tile(3, 0)], 6);
  EXPECT_EQ(path.size(), 6u);
}

TEST(PathCost, ChecksEachStep) {
  const MapData m = make_map({
      "....",
      "~~~~",
  });
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  const uint16_t ok[] = {0, 1, 2};
  EXPECT_EQ(path_cost(m, s, inf, ok, 3), 2);
  const uint16_t jump[] = {0, 2};
  EXPECT_EQ(path_cost(m, s, inf, jump, 2), -1);
  const uint16_t wet[] = {0, 4};
  EXPECT_EQ(path_cost(m, s, inf, wet, 2), -1);
  const uint16_t far[] = {0, 1, 2, 3, 2};
  EXPECT_EQ(path_cost(m, s, inf, far, 5), -1);  // 4 > 3 movement
  const uint16_t wrong_start[] = {1, 2};
  EXPECT_EQ(path_cost(m, s, inf, wrong_start, 2), -1);
}

}  // namespace
}  // namespace awbw
