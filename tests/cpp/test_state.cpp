#include <gtest/gtest.h>

#include <cstring>

#include "core/state.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;
using testing::make_state;
using testing::put;

class StateTest : public ::testing::Test {
 protected:
  MapData m = make_map({
      ".....",
      ".....",
      "~~~~~",
  });
  GameState s = make_state(m);
};

TEST_F(StateTest, AddAndRemove) {
  const int a = put(s, m, kInfantry, 0, 1, 1);
  const int b = put(s, m, kTank, 1, 2, 1);
  EXPECT_EQ(a, 0);
  EXPECT_EQ(b, 1);
  EXPECT_EQ(s.unit_at[m.tile(1, 1)], a);
  EXPECT_LT(s.units[a].id, s.units[b].id);
  EXPECT_EQ(check_state(m, s), "");
  EXPECT_EQ(put(s, m, kInfantry, 0, 3, 1), 2);
  remove_unit(s, a);
  EXPECT_EQ(s.unit_at[m.tile(1, 1)], kNoUnit);
  EXPECT_EQ(count_units(s, 0), 1);
  EXPECT_EQ(put(s, m, kMech, 0, 0, 0), a);  // lowest free slot is reused
  EXPECT_GT(s.units[a].id, s.units[b].id);  // but ids keep creation order
  EXPECT_EQ(check_state(m, s), "");
}

TEST_F(StateTest, RejectsOccupiedTile) {
  put(s, m, kInfantry, 0, 1, 1);
  Unit u{};
  u.tile = static_cast<uint16_t>(m.tile(1, 1));
  u.type = kTank;
  u.hp = 100;
  EXPECT_EQ(add_unit(s, u), -1);
}

TEST_F(StateTest, CargoFollowsTransport) {
  const int lander = put(s, m, kLander, 0, 0, 2);
  const int tank = put(s, m, kTank, 0, 0, 1);
  const int inf = put(s, m, kInfantry, 0, 1, 1);
  load_unit(s, tank, lander);
  load_unit(s, inf, lander);
  EXPECT_EQ(check_state(m, s), "");
  EXPECT_TRUE(s.units[tank].has(kCarried));
  EXPECT_EQ(s.unit_at[m.tile(0, 1)], kNoUnit);
  relocate_unit(s, lander, m.tile(3, 2));
  EXPECT_EQ(s.units[tank].tile, m.tile(3, 2));
  EXPECT_EQ(s.units[inf].tile, m.tile(3, 2));
  unload_unit(s, tank, m.tile(3, 1));            // leaves slot 1 empty
  EXPECT_EQ(s.units[lander].cargo[0], kNoUnit);  // slots don't shift, as in AWBW
  EXPECT_EQ(s.units[lander].cargo[1], inf);
  const int mech = put(s, m, kMech, 0, 4, 1);
  load_unit(s, mech, lander);  // into the free first slot
  EXPECT_EQ(s.units[lander].cargo[0], mech);
  EXPECT_EQ(check_state(m, s), "");
  remove_unit(s, lander);  // cargo goes down with it
  EXPECT_FALSE(s.units[inf].alive());
  EXPECT_TRUE(s.units[tank].alive());
  EXPECT_EQ(check_state(m, s), "");
}

TEST_F(StateTest, CheckStateFindsCorruption) {
  const int a = put(s, m, kInfantry, 0, 1, 1);
  GameState bad = s;
  bad.unit_at[m.tile(1, 1)] = kNoUnit;
  EXPECT_NE(check_state(m, bad), "");
  bad = s;
  bad.units[a].hp = 0;
  EXPECT_NE(check_state(m, bad), "");
  bad = s;
  bad.units[a].fuel = 100;
  EXPECT_NE(check_state(m, bad), "");
}

TEST_F(StateTest, CopiesAreBytes) {
  put(s, m, kInfantry, 0, 1, 1);
  GameState copy;
  std::memcpy(&copy, &s, sizeof s);
  EXPECT_EQ(std::memcmp(&copy, &s, sizeof s), 0);
  EXPECT_EQ(check_state(m, copy), "");
}

TEST(State, DynamicTerrain) {
  const MapData m = make_map({"S.X"});
  GameState s = make_state(m);
  const int seam = m.tile(0, 0), silo = m.tile(2, 0);
  EXPECT_EQ(s.prop_points[m.prop[seam]], kFullSeamHp);
  EXPECT_EQ(terrain_kind(m, s, seam), kSeam);
  EXPECT_EQ(move_cost(m, s, seam, kMoveTread, kClear), 0);
  s.prop_points[m.prop[seam]] = 0;  // broken
  EXPECT_EQ(terrain_kind(m, s, seam), kRubble);
  EXPECT_EQ(terrain_stars(m, s, seam), 1);
  EXPECT_EQ(move_cost(m, s, seam, kMoveTread, kClear), 1);
  EXPECT_EQ(terrain_kind(m, s, silo), kSilo);
  s.prop_points[m.prop[silo]] = 0;
  EXPECT_EQ(terrain_kind(m, s, silo), kSiloEmpty);
}

TEST(State, PropertyCounts) {
  const MapData m = make_map({"HCc#T"});
  const GameState s = make_state(m);
  EXPECT_EQ(count_props(m, s, 0), 3);
  EXPECT_EQ(count_props(m, s, 0, kComtower), 1);
  EXPECT_EQ(count_props(m, s, 1), 1);
  EXPECT_EQ(count_props(m, s, kNoOwner), 1);
}

}  // namespace
}  // namespace awbw
