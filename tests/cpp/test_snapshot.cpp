#include <gtest/gtest.h>

#include <vector>

#include "core/snapshot.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;

Snapshot base_snapshot(const std::vector<SnapshotBuilding>& b, const std::vector<SnapshotUnit>& u) {
  Snapshot snap{};
  snap.day = 3;
  snap.active = 1;
  snap.players[0] = SnapshotPlayer{1, testing::kOrangeStar, 5000, 1200, 270000, 630000, 0, 0};
  snap.players[1] = SnapshotPlayer{7, testing::kBlueMoon, 0, 0, 270000, 540000, 1, 0};
  snap.buildings = b.data();
  snap.n_buildings = static_cast<int>(b.size());
  snap.units = u.data();
  snap.n_units = static_cast<int>(u.size());
  return snap;
}

TEST(Snapshot, PlayersBuildingsAndUnits) {
  const MapData m = make_map({
      "H.#S",
      "..~h",
  });
  // The neutral city now belongs to Blue Moon (terrain id 43) and is half
  // captured; the seam has been hit.
  const std::vector<SnapshotBuilding> b = {
      {0, 0, 42, 20}, {2, 0, 43, 10}, {3, 0, 113, 55}, {3, 1, 47, 20}};
  const std::vector<SnapshotUnit> u = {
      {500, kTank, 1, 1, 1, 73, 40, 5, kMoved, {0, 0}},
      {300, kInfantry, 0, 1, 0, 100, 99, 0, kCapturing, {0, 0}},
  };
  GameState s;
  std::string error;
  ASSERT_TRUE(load_snapshot(m, base_snapshot(b, u), s, error)) << error;
  EXPECT_EQ(s.day, 3);
  EXPECT_EQ(s.active, 1);
  EXPECT_EQ(s.players[0].funds, 5000);
  EXPECT_EQ(s.players[1].co, 7);
  EXPECT_EQ(s.players[1].power_on, 1);
  EXPECT_EQ(s.prop_owner[m.prop[m.tile(0, 0)]], 0);
  EXPECT_EQ(s.prop_owner[m.prop[m.tile(2, 0)]], 1);
  EXPECT_EQ(s.prop_points[m.prop[m.tile(2, 0)]], 10);
  EXPECT_EQ(s.prop_points[m.prop[m.tile(3, 0)]], 55);
  EXPECT_EQ(s.prop_owner[m.prop[m.tile(3, 1)]], 1);
  // Slots follow AWBW id order.
  EXPECT_EQ(s.units[0].id, 300u);
  EXPECT_EQ(s.units[1].id, 500u);
  EXPECT_EQ(s.units[1].hp, 73);
  EXPECT_TRUE(s.units[1].has(kMoved));
  EXPECT_EQ(s.unit_at[m.tile(1, 1)], 1);
  EXPECT_EQ(s.next_id, 501u);
}

TEST(Snapshot, BrokenSeamAndCargo) {
  const MapData m = make_map({
      "...S",
      "~~~~",
  });
  const std::vector<SnapshotBuilding> b = {{3, 0, 115, 0}};  // rubble now
  const std::vector<SnapshotUnit> u = {
      {10, kLander, 0, 0, 1, 100, 90, 0, 0, {12, 11}},
      {11, kTank, 0, 0, 1, 100, 70, 9, 0, {0, 0}},
      {12, kInfantry, 0, 0, 1, 100, 99, 0, 0, {0, 0}},
  };
  GameState s;
  std::string error;
  ASSERT_TRUE(load_snapshot(m, base_snapshot(b, u), s, error)) << error;
  EXPECT_EQ(s.prop_points[m.prop[m.tile(3, 0)]], 0);
  EXPECT_EQ(terrain_kind(m, s, m.tile(3, 0)), kRubble);
  EXPECT_EQ(s.unit_at[m.tile(0, 1)], 0);
  EXPECT_EQ(s.units[0].cargo[0], 2);  // cargo order as AWBW lists it
  EXPECT_EQ(s.units[0].cargo[1], 1);
  EXPECT_TRUE(s.units[1].has(kCarried));
  EXPECT_EQ(s.units[2].carrier, 0);
}

TEST(Snapshot, Errors) {
  const MapData m = make_map({"..", ".."});
  GameState s;
  std::string error;
  std::vector<SnapshotUnit> two = {
      {1, kInfantry, 0, 0, 0, 100, 99, 0, 0, {0, 0}},
      {2, kInfantry, 1, 0, 0, 100, 99, 0, 0, {0, 0}},
  };
  EXPECT_FALSE(load_snapshot(m, base_snapshot({}, two), s, error));
  EXPECT_NE(error.find("two units"), std::string::npos);
  std::vector<SnapshotUnit> off = {{1, kInfantry, 0, 5, 0, 100, 99, 0, 0, {0, 0}}};
  EXPECT_FALSE(load_snapshot(m, base_snapshot({}, off), s, error));
  std::vector<SnapshotUnit> missing = {{1, kApc, 0, 0, 0, 100, 70, 0, 0, {7, 0}}};
  EXPECT_FALSE(load_snapshot(m, base_snapshot({}, missing), s, error));
}

}  // namespace
}  // namespace awbw
