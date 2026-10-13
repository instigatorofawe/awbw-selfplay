#include <gtest/gtest.h>

#include "core/map.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;

TEST(Map, TilesKindsAndNeighbors) {
  const MapData m = make_map({
      "H.M",
      "~#c",
  });
  EXPECT_EQ(m.width, 3);
  EXPECT_EQ(m.height, 2);
  EXPECT_EQ(m.n_tiles, 6);
  EXPECT_EQ(m.tile(2, 1), 5);
  EXPECT_EQ(m.kind[m.tile(0, 0)], kHq);
  EXPECT_EQ(m.kind[m.tile(2, 0)], kMountain);
  EXPECT_EQ(m.stars[m.tile(2, 0)], 4);
  EXPECT_EQ(m.kind[m.tile(0, 1)], kSea);
  // North, east, south, west; off the map is kOffMap.
  const int16_t* n = m.neighbor[m.tile(0, 0)];
  EXPECT_EQ(n[0], kOffMap);
  EXPECT_EQ(n[1], m.tile(1, 0));
  EXPECT_EQ(n[2], m.tile(0, 1));
  EXPECT_EQ(n[3], kOffMap);
  EXPECT_EQ(m.distance(m.tile(0, 0), m.tile(2, 1)), 3);
}

TEST(Map, PropertyListInTileOrder) {
  const MapData m = make_map({
      "H.S",
      "~#c",
  });
  ASSERT_EQ(m.n_props, 4);  // HQ, seam, neutral city, Blue Moon city
  EXPECT_EQ(m.prop_tile[0], m.tile(0, 0));
  EXPECT_EQ(m.prop_kind[0], kHq);
  EXPECT_EQ(m.prop_country[0], testing::kOrangeStar);
  EXPECT_EQ(m.prop_kind[1], kSeam);
  EXPECT_EQ(m.prop_country[2], 0);
  EXPECT_EQ(m.prop_country[3], testing::kBlueMoon);
  EXPECT_EQ(m.prop[m.tile(1, 0)], kNoProp);
  EXPECT_EQ(m.prop[m.tile(2, 1)], 3);
}

TEST(Map, MoveCostsFromTables) {
  const MapData m = make_map({"MF~"});
  const int w = 0;  // clear
  EXPECT_EQ(m.move_cost[w][kMoveFoot][0], 2);
  EXPECT_EQ(m.move_cost[w][kMoveBoot][0], 1);
  EXPECT_EQ(m.move_cost[w][kMoveTread][0], 0);  // impassable
  EXPECT_EQ(m.move_cost[w][kMoveTire][1], 3);
  EXPECT_EQ(m.move_cost[w][kMoveSea][2], 1);
  EXPECT_EQ(m.move_cost[w][kMoveAir][2], 1);
}

TEST(Map, RejectsBadInput) {
  MapData m;
  std::string error;
  const int16_t unknown[] = {1, 9999};
  EXPECT_FALSE(build_map(1, 2, 1, unknown, m, error));
  EXPECT_NE(error.find("unknown terrain"), std::string::npos);
  std::vector<int16_t> big((kMaxWidth + 1) * 2, 1);
  EXPECT_FALSE(build_map(1, kMaxWidth + 1, 2, big.data(), m, error));
}

TEST(Map, Registry) {
  std::string error;
  const MapData a = make_map({"..", ".."}, 900001);
  const MapData* p = register_map(a, error);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(find_map(900001), p);
  EXPECT_EQ(register_map(a, error), p);  // same terrain again is fine
  const MapData b = make_map({"..", ".M"}, 900001);
  EXPECT_EQ(register_map(b, error), nullptr);
  EXPECT_EQ(find_map(900002), nullptr);
}

}  // namespace
}  // namespace awbw
