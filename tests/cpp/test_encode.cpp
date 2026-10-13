#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "core/belief.h"
#include "core/encode.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;
using testing::make_state;
using testing::put;

constexpr int kAndy = 1, kSasha = 19;

struct Encoded {
  int h, w;
  std::vector<uint8_t> planes;
  std::vector<float> globals;

  Encoded(const MapData& m, const GameState& s, int h_ = 8, int w_ = 8)
      : h(h_), w(w_), planes(size_t{kNumPlanes} * h_ * w_, 0xAB), globals(kNumGlobals, -1.0f) {
    encode(m, s, planes.data(), h, w, globals.data());
  }
  int at(int plane, int x, int y) const {
    return planes[(static_cast<size_t>(plane) * h + y) * w + x];
  }
  float me(int g) const { return globals[g]; }
  float opp(int g) const { return globals[kSideGlobals + g]; }
};

int terrain(int kind, int side) { return 1 + kind + kNumTerrainKinds * side; }

class EncodeTest : public ::testing::Test {
 protected:
  // Orange Star (player 0) owns C, H and the comtower T; Blue Moon owns c and h.
  MapData m = make_map({
      "C.#~~",
      "H.S.c",
      "...Th",
  });
  GameState s = make_state(m, kAndy, kSasha);
};

TEST_F(EncodeTest, TerrainAndPropertiesFromTheMoversSide) {
  const Encoded e(m, s);
  EXPECT_EQ(e.at(kPlaneTerrain, 0, 0), terrain(kCity, 1));  // mine
  EXPECT_EQ(e.at(kPlaneTerrain, 4, 1), terrain(kCity, 2));  // theirs
  EXPECT_EQ(e.at(kPlaneTerrain, 2, 0), terrain(kCity, 0));  // neutral
  EXPECT_EQ(e.at(kPlaneTerrain, 1, 0), terrain(kPlain, 0));
  EXPECT_EQ(e.at(kPlaneTerrain, 3, 0), terrain(kSea, 0));
  EXPECT_EQ(e.at(kPlaneTerrain, 2, 1), terrain(kSeam, 0));
  EXPECT_EQ(e.at(kPlanePoints, 0, 0), kFullCapture);
  EXPECT_EQ(e.at(kPlanePoints, 2, 1), kFullSeamHp);
  EXPECT_EQ(e.at(kPlanePoints, 1, 0), 0);
  // Outside the 5 x 3 board everything is 0, whatever the buffer held.
  for (int p = 0; p < kNumPlanes; ++p) {
    EXPECT_EQ(e.at(p, 5, 0), 0);
    EXPECT_EQ(e.at(p, 0, 3), 0);
    EXPECT_EQ(e.at(p, 7, 7), 0);
  }

  s.active = 1;
  const Encoded flipped(m, s);
  EXPECT_EQ(flipped.at(kPlaneTerrain, 0, 0), terrain(kCity, 2));
  EXPECT_EQ(flipped.at(kPlaneTerrain, 4, 1), terrain(kCity, 1));
}

TEST_F(EncodeTest, BrokenSeamsAndCapturesShow) {
  s.prop_points[m.prop[m.tile(2, 1)]] = 0;  // the seam broke into rubble
  s.prop_points[m.prop[m.tile(2, 0)]] = 7;  // the neutral city is being captured
  const Encoded e(m, s);
  EXPECT_EQ(e.at(kPlaneTerrain, 2, 1), terrain(kRubble, 0));
  EXPECT_EQ(e.at(kPlanePoints, 2, 1), 0);
  EXPECT_EQ(e.at(kPlanePoints, 2, 0), 7);
}

TEST_F(EncodeTest, UnitsAndCargo) {
  const int tank = put(s, m, kTank, 0, 1, 0, 64);
  const int apc = put(s, m, kApc, 1, 1, 2);
  const int inf = put(s, m, kInfantry, 1, 0, 2, 35);
  load_unit(s, inf, apc);
  s.units[tank].fuel = 41;
  s.units[tank].ammo = 3;
  s.units[tank].flags |= kMoved;

  const Encoded e(m, s);
  EXPECT_EQ(e.at(kPlaneUnit, 1, 0), 1 + kTank);
  EXPECT_EQ(e.at(kPlaneHp, 1, 0), 7);  // displayed
  for (int k = 0; k < kHpBins; ++k) EXPECT_EQ(e.at(kPlaneHpProb + k, 1, 0), k == 3 ? 255 : 0);
  EXPECT_EQ(e.at(kPlaneFuel, 1, 0), 41);
  EXPECT_EQ(e.at(kPlaneAmmo, 1, 0), 3);
  EXPECT_EQ(e.at(kPlaneFlags, 1, 0), kMoved);
  EXPECT_EQ(e.at(kPlaneUnit, 1, 2), 1 + kNumUnitTypes + kApc);
  EXPECT_EQ(e.at(kPlaneCargo1, 1, 2), 1 + kInfantry);
  EXPECT_EQ(e.at(kPlaneCargo1Hp, 1, 2), 35);
  EXPECT_EQ(e.at(kPlaneCargo2, 1, 2), 0);
  EXPECT_EQ(e.at(kPlaneUnit, 0, 2), 0);  // the infantry rides in the APC
  EXPECT_FLOAT_EQ(e.me(kGUnits), 1 / 50.0f);
  EXPECT_FLOAT_EQ(e.opp(kGUnits), 2 / 50.0f);
  EXPECT_FLOAT_EQ(e.me(kGArmyValue), 7000 * 7 / 10 / 100000.0f);
  EXPECT_FLOAT_EQ(e.opp(kGArmyValue), (5000 + 1000 * 4 / 10) / 100000.0f);

  s.active = 1;
  const Encoded flipped(m, s);
  EXPECT_EQ(flipped.at(kPlaneUnit, 1, 0), 1 + kNumUnitTypes + kTank);
  EXPECT_EQ(flipped.at(kPlaneUnit, 1, 2), 1 + kApc);
}

TEST_F(EncodeTest, HpIsWhatPlayersKnow) {
  const int tank = put(s, m, kTank, 0, 1, 0, 64);
  const int apc = put(s, m, kApc, 0, 1, 2);
  const int inf = put(s, m, kInfantry, 0, 0, 2, 35);
  load_unit(s, inf, apc);
  const uint8_t two[kHpBins] = {0, 0, 0, 255, 255, 0, 0, 0, 0, 0};  // 6.4 or 6.5
  std::copy(two, two + kHpBins, s.hp_belief[tank]);
  belief_unknown(s, inf);  // 3.1 to 4.0
  const Encoded e(m, s);
  EXPECT_EQ(e.at(kPlaneHp, 1, 0), 7);
  for (int k = 0; k < kHpBins; ++k)
    EXPECT_EQ(e.at(kPlaneHpProb + k, 1, 0), k == 3 || k == 4 ? 128 : 0);
  EXPECT_EQ(e.at(kPlaneCargo1Hp, 1, 2), 36);  // 35.5, rounded
}

// Sonja's opponent sees her units' displays only as a distribution; she sees
// them as anyone's.
TEST_F(EncodeTest, SonjasHpIsHiddenFromHerOpponent) {
  s.players[1].co = 18;
  const int tank = put(s, m, kTank, 1, 1, 0, 64);
  const int apc = put(s, m, kApc, 1, 1, 2);
  const int inf = put(s, m, kInfantry, 1, 0, 2, 35);
  load_unit(s, inf, apc);
  const int known = put(s, m, kMech, 1, 3, 2, 50);
  const uint8_t two[kHpBins] = {0, 0, 0, 0, 0, 255, 255, 0, 0, 0};  // 6 or 7
  std::copy(two, two + kHpBins, s.hp_hidden[tank]);
  const uint8_t inf_hidden[kHpBins] = {0, 0, 255, 85, 0, 0, 0, 0, 0, 0};  // 3, or less likely 4
  std::copy(inf_hidden, inf_hidden + kHpBins, s.hp_hidden[inf]);
  const Encoded e(m, s);
  EXPECT_EQ(e.at(kPlaneHp, 1, 0), 0);
  for (int k = 0; k < kHpBins; ++k)
    EXPECT_EQ(e.at(kPlaneHpProb + k, 1, 0), k == 5 || k == 6 ? 128 : 0);
  EXPECT_EQ(e.at(kPlaneCargo1Hp, 1, 2), 28);  // 3/4 x 2.55 + 1/4 x 3.55
  EXPECT_EQ(e.at(kPlaneHp, 3, 2), 5);         // a display they know
  EXPECT_EQ(e.at(kPlaneHpProb + 9, 3, 2), 255);
  const double value = 7000 * 6.5 / 10 + 5000 + 1000 * 3.25 / 10 + 3000 * 5 / 10;
  EXPECT_NEAR(e.opp(kGArmyValue), value / 100000.0, 1e-6);

  s.active = 1;
  const Encoded mine(m, s);
  EXPECT_EQ(mine.at(kPlaneHp, 1, 0), 7);
  EXPECT_EQ(mine.at(kPlaneHpProb + 3, 1, 0), 255);
  EXPECT_EQ(mine.at(kPlaneCargo1Hp, 1, 2), 35);
  EXPECT_NEAR(mine.me(kGArmyValue), (4900 + 5000 + 400 + 1500) / 100000.0, 1e-6);
}

TEST_F(EncodeTest, HiddenEnemiesStayOffTheBoardButCount) {
  const int sub = put(s, m, kSub, 1, 4, 0, 55);
  s.units[sub].flags |= kHidden;
  const int mine = put(s, m, kStealth, 0, 1, 2);
  s.units[mine].flags |= kHidden;
  const Encoded e(m, s);
  EXPECT_EQ(e.at(kPlaneUnit, 4, 0), 0);
  EXPECT_EQ(e.at(kPlaneHp, 4, 0), 0);
  EXPECT_EQ(e.at(kPlaneUnit, 1, 2), 1 + kStealth);  // my own hidden unit
  EXPECT_EQ(e.at(kPlaneFlags, 1, 2), kHidden);
  // The totals count both, as the game page does.
  const float sub_value = kUnits[kSub].cost * 6 / 10 / 100000.0f;
  EXPECT_FLOAT_EQ(e.opp(kGUnits), 1 / 50.0f);
  EXPECT_FLOAT_EQ(e.opp(kGArmyValue), sub_value);
  EXPECT_FLOAT_EQ(e.opp(kGHiddenSubs), 1 / 10.0f);
  EXPECT_FLOAT_EQ(e.opp(kGHiddenStealths), 0.0f);
  EXPECT_FLOAT_EQ(e.opp(kGHiddenValue), sub_value);
  EXPECT_FLOAT_EQ(e.me(kGUnits), 1 / 50.0f);
  EXPECT_FLOAT_EQ(e.me(kGHiddenStealths), 1 / 10.0f);  // hidden from the opponent
  EXPECT_FLOAT_EQ(e.me(kGHiddenValue), kUnits[kStealth].cost / 100000.0f);

  put(s, m, kCruiser, 0, 3, 0);  // next to the sub
  const Encoded seen(m, s);
  EXPECT_EQ(seen.at(kPlaneUnit, 4, 0), 1 + kNumUnitTypes + kSub);
  EXPECT_EQ(seen.at(kPlaneFlags, 4, 0), kHidden);
  EXPECT_FLOAT_EQ(seen.opp(kGHiddenSubs), 0.0f);
  EXPECT_FLOAT_EQ(seen.opp(kGHiddenValue), 0.0f);
  EXPECT_FLOAT_EQ(seen.opp(kGArmyValue), sub_value);
}

TEST_F(EncodeTest, PlayerGlobals) {
  s.players[0].funds = 12000;
  s.players[0].power = 18000;
  s.players[0].cop_cost = 27000;
  s.players[0].scop_cost = 54000;
  s.players[1].power_on = 2;
  s.players[1].power_uses = 3;
  s.weather = kSnow;
  s.weather_owner = 1;
  s.settings.capture_win = 21;
  s.day = 9;
  const Encoded e(m, s);
  EXPECT_FLOAT_EQ(e.me(kGFunds), 1.2f);
  EXPECT_FLOAT_EQ(e.me(kGIncome), 0.2f);    // city + HQ; comtowers pay nothing
  EXPECT_FLOAT_EQ(e.opp(kGIncome), 0.22f);  // city + HQ with Sasha's +100 each
  EXPECT_FLOAT_EQ(e.me(kGCopCharge), 18000.0f / 27000);
  EXPECT_FLOAT_EQ(e.me(kGScopCharge), 18000.0f / 54000);
  EXPECT_FLOAT_EQ(e.me(kGCopCost), 0.27f);
  EXPECT_FLOAT_EQ(e.opp(kGScopOn), 1.0f);
  EXPECT_FLOAT_EQ(e.opp(kGCopOn), 0.0f);
  EXPECT_FLOAT_EQ(e.opp(kGPowerUses), 0.3f);
  EXPECT_FLOAT_EQ(e.me(kGProps), 3 / 30.0f);
  EXPECT_FLOAT_EQ(e.opp(kGProps), 2 / 30.0f);
  EXPECT_FLOAT_EQ(e.me(kGComtowers), 0.1f);
  EXPECT_FLOAT_EQ(e.opp(kGComtowers), 0.0f);
  EXPECT_FLOAT_EQ(e.me(kGCo + kAndy), 1.0f);
  EXPECT_FLOAT_EQ(e.opp(kGCo + kSasha), 1.0f);
  EXPECT_FLOAT_EQ(e.me(kGCo + kSasha), 0.0f);
  EXPECT_FLOAT_EQ(e.globals[kGDay], 0.3f);
  EXPECT_FLOAT_EQ(e.globals[kGWeather + int{kSnow}], 1.0f);
  EXPECT_FLOAT_EQ(e.globals[kGWeather + int{kClear}], 0.0f);
  EXPECT_FLOAT_EQ(e.globals[kGWeatherTheirs], 1.0f);
  EXPECT_FLOAT_EQ(e.globals[kGCaptureWinOn], 1.0f);
  EXPECT_FLOAT_EQ(e.globals[kGCaptureWin], 21 / 50.0f);
  EXPECT_FLOAT_EQ(e.globals[kGMovedFirst], 1.0f);

  s.active = 1;
  const Encoded flipped(m, s);
  EXPECT_FLOAT_EQ(flipped.me(kGCo + kSasha), 1.0f);
  EXPECT_FLOAT_EQ(flipped.opp(kGFunds), 1.2f);
  EXPECT_FLOAT_EQ(flipped.globals[kGWeatherMine], 1.0f);
  EXPECT_FLOAT_EQ(flipped.globals[kGMovedFirst], 0.0f);
}

TEST(Encode, GlobalNamesAreUnique) {
  std::set<std::string> names;
  for (int i = 0; i < kNumGlobals; ++i) {
    const std::string n = global_name(i);
    EXPECT_FALSE(n.empty()) << i;
    names.insert(n);
  }
  EXPECT_EQ(static_cast<int>(names.size()), kNumGlobals);
  EXPECT_EQ(global_name(kSideGlobals + kGCo + kSasha), "opp.co.Sasha");
  EXPECT_EQ(global_name(kGWeather + int{kRain}), "weather.rain");
}

}  // namespace
}  // namespace awbw
