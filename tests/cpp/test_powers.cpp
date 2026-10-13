// CO powers' immediate effects and weather.

#include <gtest/gtest.h>

#include <vector>

#include "core/actions.h"
#include "core/movegen.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;
using testing::make_state;
using testing::put;

constexpr int kAndy = 1, kDrake = 5, kOlaf = 9, kEagle = 10, kHawke = 12, kSensei = 13, kJess = 14,
              kSasha = 19, kKindle = 23, kRachel = 28, kSturm = 29, kVonBolt = 30;

Action power(int level) {
  Action a;
  a.kind = kActPower;
  a.arg = static_cast<uint8_t>(level);
  return a;
}

Action end_turn() {
  Action a;
  a.kind = kActEndTurn;
  return a;
}

// Give `player` a full meter at current prices.
void charge_up(GameState& s, int player) {
  Player& p = s.players[player];
  p.cop_cost = cop_cost(p.co, p.power_uses);
  p.scop_cost = scop_cost(p.co, p.power_uses);
  p.power = p.scop_cost;
}

// End the active player's turn and the opponent's, back to the same player.
void pass_round(const MapData& m, GameState& s) {
  ASSERT_EQ(apply(m, s, end_turn()), kOk);
  ASSERT_EQ(apply(m, s, end_turn()), kOk);
}

int cost_to(const MapData& m, const GameState& s, int slot, int x, int y) {
  Reach r;
  reachable(m, s, slot, r);
  return r.cost[m.tile(x, y)] == kUnreachable ? -1 : r.cost[m.tile(x, y)];
}

TEST(Weather, SnowAndRainRaiseMoveCosts) {
  const MapData m = make_map({"....F"});
  GameState s = make_state(m);
  const int tank = put(s, m, kTank, 0, 0, 0);
  EXPECT_EQ(cost_to(m, s, tank, 3, 0), 3);
  s.weather = kSnow;
  EXPECT_EQ(cost_to(m, s, tank, 3, 0), 6);  // plains cost treads 2 in snow
  s.weather = kRain;
  EXPECT_EQ(cost_to(m, s, tank, 3, 0), 6);
  EXPECT_EQ(cost_to(m, s, tank, 4, 0), -1);  // woods cost 3 in rain: 9 > 6 moves
}

TEST(Weather, OlafDrakeAndSturmMoveTheirOwnWay) {
  const MapData m = make_map({"....."});
  GameState s = make_state(m, kOlaf, kDrake);
  const int olaf = put(s, m, kTank, 0, 0, 0);
  const int drake = put(s, m, kTank, 1, 4, 0);
  s.weather = kSnow;
  EXPECT_EQ(cost_to(m, s, olaf, 3, 0), 3);  // Olaf ignores snow
  EXPECT_EQ(cost_to(m, s, drake, 1, 0), 6);
  s.weather = kRain;
  EXPECT_EQ(cost_to(m, s, olaf, 3, 0), 6);   // and moves in rain as others do in snow
  EXPECT_EQ(cost_to(m, s, drake, 1, 0), 3);  // Drake ignores rain

  const MapData m2 = make_map({"FFFF"});
  GameState t = make_state(m2, kSturm);
  const int sturm = put(t, m2, kTank, 0, 0, 0);
  EXPECT_EQ(cost_to(m2, t, sturm, 3, 0), 3);  // every tile costs Sturm 1 ...
  t.weather = kSnow;
  EXPECT_EQ(cost_to(m2, t, sturm, 3, 0), 6);  // ... except in snow
}

TEST(Weather, PowerWeatherEndsAtItsOwnersNextTurn) {
  const MapData m = make_map({"...."});
  GameState s = make_state(m, kOlaf);
  put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 3, 0);
  charge_up(s, 0);
  ASSERT_EQ(apply(m, s, power(1)), kOk);
  EXPECT_EQ(s.weather, kSnow);
  ASSERT_EQ(apply(m, s, end_turn()), kOk);
  EXPECT_EQ(s.weather, kSnow);  // the opponent's turn is snowy
  ASSERT_EQ(apply(m, s, end_turn()), kOk);
  EXPECT_EQ(s.weather, kClear);
  EXPECT_EQ(s.weather_owner, kNoOwner);
}

TEST(Powers, AndyHealsEveryUnitEvenInTransports) {
  const MapData m = make_map({"...."});
  GameState s = make_state(m, kAndy);
  const int tank = put(s, m, kTank, 0, 0, 0, 45);
  const int apc = put(s, m, kApc, 0, 1, 0);
  const int inf = put(s, m, kInfantry, 0, 2, 0, 63);
  load_unit(s, inf, apc);
  const int foe = put(s, m, kTank, 1, 3, 0, 45);
  charge_up(s, 0);
  ASSERT_EQ(apply(m, s, power(2)), kOk);
  EXPECT_EQ(s.units[tank].hp, 95);
  EXPECT_EQ(s.units[inf].hp, 100);
  EXPECT_EQ(s.units[foe].hp, 45);
}

TEST(Powers, HawkeHealsAndHurtsOnlyUnitsOnTheBoard) {
  const MapData m = make_map({"....."});
  GameState s = make_state(m, kHawke);
  const int own = put(s, m, kTank, 0, 0, 0, 50);
  const int apc = put(s, m, kApc, 0, 1, 0);
  const int cargo = put(s, m, kInfantry, 0, 2, 0, 50);
  load_unit(s, cargo, apc);
  const int foe = put(s, m, kTank, 1, 3, 0, 15);
  const int weak = put(s, m, kInfantry, 1, 4, 0, 8);
  charge_up(s, 0);
  ASSERT_EQ(apply(m, s, power(2)), kOk);
  EXPECT_EQ(s.units[own].hp, 70);
  EXPECT_EQ(s.units[cargo].hp, 50);
  EXPECT_EQ(s.units[foe].hp, 1);  // power damage leaves 0.1 HP
  EXPECT_EQ(s.units[weak].hp, 1);
}

TEST(Powers, DrakeHurtsAndHalvesEnemyFuelAndBringsRain) {
  const MapData m = make_map({"...."});
  GameState s = make_state(m, kDrake);
  put(s, m, kInfantry, 0, 0, 0);
  const int foe = put(s, m, kTank, 1, 3, 0);
  s.units[foe].fuel = 69;
  charge_up(s, 0);
  ASSERT_EQ(apply(m, s, power(1)), kOk);
  EXPECT_EQ(s.units[foe].hp, 90);
  EXPECT_EQ(s.units[foe].fuel, 35);  // rounded up
  EXPECT_EQ(s.weather, kClear);
  pass_round(m, s);
  charge_up(s, 0);
  ASSERT_EQ(apply(m, s, power(2)), kOk);
  EXPECT_EQ(s.units[foe].hp, 70);
  EXPECT_EQ(s.weather, kRain);
}

TEST(Powers, EagleLightningStrikeLetsAllButFootsoldiersActAgain) {
  const MapData m = make_map({"...."});
  GameState s = make_state(m, kEagle);
  const int tank = put(s, m, kTank, 0, 0, 0);
  const int inf = put(s, m, kInfantry, 0, 1, 0);
  put(s, m, kInfantry, 1, 3, 0);
  s.units[tank].flags |= kMoved;
  s.units[inf].flags |= kMoved;
  charge_up(s, 0);
  ASSERT_EQ(apply(m, s, power(2)), kOk);
  EXPECT_FALSE(s.units[tank].has(kMoved));
  EXPECT_TRUE(s.units[inf].has(kMoved));
}

TEST(Powers, SenseiPlacesFootsoldiersOnEmptyOwnedCities) {
  const MapData m = make_map({"CCc#"});
  GameState s = make_state(m, kSensei);
  put(s, m, kTank, 0, 1, 0);
  put(s, m, kInfantry, 1, 3, 0);
  charge_up(s, 0);
  ASSERT_EQ(apply(m, s, power(2)), kOk);
  const int placed = s.unit_at[m.tile(0, 0)];
  ASSERT_NE(placed, kNoUnit);
  EXPECT_EQ(s.units[placed].type, kMech);
  EXPECT_EQ(s.units[placed].hp, 90);
  EXPECT_FALSE(s.units[placed].has(kMoved));
  EXPECT_EQ(s.unit_at[m.tile(2, 0)], kNoUnit);  // the enemy's city
  EXPECT_EQ(count_units(s, 0), 2);
}

TEST(Powers, JessResuppliesEverything) {
  const MapData m = make_map({"...."});
  GameState s = make_state(m, kJess);
  const int tank = put(s, m, kTank, 0, 0, 0);
  s.units[tank].ammo = 0;
  s.units[tank].fuel = 3;
  put(s, m, kInfantry, 1, 3, 0);
  charge_up(s, 0);
  ASSERT_EQ(apply(m, s, power(1)), kOk);
  EXPECT_EQ(s.units[tank].ammo, kUnits[kTank].ammo);
  EXPECT_EQ(s.units[tank].fuel, kUnits[kTank].fuel);
}

TEST(Powers, SashaMarketCrashDrainsTheEnemyMeter) {
  const MapData m = make_map({"...."});
  GameState s = make_state(m, kSasha, kAndy);
  put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 3, 0);
  charge_up(s, 0);
  charge_up(s, 1);
  s.players[0].funds = 25000;
  const int before = s.players[1].power;
  ASSERT_EQ(apply(m, s, power(1)), kOk);
  EXPECT_EQ(s.players[1].power, before - int64_t{s.players[1].scop_cost} * 25000 / 50000);
}

TEST(Powers, WarBondsPayHalfTheValueOfDamageDealt) {
  const MapData m = make_map({"...."});
  GameState s = make_state(m, kSasha);
  const int tank = put(s, m, kTank, 0, 0, 0);
  const int foe = put(s, m, kTank, 1, 1, 0);
  charge_up(s, 0);
  ASSERT_EQ(apply(m, s, power(2)), kOk);
  s.players[0].funds = 0;
  Action fire;
  fire.kind = kActFire;
  fire.unit = static_cast<uint8_t>(tank);
  fire.to = static_cast<uint16_t>(m.tile(0, 0));
  fire.target = static_cast<uint16_t>(m.tile(1, 0));
  Detail d;
  d.att_hp = 80;
  d.def_hp = 40;  // 6 HP of a 7000 tank: 4200 of damage
  ASSERT_EQ(apply(m, s, fire, d), kOk);
  EXPECT_EQ(s.players[0].funds, 2100);
  // Her counterattacks pay too, through the opponent's turn.
  ASSERT_EQ(apply(m, s, end_turn()), kOk);
  s.players[0].funds = 0;
  fire.unit = static_cast<uint8_t>(foe);
  fire.to = static_cast<uint16_t>(m.tile(1, 0));
  fire.target = static_cast<uint16_t>(m.tile(0, 0));
  std::vector<Outcome> outs;
  combat_outcomes(combatant(m, s, foe, m.tile(1, 0)), combatant(m, s, tank, m.tile(0, 0)), 1, outs);
  ASSERT_FALSE(outs.empty());
  d.att_hp = outs[0].att_hp;
  d.def_hp = outs[0].def_hp;
  ASSERT_EQ(apply(m, s, fire, d), kOk);
  EXPECT_EQ(s.players[0].funds, (4 - displayed_hp(outs[0].att_hp)) * kUnits[kTank].cost / 20);
  EXPECT_GT(s.players[0].funds, 0);
}

TEST(Powers, KindleUrbanBlightHitsEnemiesOnUrbanTerrain) {
  const MapData m = make_map({"..#&"});
  GameState s = make_state(m, kKindle);
  put(s, m, kInfantry, 0, 0, 0);
  const int open = put(s, m, kTank, 1, 1, 0);
  const int city = put(s, m, kTank, 1, 2, 0);
  const int tower = put(s, m, kTank, 1, 3, 0);
  charge_up(s, 0);
  ASSERT_EQ(apply(m, s, power(1)), kOk);
  EXPECT_EQ(s.units[open].hp, 100);
  EXPECT_EQ(s.units[city].hp, 70);
  EXPECT_EQ(s.units[tower].hp, 70);
}

TEST(Powers, MissilesHitEveryUnitInReachAndVonBoltStuns) {
  const MapData m = make_map({"......."});
  GameState s = make_state(m, kVonBolt);
  const int own = put(s, m, kTank, 0, 1, 0);
  const int near = put(s, m, kTank, 1, 3, 0);
  const int edge = put(s, m, kTank, 1, 5, 0);
  const int far = put(s, m, kTank, 1, 6, 0);
  charge_up(s, 0);
  const uint16_t centre = static_cast<uint16_t>(m.tile(3, 0));
  Detail d;
  d.missiles = &centre;
  d.n_missiles = 1;
  ASSERT_EQ(apply(m, s, power(2), d), kOk);
  for (int slot : {own, near, edge}) {
    EXPECT_EQ(s.units[slot].hp, 70);
    EXPECT_TRUE(s.units[slot].has(kStunned));
  }
  EXPECT_EQ(s.units[far].hp, 100);
  EXPECT_FALSE(s.units[far].has(kStunned));
}

TEST(Powers, AStunnedUnitKeepsItsCaptureFlag) {
  const MapData m = make_map({"..#.."});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 1, 2, 0);
  s.units[inf].flags |= kCapturing | kStunned;
  ASSERT_EQ(apply(m, s, end_turn()), kOk);  // Blue Moon's turn starts
  EXPECT_TRUE(s.units[inf].has(kMoved));
  EXPECT_TRUE(s.units[inf].has(kCapturing));
}

TEST(Powers, MissileCountMustMatchThePower) {
  const MapData m = make_map({"...."});
  GameState s = make_state(m, kRachel);
  put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 3, 0);
  charge_up(s, 0);
  const uint16_t one = 0;
  Detail d;
  d.missiles = &one;
  d.n_missiles = 1;
  EXPECT_EQ(apply(m, s, power(2), d), kErrBadTarget);
}

TEST(Powers, SturmAimsAtTheEnemyUnitWithTheMostValueAround) {
  const MapData m = make_map({".........."});
  GameState s = make_state(m, kSturm);
  put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 2, 0);
  put(s, m, kMdTank, 1, 7, 0);
  put(s, m, kTank, 0, 8, 0);  // Sturm's own tank near the Md.Tank costs 7000 of value
  put(s, m, kTank, 1, 5, 0);
  charge_up(s, 0);
  uint16_t out[3];
  ASSERT_EQ(missile_targets(m, s, 1, out), 1);
  // On the Md.Tank: 16000 + 7000 - 7000; on the Tank: 7000 + 16000; never an empty tile.
  EXPECT_EQ(out[0], m.tile(5, 0));
}

TEST(Powers, RachelAimsAtFootsoldiersThenValueThenHp) {
  const MapData m = make_map({"..............", "..............", ".............."});
  GameState s = make_state(m, kRachel);
  put(s, m, kInfantry, 0, 13, 2);
  // Two infantry at the left, a Md.Tank at the right.
  put(s, m, kInfantry, 1, 1, 1);
  put(s, m, kInfantry, 1, 2, 1);
  put(s, m, kMdTank, 1, 10, 1);
  charge_up(s, 0);
  uint16_t out[3];
  ASSERT_EQ(missile_targets(m, s, 2, out), 3);
  // Footsoldiers: 2 x 3.0 x 4 at best; the first such tile in row order.
  EXPECT_EQ(m.y_of(out[0]), 0);
  EXPECT_LE(m.x_of(out[0]), 2);
  EXPECT_EQ(out[1], m.tile(9, 0));  // value: the Md.Tank, from the top-left of its ties
  EXPECT_EQ(m.y_of(out[2]), 0);     // HP: 6.0 by the infantry beats 3.0
  EXPECT_LE(m.x_of(out[2]), 2);
}

}  // namespace
}  // namespace awbw
