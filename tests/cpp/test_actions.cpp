#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "core/actions.h"
#include "core/movegen.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;
using testing::make_state;
using testing::put;

constexpr int kAndy = 1, kKanbei = 3, kColin = 15, kSonja = 18;

Action act(int kind, int unit, int to, int target = 0, int arg = 0) {
  Action a;
  a.kind = static_cast<uint8_t>(kind);
  a.unit = static_cast<uint8_t>(unit);
  a.to = static_cast<uint16_t>(to);
  a.target = static_cast<uint16_t>(target);
  a.arg = static_cast<uint8_t>(arg);
  return a;
}

Action player_act(int kind, int target = 0, int arg = 0) {
  return act(kind, kNoUnit, 0, target, arg);
}

bool same(const GameState& a, const GameState& b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// End the active player's turn and the opponent's, back to the same player.
void pass_round(const MapData& m, GameState& s) {
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
}

TEST(Actions, WaitTakesTheCheapestPathAndEndsTheUnitsTurn) {
  const MapData m = make_map({"......", "......"});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  ASSERT_EQ(apply(m, s, act(kActWait, inf, m.tile(2, 0))), kOk);
  EXPECT_EQ(s.units[inf].tile, m.tile(2, 0));
  EXPECT_EQ(s.units[inf].fuel, 97);
  EXPECT_TRUE(s.units[inf].has(kMoved));
  EXPECT_EQ(s.unit_at[m.tile(0, 0)], kNoUnit);
  EXPECT_EQ(apply(m, s, act(kActWait, inf, m.tile(3, 0))), kErrCannotAct);
}

TEST(Actions, ErrorsLeaveTheStateUnchanged) {
  const MapData m = make_map({"......", "......"});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  const int other = put(s, m, kInfantry, 0, 1, 0);
  const int enemy = put(s, m, kInfantry, 1, 5, 1);
  const GameState before = s;
  EXPECT_EQ(apply(m, s, act(kActWait, inf, m.tile(5, 0))), kErrUnreachable);
  EXPECT_EQ(apply(m, s, act(kActWait, inf, m.tile(1, 0))), kErrOccupied);
  EXPECT_EQ(apply(m, s, act(kActWait, enemy, m.tile(4, 1))), kErrNotYourUnit);
  EXPECT_EQ(apply(m, s, act(kActCapture, other, m.tile(2, 0))), kErrCannotCapture);
  EXPECT_TRUE(same(s, before));
}

TEST(Actions, AnExplicitPathChargesItsOwnCost) {
  const MapData m = make_map({"......", "......"});
  GameState s = make_state(m);
  const int recon = put(s, m, kRecon, 0, 0, 0);
  const std::vector<uint16_t> detour = {
      static_cast<uint16_t>(m.tile(0, 0)), static_cast<uint16_t>(m.tile(0, 1)),
      static_cast<uint16_t>(m.tile(1, 1)), static_cast<uint16_t>(m.tile(2, 1)),
      static_cast<uint16_t>(m.tile(2, 0))};
  Detail d;
  d.path = detour.data();
  d.path_len = static_cast<int>(detour.size());
  ASSERT_EQ(apply(m, s, act(kActWait, recon, m.tile(2, 0)), d), kOk);
  EXPECT_EQ(s.units[recon].fuel, 80 - 4 * 2);  // tires pay 2 on plains
  d.path_len = 2;                              // doesn't end on `to`
  GameState t = make_state(m);
  const int r2 = put(t, m, kRecon, 0, 0, 0);
  EXPECT_EQ(apply(m, t, act(kActWait, r2, m.tile(2, 0)), d), kErrBadPath);
}

TEST(Actions, AnUnseenHiddenEnemyTrapsTheMover) {
  const MapData m = make_map({"======"});
  GameState s = make_state(m);
  const int recon = put(s, m, kRecon, 0, 0, 0);
  const int stealth = put(s, m, kStealth, 1, 3, 0);
  s.units[stealth].flags |= kHidden;
  EXPECT_FALSE(visible_to(m, s, stealth, 0));
  ASSERT_EQ(apply(m, s, act(kActWait, recon, m.tile(5, 0))), kTrapped);
  EXPECT_EQ(s.units[recon].tile, m.tile(2, 0));  // stopped next to it
  EXPECT_EQ(s.units[recon].fuel, 78);            // paid only for the tiles it crossed
  EXPECT_TRUE(s.units[recon].has(kMoved));
  EXPECT_TRUE(visible_to(m, s, stealth, 0));  // now adjacent
}

TEST(Actions, AVisibleEnemyBlocks) {
  const MapData m = make_map({"======"});
  GameState s = make_state(m);
  const int recon = put(s, m, kRecon, 0, 0, 0);
  put(s, m, kInfantry, 1, 3, 0);
  EXPECT_EQ(apply(m, s, act(kActWait, recon, m.tile(5, 0))), kErrUnreachable);
}

class FireTest : public ::testing::Test {
 protected:
  MapData m = make_map({"......", "......"});
  GameState s = make_state(m);
  int tank = put(s, m, kTank, 0, 0, 0);
  int target = put(s, m, kTank, 1, 2, 0);

  std::vector<Outcome> outcomes(int from) {
    std::vector<Outcome> out;
    combat_outcomes(combatant(m, s, tank, from), combatant(m, s, target, m.tile(2, 0)),
                    m.distance(from, m.tile(2, 0)), out);
    return out;
  }
};

TEST_F(FireTest, ForcedResultSpendsAmmoAndChargesMeters) {
  s.players[0].cop_cost = s.players[1].cop_cost = cop_cost(kAndy, 0);
  s.players[0].scop_cost = s.players[1].scop_cost = scop_cost(kAndy, 0);
  const std::vector<Outcome> outs = outcomes(m.tile(1, 0));
  ASSERT_FALSE(outs.empty());
  const Outcome o = outs.front();
  Detail d;
  d.att_hp = o.att_hp;
  d.def_hp = o.def_hp;
  ASSERT_EQ(apply(m, s, act(kActFire, tank, m.tile(1, 0), m.tile(2, 0)), d), kOk);
  EXPECT_EQ(s.units[tank].hp, o.att_hp);
  EXPECT_EQ(s.units[target].hp, o.def_hp);
  EXPECT_EQ(s.units[tank].ammo, 8);
  EXPECT_EQ(s.units[target].ammo, 8);  // countered with its cannon
  // Meter: own displayed HP lost at full cost, plus half the cost of what you dealt.
  const int lost_att = (10 - displayed_hp(o.att_hp)) * 7000;
  const int lost_def = (10 - displayed_hp(o.def_hp)) * 7000;
  EXPECT_EQ(s.players[0].power, lost_att + lost_def / 2);
  EXPECT_EQ(s.players[1].power, lost_def + lost_att / 2);
  EXPECT_TRUE(s.units[tank].has(kMoved));
}

TEST_F(FireTest, ImpossibleForcedResultIsRejected) {
  const GameState before = s;
  Detail d;
  d.att_hp = 100;
  d.def_hp = 100;
  EXPECT_EQ(apply(m, s, act(kActFire, tank, m.tile(1, 0), m.tile(2, 0)), d), kErrBadOutcome);
  EXPECT_TRUE(same(s, before));
}

TEST_F(FireTest, RolledResultIsAPossibleOne) {
  const std::vector<Outcome> outs = outcomes(m.tile(1, 0));
  for (uint64_t seed = 0; seed < 50; ++seed) {
    GameState t = s;
    rng_seed(t.rng, seed);
    ASSERT_EQ(apply(m, t, act(kActFire, tank, m.tile(1, 0), m.tile(2, 0))), kOk);
    const bool found = std::any_of(outs.begin(), outs.end(), [&](const Outcome& o) {
      return o.att_hp == t.units[tank].hp && o.def_hp == t.units[target].hp;
    });
    EXPECT_TRUE(found) << "seed " << seed;
  }
}

TEST_F(FireTest, OutOfRangeAndOwnUnitsAreRejected) {
  EXPECT_EQ(apply(m, s, act(kActFire, tank, m.tile(0, 1), m.tile(2, 0))), kErrCannotFire);
  const int friend_ = put(s, m, kInfantry, 0, 1, 1);
  EXPECT_EQ(apply(m, s, act(kActFire, tank, m.tile(0, 0), m.tile(1, 1))), kErrBadTarget);
  (void)friend_;
}

TEST(Actions, IndirectUnitsCantMoveAndFire) {
  const MapData m = make_map({"......"});
  GameState s = make_state(m);
  const int art = put(s, m, kArtillery, 0, 0, 0);
  put(s, m, kInfantry, 1, 3, 0);
  EXPECT_EQ(apply(m, s, act(kActFire, art, m.tile(1, 0), m.tile(3, 0))), kErrCannotFire);
  EXPECT_EQ(apply(m, s, act(kActFire, art, m.tile(0, 0), m.tile(3, 0))), kOk);  // range 2-3
}

TEST(Actions, ArtilleryFiresFromWhereItStands) {
  const MapData m = make_map({"......"});
  GameState s = make_state(m);
  const int art = put(s, m, kArtillery, 0, 0, 0);
  const int inf = put(s, m, kInfantry, 1, 2, 0);
  ASSERT_EQ(apply(m, s, act(kActFire, art, m.tile(0, 0), m.tile(2, 0))), kOk);
  EXPECT_EQ(s.units[art].hp, 100);  // no counter at range
  EXPECT_EQ(s.units[art].ammo, 8);
  EXPECT_LT(s.units[inf].hp, 100);
}

TEST(Actions, DestroyingTheLastUnitWinsByRout) {
  const MapData m = make_map({"......"});
  GameState s = make_state(m);
  const int tank = put(s, m, kTank, 0, 0, 0);
  put(s, m, kInfantry, 1, 1, 0, 1);
  ASSERT_EQ(apply(m, s, act(kActFire, tank, m.tile(0, 0), m.tile(1, 0))), kOk);
  EXPECT_EQ(s.unit_at[m.tile(1, 0)], kNoUnit);
  EXPECT_EQ(s.winner, 0);
  EXPECT_TRUE(s.players[1].eliminated);
  EXPECT_EQ(apply(m, s, player_act(kActEndTurn)), kErrGameOver);
}

TEST(Actions, HiddenAttackerTakesNoCounterButAmmoIsSpent) {
  const MapData m = make_map({"......"});
  GameState s = make_state(m);
  const int stealth = put(s, m, kStealth, 0, 0, 0);
  s.units[stealth].flags |= kHidden;
  const int aa = put(s, m, kAntiAir, 1, 2, 0);
  ASSERT_EQ(apply(m, s, act(kActFire, stealth, m.tile(1, 0), m.tile(2, 0))), kOk);
  EXPECT_EQ(s.units[stealth].hp, 100);
  EXPECT_EQ(s.units[aa].ammo, 8);
}

TEST(Actions, SonjaCounterBonusAddsToAttack) {
  // Sonja's 1.5x counters add 50 to the attack value; luck isn't scaled.
  Combatant mech;
  mech.type = kMech;
  mech.hp = 40;
  mech.co = kSonja;
  Combatant inf;
  inf.type = kInfantry;
  inf.co = kAndy;
  inf.terrain = kRoad;
  // (65 * 150 / 100 + luck) * 4/10, luck 0: 39.
  EXPECT_EQ(damage(mech, inf, 0, true), 39);
  EXPECT_EQ(damage(mech, inf, 0, false), 26);
}

TEST(Actions, CaptureTakesTwoTurnsAtFullHp) {
  const MapData m = make_map({".#...."});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  const int city = m.prop[m.tile(1, 0)];
  ASSERT_EQ(apply(m, s, act(kActCapture, inf, m.tile(1, 0))), kOk);
  EXPECT_EQ(s.prop_points[city], 10);
  EXPECT_EQ(s.prop_owner[city], kNoOwner);
  EXPECT_TRUE(s.units[inf].has(kCapturing));
  pass_round(m, s);
  EXPECT_FALSE(s.units[inf].has(kCapturing));  // AWBW's per-turn flag
  ASSERT_EQ(apply(m, s, act(kActCapture, inf, m.tile(1, 0))), kOk);
  EXPECT_EQ(s.prop_owner[city], 0);
  EXPECT_EQ(s.prop_points[city], kFullCapture);
}

TEST(Actions, CapturePointsFollowDisplayedHpAndResetWhenTheCapturerLeaves) {
  const MapData m = make_map({".#...."});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 0, 0, 45);
  const int city = m.prop[m.tile(1, 0)];
  ASSERT_EQ(apply(m, s, act(kActCapture, inf, m.tile(1, 0))), kOk);
  EXPECT_EQ(s.prop_points[city], 15);
  pass_round(m, s);
  ASSERT_EQ(apply(m, s, act(kActWait, inf, m.tile(2, 0))), kOk);
  EXPECT_EQ(s.prop_points[city], kFullCapture);
}

TEST(Actions, TakingTheHqWins) {
  const MapData m = make_map({".h...b"});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 5, 0);
  ASSERT_EQ(apply(m, s, act(kActCapture, inf, m.tile(1, 0))), kOk);
  pass_round(m, s);
  ASSERT_EQ(apply(m, s, act(kActCapture, inf, m.tile(1, 0))), kOk);
  EXPECT_EQ(s.winner, 0);
}

TEST(Actions, TakingTheLastLabOfAPlayerWithoutAnHqWins) {
  const MapData m = make_map({".l...."});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 5, 0);
  ASSERT_EQ(apply(m, s, act(kActCapture, inf, m.tile(1, 0))), kOk);
  pass_round(m, s);
  ASSERT_EQ(apply(m, s, act(kActCapture, inf, m.tile(1, 0))), kOk);
  EXPECT_EQ(s.winner, 0);
}

TEST(Actions, CaptureLimitIgnoresComtowersAndLabs) {
  const MapData m = make_map({".#TLC.", "......"});
  GameState s = make_state(m);
  s.settings.capture_win = 2;  // the city to take, and the one owned (not the tower or lab)
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 5, 1);
  ASSERT_EQ(apply(m, s, act(kActCapture, inf, m.tile(1, 0))), kOk);
  pass_round(m, s);
  ASSERT_EQ(apply(m, s, act(kActCapture, inf, m.tile(1, 0))), kOk);
  EXPECT_EQ(s.winner, 0);
  GameState t = make_state(m);
  t.settings.capture_win = 3;
  const int inf2 = put(t, m, kInfantry, 0, 0, 0);
  put(t, m, kInfantry, 1, 5, 1);
  ASSERT_EQ(apply(m, t, act(kActCapture, inf2, m.tile(1, 0))), kOk);
  pass_round(m, t);
  ASSERT_EQ(apply(m, t, act(kActCapture, inf2, m.tile(1, 0))), kOk);
  EXPECT_EQ(t.winner, kNoOwner);  // 4 properties, but only 2 count
}

TEST(Actions, DayLimitGoesToTheMostPropertiesCountingComtowersAndLabs) {
  const MapData m = make_map({"CC.ctl", "......"});
  GameState s = make_state(m);
  s.settings.day_limit = 2;
  put(s, m, kInfantry, 0, 2, 0);
  put(s, m, kInfantry, 1, 2, 1);
  pass_round(m, s);
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  EXPECT_EQ(s.winner, kNoOwner);  // day 2 is played out
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  EXPECT_EQ(s.winner, 1);  // tower and lab: 3 properties to 2
  EXPECT_EQ(apply(m, s, player_act(kActEndTurn)), kErrGameOver);
}

TEST(Actions, DayLimitTieIsADraw) {
  const MapData m = make_map({"CT.cl", "....."});
  GameState s = make_state(m);
  s.settings.day_limit = 1;
  put(s, m, kInfantry, 0, 2, 0);
  put(s, m, kInfantry, 1, 2, 1);
  pass_round(m, s);
  EXPECT_EQ(s.winner, kDraw);
}

class TransportTest : public ::testing::Test {
 protected:
  MapData m = make_map({"......", "......"});
  GameState s = make_state(m);
  int inf = put(s, m, kInfantry, 0, 0, 0);
  int apc = put(s, m, kApc, 0, 1, 0);
};

TEST_F(TransportTest, LoadThenUnloadInPlaceKeepsTheTransportsAction) {
  ASSERT_EQ(apply(m, s, act(kActLoad, inf, m.tile(1, 0))), kOk);
  EXPECT_TRUE(s.units[inf].has(kCarried));
  EXPECT_EQ(s.units[apc].cargo[0], inf);
  EXPECT_EQ(s.unit_at[m.tile(0, 0)], kNoUnit);
  ASSERT_EQ(apply(m, s, act(kActUnload, apc, m.tile(1, 0), m.tile(2, 0), 0)), kOk);
  EXPECT_EQ(s.units[inf].tile, m.tile(2, 0));
  EXPECT_TRUE(s.units[inf].has(kMoved));
  EXPECT_FALSE(s.units[apc].has(kMoved));
  EXPECT_EQ(apply(m, s, act(kActWait, apc, m.tile(4, 0))), kOk);
  EXPECT_EQ(check_state(m, s), "");
}

TEST_F(TransportTest, UnloadAfterMovingEndsTheTransportsTurnAndMayUseTheTileItLeft) {
  ASSERT_EQ(apply(m, s, act(kActLoad, inf, m.tile(1, 0))), kOk);
  ASSERT_EQ(apply(m, s, act(kActUnload, apc, m.tile(2, 0), m.tile(1, 0), 0)), kOk);
  EXPECT_EQ(s.units[apc].tile, m.tile(2, 0));
  EXPECT_EQ(s.units[inf].tile, m.tile(1, 0));
  EXPECT_TRUE(s.units[apc].has(kMoved));
  EXPECT_EQ(check_state(m, s), "");
}

TEST_F(TransportTest, CannotLoadWhatItCantCarry) {
  const int tank = put(s, m, kTank, 0, 0, 1);
  EXPECT_EQ(apply(m, s, act(kActLoad, tank, m.tile(1, 0))), kErrCannotLoad);
}

TEST(Actions, CargoSlotsDontShift) {
  const MapData m = make_map({"__~~~~", "......"});
  GameState s = make_state(m);
  const int lander = put(s, m, kLander, 0, 1, 0);
  const int a = put(s, m, kInfantry, 0, 0, 1);
  const int b = put(s, m, kTank, 0, 1, 1);
  const int c = put(s, m, kMech, 0, 2, 1);
  ASSERT_EQ(apply(m, s, act(kActLoad, a, m.tile(1, 0))), kOk);
  ASSERT_EQ(apply(m, s, act(kActLoad, b, m.tile(1, 0))), kOk);
  ASSERT_EQ(apply(m, s, act(kActUnload, lander, m.tile(1, 0), m.tile(0, 0), 0)), kOk);
  EXPECT_EQ(s.units[lander].cargo[0], kNoUnit);
  EXPECT_EQ(s.units[lander].cargo[1], b);
  ASSERT_EQ(apply(m, s, act(kActLoad, c, m.tile(1, 0))), kOk);
  EXPECT_EQ(s.units[lander].cargo[0], c);
}

TEST(Actions, JoinAddsDisplayedHpAndRefundsTheExcess) {
  const MapData m = make_map({"......"});
  GameState s = make_state(m);
  const int a = put(s, m, kInfantry, 0, 0, 0, 55);  // shows 6
  const int b = put(s, m, kInfantry, 0, 1, 0, 61);  // shows 7
  s.units[a].fuel = 60;
  s.units[b].fuel = 50;
  s.players[0].funds = 1000;
  ASSERT_EQ(apply(m, s, act(kActJoin, a, m.tile(1, 0))), kOk);
  EXPECT_FALSE(s.units[a].alive());
  EXPECT_EQ(s.units[b].hp, 100);
  EXPECT_EQ(s.units[b].fuel, 99);
  EXPECT_EQ(s.players[0].funds, 1000 + 3 * 100);
  EXPECT_TRUE(s.units[b].has(kMoved));
}

TEST(Actions, JoinNeedsADamagedUnitOfTheSameType) {
  const MapData m = make_map({"......"});
  GameState s = make_state(m);
  const int a = put(s, m, kInfantry, 0, 0, 0, 50);
  put(s, m, kInfantry, 0, 1, 0);  // full HP
  put(s, m, kMech, 0, 2, 0, 50);  // another type
  EXPECT_EQ(apply(m, s, act(kActJoin, a, m.tile(1, 0))), kErrCannotJoin);
  EXPECT_EQ(apply(m, s, act(kActJoin, a, m.tile(2, 0))), kErrCannotJoin);
}

TEST(Actions, SupplyRefillsNeighboursAndTheirCargo) {
  const MapData m = make_map({"......", "......"});
  GameState s = make_state(m);
  const int apc = put(s, m, kApc, 0, 0, 0);
  const int tank = put(s, m, kTank, 0, 2, 0);
  const int copter = put(s, m, kTCopter, 0, 1, 1);
  const int inf = put(s, m, kInfantry, 0, 0, 1);
  load_unit(s, inf, copter);
  s.units[tank].ammo = 2;
  s.units[tank].fuel = 10;
  s.units[inf].fuel = 5;
  GameState lonely = s;
  ASSERT_EQ(apply(m, s, act(kActSupply, apc, m.tile(1, 0))), kOk);
  EXPECT_EQ(s.units[tank].ammo, 9);
  EXPECT_EQ(s.units[tank].fuel, 70);
  EXPECT_EQ(s.units[inf].fuel, 99);  // inside the copter next door
  const int far_apc = put(lonely, m, kApc, 0, 5, 0);
  EXPECT_EQ(apply(m, lonely, act(kActSupply, far_apc, m.tile(5, 0))), kErrCannotSupply);
}

TEST(Actions, BlackBoatRepairsOneHpForATenthOfTheCost) {
  const MapData m = make_map({"~_....", "~_...."});
  GameState s = make_state(m);
  const int boat = put(s, m, kBlackBoat, 0, 0, 0);
  const int tank = put(s, m, kTank, 0, 1, 0, 55);
  const int topped = put(s, m, kTank, 0, 1, 1, 96);
  s.players[0].funds = 1000;
  ASSERT_EQ(apply(m, s, act(kActRepair, boat, m.tile(0, 0), m.tile(1, 0))), kOk);
  EXPECT_EQ(s.units[tank].hp, 65);
  EXPECT_EQ(s.players[0].funds, 300);
  pass_round(m, s);
  s.players[0].funds = 0;
  ASSERT_EQ(apply(m, s, act(kActRepair, boat, m.tile(0, 1), m.tile(1, 1))), kOk);
  EXPECT_EQ(s.units[topped].hp, 100);  // 9.6 -> 10 shows no gain, so it's free
}

TEST(Actions, BuildChargesTheCoPriceAndTheUnitWaits) {
  const MapData m = make_map({"BA...."});
  GameState s = make_state(m, kKanbei);
  s.players[0].funds = 10000;
  ASSERT_EQ(apply(m, s, player_act(kActBuild, m.tile(0, 0), kTank)), kOk);
  EXPECT_EQ(s.players[0].funds, 10000 - 8400);  // Kanbei pays 120%
  const int tank = s.unit_at[m.tile(0, 0)];
  ASSERT_NE(tank, kNoUnit);
  EXPECT_TRUE(s.units[tank].has(kMoved));
  EXPECT_EQ(s.units[tank].ammo, 9);
  EXPECT_EQ(apply(m, s, player_act(kActBuild, m.tile(0, 0), kInfantry)), kErrOccupied);
  EXPECT_EQ(apply(m, s, player_act(kActBuild, m.tile(1, 0), kTank)), kErrCannotBuild);
  EXPECT_EQ(apply(m, s, player_act(kActBuild, m.tile(1, 0), kFighter)), kErrFunds);
}

TEST(Actions, PowersCostMeterAndGetDearer) {
  const MapData m = make_map({"......"});
  GameState s = make_state(m, kAndy);
  Player& p = s.players[0];
  p.cop_cost = cop_cost(kAndy, 0);
  p.scop_cost = scop_cost(kAndy, 0);
  EXPECT_EQ(p.cop_cost, 3 * kStarValue);
  EXPECT_EQ(apply(m, s, player_act(kActPower, 0, 1)), kErrPower);
  p.power = p.cop_cost + 5000;
  ASSERT_EQ(apply(m, s, player_act(kActPower, 0, 1)), kOk);
  EXPECT_EQ(p.power, 5000);
  EXPECT_EQ(p.power_on, 1);
  EXPECT_EQ(p.cop_cost, 3 * kStarValue * 6 / 5);
  EXPECT_EQ(p.scop_cost, 6 * kStarValue * 6 / 5);
  EXPECT_EQ(apply(m, s, player_act(kActPower, 0, 2)), kErrPower);  // already on
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  EXPECT_EQ(s.players[0].power_on, 1);  // lasts through the opponent's turn
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  EXPECT_EQ(s.players[0].power_on, 0);
  EXPECT_EQ(cop_cost(kAndy, 50), 3 * kStarValue * 3);  // capped at 10 uses
}

TEST(Actions, ColinsGoldRushMultipliesFunds) {
  const MapData m = make_map({"......"});
  GameState s = make_state(m, kColin);
  s.players[0].cop_cost = cop_cost(kColin, 0);
  s.players[0].scop_cost = scop_cost(kColin, 0);
  s.players[0].power = s.players[0].cop_cost;
  s.players[0].funds = 10000;
  ASSERT_EQ(apply(m, s, player_act(kActPower, 0, 1)), kOk);
  EXPECT_EQ(s.players[0].funds, 15000);
}

TEST(Actions, SeamsBreakIntoRubble) {
  const MapData m = make_map({"....S."});
  GameState s = make_state(m);
  const int art = put(s, m, kArtillery, 0, 2, 0);
  const int seam = m.prop[m.tile(4, 0)];
  ASSERT_EQ(apply(m, s, act(kActAttackSeam, art, m.tile(2, 0), m.tile(4, 0))), kOk);
  EXPECT_EQ(s.prop_points[seam], kFullSeamHp - 40);  // base 40 vs a Neotank, no luck
  EXPECT_EQ(s.units[art].ammo, 8);
  for (int i = 0; i < 2; ++i) {
    pass_round(m, s);
    ASSERT_EQ(apply(m, s, act(kActAttackSeam, art, m.tile(2, 0), m.tile(4, 0))), kOk);
  }
  EXPECT_EQ(s.prop_points[seam], 0);
  EXPECT_EQ(terrain_kind(m, s, m.tile(4, 0)), kRubble);
  pass_round(m, s);
  EXPECT_EQ(apply(m, s, act(kActAttackSeam, art, m.tile(2, 0), m.tile(4, 0))), kErrBadTarget);
}

TEST(Actions, HideAndUnhide) {
  const MapData m = make_map({"~~~~~~"});
  GameState s = make_state(m);
  const int sub = put(s, m, kSub, 0, 0, 0);
  ASSERT_EQ(apply(m, s, act(kActHide, sub, m.tile(1, 0))), kOk);
  EXPECT_TRUE(s.units[sub].has(kHidden));
  pass_round(m, s);
  EXPECT_EQ(apply(m, s, act(kActHide, sub, m.tile(1, 0))), kErrCannotHide);
  ASSERT_EQ(apply(m, s, act(kActUnhide, sub, m.tile(1, 0))), kOk);
  EXPECT_FALSE(s.units[sub].has(kHidden));
}

TEST(Actions, DeleteAndResign) {
  const MapData m = make_map({"......"});
  GameState s = make_state(m);
  const int a = put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 0, 1, 0);
  put(s, m, kInfantry, 1, 5, 0);
  ASSERT_EQ(apply(m, s, act(kActDelete, a, 0)), kOk);
  EXPECT_FALSE(s.units[a].alive());
  EXPECT_EQ(s.winner, kNoOwner);
  ASSERT_EQ(apply(m, s, player_act(kActResign)), kOk);
  EXPECT_EQ(s.winner, 1);
}

class TurnStartTest : public ::testing::Test {
 protected:
  // Orange Star (player 0) owns the HQ, two cities, a comtower and an airport.
  MapData m = make_map({"HCCTA.", "......", "......"});
  GameState s = make_state(m);

  void SetUp() override {
    s.day = 2;
    s.active = 1;  // Blue Moon ends its turn; Orange Star's begins
  }
};

TEST_F(TurnStartTest, IncomeComesFromPayingProperties) {
  s.players[0].funds = 500;
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  EXPECT_EQ(s.active, 0);
  EXPECT_EQ(s.day, 3);
  EXPECT_EQ(s.players[0].funds, 500 + 4 * 1000);  // HQ, cities, airport; not the tower
}

TEST_F(TurnStartTest, RepairsGoColumnByColumnAndSkipWhatTheyCantAfford) {
  const int c = put(s, m, kTank, 0, 0, 0, 50);      // 5 -> 7 costs 1400: unaffordable, skipped
  const int a = put(s, m, kInfantry, 0, 1, 0, 55);  // 6 -> 8: 200
  const int b = put(s, m, kTank, 0, 2, 0, 95);      // 10 -> 10: free
  s.units[b].ammo = 0;
  s.players[0].funds = -4000 + 200 + 700;  // after 4000 income: 900
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  EXPECT_EQ(s.units[c].hp, 50);  // no 1 HP repair for the 700 it could pay
  EXPECT_EQ(s.units[a].hp, 75);
  EXPECT_EQ(s.units[b].hp, 100);
  EXPECT_EQ(s.units[b].ammo, 9);
  EXPECT_EQ(s.players[0].funds, 700);
}

TEST(Actions, RepairsFollowColumnsNotCreationOrder) {
  // Two damaged tanks on Orange Star cities; funds for one repair. The one in
  // the left column is repaired although it was created second.
  const MapData m = make_map({"..C", "C.."});
  GameState s = make_state(m);
  const int right = put(s, m, kTank, 0, 2, 0, 50);
  const int left = put(s, m, kTank, 0, 0, 1, 50);
  s.players[0].funds = -2000 + 1400;
  s.day = 2;
  s.active = 1;
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  EXPECT_EQ(s.units[left].hp, 70);
  EXPECT_EQ(s.units[right].hp, 50);
  EXPECT_EQ(s.players[0].funds, 0);
}

TEST_F(TurnStartTest, AirUnitsBurnFuelAndCrashWhenEmpty) {
  const int fighter = put(s, m, kFighter, 0, 0, 1);
  const int copter = put(s, m, kBCopter, 0, 1, 1);
  const int parked = put(s, m, kFighter, 0, 4, 0);  // on its own airport
  const int keep = put(s, m, kInfantry, 0, 5, 2);
  s.units[fighter].fuel = 5;
  s.units[copter].fuel = 10;
  s.units[parked].fuel = 3;
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  EXPECT_FALSE(s.units[fighter].alive());
  EXPECT_EQ(s.units[copter].fuel, 8);
  EXPECT_EQ(s.units[parked].fuel, 99);
  EXPECT_TRUE(s.units[keep].alive());
}

TEST_F(TurnStartTest, NoDailyFuelOnDayOne) {
  s.day = 1;
  s.active = 0;
  const int copter = put(s, m, kBCopter, 1, 1, 1);
  s.units[copter].fuel = 10;
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);  // Blue Moon's day 1 begins
  EXPECT_EQ(s.units[copter].fuel, 10);
}

TEST_F(TurnStartTest, ApcsResupplyTheirNeighbours) {
  put(s, m, kApc, 0, 1, 2);
  const int tank = put(s, m, kTank, 0, 2, 2);
  s.units[tank].ammo = 1;
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  EXPECT_EQ(s.units[tank].ammo, 9);
}

TEST_F(TurnStartTest, MovedFlagsClearForTheNextTurn) {
  const int inf = put(s, m, kInfantry, 1, 5, 2);
  s.units[inf].flags |= kMoved;
  const int stunned = put(s, m, kInfantry, 0, 5, 1);
  s.units[stunned].flags |= kStunned;
  ASSERT_EQ(apply(m, s, player_act(kActEndTurn)), kOk);
  EXPECT_FALSE(s.units[inf].has(kMoved));
  EXPECT_TRUE(s.units[stunned].has(kMoved));  // stunned units sit this turn out
  EXPECT_FALSE(s.units[stunned].has(kStunned));
}

}  // namespace
}  // namespace awbw
