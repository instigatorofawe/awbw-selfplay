// HP beliefs: what watching every turn tells players about internal HP.

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <vector>

#include "core/actions.h"
#include "core/belief.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;
using testing::make_state;
using testing::put;

constexpr int kAndy = 1, kSonja = 18, kNell = 24, kFlak = 25;

std::vector<double> probs(const GameState& s, int slot) {
  double p[kHpBins];
  belief_probs(s, slot, p);
  return {p, p + kHpBins};
}

void set_belief(GameState& s, int slot, std::vector<int> w) {
  ASSERT_EQ(w.size(), size_t{kHpBins});
  std::copy(w.begin(), w.end(), s.hp_belief[slot]);
}

// The posterior by brute force: every pair of internal HPs the two displays
// allow, weighted by the units' beliefs, through combat_outcomes, keeping the
// outcomes that show the displayed HPs seen. Returns each unit's distribution
// over its final internal HP.
std::pair<std::map<int, double>, std::map<int, double>> brute_force(const GameState& s,
                                                                    int att_slot, int def_slot,
                                                                    Combatant att, Combatant def,
                                                                    int dist, const Outcome& seen) {
  std::map<int, double> pa, pd;
  double total = 0;
  const int att_base = hp_bin_base(att.hp), def_base = hp_bin_base(def.hp);
  std::vector<Outcome> outs;
  for (int i = 0; i < kHpBins; ++i) {
    for (int j = 0; j < kHpBins; ++j) {
      const double prior = static_cast<double>(s.hp_belief[att_slot][i]) * s.hp_belief[def_slot][j];
      if (prior == 0) continue;
      att.hp = static_cast<uint8_t>(att_base + i);
      def.hp = static_cast<uint8_t>(def_base + j);
      combat_outcomes(att, def, dist, outs);
      for (const Outcome& o : outs) {
        if (displayed_hp(o.att_hp) != displayed_hp(seen.att_hp) ||
            displayed_hp(o.def_hp) != displayed_hp(seen.def_hp))
          continue;
        pa[o.att_hp] += prior * o.prob;
        pd[o.def_hp] += prior * o.prob;
        total += prior * o.prob;
      }
    }
  }
  for (auto* m : {&pa, &pd}) {
    for (auto& [hp, p] : *m) p /= total;
  }
  return {pa, pd};
}

void expect_matches(const GameState& s, int slot, const std::map<int, double>& want) {
  const Unit& u = s.units[slot];
  const std::vector<double> got = probs(s, slot);
  for (int k = 0; k < kHpBins; ++k) {
    const int hp = hp_bin_base(u.hp) + k;
    const double w = want.count(hp) ? want.at(hp) : 0.0;
    EXPECT_NEAR(got[k], w, 0.01) << "internal HP " << hp;
    EXPECT_EQ(got[k] > 0, w > 0) << "internal HP " << hp;
  }
}

struct FightCase {
  const char* name;
  int att_type, def_type;
  int att_hp, def_hp;
  int att_co, att_power, def_co, def_power;
  int def_x;  // the attacker stands on (0, 0)
};

class FightBeliefs : public ::testing::TestWithParam<FightCase> {};

// The engine's two independent updates give the joint posterior's marginals.
TEST_P(FightBeliefs, MatchBruteForce) {
  const FightCase& c = GetParam();
  const MapData m = make_map({".F.M."});
  GameState s = make_state(m, c.att_co, c.def_co);
  s.players[0].power_on = static_cast<uint8_t>(c.att_power);
  s.players[1].power_on = static_cast<uint8_t>(c.def_power);
  const int a = put(s, m, c.att_type, 0, 0, 0, c.att_hp);
  const int d = put(s, m, c.def_type, 1, c.def_x, 0, c.def_hp);
  // Unequal priors that allow the true HPs.
  for (int k = 0; k < kHpBins; ++k) {
    s.hp_belief[a][k] = static_cast<uint8_t>(40 + 20 * k);
    s.hp_belief[d][k] = static_cast<uint8_t>(k % 3 == 0 ? 0 : 255 - 15 * k);
  }
  s.hp_belief[d][c.def_hp - hp_bin_base(c.def_hp)] = 200;
  ASSERT_EQ(check_state(m, s), "");

  const Combatant att = combatant(m, s, a, s.units[a].tile);
  const Combatant def = combatant(m, s, d, s.units[d].tile);
  const int dist = c.def_x;
  std::vector<Outcome> outs;
  ASSERT_TRUE(combat_outcomes(att, def, dist, outs));
  // Try every result the true HPs can give.
  for (const Outcome& o : outs) {
    GameState t = s;
    const auto [pa, pd] = brute_force(t, a, d, att, def, dist, o);
    Action act;
    act.kind = kActFire;
    act.unit = static_cast<uint8_t>(a);
    act.to = s.units[a].tile;
    act.target = s.units[d].tile;
    Detail det;
    det.att_hp = o.att_hp;
    det.def_hp = o.def_hp;
    ASSERT_EQ(apply(m, t, act, det), kOk);
    ASSERT_EQ(check_state(m, t), "");
    if (o.att_hp) expect_matches(t, a, pa);
    if (o.def_hp) expect_matches(t, d, pd);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Fights, FightBeliefs,
    ::testing::Values(
        FightCase{"TankOnTank", kTank, kTank, 67, 45, kAndy, 0, kAndy, 0, 1},
        FightCase{"InfantryInWood", kMech, kInfantry, 100, 83, kAndy, 0, kAndy, 0, 1},
        FightCase{"ArtilleryTakesNoCounter", kArtillery, kTank, 52, 78, kAndy, 0, kAndy, 0, 2},
        FightCase{"SonjaStrikesFirst", kTank, kMdTank, 91, 34, kAndy, 0, kSonja, 2, 1},
        FightCase{"NellsLuck", kInfantry, kInfantry, 100, 100, kNell, 2, kAndy, 0, 1},
        FightCase{"FlaksBadLuck", kTank, kRecon, 88, 57, kFlak, 2, kAndy, 0, 1}),
    [](const ::testing::TestParamInfo<FightCase>& info) { return info.param.name; });

TEST(Belief, ShiftsKeepTheirShapeUntilClamped) {
  const MapData m = make_map({"..."});
  GameState s = make_state(m);
  const int u = put(s, m, kTank, 0, 0, 0, 64);
  set_belief(s, u, {0, 0, 0, 255, 128, 0, 0, 0, 0, 0});  // 6.4, or less likely 6.5
  change_hp(s, u, 20);
  EXPECT_EQ(s.units[u].hp, 84);
  EXPECT_EQ(std::vector<int>(s.hp_belief[u], s.hp_belief[u] + kHpBins),
            (std::vector<int>{0, 0, 0, 255, 128, 0, 0, 0, 0, 0}));
  change_hp(s, u, 20);  // 10.4 and 10.5 both stop at 10.0
  EXPECT_EQ(s.units[u].hp, 100);
  EXPECT_EQ(probs(s, u)[9], 1.0);

  set_belief(s, u, std::vector<int>(kHpBins, 255));  // 9.1 to 10.0
  change_hp(s, u, -90);                              // 0.1 to 1.0, kept at 0.1 or more
  EXPECT_EQ(s.units[u].hp, 10);
  EXPECT_EQ(std::vector<int>(s.hp_belief[u], s.hp_belief[u] + kHpBins),
            std::vector<int>(kHpBins, 255));
  change_hp(s, u, -30);  // all pile up on 0.1
  EXPECT_EQ(s.units[u].hp, 1);
  EXPECT_EQ(probs(s, u)[0], 1.0);
  EXPECT_EQ(check_state(m, s), "");

  // A shift that isn't a multiple of 10 (no AWBW effect does this) splits the
  // bin; the display seen keeps only part of it.
  set_belief(s, u, std::vector<int>(kHpBins, 255));
  change_hp(s, u, 14);  // 1 -> 15: candidates 1..10 land on 15..24, 15..20 shown as 2
  EXPECT_EQ(s.units[u].hp, 15);
  EXPECT_EQ(std::vector<int>(s.hp_belief[u], s.hp_belief[u] + kHpBins),
            (std::vector<int>{0, 0, 0, 0, 255, 255, 255, 255, 255, 255}));
}

TEST(Belief, BuildsJoinsAndRepairsAreSeen) {
  const MapData m = make_map({"B..", ".b."});
  GameState s = make_state(m, kAndy, kAndy);
  s.players[0].funds = 20000;
  s.players[1].funds = 20000;
  Action build;
  build.kind = kActBuild;
  build.target = static_cast<uint16_t>(m.tile(0, 0));
  build.arg = kInfantry;
  ASSERT_EQ(apply(m, s, build), kOk);
  const int fresh = s.unit_at[m.tile(0, 0)];
  EXPECT_EQ(probs(s, fresh)[9], 1.0);

  const int a = put(s, m, kInfantry, 0, 1, 0, 34);
  const int b = put(s, m, kInfantry, 0, 2, 0, 47);
  belief_unknown(s, a);
  belief_unknown(s, b);
  Action join;
  join.kind = kActJoin;
  join.unit = static_cast<uint8_t>(a);
  join.to = static_cast<uint16_t>(m.tile(2, 0));
  ASSERT_EQ(apply(m, s, join), kOk);
  EXPECT_EQ(s.units[b].hp, 90);  // 4 + 5 displayed
  EXPECT_EQ(probs(s, b)[9], 1.0);
  EXPECT_EQ(std::vector<int>(s.hp_belief[a], s.hp_belief[a] + kHpBins),
            std::vector<int>(kHpBins, 0));  // the slot is free

  // Repair at the start of Blue Moon's turn, on its base: +2 HP, the belief
  // carried along.
  const int hurt = put(s, m, kTank, 1, 1, 1, 33);
  set_belief(s, hurt, {0, 255, 255, 0, 0, 0, 0, 0, 0, 0});  // 3.2 or 3.3
  ASSERT_EQ(apply(m, s, Action{}), kOk);
  EXPECT_EQ(s.units[hurt].hp, 53);
  EXPECT_EQ(std::vector<int>(s.hp_belief[hurt], s.hp_belief[hurt] + kHpBins),
            (std::vector<int>{0, 255, 255, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(check_state(m, s), "");
}

// Random fights, heals and power damage never leave a belief that rules out
// the unit's real HP, whatever the rounding.
TEST(Belief, NeverRulesOutTheTruth) {
  const MapData m = make_map({"..F.M"});
  const int cos[] = {kAndy, kSonja, kNell, kFlak};
  const int types[] = {kInfantry, kMech, kTank, kMdTank};
  uint64_t rng[2] = {12345, 67890};
  int fights = 0;
  for (int round = 0; round < 1000; ++round) {
    GameState s = make_state(m, cos[rng_below(rng, 4)], cos[rng_below(rng, 4)]);
    s.players[0].power_on = static_cast<uint8_t>(rng_below(rng, 3));
    s.players[1].power_on = static_cast<uint8_t>(rng_below(rng, 3));
    rng_seed(s.rng, round);
    const int a = put(s, m, types[rng_below(rng, 4)], 0, 1, 0);
    const int d = put(s, m, types[rng_below(rng, 4)], 1, 2, 0);
    for (int step = 0; step < 40 && s.units[a].alive() && s.units[d].alive(); ++step) {
      s.units[a].flags = 0;
      s.units[a].ammo = kUnits[s.units[a].type].ammo;
      s.units[d].ammo = kUnits[s.units[d].type].ammo;
      // Heals, and now and then power damage, between fights.
      if (rng_below(rng, 2) == 0) change_hp(s, d, 10 * static_cast<int>(rng_below(rng, 5)));
      if (rng_below(rng, 2) == 0) change_hp(s, a, 10 * static_cast<int>(rng_below(rng, 5)) - 10);
      Action act;
      act.kind = kActFire;
      act.unit = static_cast<uint8_t>(a);
      act.to = s.units[a].tile;
      act.target = s.units[d].tile;
      ASSERT_EQ(apply(m, s, act), kOk);
      ++fights;
      ASSERT_EQ(check_state(m, s), "") << "round " << round << " step " << step;
    }
  }
  EXPECT_GT(fights, 2000);
}

TEST(Belief, CheckStateCatchesABeliefThatRulesOutTheHp) {
  const MapData m = make_map({"..."});
  GameState s = make_state(m);
  const int u = put(s, m, kTank, 0, 0, 0, 64);
  EXPECT_EQ(check_state(m, s), "");
  set_belief(s, u, {255, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  EXPECT_NE(check_state(m, s), "");
}

}  // namespace
}  // namespace awbw
