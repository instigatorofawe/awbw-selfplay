// Hidden HP: what Sonja's opponent knows of her units' displayed HP.

#include <gtest/gtest.h>

#include <vector>

#include "core/actions.h"
#include "core/belief.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;
using testing::make_state;
using testing::put;

constexpr int kAndy = 1, kSonja = 18, kSasha = 19;

std::vector<double> hidden(const GameState& s, int slot) {
  double p[kHpBins];
  hidden_probs(s, slot, p);
  return {p, p + kHpBins};
}

void set_hidden(GameState& s, int slot, std::vector<int> w) {
  ASSERT_EQ(w.size(), size_t{kHpBins});
  std::copy(w.begin(), w.end(), s.hp_hidden[slot]);
}

void expect_hidden(const GameState& s, int slot, const std::vector<double>& want) {
  const std::vector<double> got = hidden(s, slot);
  for (int d = 0; d < kHpBins; ++d) {
    EXPECT_NEAR(got[d], want[d], 0.01) << "displayed HP " << d + 1;
    EXPECT_EQ(got[d] > 0, want[d] > 0) << "displayed HP " << d + 1;
  }
}

std::vector<double> normalized(std::vector<double> p) {
  double total = 0;
  for (double x : p) total += x;
  for (double& x : p) x /= total;
  return p;
}

// What v's owner sees after an action: v's displayed HP (0 once gone), whether
// h is there, and both players' power meters and funds.
struct Seen {
  int v_shown;
  bool h_alive;
  int32_t power[2], funds[2];
  bool operator==(const Seen&) const = default;
};
Seen seen(const GameState& s, int h, int v) {
  return {s.units[v].alive() ? displayed_hp(s.units[v].hp) : 0,
          s.units[h].alive(),
          {s.players[0].power, s.players[1].power},
          {s.players[0].funds, s.players[1].funds}};
}

// The opponent's posterior over hidden unit h's display after `act` ended in
// `result`, by brute force: every internal HP of h they allow (uniform within
// a display, or the watcher's belief while they know it) and of their own unit
// v (the watcher's belief), through every result combat_outcomes gives, keeping
// those that look the same to them.
std::vector<double> brute_force_fight(const MapData& m, const GameState& s, const Action& act,
                                      const Outcome& result, int h, int v) {
  GameState truth = s;
  Detail det;
  det.att_hp = result.att_hp;
  det.def_hp = result.def_hp;
  EXPECT_EQ(apply(m, truth, act, det), kOk);
  const Seen want = seen(truth, h, v);
  const bool known = hidden_known(s, h);
  int v_total = 0;
  for (int k = 0; k < kHpBins; ++k) v_total += s.hp_belief[v][k];
  std::vector<double> post(kHpBins);
  std::vector<Outcome> outs;
  for (int d = 1; d <= kHpBins; ++d) {
    if (s.hp_hidden[h][d - 1] == 0) continue;
    int h_total = 0;
    for (int k = 0; k < kHpBins; ++k) h_total += s.hp_belief[h][k];
    for (int k = 0; k < kHpBins; ++k) {
      const double wk = known ? static_cast<double>(s.hp_belief[h][k]) / h_total : 0.1;
      if (wk == 0) continue;
      for (int j = 0; j < kHpBins; ++j) {
        if (s.hp_belief[v][j] == 0) continue;
        const double prior =
            s.hp_hidden[h][d - 1] * wk * static_cast<double>(s.hp_belief[v][j]) / v_total;
        GameState c = s;
        c.units[h].hp = static_cast<uint8_t>(display_base(d) + k);
        c.units[v].hp = static_cast<uint8_t>(hp_bin_base(s.units[v].hp) + j);
        belief_exact(c, h);
        belief_exact(c, v);
        hidden_exact(c, h);
        const Combatant att = combatant(m, c, act.unit, act.to);
        const Combatant def = combatant(m, c, c.unit_at[act.target], act.target);
        combat_outcomes(att, def, m.distance(act.to, act.target), outs);
        for (const Outcome& o : outs) {
          GameState r = c;
          Detail od;
          od.att_hp = o.att_hp;
          od.def_hp = o.def_hp;
          EXPECT_EQ(apply(m, r, act, od), kOk);
          if (seen(r, h, v) == want) post[displayed_hp(r.units[h].hp) - 1] += prior * o.prob;
        }
      }
    }
  }
  return normalized(post);
}

struct FightCase {
  const char* name;
  int att_type, def_type;
  int att_co, att_power, def_co, def_power;
  bool att_meter_full, def_meter_full;
  int def_x;         // the attacker stands on (0, 0)
  bool meters_tell;  // whether the meters give every drop away
};

class HiddenFight : public ::testing::TestWithParam<FightCase> {};

TEST_P(HiddenFight, MatchesBruteForce) {
  const FightCase& c = GetParam();
  const MapData m = make_map({".F.M."});
  GameState s = make_state(m, c.att_co, c.def_co);
  const bool full[2] = {c.att_meter_full, c.def_meter_full};
  for (int p = 0; p < 2; ++p) {
    s.players[p].cop_cost = 90000;
    s.players[p].scop_cost = 180000;
    s.players[p].power = full[p] ? 180000 : 1000;
  }
  s.players[0].power_on = static_cast<uint8_t>(c.att_power);
  s.players[1].power_on = static_cast<uint8_t>(c.def_power);
  const int a = put(s, m, c.att_type, 0, 0, 0, 83);
  const int d = put(s, m, c.def_type, 1, c.def_x, 0, 57);
  // The watcher's beliefs, and the opponent's over a hidden unit's display.
  const uint8_t ba[kHpBins] = {0, 100, 255, 180, 0, 0, 0, 0, 0, 0};  // 8.2 to 8.4
  const uint8_t bd[kHpBins] = {0, 0, 0, 0, 0, 200, 255, 30, 0, 0};   // 5.6 to 5.8
  std::copy(ba, ba + kHpBins, s.hp_belief[a]);
  std::copy(bd, bd + kHpBins, s.hp_belief[d]);
  Action act;
  act.kind = kActFire;
  act.unit = static_cast<uint8_t>(a);
  act.to = s.units[a].tile;
  act.target = s.units[d].tile;
  const Combatant att = combatant(m, s, a, act.to);
  const Combatant def = combatant(m, s, d, act.target);
  std::vector<Outcome> outs;
  ASSERT_TRUE(combat_outcomes(att, def, c.def_x, outs));
  int checked = 0;
  // Displays the opponent knows (the watcher's belief within them), then
  // displays they don't.
  for (const bool known : {true, false}) {
    if (!known) {
      set_hidden(s, a, {0, 0, 0, 0, 0, 0, 60, 255, 140, 0});
      set_hidden(s, d, {0, 0, 0, 30, 200, 255, 90, 0, 0, 0});
    }
    ASSERT_EQ(check_state(m, s), "");
    for (const Outcome& o : outs) {
      GameState t = s;
      Detail det;
      det.att_hp = o.att_hp;
      det.def_hp = o.def_hp;
      ASSERT_EQ(apply(m, t, act, det), kOk);
      ASSERT_EQ(check_state(m, t), "");
      for (const auto [h, v, alive] :
           {std::tuple{a, d, o.att_hp > 0}, std::tuple{d, a, o.def_hp > 0}}) {
        if (!alive || !hides_hp(s, s.units[h].owner)) continue;
        SCOPED_TRACE(::testing::Message()
                     << (known ? "known" : "unknown") << " display, result " << int{o.att_hp} << "/"
                     << int{o.def_hp} << ", hidden unit in slot " << h);
        expect_hidden(t, h, brute_force_fight(m, s, act, o, h, v));
        // The meters show the drop, which tells the display only if the one before was known.
        if (c.meters_tell && known) EXPECT_TRUE(hidden_known(t, h));
        ++checked;
      }
    }
  }
  EXPECT_GT(checked, 0);
}

INSTANTIATE_TEST_SUITE_P(
    Fights, HiddenFight,
    ::testing::Values(
        FightCase{"MetersMove", kTank, kTank, kAndy, 0, kSonja, 0, false, false, 1, true},
        FightCase{"OneMeterMoves", kTank, kTank, kAndy, 0, kSonja, 0, true, false, 1, true},
        FightCase{"MetersFull", kTank, kTank, kAndy, 0, kSonja, 0, true, true, 1, false},
        FightCase{"SonjaAttacks", kTank, kMdTank, kSonja, 0, kAndy, 0, true, true, 1, false},
        FightCase{"CounterBreak", kTank, kTank, kAndy, 0, kSonja, 2, true, false, 1, false},
        FightCase{"ArtilleryOnSonja", kArtillery, kMdTank, kAndy, 0, kSonja, 0, true, true, 2,
                  false},
        FightCase{"SonjasArtillery", kArtillery, kTank, kSonja, 0, kAndy, 0, true, true, 2, false},
        FightCase{"InWoods", kMech, kInfantry, kAndy, 0, kSonja, 0, true, true, 1, false},
        FightCase{"Mirror", kTank, kTank, kSonja, 0, kSonja, 2, true, true, 1, false},
        FightCase{"WarBonds", kTank, kTank, kSasha, 2, kSonja, 0, false, true, 1, true}),
    [](const ::testing::TestParamInfo<FightCase>& info) { return info.param.name; });

// While the display is known, the watcher's belief within it is the opponent's.
TEST(HiddenHp, KnownDisplayUsesTheWatchersBelief) {
  const MapData m = make_map({"..."});
  GameState s = make_state(m, kAndy, kSonja);
  const int a = put(s, m, kTank, 0, 0, 0);
  const int d = put(s, m, kTank, 1, 1, 0, 64);
  const uint8_t bd[kHpBins] = {0, 0, 0, 255, 0, 0, 0, 0, 0, 255};  // 6.4 or 7.0
  std::copy(bd, bd + kHpBins, s.hp_belief[d]);
  ASSERT_TRUE(hidden_known(s, d));
  Action act;
  act.kind = kActFire;
  act.unit = static_cast<uint8_t>(a);
  act.to = s.units[a].tile;
  act.target = s.units[d].tile;
  std::vector<Outcome> outs;
  combat_outcomes(combatant(m, s, a, act.to), combatant(m, s, d, act.target), 1, outs);
  for (const Outcome& o : outs) {
    if (o.def_hp == 0) continue;
    GameState t = s;
    Detail det;
    det.att_hp = o.att_hp;
    det.def_hp = o.def_hp;
    ASSERT_EQ(apply(m, t, act, det), kOk);
    expect_hidden(t, d, brute_force_fight(m, s, act, o, d, a));
  }
}

// Repairs show only as the funds Sonja paid for all of them, and a repair she
// can't afford is skipped for the next.
TEST(HiddenHp, RepairsShowOnlyInFunds) {
  const MapData m = make_map({"cbc.", "...."});
  for (const int funds : {5000, -1300, -2500}) {  // after income: 8000, 1700, 500
    SCOPED_TRACE(funds);
    GameState s = make_state(m, kAndy, kSonja);
    s.players[1].funds = funds + 2500;
    const int t1 = put(s, m, kTank, 1, 0, 0, 85);  // pays 700 if it can
    const int t2 = put(s, m, kInfantry, 1, 1, 0, 64);
    const int t3 = put(s, m, kTank, 1, 2, 0, 100);  // free
    s.players[1].funds -= 3000;                     // income: 3 properties
    set_hidden(s, t1, {0, 0, 0, 0, 0, 0, 0, 255, 255, 100});
    set_hidden(s, t2, {0, 0, 0, 0, 0, 255, 255, 0, 255, 255});
    set_hidden(s, t3, {0, 0, 0, 0, 0, 0, 0, 0, 200, 255});
    ASSERT_EQ(check_state(m, s), "");
    s.players[1].funds += 3000;

    GameState truth = s;
    ASSERT_EQ(apply(m, truth, Action{}), kOk);  // Sonja's turn starts
    ASSERT_EQ(check_state(m, truth), "");

    // Brute force: every combination of displays, through the engine.
    const int slots[3] = {t1, t2, t3};
    std::vector<std::vector<double>> post(3, std::vector<double>(kHpBins));
    for (int d1 = 1; d1 <= kHpBins; ++d1) {
      for (int d2 = 1; d2 <= kHpBins; ++d2) {
        for (int d3 = 1; d3 <= kHpBins; ++d3) {
          const int ds[3] = {d1, d2, d3};
          double prior = 1;
          GameState c = s;
          for (int i = 0; i < 3; ++i) {
            prior *= s.hp_hidden[slots[i]][ds[i] - 1];
            c.units[slots[i]].hp = static_cast<uint8_t>(10 * ds[i]);
          }
          if (prior == 0) continue;
          ASSERT_EQ(apply(m, c, Action{}), kOk);
          if (c.players[1].funds != truth.players[1].funds) continue;
          for (int i = 0; i < 3; ++i) post[i][displayed_hp(c.units[slots[i]].hp) - 1] += prior;
        }
      }
    }
    for (int i = 0; i < 3; ++i) {
      SCOPED_TRACE(i);
      expect_hidden(truth, slots[i], normalized(post[i]));
    }
  }
}

TEST(HiddenHp, BlackBoatRepairsShowInFunds) {
  const MapData m = make_map({"~~~"});
  GameState s = make_state(m, kAndy, kSonja);
  s.active = 1;
  s.players[1].funds = 5000;
  const int boat = put(s, m, kBlackBoat, 1, 0, 0);
  const int lander = put(s, m, kLander, 1, 1, 0, 95);
  set_hidden(s, lander, {0, 0, 0, 0, 0, 0, 0, 0, 255, 255});
  Action act;
  act.kind = kActRepair;
  act.unit = static_cast<uint8_t>(boat);
  act.to = s.units[boat].tile;
  act.target = s.units[lander].tile;
  ASSERT_EQ(apply(m, s, act), kOk);
  EXPECT_EQ(s.players[1].funds, 5000);  // topping up 9.5 is free
  expect_hidden(s, lander, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
}

TEST(HiddenHp, JoinsAddDisplays) {
  const MapData m = make_map({"..."});
  for (const auto [hp_a, hp_b] : {std::pair{34, 47}, std::pair{64, 57}}) {
    GameState s = make_state(m, kAndy, kSonja);
    s.active = 1;
    const int a = put(s, m, kInfantry, 1, 0, 0, hp_a);
    const int b = put(s, m, kInfantry, 1, 1, 0, hp_b);
    set_hidden(s, a, {0, 0, 255, 255, 0, 0, 255, 0, 0, 0});
    set_hidden(s, b, {0, 0, 0, 0, 255, 128, 0, 0, 0, 255});
    ASSERT_EQ(check_state(m, s), "");
    Action join;
    join.kind = kActJoin;
    join.unit = static_cast<uint8_t>(a);
    join.to = s.units[b].tile;
    GameState truth = s;
    ASSERT_EQ(apply(m, truth, join), kOk);
    std::vector<double> post(kHpBins);
    for (int d1 = 1; d1 <= kHpBins; ++d1) {
      for (int d2 = 1; d2 < kHpBins; ++d2) {  // a 10 can't be joined
        const double prior = static_cast<double>(s.hp_hidden[a][d1 - 1]) * s.hp_hidden[b][d2 - 1];
        if (prior == 0) continue;
        GameState c = s;
        c.units[a].hp = static_cast<uint8_t>(10 * d1);
        c.units[b].hp = static_cast<uint8_t>(10 * d2);
        ASSERT_EQ(apply(m, c, join), kOk);
        if (c.players[1].funds == truth.players[1].funds)
          post[displayed_hp(c.units[b].hp) - 1] += prior;
      }
    }
    expect_hidden(truth, b, normalized(post));
    EXPECT_EQ(check_state(m, truth), "");
  }
}

TEST(HiddenHp, CapturesTakeTheDisplayOffThePoints) {
  const MapData m = make_map({"#.#"});
  GameState s = make_state(m, kAndy, kSonja);
  s.active = 1;
  const int a = put(s, m, kInfantry, 1, 0, 0, 47);
  set_hidden(s, a, {0, 0, 255, 255, 255, 255, 255, 0, 0, 0});
  Action cap;
  cap.kind = kActCapture;
  cap.unit = static_cast<uint8_t>(a);
  cap.to = s.units[a].tile;
  ASSERT_EQ(apply(m, s, cap), kOk);  // 20 -> 15
  expect_hidden(s, a, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0});

  // Finishing a capture says only that the display was enough.
  const int b = put(s, m, kInfantry, 1, 2, 0, 66);
  set_hidden(s, b, {0, 255, 255, 255, 255, 255, 255, 0, 0, 0});
  s.prop_points[m.prop[s.units[b].tile]] = 5;
  cap.unit = static_cast<uint8_t>(b);
  cap.to = s.units[b].tile;
  ASSERT_EQ(apply(m, s, cap), kOk);
  expect_hidden(s, b, {0, 0, 0, 0, 1 / 3.0, 1 / 3.0, 1 / 3.0, 0, 0, 0});
  EXPECT_EQ(check_state(m, s), "");
}

TEST(HiddenHp, PowerDamageShiftsInPlainSight) {
  const MapData m = make_map({"..."});
  GameState s = make_state(m, kAndy, kSonja);
  const int u = put(s, m, kTank, 1, 0, 0, 24);
  set_hidden(s, u, {255, 128, 64, 0, 255, 0, 0, 0, 0, 0});
  change_hp(s, u, -20);  // 1, 2 and 3 all end on 1
  EXPECT_EQ(s.units[u].hp, 4);
  expect_hidden(s, u, normalized({255 + 128 + 64, 0, 255, 0, 0, 0, 0, 0, 0, 0}));
  // Units of a player who doesn't hide HP keep no hidden belief.
  const int v = put(s, m, kTank, 0, 1, 0, 24);
  std::fill(s.hp_hidden[v], s.hp_hidden[v] + kHpBins, 0);
  change_hp(s, v, -20);
  EXPECT_EQ(check_state(m, s), "");
}

// Random fights, with meters that move or not, heals and power damage never
// leave a hidden belief that rules out the unit's display.
TEST(HiddenHp, NeverRulesOutTheTruth) {
  const MapData m = make_map({"..F.M"});
  const int types[] = {kInfantry, kMech, kTank, kMdTank};
  uint64_t rng[2] = {54321, 9876};
  int fights = 0, unknown = 0;
  for (int round = 0; round < 1000; ++round) {
    GameState s = make_state(m, rng_below(rng, 2) ? kSonja : kAndy, kSonja);
    rng_seed(s.rng, round);
    const int a = put(s, m, types[rng_below(rng, 4)], 0, 1, 0);
    const int d = put(s, m, types[rng_below(rng, 4)], 1, 2, 0);
    for (int step = 0; step < 40 && s.units[a].alive() && s.units[d].alive(); ++step) {
      for (Player& p : s.players) {
        p.cop_cost = 90000;
        p.scop_cost = 180000;
        p.power = rng_below(rng, 3) ? 180000 : 0;
        p.power_on = static_cast<uint8_t>(rng_below(rng, 4) == 0 ? 2 : 0);
      }
      s.units[a].flags = 0;
      s.units[a].ammo = kUnits[s.units[a].type].ammo;
      s.units[d].ammo = kUnits[s.units[d].type].ammo;
      if (rng_below(rng, 3) == 0) change_hp(s, d, 10 * static_cast<int>(rng_below(rng, 5)) - 20);
      Action act;
      act.kind = kActFire;
      act.unit = static_cast<uint8_t>(a);
      act.to = s.units[a].tile;
      act.target = s.units[d].tile;
      ASSERT_EQ(apply(m, s, act), kOk);
      ++fights;
      ASSERT_EQ(check_state(m, s), "") << "round " << round << " step " << step;
      unknown += s.units[d].alive() && !hidden_known(s, d);
    }
  }
  EXPECT_GT(fights, 2000);
  EXPECT_GT(unknown, 500);
}

TEST(HiddenHp, CheckStateCatchesABeliefThatRulesOutTheDisplay) {
  const MapData m = make_map({"..."});
  GameState s = make_state(m, kAndy, kSonja);
  const int u = put(s, m, kTank, 1, 0, 0, 64);
  EXPECT_EQ(check_state(m, s), "");
  set_hidden(s, u, {255, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  EXPECT_NE(check_state(m, s), "");
}

}  // namespace
}  // namespace awbw
