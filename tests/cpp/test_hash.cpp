#include <gtest/gtest.h>

#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "core/actions.h"
#include "core/belief.h"
#include "core/hash.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::make_map;
using testing::make_state;
using testing::put;

class HashTest : public ::testing::Test {
 protected:
  MapData m = make_map({
      "B.B...",
      "......",
      "....#.",
      "~~~~~~",
  });
  GameState s = make_state(m);

  Action wait(int slot, int x, int y) const {
    Action a;
    a.kind = kActWait;
    a.unit = static_cast<uint8_t>(slot);
    a.to = static_cast<uint16_t>(m.tile(x, y));
    return a;
  }
  Action build(int x, int y, int type) const {
    Action a;
    a.kind = kActBuild;
    a.target = static_cast<uint16_t>(m.tile(x, y));
    a.arg = static_cast<uint8_t>(type);
    return a;
  }
};

TEST_F(HashTest, MovesInEitherOrderGiveOneKey) {
  const int inf = put(s, m, kInfantry, 0, 1, 1);
  const int tank = put(s, m, kTank, 0, 3, 1);
  put(s, m, kTank, 1, 5, 2);
  GameState a = s, b = s;
  ASSERT_EQ(apply(m, a, wait(inf, 1, 2)), kOk);
  ASSERT_EQ(apply(m, a, wait(tank, 3, 2)), kOk);
  ASSERT_EQ(apply(m, b, wait(tank, 3, 2)), kOk);
  ASSERT_EQ(apply(m, b, wait(inf, 1, 2)), kOk);
  EXPECT_EQ(state_hash(a), state_hash(b));
  EXPECT_NE(state_hash(a), state_hash(s));

  GameState c = s;  // the same units, but not the same places
  ASSERT_EQ(apply(m, c, wait(inf, 1, 2)), kOk);
  ASSERT_EQ(apply(m, c, wait(tank, 4, 1)), kOk);
  EXPECT_NE(state_hash(a), state_hash(c));
}

TEST_F(HashTest, BuildsInEitherOrderGiveOneKey) {
  s.players[0].funds = 20000;
  GameState a = s, b = s;
  ASSERT_EQ(apply(m, a, build(0, 0, kInfantry)), kOk);
  ASSERT_EQ(apply(m, a, build(2, 0, kTank)), kOk);
  ASSERT_EQ(apply(m, b, build(2, 0, kTank)), kOk);
  ASSERT_EQ(apply(m, b, build(0, 0, kInfantry)), kOk);
  // Different slots and ids, so different bytes: the key doesn't see them.
  ASSERT_NE(std::memcmp(&a, &b, sizeof a), 0);
  EXPECT_EQ(state_hash(a), state_hash(b));

  GameState c = s;  // the same units on each other's bases
  ASSERT_EQ(apply(m, c, build(2, 0, kInfantry)), kOk);
  ASSERT_EQ(apply(m, c, build(0, 0, kTank)), kOk);
  EXPECT_NE(state_hash(a), state_hash(c));
}

TEST_F(HashTest, IgnoresRngIdsAndFreeSlots) {
  const int inf = put(s, m, kInfantry, 0, 1, 1);
  const int tank = put(s, m, kTank, 1, 3, 1);
  const uint64_t key = state_hash(s);
  GameState t = s;
  t.rng[0] ^= 12345;
  t.next_id += 7;
  t.units[inf].id += 100;
  t.units[tank].id = 3;
  EXPECT_EQ(state_hash(t), key);

  // A unit that came and went leaves its slot free: the state plays the same.
  GameState u = s;
  const int gone = put(u, m, kMech, 0, 4, 1);
  remove_unit(u, gone);
  EXPECT_EQ(state_hash(u), key);
}

TEST_F(HashTest, SlotsDontCount) {
  GameState t = make_state(m);
  put(s, m, kInfantry, 0, 1, 1);
  put(s, m, kTank, 1, 3, 1, 57);
  put(s, m, kLander, 0, 2, 3);
  put(t, m, kLander, 0, 2, 3);
  put(t, m, kTank, 1, 3, 1, 57);
  put(t, m, kInfantry, 0, 1, 1);
  EXPECT_EQ(state_hash(s), state_hash(t));
}

TEST_F(HashTest, CargoCountsBySlot) {
  const int lander = put(s, m, kLander, 0, 1, 3);
  const int inf = put(s, m, kInfantry, 0, 1, 2);
  const int mech = put(s, m, kMech, 0, 2, 2);
  GameState a = s, b = s;
  load_unit(a, inf, lander);
  load_unit(a, mech, lander);
  load_unit(b, mech, lander);
  load_unit(b, inf, lander);
  EXPECT_NE(state_hash(a), state_hash(b));  // Unload1 and Unload2 drop different units

  // The mech in the second slot (the first freed, as AWBW leaves it) or the
  // first: its unload is a different id.
  GameState c = a, d = s;
  unload_unit(c, inf, m.tile(1, 2));
  load_unit(d, mech, lander);
  EXPECT_NE(state_hash(c), state_hash(d));
  EXPECT_NE(state_hash(c), state_hash(s));
}

TEST_F(HashTest, EveryRuleInputCounts) {
  const int inf = put(s, m, kInfantry, 0, 1, 1);
  put(s, m, kTank, 1, 3, 1);
  const uint64_t key = state_hash(s);
  const std::vector<std::pair<std::string, std::function<void(GameState&)>>> edits = {
      {"hp", [&](GameState& t) { t.units[inf].hp = 95; }},
      {"fuel", [&](GameState& t) { --t.units[inf].fuel; }},
      {"ammo", [&](GameState& t) { t.units[inf].ammo = 1; }},
      {"flags", [&](GameState& t) { t.units[inf].flags |= kMoved; }},
      {"type", [&](GameState& t) { t.units[inf].type = kMech; }},
      {"owner", [&](GameState& t) { t.units[inf].owner = 1; }},
      {"tile", [&](GameState& t) { relocate_unit(t, inf, m.tile(0, 1)); }},
      {"belief", [&](GameState& t) { belief_unknown(t, inf); }},
      {"funds", [](GameState& t) { t.players[0].funds += 1000; }},
      {"power", [](GameState& t) { t.players[1].power += 1; }},
      {"power_on", [](GameState& t) { t.players[0].power_on = 1; }},
      {"power_uses", [](GameState& t) { t.players[1].power_uses = 1; }},
      {"co", [](GameState& t) { t.players[1].co = 2; }},
      {"day", [](GameState& t) { ++t.day; }},
      {"active", [](GameState& t) { t.active = 1; }},
      {"weather", [](GameState& t) { t.weather = 1; }},
      {"winner", [](GameState& t) { t.winner = 0; }},
      {"prop_owner", [&](GameState& t) { t.prop_owner[m.n_props - 1] = 0; }},
      {"prop_points", [&](GameState& t) { t.prop_points[m.n_props - 1] = 10; }},
      {"day_limit", [](GameState& t) { t.settings.day_limit = 30; }},
  };
  for (const auto& [name, edit] : edits) {
    GameState t = s;
    edit(t);
    EXPECT_NE(state_hash(t), key) << name;
  }
}

TEST_F(HashTest, HiddenHpCountsForSonjaOnly) {
  GameState sonja = make_state(m, 1, kCoSonja);
  const int a = put(s, m, kTank, 1, 3, 1);
  const int b = put(sonja, m, kTank, 1, 3, 1);
  GameState t = s, u = sonja;
  hidden_unknown(t, a);
  hidden_unknown(u, b);
  EXPECT_EQ(state_hash(t), state_hash(s));  // unused for other COs' units
  EXPECT_NE(state_hash(u), state_hash(sonja));
}

TEST_F(HashTest, SelectionsByTileNotSlot) {
  const int inf = put(s, m, kInfantry, 0, 1, 1);
  EXPECT_EQ(state_hash(s, Selection{}), state_hash(s));
  const Selection picked{kPhaseSelected, static_cast<uint8_t>(inf),
                         static_cast<uint16_t>(m.tile(1, 1))};
  Selection moved = picked;
  moved.phase = kPhaseMoved;
  moved.tile = static_cast<uint16_t>(m.tile(1, 2));
  EXPECT_NE(state_hash(s, picked), state_hash(s));
  EXPECT_NE(state_hash(s, picked), state_hash(s, moved));

  // The same infantry in another slot, picked: the same decision.
  GameState t = make_state(m);
  put(t, m, kMech, 1, 5, 2);
  const int other = put(t, m, kInfantry, 0, 1, 1);
  put(s, m, kMech, 1, 5, 2);
  Selection again = picked;
  again.unit = static_cast<uint8_t>(other);
  ASSERT_NE(other, inf);
  EXPECT_EQ(state_hash(t, again), state_hash(s, picked));

  const Selection factory{kPhaseFactory, kNoUnit, static_cast<uint16_t>(m.tile(0, 0))};
  Selection other_factory = factory;
  other_factory.tile = static_cast<uint16_t>(m.tile(2, 0));
  EXPECT_NE(state_hash(s, factory), state_hash(s, other_factory));
}

}  // namespace
}  // namespace awbw
