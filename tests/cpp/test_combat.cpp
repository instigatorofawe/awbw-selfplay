#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "core/combat.h"

namespace awbw {
namespace {

Combatant unit(int type, int hp = 100, int co = 1, int power = 0, int terrain = kPlain,
               int stars = 1) {
  Combatant c;
  c.type = static_cast<uint8_t>(type);
  c.hp = static_cast<uint8_t>(hp);
  c.ammo = kUnits[type].ammo;
  c.co = static_cast<uint8_t>(co);
  c.power = static_cast<uint8_t>(power);
  c.terrain = static_cast<uint8_t>(terrain);
  c.stars = static_cast<uint8_t>(stars);
  return c;
}

TEST(Combat, TankVsTankOnPlains) {
  std::vector<int> got;
  for (int luck = 0; luck < 10; ++luck)
    got.push_back(damage(unit(kTank), unit(kTank), luck, false));
  EXPECT_EQ(got, (std::vector<int>{49, 50, 51, 52, 53, 54, 54, 55, 56, 57}));
}

TEST(Combat, ExactX95RoundsUp) {
  // Replay 1738979: raw (1 + 9) * 0.5 * 0.99 = 4.95 deals 5.
  const Combatant att = unit(kInfantry, 44, 5);
  const Combatant def = unit(kMdTank, 5, 16);
  EXPECT_EQ(damage(att, def, 9, false), 5);
  EXPECT_EQ(damage(att, def, 8, false), 4);
}

TEST(Combat, WeaponsAndRange) {
  EXPECT_EQ(choose_weapon(unit(kTank), kInfantry), kSecondaryWeapon);
  EXPECT_EQ(choose_weapon(unit(kTank), kTank), kPrimaryWeapon);
  Combatant dry = unit(kTank);
  dry.ammo = 0;
  EXPECT_EQ(choose_weapon(dry, kTank), kSecondaryWeapon);
  EXPECT_EQ(choose_weapon(unit(kInfantry), kBomber), kNoWeapon);
  EXPECT_FALSE(can_attack(unit(kArtillery), unit(kTank), 1));
  EXPECT_TRUE(can_attack(unit(kArtillery), unit(kTank), 3));
  EXPECT_TRUE(can_attack(unit(kArtillery, 100, 2), unit(kTank), 4));  // Grit +1
}

TEST(Combat, OutcomesSumToOne) {
  std::vector<Outcome> out;
  ASSERT_TRUE(combat_outcomes(unit(kTank), unit(kTank), 1, out));
  const double total = std::accumulate(out.begin(), out.end(), 0.0,
                                       [](double a, const Outcome& o) { return a + o.prob; });
  EXPECT_NEAR(total, 1.0, 1e-12);
  for (const Outcome& o : out) EXPECT_LT(o.att_hp, 100);  // the survivor always counters
  EXPECT_FALSE(combat_outcomes(unit(kInfantry), unit(kBomber), 1, out));
}

TEST(Combat, NoCounterFromIndirectOrDead) {
  std::vector<Outcome> out;
  ASSERT_TRUE(combat_outcomes(unit(kMegaTank), unit(kInfantry, 10), 1, out));
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].att_hp, 100);
  EXPECT_EQ(out[0].def_hp, 0);
  ASSERT_TRUE(combat_outcomes(unit(kArtillery), unit(kArtillery), 2, out));
  for (const Outcome& o : out) EXPECT_EQ(o.att_hp, 100);
}

}  // namespace
}  // namespace awbw
