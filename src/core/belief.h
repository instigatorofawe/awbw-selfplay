#pragma once

// HP beliefs (GameState::hp_belief): what a player who watched every turn
// knows of each unit's internal HP.
//
// AWBW shows players only displayed HP (1..10), their own units' too. Nothing
// else gives the rest away: power meters, Sasha's War Bonds and the army value
// count displayed HP, and the damage formula reads only displayed HPs and
// luck. So a watcher knows a unit's internal HP as a posterior over the 10
// values its display allows, and the posteriors of different units are
// independent. The engine keeps them up to date:
//
// - A unit built or placed by a power, or two units joined, has a known HP.
// - Heals, repairs and the damage of powers, bombs and silos move HP by an
//   amount both players see (clamped to 1..100): the posterior moves with it.
// - A strike deals damage that depends only on luck once both displayed HPs
//   are known, and both players see the result: Bayes' rule over the luck
//   rolls (the counterattack is a second strike, at the defender's new
//   displayed HP).
//
// One hint is left out: AWBW aims Rachel's and Von Bolt's missiles by internal
// HP, so where they land says a little about the HPs around them.

#include "core/combat.h"
#include "core/state.h"

namespace awbw {

inline constexpr int kBeliefMax = 255;

// Lowest internal HP with the same display as `hp` (1..100): hp_belief[slot][k]
// weighs internal HP hp_bin_base(hp) + k.
constexpr int hp_bin_base(int hp) { return 10 * (displayed_hp(hp) - 1) + 1; }

// The unit's HP is known.
void belief_exact(GameState& s, int slot);
// Nothing is known beyond the display: every HP it allows is equally likely.
void belief_unknown(GameState& s, int slot);
// Probability that the unit's internal HP is hp_bin_base(hp) + k, for k < kHpBins.
void belief_probs(const GameState& s, int slot, double (&p)[kHpBins]);
// Expected internal HP.
double belief_mean(const GameState& s, int slot);

// Move the unit's HP to clamp(hp + amount, 1, 100) in plain sight.
void change_hp(GameState& s, int slot, int amount);

// After a fight between the units in `att_slot` and `def_slot` (described by
// `att` and `def` as they were before it) ended in `o`, condition both units'
// beliefs on the displayed HPs both players saw. Call it before a destroyed
// unit's slot is freed; units' HPs need not be updated yet.
void fight_beliefs(GameState& s, int att_slot, int def_slot, const Combatant& att,
                   const Combatant& def, int dist, const Outcome& o);

}  // namespace awbw
