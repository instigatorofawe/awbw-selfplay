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
//
// Hidden HP (GameState::hp_hidden). Sonja's units show no HP at all to her
// opponent (and her army value shows as 0), so the opponent knows each one's
// displayed HP as a posterior too. It narrows on what they do see:
//
// - Builds and joins: a joined unit's display is the sum of two (at most 10),
//   the unit joined must have been below 10, and funds show any refund.
// - Plain-sight HP changes (power damage and heals, bombs, silos) move it.
// - Repairs show only in her funds, as the total paid for all of them.
// - Captures take the capturer's displayed HP off the property's points.
// - In a fight the opponent sees their own unit's displayed HP and which
//   units survived, and both power meters (and War Bonds) move by the
//   displayed HP each side lost, unless a power is on or a meter is full.
//
// Within a display the opponent's belief over internal HP is taken as the
// watcher's (hp_belief) while they know the display and as uniform while they
// don't, and their own units' internal HP as the watcher's. So it is exact
// for a fight from known displays, which covers most of them (the meters give
// the drop away), but not quite after a stretch they lost track.

#include "core/combat.h"
#include "core/state.h"

namespace awbw {

inline constexpr int kBeliefMax = 255;

// Lowest internal HP showing displayed HP d (1..10).
constexpr int display_base(int d) { return 10 * (d - 1) + 1; }
// Lowest internal HP with the same display as `hp` (1..100): hp_belief[slot][k]
// weighs internal HP hp_bin_base(hp) + k.
constexpr int hp_bin_base(int hp) { return display_base(displayed_hp(hp)); }

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

// --- Hidden HP ---------------------------------------------------------------

inline constexpr int kCoSonja = 18;

// Whether `player`'s units show no HP to the other player.
inline bool hides_hp(const GameState& s, int player) { return s.players[player].co == kCoSonja; }

// The opponent knows the unit's displayed HP.
void hidden_exact(GameState& s, int slot);
// Any display is equally likely.
void hidden_unknown(GameState& s, int slot);
// Whether hp_hidden allows a single display.
bool hidden_known(const GameState& s, int slot);
// Probability that the displayed HP is d + 1, for d < kHpBins.
void hidden_probs(const GameState& s, int slot, double (&p)[kHpBins]);

// Whether `viewer` knows the unit's displayed HP.
bool display_known_to(const GameState& s, int slot, int viewer);
// Expected displayed HP and internal HP, as `viewer` knows them.
double display_mean_for(const GameState& s, int slot, int viewer);
double hp_mean_for(const GameState& s, int slot, int viewer);

// change_hp, but only the unit's owner sees it: a repair, which Sonja's
// opponent knows of only through her funds (hidden_repairs).
void repair_hp(GameState& s, int slot, int amount);

// After repairs that added `gain` (a multiple of 10) internal HP to each of the
// `n` units in `slots` in turn, each one only if its price (per displayed HP
// gained) was within the funds left, starting from `funds` and spending
// `spent`: condition the hidden beliefs of the units (all of one player, who
// hides HP) on the total. Call it before or after their HPs change.
void hidden_repairs(GameState& s, const int* slots, int n, int gain, int funds, int spent);

// Before `mover` joins `into` (both hiding HP), which refunded `refund`:
// `into` gets the display of the sum.
void hidden_join(GameState& s, int mover, int into, int refund);

// Rule out displays: the opponent saw something only `ok[d - 1]` displays d allow.
void hidden_restrict(GameState& s, int slot, const bool (&ok)[kHpBins]);

// Before fight_beliefs (same arguments): condition the hidden belief of each
// unit whose owner hides HP and that survived on what the other player saw.
// `att_drop_ok[k]` (`def_drop_ok[k]`) says whether the attacker (defender)
// losing k displayed HP agrees with the power meters and funds seen afterwards,
// given what the other unit lost.
void hidden_fight(GameState& s, int att_slot, int def_slot, const Combatant& att,
                  const Combatant& def, int dist, const Outcome& o,
                  const bool (&att_drop_ok)[kHpBins + 1], const bool (&def_drop_ok)[kHpBins + 1]);

}  // namespace awbw
