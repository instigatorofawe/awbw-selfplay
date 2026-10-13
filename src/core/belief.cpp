#include "core/belief.h"

#include <algorithm>
#include <cstring>

namespace awbw {

namespace {

using Weights = uint8_t[kHpBins];

// Scale unnormalized weights so the largest is kBeliefMax, keeping every
// possible HP at least 1. If nothing is possible (what was seen contradicts
// the belief, which the engine never does) nothing is known.
void store(Weights& w, const uint64_t (&q)[kHpBins]) {
  const uint64_t top = *std::max_element(q, q + kHpBins);
  for (int k = 0; k < kHpBins; ++k) {
    w[k] = top == 0 ? kBeliefMax
           : q[k] == 0
               ? 0
               : static_cast<uint8_t>(std::max<uint64_t>(1, (q[k] * kBeliefMax + top / 2) / top));
  }
}

// Condition `w` (over the bin of `before`) on a strike by `att` on `def` that
// left the struck unit at `after` > 0. def.hp is `before`.
void condition(Weights& w, int before, int after, const Combatant& att, const Combatant& def,
               bool counter) {
  Luck luck;
  luck_distribution(att, luck);
  int dmg[256];
  luck_damages(att, def, counter, luck, dmg);
  const int shift = hp_bin_base(before) - hp_bin_base(after);
  uint64_t q[kHpBins] = {};
  for (int i = 0; i < luck.n; ++i) {
    // Candidate HP base(before) + k ends at base(after) + k + shift - dmg.
    const int off = shift - dmg[i];
    for (int k = std::max(0, -off); k < std::min(kHpBins, kHpBins - off); ++k)
      q[k + off] += uint64_t{w[k]} * luck.weight[i];
  }
  store(w, q);
}

}  // namespace

void belief_exact(GameState& s, int slot) {
  Weights& w = s.hp_belief[slot];
  std::memset(w, 0, sizeof w);
  w[s.units[slot].hp - hp_bin_base(s.units[slot].hp)] = kBeliefMax;
}

void belief_unknown(GameState& s, int slot) { std::memset(s.hp_belief[slot], kBeliefMax, kHpBins); }

void belief_probs(const GameState& s, int slot, double (&p)[kHpBins]) {
  const Weights& w = s.hp_belief[slot];
  int total = 0;
  for (int k = 0; k < kHpBins; ++k) total += w[k];
  for (int k = 0; k < kHpBins; ++k) p[k] = total ? static_cast<double>(w[k]) / total : 0.0;
}

double belief_mean(const GameState& s, int slot) {
  double p[kHpBins];
  belief_probs(s, slot, p);
  double mean = hp_bin_base(s.units[slot].hp);
  for (int k = 0; k < kHpBins; ++k) mean += k * p[k];
  return mean;
}

void change_hp(GameState& s, int slot, int amount) {
  Unit& u = s.units[slot];
  const int before = u.hp, after = std::clamp(before + amount, 1, 100);
  u.hp = static_cast<uint8_t>(after);
  const int from = hp_bin_base(before), to = hp_bin_base(after);
  if (from + amount == to) return;  // every candidate moves by `amount`, unclamped
  // Clamping piles candidates onto 1 or 100; an amount that isn't a multiple of
  // 10 spills some into another display, which both players see isn't theirs.
  Weights& w = s.hp_belief[slot];
  uint64_t q[kHpBins] = {};
  for (int k = 0; k < kHpBins; ++k) {
    const int h = std::clamp(from + k + amount, 1, 100) - to;
    if (h >= 0 && h < kHpBins) q[h] += w[k];
  }
  store(w, q);
}

void fight_beliefs(GameState& s, int att_slot, int def_slot, const Combatant& att,
                   const Combatant& def, int dist, const Outcome& o) {
  // The strikes in combat_outcomes' order: `a` first, then `b` back if able and alive.
  const bool def_first = strikes_first(def) && can_counter(def, att, dist);
  const Combatant& a = def_first ? def : att;
  const Combatant& b = def_first ? att : def;
  const int a_slot = def_first ? def_slot : att_slot;
  const int b_slot = def_first ? att_slot : def_slot;
  const int a_after = def_first ? o.def_hp : o.att_hp;
  const int b_after = def_first ? o.att_hp : o.def_hp;
  if (b_after == 0) return;
  condition(s.hp_belief[b_slot], b.hp, b_after, a, b, false);
  if (a_after == 0 || !(def_first || can_counter(b, a, dist))) return;
  Combatant b2 = b;
  b2.hp = static_cast<uint8_t>(b_after);
  condition(s.hp_belief[a_slot], a.hp, a_after, b2, a, true);
}

}  // namespace awbw
