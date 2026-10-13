#include "core/belief.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

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

// store() from real weights.
void store(Weights& w, const double (&q)[kHpBins]) {
  const double top = *std::max_element(q, q + kHpBins);
  for (int k = 0; k < kHpBins; ++k) {
    w[k] = !(top > 0)  ? kBeliefMax
           : q[k] <= 0 ? 0
                       : static_cast<uint8_t>(
                             std::clamp<long>(std::lround(q[k] / top * kBeliefMax), 1, kBeliefMax));
  }
}

void move_hp(GameState& s, int slot, int amount) {
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

// Weights summing to 1 over the internal HPs behind a display of the unit: the
// watcher's (over its own display) if `known`, else uniform. The opponent's
// belief within a display of a hidden unit, known while they know the display.
void within(const GameState& s, int slot, bool known, double (&w)[kHpBins]) {
  if (!known) {
    std::fill(w, w + kHpBins, 1.0 / kHpBins);
    return;
  }
  const Weights& b = s.hp_belief[slot];
  int total = 0;
  for (int k = 0; k < kHpBins; ++k) total += b[k];
  for (int k = 0; k < kHpBins; ++k) w[k] = static_cast<double>(b[k]) / total;
}

void hidden_shift(GameState& s, int slot, int amount) {
  const Weights& h = s.hp_hidden[slot];
  double w[kHpBins];
  within(s, slot, hidden_known(s, slot), w);
  double q[kHpBins] = {};
  for (int d = 1; d <= kHpBins; ++d) {
    if (h[d - 1] == 0) continue;
    for (int k = 0; k < kHpBins; ++k) {
      const int hp = std::clamp(display_base(d) + k + amount, 1, 100);
      q[displayed_hp(hp) - 1] += h[d - 1] * w[k];
    }
  }
  store(s.hp_hidden[slot], q);
}

// Condition hidden unit `h_slot`'s belief on its fight with `v_slot`, as
// v's owner saw it: v's displayed HP afterwards (`v_after` internal, 0 if
// destroyed), h surviving, and the meters (`drop_ok`). `h_first` says who
// struck first; `b_counters`, whether the other strikes back if it survives.
void see_fight(GameState& s, int h_slot, int v_slot, const Combatant& hc, const Combatant& vc,
               bool h_first, bool b_counters, int v_after, const bool (&drop_ok)[kHpBins + 1]) {
  const Weights& hq = s.hp_hidden[h_slot];
  double w[kHpBins], vw[kHpBins];
  within(s, h_slot, hidden_known(s, h_slot), w);
  within(s, v_slot, true, vw);  // their own unit
  const int v_base = hp_bin_base(vc.hp);
  const int v_shown = displayed_hp(v_after);
  Luck la, lb;
  luck_distribution(h_first ? hc : vc, la);
  luck_distribution(h_first ? vc : hc, lb);
  // Weight of v ending as seen when struck for dmg[i] at luck i.
  auto v_likelihood = [&](const int* dmg, const Luck& luck) {
    double l = 0;
    for (int k = 0; k < kHpBins; ++k) {
      if (vw[k] == 0) continue;
      for (int i = 0; i < luck.n; ++i) {
        const int r = v_base + k - dmg[i];
        if (v_after == 0 ? r <= 0 : r > 0 && displayed_hp(r) == v_shown)
          l += vw[k] * luck.weight[i];
      }
    }
    return l;
  };

  double q[kHpBins] = {};
  int dmg[256];
  Combatant hd = hc;
  if (h_first) {
    for (int d = 1; d <= kHpBins; ++d) {
      if (hq[d - 1] == 0) continue;
      hd.hp = static_cast<uint8_t>(10 * d);
      luck_damages(hd, vc, false, la, dmg);
      const double pd = hq[d - 1] * v_likelihood(dmg, la);
      if (pd == 0) continue;
      if (v_after == 0 || !b_counters) {
        if (drop_ok[0]) q[d - 1] += pd;
        continue;
      }
      Combatant v2 = vc;
      v2.hp = static_cast<uint8_t>(v_after);
      luck_damages(v2, hd, true, lb, dmg);
      for (int k = 0; k < kHpBins; ++k) {
        if (w[k] == 0) continue;
        for (int j = 0; j < lb.n; ++j) {
          const int r = display_base(d) + k - dmg[j];
          if (r <= 0) continue;  // h survived
          const int e = displayed_hp(r);
          if (drop_ok[d - e]) q[e - 1] += pd * w[k] * lb.weight[j];
        }
      }
    }
  } else {
    double reply[kHpBins + 1];  // weight of v's result when h strikes back from display e
    bool have[kHpBins + 1] = {};
    int dmg2[256];
    for (int d = 1; d <= kHpBins; ++d) {
      if (hq[d - 1] == 0) continue;
      hd.hp = static_cast<uint8_t>(10 * d);
      luck_damages(vc, hd, false, la, dmg);
      for (int k = 0; k < kHpBins; ++k) {
        if (w[k] == 0) continue;
        for (int i = 0; i < la.n; ++i) {
          const int r = display_base(d) + k - dmg[i];
          if (r <= 0) continue;
          const int e = displayed_hp(r);
          if (!drop_ok[d - e]) continue;
          double l = 1;  // with no reply v is untouched
          if (b_counters) {
            if (!have[e]) {
              Combatant he = hc;
              he.hp = static_cast<uint8_t>(10 * e);
              luck_damages(he, vc, true, lb, dmg2);
              reply[e] = v_likelihood(dmg2, lb);
              have[e] = true;
            }
            l = reply[e];
          }
          q[e - 1] += hq[d - 1] * w[k] * la.weight[i] * l;
        }
      }
    }
  }
  store(s.hp_hidden[h_slot], q);
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
  if (hides_hp(s, s.units[slot].owner)) hidden_shift(s, slot, amount);
  move_hp(s, slot, amount);
}

void repair_hp(GameState& s, int slot, int amount) { move_hp(s, slot, amount); }

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

void hidden_exact(GameState& s, int slot) {
  Weights& w = s.hp_hidden[slot];
  std::memset(w, 0, sizeof w);
  w[displayed_hp(s.units[slot].hp) - 1] = kBeliefMax;
}

void hidden_unknown(GameState& s, int slot) { std::memset(s.hp_hidden[slot], kBeliefMax, kHpBins); }

bool hidden_known(const GameState& s, int slot) {
  int n = 0;
  for (int d = 0; d < kHpBins; ++d) n += s.hp_hidden[slot][d] != 0;
  return n == 1;
}

void hidden_probs(const GameState& s, int slot, double (&p)[kHpBins]) {
  const Weights& w = s.hp_hidden[slot];
  int total = 0;
  for (int d = 0; d < kHpBins; ++d) total += w[d];
  for (int d = 0; d < kHpBins; ++d) p[d] = total ? static_cast<double>(w[d]) / total : 0.0;
}

bool display_known_to(const GameState& s, int slot, int viewer) {
  const int owner = s.units[slot].owner;
  return owner == viewer || !hides_hp(s, owner) || hidden_known(s, slot);
}

double display_mean_for(const GameState& s, int slot, int viewer) {
  if (display_known_to(s, slot, viewer)) return displayed_hp(s.units[slot].hp);
  double p[kHpBins];
  hidden_probs(s, slot, p);
  double mean = 0;
  for (int d = 0; d < kHpBins; ++d) mean += (d + 1) * p[d];
  return mean;
}

double hp_mean_for(const GameState& s, int slot, int viewer) {
  if (display_known_to(s, slot, viewer)) return belief_mean(s, slot);
  double p[kHpBins];
  hidden_probs(s, slot, p);
  double mean = 0;
  for (int d = 0; d < kHpBins; ++d) mean += (display_base(d + 1) + 4.5) * p[d];
  return mean;
}

void hidden_repairs(GameState& s, const int* slots, int n, int gain, int funds, int spent) {
  // A unit's price depends only on its display (gain is a multiple of 10), so
  // each unit has a few options: the prices of its possible displays.
  constexpr int kMaxJoint = 1 << 12;
  struct Option {
    int price = 0;
    double mass = 0;      // probability of the displays with this price
    double seen[2] = {};  // weight of the totals that match, by whether it was paid
  };
  std::vector<std::array<Option, kHpBins>> opts(n);
  std::vector<int> n_opts(n), pick(n);
  std::vector<char> paid(n);
  auto healed = [&](int d) { return std::min(kHpBins, d + gain / 10); };
  auto price_of = [&](int i, int d) {
    return (healed(d) - d) * kUnits[s.units[slots[i]].type].cost / 10;
  };
  auto option_of = [&](int i, int d) {
    int c = 0;
    while (opts[i][c].price != price_of(i, d)) ++c;
    return c;
  };
  long joint = 1;
  for (int i = 0; i < n; ++i) {
    double p[kHpBins];
    hidden_probs(s, slots[i], p);
    for (int d = 1; d <= kHpBins; ++d) {
      if (p[d - 1] == 0) continue;
      const int price = price_of(i, d);
      int c = 0;
      while (c < n_opts[i] && opts[i][c].price != price) ++c;
      if (c == n_opts[i]) opts[i][n_opts[i]++].price = price;
      opts[i][c].mass += p[d - 1];
    }
    joint = std::min<long>(joint * n_opts[i], kMaxJoint + 1);
  }

  if (joint <= kMaxJoint) {
    // Every combination of prices, paid in turn while the funds last.
    for (long j = 0; j < joint; ++j) {
      long r = j;
      double weight = 1;
      int left = funds;
      for (int i = 0; i < n; ++i) {
        pick[i] = static_cast<int>(r % n_opts[i]);
        r /= n_opts[i];
        const Option& o = opts[i][pick[i]];
        weight *= o.mass;
        paid[i] = o.price <= left;
        if (paid[i]) left -= o.price;
      }
      if (funds - left != spent) continue;
      for (int i = 0; i < n; ++i) opts[i][pick[i]].seen[paid[i] ? 1 : 0] += weight;
    }
  } else {
    // Too many to try: allow paid and, where it costs anything, unpaid.
    for (int i = 0; i < n; ++i) {
      for (int c = 0; c < n_opts[i]; ++c) {
        opts[i][c].seen[1] = opts[i][c].mass;
        opts[i][c].seen[0] = opts[i][c].price > 0 ? opts[i][c].mass : 0;
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    const Weights& h = s.hp_hidden[slots[i]];
    double q[kHpBins] = {};
    for (int d = 1; d <= kHpBins; ++d) {
      if (h[d - 1] == 0) continue;
      const Option& o = opts[i][option_of(i, d)];
      q[healed(d) - 1] += h[d - 1] * o.seen[1] / o.mass;
      q[d - 1] += h[d - 1] * o.seen[0] / o.mass;
    }
    store(s.hp_hidden[slots[i]], q);
  }
}

void hidden_join(GameState& s, int mover, int into, int refund) {
  const Weights& a = s.hp_hidden[mover];
  const Weights& b = s.hp_hidden[into];
  const int cost = kUnits[s.units[into].type].cost;
  uint64_t q[kHpBins] = {};
  for (int d1 = 1; d1 <= kHpBins; ++d1) {
    for (int d2 = 1; d2 < kHpBins; ++d2) {  // a unit at 10 can't be joined
      const int shown = d1 + d2;
      if (std::max(0, shown - kHpBins) * cost / 10 != refund) continue;
      q[std::min(kHpBins, shown) - 1] += uint64_t{a[d1 - 1]} * b[d2 - 1];
    }
  }
  store(s.hp_hidden[into], q);
}

void hidden_restrict(GameState& s, int slot, const bool (&ok)[kHpBins]) {
  uint64_t q[kHpBins];
  for (int d = 0; d < kHpBins; ++d) q[d] = ok[d] ? s.hp_hidden[slot][d] : 0;
  store(s.hp_hidden[slot], q);
}

void hidden_fight(GameState& s, int att_slot, int def_slot, const Combatant& att,
                  const Combatant& def, int dist, const Outcome& o,
                  const bool (&att_drop_ok)[kHpBins + 1], const bool (&def_drop_ok)[kHpBins + 1]) {
  const bool def_first = strikes_first(def) && can_counter(def, att, dist);
  const bool b_counters = def_first || can_counter(def, att, dist);
  if (o.att_hp > 0 && hides_hp(s, s.units[att_slot].owner))
    see_fight(s, att_slot, def_slot, att, def, !def_first, b_counters, o.def_hp, att_drop_ok);
  if (o.def_hp > 0 && hides_hp(s, s.units[def_slot].owner))
    see_fight(s, def_slot, att_slot, def, att, def_first, b_counters, o.att_hp, def_drop_ok);
}

}  // namespace awbw
