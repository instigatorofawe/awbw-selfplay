#include "core/combat.h"

#include <algorithm>

namespace awbw {

namespace {

const CoStateSpec& co_state(const Combatant& c) { return kCos[c.co].state[c.power]; }

bool is_air(int type) {
  const int cls = kUnits[type].cls;
  return cls == kCopter || cls == kPlane;
}

// Distribution of net luck (good roll minus bad roll) for a CO state: value
// lo + i has probability weight[i] / total.
struct Luck {
  int lo = 0;
  int n = 0;
  int total = 0;
  int weight[256] = {};
};

void luck_distribution(const Combatant& c, Luck& luck) {
  const CoStateSpec& s = co_state(c);
  const int good = s.luck_good, bad = s.luck_bad;
  luck.lo = -bad;
  luck.n = good + bad + 1;
  luck.total = (good + 1) * (bad + 1);
  std::fill(luck.weight, luck.weight + luck.n, 0);
  for (int g = 0; g <= good; ++g) {
    for (int b = 0; b <= bad; ++b) ++luck.weight[g - b + bad];
  }
}

bool can_find(int att_type, const Combatant& def) {
  if (!def.hidden) return true;
  if (def.type == kStealth) return att_type == kFighter || att_type == kStealth;
  if (def.type == kSub) return att_type == kSub || att_type == kCruiser;
  return true;
}

int roll_luck(const Combatant& c, uint64_t (&rng)[2]) {
  const CoStateSpec& s = co_state(c);
  return rng_below(rng, s.luck_good + 1) - rng_below(rng, s.luck_bad + 1);
}

}  // namespace

// Only direct units counterattack, and only against adjacent attackers.
bool can_counter(const Combatant& def, const Combatant& att, int dist) {
  return kUnits[def.type].max_range == 1 && can_attack(def, att, dist);
}

// Sonja's Counter Break: a unit that could counter strikes first instead.
bool strikes_first(const Combatant& def) { return def.co == 18 && def.power == 2; }

Weapon choose_weapon(const Combatant& att, int def_type) {
  if (kPrimaryDamage[att.type][def_type] != kNoDamage && att.ammo > 0) return kPrimaryWeapon;
  if (kSecondaryDamage[att.type][def_type] != kNoDamage) return kSecondaryWeapon;
  return kNoWeapon;
}

int max_range(const Combatant& c) {
  const UnitSpec& u = kUnits[c.type];
  if (u.max_range <= 1) return u.max_range;
  return std::max<int>(u.min_range, u.max_range + co_state(c).range[c.type]);
}

bool can_attack(const Combatant& att, const Combatant& def, int dist) {
  const UnitSpec& u = kUnits[att.type];
  return dist >= u.min_range && dist <= max_range(att) &&
         choose_weapon(att, def.type) != kNoWeapon && can_find(att.type, def);
}

int defense_stars(const Combatant& c) {
  if (is_air(c.type)) return 0;
  return c.stars * co_state(c).star_mult;
}

int attack_value(const Combatant& att, const Combatant& def) {
  (void)def;
  const CoStateSpec& s = co_state(att);
  int v =
      100 + s.atk[att.type] + s.terrain_atk[att.terrain] + att.towers * s.tower_atk + att.extra_atk;
  if (att.power) v += kPowerBonus;
  if (s.atk_per_star) v += s.atk_per_star * defense_stars(att);
  return v;
}

int defense_value(const Combatant& def, const Combatant& att) {
  const CoStateSpec& s = co_state(def);
  int v = 100 + s.def[def.type] + def.towers * s.tower_def;
  if (def.power) v += kPowerBonus;
  if (kUnits[att.type].max_range > 1) v += s.def_vs_indirect;
  return v;
}

int damage(const Combatant& att, const Combatant& def, int luck, bool counter) {
  const Weapon w = choose_weapon(att, def.type);
  if (w == kNoWeapon) return 0;
  const int base = (w == kPrimaryWeapon ? kPrimaryDamage : kSecondaryDamage)[att.type][def.type];
  // awbw.fandom.com/wiki/Damage_Formula, with displayed HPs:
  //   raw = (B * ATK / 100 + luck) * (attacker HP / 10) * (200 - (DEF + stars * defender HP)) / 100
  // AWBW rounds raw up to a multiple of 0.05 and then down to an integer.
  // Replays also show an exact x.95 rounding up (floating point in AWBW), so
  // the result is floor(raw + 0.05). In integers raw = n / 100000.
  // A counterattack bonus (Sonja, Kanbei's SCOP) adds to the attack value; it
  // doesn't scale luck (checked against internal HPs in replays).
  const int atk = attack_value(att, def) + (counter ? co_state(att).counter - 100 : 0);
  const int64_t offense = int64_t{base} * atk + 100 * int64_t{luck};
  const int64_t defense = 200 - defense_value(def, att) - defense_stars(def) * displayed_hp(def.hp);
  if (offense <= 0 || defense <= 0) return 0;
  const int64_t n = offense * displayed_hp(att.hp) * defense;
  return static_cast<int>(std::min<int64_t>((n + 5000) / 100000, 255));
}

bool combat_outcomes(const Combatant& att, const Combatant& def, int dist,
                     std::vector<Outcome>& out) {
  out.clear();
  if (!can_attack(att, def, dist)) return false;
  const bool first_is_def = strikes_first(def) && can_counter(def, att, dist);
  const Combatant& a = first_is_def ? def : att;  // strikes first
  const Combatant& b = first_is_def ? att : def;  // strikes back, if able and alive

  Luck la, lb;
  luck_distribution(a, la);
  luck_distribution(b, lb);
  const bool b_counters = first_is_def || can_counter(b, a, dist);

  // Counter Break's pre-emptive strike does not get counterattack multipliers
  // (checked against replays), so only a normal counter is a "counter".
  const bool a_counter = false;
  const bool b_counter = !first_is_def;

  // Damage is monotone in luck, so equal results from the first strike are
  // contiguous: group them, then expand each group by the reply's luck.
  int i = 0;
  while (i < la.n) {
    const int b_hp = std::max(0, b.hp - damage(a, b, la.lo + i, a_counter));
    int weight = 0;
    int j = i;
    while (j < la.n && std::max(0, b.hp - damage(a, b, la.lo + j, a_counter)) == b_hp)
      weight += la.weight[j++];
    const double p = static_cast<double>(weight) / la.total;
    i = j;
    if (b_hp == 0 || !b_counters) {
      out.push_back(first_is_def ? Outcome{static_cast<uint8_t>(b_hp), a.hp, p}
                                 : Outcome{a.hp, static_cast<uint8_t>(b_hp), p});
      continue;
    }
    Combatant b2 = b;
    b2.hp = static_cast<uint8_t>(b_hp);
    int k = 0;
    while (k < lb.n) {
      const int a_hp = std::max(0, a.hp - damage(b2, a, lb.lo + k, b_counter));
      int w2 = 0;
      int m = k;
      while (m < lb.n && std::max(0, a.hp - damage(b2, a, lb.lo + m, b_counter)) == a_hp)
        w2 += lb.weight[m++];
      k = m;
      const double q = p * w2 / lb.total;
      out.push_back(first_is_def
                        ? Outcome{static_cast<uint8_t>(b_hp), static_cast<uint8_t>(a_hp), q}
                        : Outcome{static_cast<uint8_t>(a_hp), static_cast<uint8_t>(b_hp), q});
    }
  }
  return true;
}

bool sample_fight(const Combatant& att, const Combatant& def, int dist, uint64_t (&rng)[2],
                  Outcome& out) {
  if (!can_attack(att, def, dist)) return false;
  const bool first_is_def = strikes_first(def) && can_counter(def, att, dist);
  const Combatant& a = first_is_def ? def : att;
  Combatant b = first_is_def ? att : def;
  const bool b_counters = first_is_def || can_counter(b, a, dist);
  const int b_hp = std::max(0, b.hp - damage(a, b, roll_luck(a, rng), false));
  int a_hp = a.hp;
  if (b_hp > 0 && b_counters) {
    b.hp = static_cast<uint8_t>(b_hp);
    a_hp = std::max(0, a.hp - damage(b, a, roll_luck(b, rng), !first_is_def));
  }
  out = first_is_def ? Outcome{static_cast<uint8_t>(b_hp), static_cast<uint8_t>(a_hp), 1.0}
                     : Outcome{static_cast<uint8_t>(a_hp), static_cast<uint8_t>(b_hp), 1.0};
  return true;
}

Combatant seam_target() {
  Combatant c;
  c.type = kNeotank;
  return c;
}

int seam_damage(const Combatant& att) {
  const Weapon w = choose_weapon(att, kNeotank);
  if (w == kNoWeapon) return 0;
  const int base = (w == kPrimaryWeapon ? kPrimaryDamage : kSecondaryDamage)[att.type][kNeotank];
  // The damage formula with no luck and defense 100 at 0 stars: (200 - 100) / 100 = 1.
  const int64_t n = int64_t{base} * attack_value(att, seam_target()) * displayed_hp(att.hp) * 100;
  return n <= 0 ? 0 : static_cast<int>((n + 5000) / 100000);
}

}  // namespace awbw
