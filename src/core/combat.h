#pragma once

#include <cstdint>
#include <vector>

#include "core/tables.h"

namespace awbw {

// Displayed HP (1..10) for an internal HP (1..100); 0 stays 0.
constexpr int displayed_hp(int hp) { return (hp + 9) / 10; }

// One side of a fight: everything the damage formula reads about a unit, its
// tile and its CO. Built from a GameState by the engine, or directly by tests
// and the replay checker.
struct Combatant {
  uint8_t type = 0;       // UnitType
  uint8_t hp = 100;       // internal HP, 1..100
  uint8_t ammo = 0;       // primary-weapon ammo
  uint8_t co = 0;         // AWBW CO id
  uint8_t power = 0;      // 0 none, 1 COP, 2 SCOP
  uint8_t terrain = 0;    // TerrainKind of the unit's tile
  uint8_t stars = 0;      // that tile's defense stars (before air/Lash adjustments)
  uint8_t towers = 0;     // comtowers the unit's owner holds
  int16_t extra_atk = 0;  // CO attack % computed from game state (Colin, Kindle SCOPs)
  uint8_t hidden = 0;     // dived Sub or hidden Stealth
};

enum Weapon : int8_t { kNoWeapon = -1, kPrimaryWeapon = 0, kSecondaryWeapon = 1 };

// The weapon `att` uses against a unit of type `def_type`: the ammo weapon if it
// can hit and has ammo, else the weapon that needs none, else kNoWeapon.
Weapon choose_weapon(const Combatant& att, int def_type);

// Whether `att` can attack `def` at Manhattan distance `dist`: range (with CO
// bonuses), a usable weapon, and, for a hidden target, being one of the units
// that can find it (Fighters and Stealths for a Stealth; Subs and Cruisers for
// a Sub).
bool can_attack(const Combatant& att, const Combatant& def, int dist);

// Max attack range after CO modifiers (never below the minimum range).
int max_range(const Combatant& c);

// Attack % (100 = no modifiers) of `att` hitting `def`.
int attack_value(const Combatant& att, const Combatant& def);
// Defense % (100 = no modifiers) of `def` when hit by `att`.
int defense_value(const Combatant& def, const Combatant& att);
// Terrain stars that count for `c`'s defense (0 for air units).
int defense_stars(const Combatant& c);

// Damage in internal HP that `att` deals to `def` with a given net luck roll
// (good roll minus bad roll). `counter` applies counterattack multipliers.
int damage(const Combatant& att, const Combatant& def, int luck, bool counter);

struct Outcome {
  uint8_t att_hp;  // attacker internal HP afterwards; 0 = destroyed
  uint8_t def_hp;  // defender internal HP afterwards; 0 = destroyed
  double prob;
};

// Exact distribution over results when `att` attacks `def` from distance
// `dist`, including the counterattack. Outcomes with the same (att_hp, def_hp)
// are merged. `out` is cleared first. Returns false if `att` cannot attack.
bool combat_outcomes(const Combatant& att, const Combatant& def, int dist,
                     std::vector<Outcome>& out);

}  // namespace awbw
