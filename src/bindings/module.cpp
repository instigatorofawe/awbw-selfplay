#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <tuple>
#include <vector>

#include "core/combat.h"
#include "core/config.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
  m.doc() = "AWBW engine core (C++)";
  m.attr("MAX_WIDTH") = awbw::kMaxWidth;
  m.attr("MAX_HEIGHT") = awbw::kMaxHeight;
  m.attr("MAX_TILES") = awbw::kMaxTiles;
  m.attr("MAX_PLAYERS") = awbw::kMaxPlayers;

  nb::class_<awbw::Combatant>(
      m, "Combatant",
      "One side of a fight: unit, tile and CO state as the damage formula sees them.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](awbw::Combatant* c, int type, int hp, int ammo, int co, int power, int terrain,
             int stars, int towers, int extra_atk, bool hidden) {
            new (c) awbw::Combatant{static_cast<uint8_t>(type),      static_cast<uint8_t>(hp),
                                    static_cast<uint8_t>(ammo),      static_cast<uint8_t>(co),
                                    static_cast<uint8_t>(power),     static_cast<uint8_t>(terrain),
                                    static_cast<uint8_t>(stars),     static_cast<uint8_t>(towers),
                                    static_cast<int16_t>(extra_atk), static_cast<uint8_t>(hidden)};
          },
          "type"_a, "hp"_a, "ammo"_a = 0, "co"_a = 0, "power"_a = 0, "terrain"_a = 0, "stars"_a = 0,
          "towers"_a = 0, "extra_atk"_a = 0, "hidden"_a = false)
      .def_rw("type", &awbw::Combatant::type, "engine unit index")
      .def_rw("hp", &awbw::Combatant::hp, "internal HP, 1..100")
      .def_rw("ammo", &awbw::Combatant::ammo)
      .def_rw("co", &awbw::Combatant::co, "AWBW CO id")
      .def_rw("power", &awbw::Combatant::power, "0 none, 1 COP, 2 SCOP")
      .def_rw("terrain", &awbw::Combatant::terrain, "terrain kind index of the unit's tile")
      .def_rw("stars", &awbw::Combatant::stars, "defense stars of the unit's tile")
      .def_rw("towers", &awbw::Combatant::towers, "comtowers owned by the unit's player")
      .def_rw("extra_atk", &awbw::Combatant::extra_atk, "attack % from state-dependent CO effects")
      .def_rw("hidden", &awbw::Combatant::hidden, "dived Sub or hidden Stealth");

  m.def("attack_value", &awbw::attack_value, "att"_a, "defender"_a);
  m.def("defense_value", &awbw::defense_value, "defender"_a, "att"_a);
  m.def("damage", &awbw::damage, "att"_a, "defender"_a, "luck"_a, "counter"_a = false,
        "Damage in internal HP for a given net luck roll.");
  m.def("can_attack", &awbw::can_attack, "att"_a, "defender"_a, "dist"_a);
  m.def(
      "weapon",
      [](const awbw::Combatant& att, int def_type) {
        return static_cast<int>(awbw::choose_weapon(att, def_type));
      },
      "att"_a, "def_type"_a, "-1 none, 0 ammo weapon, 1 weapon without ammo");
  m.def(
      "combat_outcomes",
      [](const awbw::Combatant& att, const awbw::Combatant& def, int dist) {
        std::vector<awbw::Outcome> out;
        awbw::combat_outcomes(att, def, dist, out);
        std::vector<std::tuple<int, int, double>> res;
        res.reserve(out.size());
        for (const auto& o : out) res.emplace_back(o.att_hp, o.def_hp, o.prob);
        return res;
      },
      "att"_a, "defender"_a, "dist"_a,
      "Exact (attacker_hp, defender_hp, probability) results of an attack, counterattack "
      "included.");
}
