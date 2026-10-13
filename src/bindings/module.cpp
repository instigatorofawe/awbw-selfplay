#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/actions.h"
#include "core/combat.h"
#include "core/config.h"
#include "core/map.h"
#include "core/movegen.h"
#include "core/snapshot.h"
#include "core/state.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T, size_t Cols>
using Rows = nb::ndarray<const T, nb::shape<-1, Cols>, nb::c_contig, nb::device::cpu>;

const awbw::MapData& map_for(const awbw::GameState& s) {
  const awbw::MapData* m = awbw::find_map(s.map_id);
  if (!m) throw nb::value_error(("map " + std::to_string(s.map_id) + " is not registered").c_str());
  return *m;
}

void check_slot(const awbw::GameState& s, int slot) {
  if (slot < 0 || slot >= awbw::kMaxUnits || !s.units[slot].alive())
    throw nb::index_error(("no unit in slot " + std::to_string(slot)).c_str());
}

using XY = std::tuple<int, int>;

uint16_t tile_of(const awbw::MapData& md, XY xy) {
  auto [x, y] = xy;
  if (x < 0 || y < 0 || x >= md.width || y >= md.height)
    throw nb::index_error(
        ("(" + std::to_string(x) + ", " + std::to_string(y) + ") is off the map").c_str());
  return static_cast<uint16_t>(md.tile(x, y));
}

int apply_action(awbw::GameState& s, int kind, int unit, std::optional<XY> to,
                 std::optional<XY> target, int arg, std::optional<std::vector<XY>> path, int att_hp,
                 int def_hp) {
  const awbw::MapData& md = map_for(s);
  if (kind < 0 || kind >= awbw::kNumActionKinds) throw nb::value_error("unknown action kind");
  awbw::Action a;
  a.kind = static_cast<uint8_t>(kind);
  a.unit = unit < 0 || unit >= awbw::kMaxUnits ? awbw::kNoUnit : static_cast<uint8_t>(unit);
  if (to) {
    a.to = tile_of(md, *to);
  } else if (a.unit != awbw::kNoUnit && s.units[a.unit].alive()) {
    a.to = s.units[a.unit].tile;
  }
  if (target) a.target = tile_of(md, *target);
  a.arg = static_cast<uint8_t>(arg);
  awbw::Detail d;
  std::vector<uint16_t> tiles;
  if (path) {
    for (const XY& xy : *path) tiles.push_back(tile_of(md, xy));
    d.path = tiles.data();
    d.path_len = static_cast<int>(tiles.size());
  }
  d.att_hp = att_hp;
  d.def_hp = def_hp;
  return awbw::apply(md, s, a, d);
}

awbw::GameState load_snapshot(const awbw::MapData& m, const awbw::Settings& settings, int day,
                              int active, Rows<int64_t, 8> players, Rows<int64_t, 4> buildings,
                              Rows<int64_t, 11> units) {
  if (players.shape(0) != awbw::kMaxPlayers) throw nb::value_error("players must have 2 rows");
  awbw::Snapshot snap{};
  snap.settings = settings;
  snap.day = day;
  snap.active = active;
  auto pv = players.view();
  for (int i = 0; i < awbw::kMaxPlayers; ++i) {
    auto c = [&](int j) { return static_cast<int>(pv(i, j)); };
    snap.players[i] = awbw::SnapshotPlayer{c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7)};
  }
  std::vector<awbw::SnapshotBuilding> b(buildings.shape(0));
  auto bv = buildings.view();
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = {static_cast<int>(bv(i, 0)), static_cast<int>(bv(i, 1)), static_cast<int>(bv(i, 2)),
            static_cast<int>(bv(i, 3))};
  }
  std::vector<awbw::SnapshotUnit> u(units.shape(0));
  auto uv = units.view();
  for (size_t i = 0; i < u.size(); ++i) {
    auto c = [&](int j) { return static_cast<int>(uv(i, j)); };
    u[i] = {static_cast<uint32_t>(uv(i, 0)),
            c(1),
            c(2),
            c(3),
            c(4),
            c(5),
            c(6),
            c(7),
            c(8),
            {static_cast<uint32_t>(uv(i, 9)), static_cast<uint32_t>(uv(i, 10))}};
  }
  snap.buildings = b.data();
  snap.n_buildings = static_cast<int>(b.size());
  snap.units = u.data();
  snap.n_units = static_cast<int>(u.size());
  awbw::GameState s;
  std::string error;
  if (!awbw::load_snapshot(m, snap, s, error)) throw nb::value_error(error.c_str());
  return s;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "AWBW engine core (C++)";
  m.attr("MAX_WIDTH") = awbw::kMaxWidth;
  m.attr("MAX_HEIGHT") = awbw::kMaxHeight;
  m.attr("MAX_TILES") = awbw::kMaxTiles;
  m.attr("MAX_PLAYERS") = awbw::kMaxPlayers;
  m.attr("MAX_UNITS") = awbw::kMaxUnits;
  m.attr("MAX_PROPS") = awbw::kMaxProps;
  m.attr("STATE_BYTES") = sizeof(awbw::GameState);
  m.attr("NO_UNIT") = awbw::kNoUnit;
  m.attr("NO_OWNER") = awbw::kNoOwner;
  m.attr("UNIT_MOVED") = static_cast<int>(awbw::kMoved);
  m.attr("UNIT_HIDDEN") = static_cast<int>(awbw::kHidden);
  m.attr("UNIT_CARRIED") = static_cast<int>(awbw::kCarried);
  m.attr("UNIT_STUNNED") = static_cast<int>(awbw::kStunned);
  m.attr("UNIT_CAPTURING") = static_cast<int>(awbw::kCapturing);
  m.attr("STAR_VALUE") = awbw::kStarValue;

  {
    nb::dict kinds;
    const char* names[awbw::kNumActionKinds] = {
        "Wait",   "Fire",   "Capture", "Load",   "Join",       "Unload",
        "Supply", "Repair", "Hide",    "Unhide", "AttackSeam", "Explode",
        "Launch", "Build",  "Power",   "Delete", "EndTurn",    "Resign"};
    for (int k = 0; k < awbw::kNumActionKinds; ++k) kinds[names[k]] = k;
    m.attr("ACTIONS") = kinds;
  }
  m.attr("OK") = static_cast<int>(awbw::kOk);
  m.attr("TRAPPED") = static_cast<int>(awbw::kTrapped);
  m.def(
      "status_name",
      [](int st) { return std::string(awbw::status_name(static_cast<awbw::Status>(st))); },
      "status"_a, "What an apply() status code means.");

  nb::class_<awbw::MapData>(m, "MapData", "A map's static data, shared by every game on it.")
      .def_ro("id", &awbw::MapData::id)
      .def_ro("width", &awbw::MapData::width)
      .def_ro("height", &awbw::MapData::height)
      .def_ro("n_props", &awbw::MapData::n_props)
      .def(
          "props",
          [](const awbw::MapData& md) {
            std::vector<std::tuple<int, int, int, int>> out;
            for (int p = 0; p < md.n_props; ++p) {
              out.emplace_back(md.x_of(md.prop_tile[p]), md.y_of(md.prop_tile[p]), md.prop_kind[p],
                               md.prop_country[p]);
            }
            return out;
          },
          "(x, y, terrain kind, starting country) for each tile with per-game state.")
      .def("tile", &awbw::MapData::tile, "x"_a, "y"_a);

  m.def(
      "register_map",
      [](uint32_t id,
         nb::ndarray<const int16_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> terrain) {
        auto md = std::make_unique<awbw::MapData>();
        std::string error;
        if (!awbw::build_map(id, static_cast<int>(terrain.shape(1)),
                             static_cast<int>(terrain.shape(0)), terrain.data(), *md, error))
          throw nb::value_error(error.c_str());
        const awbw::MapData* reg = awbw::register_map(*md, error);
        if (!reg) throw nb::value_error(error.c_str());
        return reg;
      },
      "map_id"_a, "terrain"_a, nb::rv_policy::reference,
      "Build a map from AWBW terrain ids [height, width] and register it under `map_id`.");
  m.def("find_map", &awbw::find_map, "map_id"_a, nb::rv_policy::reference,
        "The registered map with this id, or None.");

  nb::class_<awbw::Settings>(m, "Settings", "Game options that vary between AWBW games.")
      .def(nb::init<>())
      .def_rw("funds_per_property", &awbw::Settings::funds_per_property)
      .def_rw("capture_win", &awbw::Settings::capture_win)
      .def_rw("day_limit", &awbw::Settings::day_limit)
      .def_rw("unit_limit", &awbw::Settings::unit_limit)
      .def_rw("use_powers", &awbw::Settings::use_powers)
      .def_rw("banned_units", &awbw::Settings::banned_units);

  nb::class_<awbw::Unit>(m, "Unit")
      .def_ro("id", &awbw::Unit::id)
      .def_ro("tile", &awbw::Unit::tile)
      .def_ro("type", &awbw::Unit::type)
      .def_ro("owner", &awbw::Unit::owner)
      .def_ro("hp", &awbw::Unit::hp)
      .def_ro("fuel", &awbw::Unit::fuel)
      .def_ro("ammo", &awbw::Unit::ammo)
      .def_ro("flags", &awbw::Unit::flags)
      .def_ro("carrier", &awbw::Unit::carrier)
      .def_prop_ro(
          "cargo",
          [](const awbw::Unit& u) {
            auto slot = [](uint8_t c) { return c == awbw::kNoUnit ? -1 : static_cast<int>(c); };
            return std::make_tuple(slot(u.cargo[0]), slot(u.cargo[1]));
          },
          "Slots of the units in cargo slots 1 and 2; -1 = empty (slots don't shift).");

  nb::class_<awbw::Player>(m, "Player")
      .def_ro("funds", &awbw::Player::funds)
      .def_ro("power", &awbw::Player::power)
      .def_ro("cop_cost", &awbw::Player::cop_cost)
      .def_ro("scop_cost", &awbw::Player::scop_cost)
      .def_ro("co", &awbw::Player::co)
      .def_ro("country", &awbw::Player::country)
      .def_ro("power_on", &awbw::Player::power_on)
      .def_ro("power_uses", &awbw::Player::power_uses)
      .def_ro("eliminated", &awbw::Player::eliminated)
      .def_ro("power_atk", &awbw::Player::power_atk);

  nb::class_<awbw::GameState>(m, "GameState", "A game position: a fixed-size plain value.")
      .def("copy", [](const awbw::GameState& s) { return s; })
      .def("to_bytes",
           [](const awbw::GameState& s) {
             return nb::bytes(reinterpret_cast<const char*>(&s), sizeof s);
           })
      .def_static(
          "from_bytes",
          [](nb::bytes b) {
            if (b.size() != sizeof(awbw::GameState))
              throw nb::value_error("wrong size for a GameState");
            awbw::GameState s;
            std::memcpy(&s, b.c_str(), sizeof s);
            return s;
          },
          "data"_a)
      .def_ro("map_id", &awbw::GameState::map_id)
      .def_ro("settings", &awbw::GameState::settings)
      .def_ro("day", &awbw::GameState::day)
      .def_ro("active", &awbw::GameState::active)
      .def_ro("winner", &awbw::GameState::winner)
      .def(
          "seed", [](awbw::GameState& s, uint64_t seed) { awbw::rng_seed(s.rng, seed); }, "seed"_a,
          "Reseed the state's RNG (combat luck).")
      .def_prop_ro("players",
                   [](const awbw::GameState& s) {
                     return std::vector<awbw::Player>(s.players, s.players + awbw::kMaxPlayers);
                   })
      .def(
          "unit",
          [](const awbw::GameState& s, int slot) {
            check_slot(s, slot);
            return s.units[slot];
          },
          "slot"_a)
      .def(
          "unit_slots",
          [](const awbw::GameState& s) {
            std::vector<int> out;
            for (int i = 0; i < awbw::kMaxUnits; ++i) {
              if (s.units[i].alive()) out.push_back(i);
            }
            return out;
          },
          "Slots holding a unit, in slot order.")
      .def(
          "find_unit",
          [](const awbw::GameState& s, uint32_t id) {
            for (int i = 0; i < awbw::kMaxUnits; ++i) {
              if (s.units[i].id == id && id != 0) return i;
            }
            return -1;
          },
          "id"_a, "Slot of the unit with this id, or -1.")
      .def(
          "unit_rows",
          [](const awbw::GameState& s) {
            const awbw::MapData& md = map_for(s);
            std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int>> out;
            for (int i = 0; i < awbw::kMaxUnits; ++i) {
              const awbw::Unit& u = s.units[i];
              if (!u.alive()) continue;
              auto id_at = [&](uint8_t slot) {
                return slot == awbw::kNoUnit ? 0 : static_cast<int>(s.units[slot].id);
              };
              out.emplace_back(static_cast<int>(u.id), u.type, u.owner, md.x_of(u.tile),
                               md.y_of(u.tile), u.hp, u.fuel, u.ammo, u.flags, id_at(u.cargo[0]),
                               id_at(u.cargo[1]));
            }
            return out;
          },
          "(id, type, owner, x, y, hp, fuel, ammo, flags, cargo1_id, cargo2_id) for every unit, "
          "in slot order; cargo ids are 0 when empty.")
      .def(
          "unit_at",
          [](const awbw::GameState& s, int x, int y) {
            const awbw::MapData& md = map_for(s);
            if (x < 0 || y < 0 || x >= md.width || y >= md.height)
              throw nb::index_error("off the map");
            const int slot = s.unit_at[md.tile(x, y)];
            return slot == awbw::kNoUnit ? -1 : slot;
          },
          "x"_a, "y"_a, "Slot of the uncarried unit on a tile, or -1.")
      .def(
          "prop_owner", [](const awbw::GameState& s, int p) { return s.prop_owner[p]; }, "prop"_a)
      .def(
          "prop_points", [](const awbw::GameState& s, int p) { return s.prop_points[p]; }, "prop"_a)
      .def(
          "check", [](const awbw::GameState& s) { return awbw::check_state(map_for(s), s); },
          "Consistency problems, or '' if the state is sound.")
      .def("__getstate__",
           [](const awbw::GameState& s) {
             return nb::bytes(reinterpret_cast<const char*>(&s), sizeof s);
           })
      .def("__setstate__", [](awbw::GameState& s, nb::bytes b) {
        if (b.size() != sizeof(awbw::GameState))
          throw nb::value_error("wrong size for a GameState");
        new (&s) awbw::GameState;
        std::memcpy(&s, b.c_str(), sizeof s);
      });

  m.def("load_snapshot", &load_snapshot, "map"_a, "settings"_a, "day"_a, "active"_a, "players"_a,
        "buildings"_a, "units"_a,
        "GameState from a flattened AWBW snapshot. players: [2, 8] (co, country, funds, power, "
        "cop_cost, scop_cost, power_on, eliminated); buildings: [n, 4] (x, y, terrain_id, "
        "capture); units: [n, 11] (id, type, owner, x, y, hp, fuel, ammo, flags, cargo1_id, "
        "cargo2_id).");

  m.def(
      "reachable",
      [](const awbw::GameState& s, int slot) {
        check_slot(s, slot);
        const awbw::MapData& md = map_for(s);
        auto r = std::make_unique<awbw::Reach>();
        awbw::reachable(md, s, slot, *r);
        std::vector<std::tuple<int, int, int>> out;
        out.reserve(r->n);
        for (int i = 0; i < r->n; ++i) {
          const int t = r->tiles[i];
          out.emplace_back(md.x_of(t), md.y_of(t), r->cost[t]);
        }
        return out;
      },
      "state"_a, "slot"_a,
      "(x, y, cost) of every tile the unit can move through this turn, cheapest first.");
  m.def(
      "path_cost",
      [](const awbw::GameState& s, int slot, const std::vector<std::tuple<int, int>>& path) {
        check_slot(s, slot);
        const awbw::MapData& md = map_for(s);
        std::vector<uint16_t> tiles;
        for (auto [x, y] : path) {
          if (x < 0 || y < 0 || x >= md.width || y >= md.height) return -1;
          tiles.push_back(static_cast<uint16_t>(md.tile(x, y)));
        }
        return awbw::path_cost(md, s, slot, tiles.data(), static_cast<int>(tiles.size()));
      },
      "state"_a, "slot"_a, "path"_a,
      "Movement cost of a path of (x, y) starting on the unit's tile, or -1 if it is not a legal "
      "move.");
  m.def(
      "move_points",
      [](const awbw::GameState& s, int slot) {
        check_slot(s, slot);
        return awbw::move_points(s, s.units[slot]);
      },
      "state"_a, "slot"_a, "Movement points with CO bonuses, before the fuel cap.");

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

  m.def("apply", &apply_action, "state"_a, "kind"_a, "unit"_a = -1, "to"_a = nb::none(),
        "target"_a = nb::none(), "arg"_a = 0, "path"_a = nb::none(), "att_hp"_a = -1,
        "def_hp"_a = -1,
        "Carry out one action for the active player, in place. kind: ACTIONS[name]; unit: slot; "
        "to / target: (x, y); path: [(x, y), ...] from the unit's tile to `to` (default: the "
        "cheapest path its owner can see); att_hp / def_hp force a fight's result. Returns OK, "
        "TRAPPED or an error status (state unchanged); see status_name.");
  m.def(
      "combatant",
      [](const awbw::GameState& s, int slot, std::optional<XY> at) {
        check_slot(s, slot);
        const awbw::MapData& md = map_for(s);
        return awbw::combatant(md, s, slot, at ? tile_of(md, *at) : s.units[slot].tile);
      },
      "state"_a, "slot"_a, "at"_a = nb::none(),
      "The unit as the damage formula sees it, standing on `at` (default: its tile).");
  m.def(
      "fight_outcomes",
      [](const awbw::GameState& s, int slot, XY from, XY target) {
        check_slot(s, slot);
        const awbw::MapData& md = map_for(s);
        const int from_t = tile_of(md, from), target_t = tile_of(md, target);
        const int def = s.unit_at[target_t];
        std::vector<std::tuple<int, int, double>> res;
        if (def == awbw::kNoUnit) return res;
        std::vector<awbw::Outcome> out;
        awbw::combat_outcomes(awbw::combatant(md, s, slot, from_t),
                              awbw::combatant(md, s, def, target_t), md.distance(from_t, target_t),
                              out);
        for (const auto& o : out) res.emplace_back(o.att_hp, o.def_hp, o.prob);
        return res;
      },
      "state"_a, "slot"_a, "from_"_a, "target"_a,
      "(attacker_hp, defender_hp, probability) results of the unit attacking from `from_` the "
      "unit on `target`.");
  m.def("cop_cost", &awbw::cop_cost, "co"_a, "uses"_a, "COP cost after `uses` power uses.");
  m.def("scop_cost", &awbw::scop_cost, "co"_a, "uses"_a, "SCOP cost after `uses` power uses.");
}
