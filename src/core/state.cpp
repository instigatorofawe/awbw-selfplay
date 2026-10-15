#include "core/state.h"

#include <cstring>

#include "core/belief.h"

namespace awbw {

void init_state(const MapData& map, GameState& s) {
  std::memset(&s, 0, sizeof s);
  s.map_id = map.id;
  s.settings = Settings{};
  s.next_id = 1;
  s.day = 1;
  s.weather = kClear;
  s.weather_owner = kNoOwner;
  s.winner = kNoOwner;
  std::memset(s.unit_at, kNoUnit, sizeof s.unit_at);
  std::memset(s.prop_owner, kNoOwner, sizeof s.prop_owner);
  for (int p = 0; p < map.n_props; ++p) {
    const int kind = map.prop_kind[p];
    s.prop_points[p] = kind == kSeam ? kFullSeamHp : kind == kSilo ? 1 : kFullCapture;
  }
  for (Player& pl : s.players) pl.power_atk = 0;
  for (Unit& u : s.units) {
    u.cargo[0] = u.cargo[1] = u.carrier = kNoUnit;
  }
  s.rng[0] = 0x9E3779B97F4A7C15ull;
  s.rng[1] = 0xD1B54A32D192ED03ull;
}

int alloc_unit(const GameState& s) {
  for (int i = 0; i < kMaxUnits; ++i) {
    if (!s.units[i].alive()) return i;
  }
  return -1;
}

int add_unit(GameState& s, const Unit& u) {
  const int slot = alloc_unit(s);
  if (slot < 0 || s.unit_at[u.tile] != kNoUnit) return -1;
  Unit& n = s.units[slot];
  n = u;
  n.flags &= static_cast<uint8_t>(~kCarried);
  n.cargo[0] = n.cargo[1] = n.carrier = kNoUnit;
  if (n.id == 0) n.id = s.next_id;
  if (n.id >= s.next_id) s.next_id = n.id + 1;
  s.unit_at[n.tile] = static_cast<uint8_t>(slot);
  belief_exact(s, slot);
  hidden_exact(s, slot);
  return slot;
}

void remove_unit(GameState& s, int slot) {
  Unit& u = s.units[slot];
  for (uint8_t c : u.cargo) {
    if (c != kNoUnit) remove_unit(s, c);
  }
  if (u.has(kCarried)) {
    Unit& t = s.units[u.carrier];
    for (uint8_t& c : t.cargo) {
      if (c == slot) c = kNoUnit;
    }
  } else if (s.unit_at[u.tile] == slot) {
    s.unit_at[u.tile] = kNoUnit;
  }
  std::memset(&u, 0, sizeof u);
  u.cargo[0] = u.cargo[1] = u.carrier = kNoUnit;
  std::memset(s.hp_belief[slot], 0, kHpBins);
  std::memset(s.hp_hidden[slot], 0, kHpBins);
}

// A transport's cargo stands on its tile, and the cargo's cargo too (AWBW
// lets a loaded APC board a Lander).
static void move_cargo(GameState& s, const Unit& t, int tile) {
  for (uint8_t c : t.cargo) {
    if (c == kNoUnit) continue;
    s.units[c].tile = static_cast<uint16_t>(tile);
    move_cargo(s, s.units[c], tile);
  }
}

void relocate_unit(GameState& s, int slot, int tile) {
  Unit& u = s.units[slot];
  if (u.tile == tile) return;
  s.unit_at[u.tile] = kNoUnit;
  s.unit_at[tile] = static_cast<uint8_t>(slot);
  u.tile = static_cast<uint16_t>(tile);
  move_cargo(s, u, tile);
}

void load_unit(GameState& s, int slot, int transport) {
  Unit& u = s.units[slot];
  Unit& t = s.units[transport];
  if (s.unit_at[u.tile] == slot) s.unit_at[u.tile] = kNoUnit;
  u.tile = t.tile;
  move_cargo(s, u, t.tile);
  u.flags |= kCarried;
  u.carrier = static_cast<uint8_t>(transport);
  t.cargo[t.cargo[0] == kNoUnit ? 0 : 1] = static_cast<uint8_t>(slot);
}

void unload_unit(GameState& s, int slot, int tile) {
  Unit& u = s.units[slot];
  Unit& t = s.units[u.carrier];
  for (uint8_t& c : t.cargo) {
    if (c == slot) c = kNoUnit;  // the other cargo keeps its slot, as in AWBW
  }
  u.flags &= static_cast<uint8_t>(~kCarried);
  u.carrier = kNoUnit;
  u.tile = static_cast<uint16_t>(tile);
  move_cargo(s, u, tile);
  s.unit_at[tile] = static_cast<uint8_t>(slot);
}

int terrain_kind(const MapData& m, const GameState& s, int tile) {
  const int kind = m.kind[tile];
  if (kind == kSeam && s.prop_points[m.prop[tile]] == 0) return kRubble;
  if (kind == kSilo && s.prop_points[m.prop[tile]] == 0) return kSiloEmpty;
  return kind;
}

int terrain_stars(const MapData& m, const GameState& s, int tile) {
  if (m.kind[tile] == kSeam && s.prop_points[m.prop[tile]] == 0) return 1;  // pipe rubble
  return m.stars[tile];
}

int move_cost(const MapData& m, const GameState& s, int tile, int move_type, int weather) {
  if (m.kind[tile] == kSeam && s.prop_points[m.prop[tile]] == 0)
    return kMoveCost[weather][kRubble][move_type];
  return m.move_cost[weather][move_type][tile];
}

int weather_for(const GameState& s, int player) {
  const Player& p = s.players[player];
  return kCos[p.co].state[p.power_on].weather_as[s.weather];
}

int count_units(const GameState& s, int owner) {
  int n = 0;
  for (const Unit& u : s.units) n += u.alive() && u.owner == owner;
  return n;
}

int count_props(const MapData& m, const GameState& s, int owner, int kind) {
  int n = 0;
  for (int p = 0; p < m.n_props; ++p) {
    n += s.prop_owner[p] == owner && (kind < 0 || m.prop_kind[p] == kind);
  }
  return n;
}

std::string check_state(const MapData& m, const GameState& s) {
  auto unit_msg = [](int slot, const char* what) {
    return "unit slot " + std::to_string(slot) + ": " + what;
  };
  if (s.map_id != m.id) return "state is for map " + std::to_string(s.map_id);
  if (s.active >= kMaxPlayers) return "bad active player";
  if (s.weather >= kNumWeathers) return "bad weather";
  if (s.weather_owner >= kMaxPlayers && s.weather_owner != kNoOwner) return "bad weather owner";
  for (int p = 0; p < kMaxPlayers; ++p) {
    const Player& pl = s.players[p];
    if (pl.co > kMaxCoId || !kCos[pl.co].valid) return "player " + std::to_string(p) + ": bad CO";
    if (pl.power_on > 2) return "player " + std::to_string(p) + ": bad power state";
  }
  for (int i = 0; i < m.n_props; ++i) {
    if (s.prop_owner[i] >= kMaxPlayers && s.prop_owner[i] != kNoOwner)
      return "property " + std::to_string(i) + ": bad owner";
  }
  for (int t = 0; t < m.n_tiles; ++t) {
    const int slot = s.unit_at[t];
    if (slot == kNoUnit) continue;
    if (slot >= kMaxUnits) return "tile " + std::to_string(t) + " points past the unit table";
    const Unit& u = s.units[slot];
    if (!u.alive() || u.tile != t || u.has(kCarried))
      return "tile " + std::to_string(t) + " points at a unit that is not there";
  }
  for (int t = m.n_tiles; t < kMaxTiles; ++t) {
    if (s.unit_at[t] != kNoUnit) return "unit off the map";
  }
  for (int i = 0; i < kMaxUnits; ++i) {
    const Unit& u = s.units[i];
    if (!u.alive()) continue;
    if (u.type >= kNumUnitTypes) return unit_msg(i, "bad type");
    if (u.owner >= kMaxPlayers) return unit_msg(i, "bad owner");
    if (u.hp < 1 || u.hp > 100) return unit_msg(i, "HP out of range");
    if (s.hp_belief[i][u.hp - hp_bin_base(u.hp)] == 0)
      return unit_msg(i, "HP belief rules out its HP");
    if (hides_hp(s, u.owner) && s.hp_hidden[i][displayed_hp(u.hp) - 1] == 0)
      return unit_msg(i, "hidden-HP belief rules out its displayed HP");
    if (u.tile >= m.n_tiles) return unit_msg(i, "off the map");
    const UnitSpec& spec = kUnits[u.type];
    if (u.fuel > spec.fuel) return unit_msg(i, "fuel above max");
    if (u.ammo > spec.ammo) return unit_msg(i, "ammo above max");
    if (u.id >= s.next_id) return unit_msg(i, "id not below next_id");
    if (u.has(kCarried)) {
      if (u.carrier >= kMaxUnits) return unit_msg(i, "carried with no carrier");
      const Unit& t = s.units[u.carrier];
      if (!t.alive() || (t.cargo[0] != i && t.cargo[1] != i))
        return unit_msg(i, "carrier does not list it");
      if (t.tile != u.tile) return unit_msg(i, "not on its carrier's tile");
      if (!((kUnits[t.type].carry_mask >> u.type) & 1))
        return unit_msg(i, "carrier cannot carry it");
      if (t.owner != u.owner) return unit_msg(i, "carried by the enemy");
    } else if (s.unit_at[u.tile] != i) {
      return unit_msg(i, "not on the occupancy grid");
    }
    const int slots = kUnits[u.type].carry_slots;
    for (int c = 0; c < 2; ++c) {
      const int k = u.cargo[c];
      if (k == kNoUnit) continue;
      if (c >= slots) return unit_msg(i, "more cargo than slots");
      if (k >= kMaxUnits || !s.units[k].alive() || s.units[k].carrier != i)
        return unit_msg(i, "cargo link broken");
    }
  }
  return "";
}

}  // namespace awbw
