#include "core/snapshot.h"

#include <algorithm>
#include <vector>

namespace awbw {

namespace {

std::string at(int x, int y) {
  return " at (" + std::to_string(x) + ", " + std::to_string(y) + ")";
}

}  // namespace

bool load_snapshot(const MapData& m, const Snapshot& snap, GameState& s, std::string& error) {
  init_state(m, s);
  s.settings = snap.settings;
  s.day = static_cast<uint16_t>(snap.day);
  s.active = static_cast<uint8_t>(snap.active);
  for (int i = 0; i < kMaxPlayers; ++i) {
    const SnapshotPlayer& sp = snap.players[i];
    Player& p = s.players[i];
    if (sp.co < 0 || sp.co > kMaxCoId || !kCos[sp.co].valid) {
      error = "unknown CO id " + std::to_string(sp.co);
      return false;
    }
    p.co = static_cast<uint8_t>(sp.co);
    p.country = static_cast<uint8_t>(sp.country);
    p.funds = sp.funds;
    p.power = sp.power;
    p.cop_cost = sp.cop_cost;
    p.scop_cost = sp.scop_cost;
    p.power_on = static_cast<uint8_t>(sp.power_on);
    p.eliminated = static_cast<uint8_t>(sp.eliminated);
  }

  for (int i = 0; i < snap.n_buildings; ++i) {
    const SnapshotBuilding& b = snap.buildings[i];
    if (b.x < 0 || b.y < 0 || b.x >= m.width || b.y >= m.height) {
      error = "building off the map" + at(b.x, b.y);
      return false;
    }
    const int prop = m.prop[m.tile(b.x, b.y)];
    if (prop == kNoProp) continue;  // AWBW also lists pipes and empty silos
    if (b.terrain_id < 0 || b.terrain_id > kMaxTerrainId || !kTerrain[b.terrain_id].valid) {
      error = "unknown terrain id " + std::to_string(b.terrain_id) + at(b.x, b.y);
      return false;
    }
    const TerrainSpec& t = kTerrain[b.terrain_id];
    switch (m.prop_kind[prop]) {
      case kSeam:
        s.prop_points[prop] =
            static_cast<uint8_t>(t.kind == kRubble ? 0 : std::clamp(b.points, 1, 99));
        break;
      case kSilo:
        s.prop_points[prop] = t.kind == kSilo ? 1 : 0;
        break;
      default:
        if (t.kind != m.prop_kind[prop]) {
          error = "building kind changed" + at(b.x, b.y);
          return false;
        }
        s.prop_points[prop] = static_cast<uint8_t>(std::clamp(b.points, 0, kFullCapture));
        s.prop_owner[prop] = kNoOwner;
        for (int pl = 0; pl < kMaxPlayers; ++pl) {
          if (t.country != 0 && s.players[pl].country == t.country)
            s.prop_owner[prop] = static_cast<uint8_t>(pl);
        }
    }
  }

  std::vector<const SnapshotUnit*> order(snap.n_units);
  for (int i = 0; i < snap.n_units; ++i) order[i] = &snap.units[i];
  std::sort(order.begin(), order.end(), [](auto a, auto b) { return a->id < b->id; });
  std::vector<uint32_t> carried;  // AWBW ids listed as cargo
  for (const SnapshotUnit* u : order) {
    for (uint32_t c : u->cargo) {
      if (c) carried.push_back(c);
    }
  }
  auto is_carried = [&](uint32_t id) {
    return std::find(carried.begin(), carried.end(), id) != carried.end();
  };

  // Every unit gets a slot in id order; carried units are linked afterwards.
  if (snap.n_units > kMaxUnits) {
    error = std::to_string(snap.n_units) + " units; the engine holds " + std::to_string(kMaxUnits);
    return false;
  }
  for (int slot = 0; slot < static_cast<int>(order.size()); ++slot) {
    const SnapshotUnit& su = *order[slot];
    if (su.id == 0 || su.type < 0 || su.type >= kNumUnitTypes || su.owner < 0 ||
        su.owner >= kMaxPlayers || su.hp < 1 || su.hp > 100) {
      error = "bad unit " + std::to_string(su.id);
      return false;
    }
    if (su.x < 0 || su.y < 0 || su.x >= m.width || su.y >= m.height) {
      error = "unit " + std::to_string(su.id) + " off the map" + at(su.x, su.y);
      return false;
    }
    const int tile = m.tile(su.x, su.y);
    Unit& u = s.units[slot];
    u.id = su.id;
    u.tile = static_cast<uint16_t>(tile);
    u.type = static_cast<uint8_t>(su.type);
    u.owner = static_cast<uint8_t>(su.owner);
    u.hp = static_cast<uint8_t>(su.hp);
    u.fuel = static_cast<uint8_t>(std::clamp<int>(su.fuel, 0, kUnits[su.type].fuel));
    u.ammo = static_cast<uint8_t>(std::clamp<int>(su.ammo, 0, kUnits[su.type].ammo));
    u.flags = static_cast<uint8_t>(su.flags & ~kCarried);
    if (is_carried(su.id)) {
      u.flags |= kCarried;
      continue;
    }
    if (s.unit_at[tile] != kNoUnit) {
      error = "two units" + at(su.x, su.y);
      return false;
    }
    s.unit_at[tile] = static_cast<uint8_t>(slot);
  }
  for (int slot = 0; slot < static_cast<int>(order.size()); ++slot) {
    const SnapshotUnit& su = *order[slot];
    for (int c = 0; c < 2; ++c) {
      if (!su.cargo[c]) continue;
      auto it =
          std::find_if(order.begin(), order.end(), [&](auto o) { return o->id == su.cargo[c]; });
      if (it == order.end()) {
        error = "unit " + std::to_string(su.id) + " carries missing unit " +
                std::to_string(su.cargo[c]);
        return false;
      }
      const int k = static_cast<int>(it - order.begin());
      Unit& t = s.units[slot];
      t.cargo[t.cargo[0] == kNoUnit ? 0 : 1] = static_cast<uint8_t>(k);
      s.units[k].carrier = static_cast<uint8_t>(slot);
      s.units[k].tile = t.tile;
    }
  }
  for (const Unit& u : s.units) {
    if (u.alive() && u.id >= s.next_id) s.next_id = u.id + 1;
  }
  error = check_state(m, s);
  return error.empty();
}

}  // namespace awbw
