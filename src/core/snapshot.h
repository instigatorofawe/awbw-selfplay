#pragma once

#include <cstdint>
#include <string>

#include "core/map.h"
#include "core/state.h"

namespace awbw {

// One AWBW turn-start snapshot, flattened by the Python replay parser.
struct SnapshotPlayer {
  int co;
  int country;
  int funds;
  int power;  // co_power
  int cop_cost;
  int scop_cost;
  int power_on;  // 0 none, 1 COP, 2 SCOP
  int eliminated;
};

struct SnapshotBuilding {
  int x, y;
  int terrain_id;  // current AWBW terrain id: owner country, or broken seam
  int points;      // AWBW "capture": capture points left, or a pipe seam's HP
};

struct SnapshotUnit {
  uint32_t id;  // AWBW unit id
  int type;     // UnitType
  int owner;    // player index
  int x, y;
  int hp;  // internal HP
  int fuel;
  int ammo;
  int flags;          // UnitFlag bits except kCarried
  uint32_t cargo[2];  // AWBW ids of loaded units, 0 = none
};

struct Snapshot {
  Settings settings;
  int day;
  int active;         // player index
  int weather;        // Weather
  int weather_owner;  // player index whose power set the weather, or -1
  SnapshotPlayer players[kMaxPlayers];
  const SnapshotBuilding* buildings;
  int n_buildings;
  const SnapshotUnit* units;
  int n_units;
};

// Build a GameState from a snapshot. Units are placed in id order, so slot
// order is creation order, and every unit's HP is known exactly (snapshots give
// internal HP; see belief.h for what players know). Returns false with a
// message in `error` if the snapshot does not fit the map or the engine's
// limits.
bool load_snapshot(const MapData& m, const Snapshot& snap, GameState& s, std::string& error);

}  // namespace awbw
