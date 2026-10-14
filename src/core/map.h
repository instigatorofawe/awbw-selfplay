#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/tables.h"

namespace awbw {

inline constexpr uint8_t kNoProp = 0xFF;
inline constexpr int16_t kOffMap = -1;
inline constexpr int kNumDirs = 4;  // north, east, south, west
inline constexpr uint16_t kNoRow = 0xFFFF;

// Unit types that move alike: the same movement type and points, before CO
// bonuses. Classes are numbered in order of their first unit type; Tank, APC,
// Anti-Air and Neotank share one, and there are 16 in all.
struct MoverClasses {
  int n = 0;
  uint8_t move_type[kNumUnitTypes] = {};
  uint8_t move[kNumUnitTypes] = {};
  uint8_t of_type[kNumUnitTypes] = {};  // each unit type's class
};

constexpr MoverClasses make_mover_classes() {
  MoverClasses c;
  for (int u = 0; u < kNumUnitTypes; ++u) {
    int i = 0;
    while (i < c.n && (c.move_type[i] != kUnits[u].move_type || c.move[i] != kUnits[u].move)) ++i;
    if (i == c.n) {
      c.move_type[i] = kUnits[u].move_type;
      c.move[i] = kUnits[u].move;
      ++c.n;
    }
    c.of_type[u] = static_cast<uint8_t>(i);
  }
  return c;
}

inline constexpr MoverClasses kMovers = make_mover_classes();
inline constexpr int kNumMoverClasses = kMovers.n;
static_assert(kNumMoverClasses == 16);

// A map's static data, built once and shared by every game on it. Tiles are
// numbered y * width + x. Anything that changes during a game (owners,
// capture points, broken pipe seams, fired silos) lives in GameState, indexed
// by the map's property list.
struct MapData {
  uint32_t id = 0;
  int width = 0;
  int height = 0;
  int n_tiles = 0;
  uint16_t terrain_id[kMaxTiles];  // AWBW terrain id at game start
  uint8_t kind[kMaxTiles];         // TerrainKind at game start
  uint8_t stars[kMaxTiles];        // defense stars at game start
  uint8_t prop[kMaxTiles];         // index into the property list, or kNoProp
  int16_t neighbor[kMaxTiles][kNumDirs];
  // Movement cost [weather][move type][tile] of the starting terrain; 0 = impassable.
  uint8_t move_cost[kNumWeathers][kNumMoveTypes][kMaxTiles];

  // Tiles with per-game state, in tile order: capturable properties (owner and
  // capture points), pipe seams (HP) and loaded missile silos.
  int n_props = 0;
  uint16_t prop_tile[kMaxProps];
  uint8_t prop_kind[kMaxProps];
  uint8_t prop_country[kMaxProps];  // owning country at game start; 0 = neutral
  // Property indices column by column (x, then y): the order AWBW repairs in.
  uint8_t prop_by_column[kMaxProps];

  // Turns of moving from each factory to every tile, for each mover class it
  // builds, on the starting terrain in clear weather without CO bonuses
  // (movement.h's turns_from, kUnreachable where it can't go): row
  // factory_row[prop][class] of factory_turns, n_tiles entries each, or
  // kNoRow. Hachi's cities, which build only under her SCOP, have none.
  uint16_t factory_row[kMaxProps][kNumMoverClasses];
  std::vector<uint8_t> factory_turns;

  int tile(int x, int y) const { return y * width + x; }
  int x_of(int t) const { return t % width; }
  int y_of(int t) const { return t / width; }
  int distance(int a, int b) const;
  // Row `row` of factory_turns.
  const uint8_t* turns_row(int row) const {
    return factory_turns.data() + static_cast<size_t>(row) * n_tiles;
  }
};

bool is_capturable(int kind);
// Properties that pay income at the start of their owner's turn.
bool pays_income(int kind);
// Whether a factory of `kind` (base, airport, port) builds units of `move_type`.
bool factory_builds(int kind, int move_type);

// Build a map from its AWBW terrain ids (row-major, height x width). Returns
// false with a message in `error` if the map is too big, uses an unknown
// terrain id, or has too many properties.
bool build_map(uint32_t id, int width, int height, const int16_t* terrain_ids, MapData& out,
               std::string& error);

// Process-wide registry, so a GameState can name its map by id. Maps are never
// removed; registering an id again with different terrain is an error.
const MapData* register_map(const MapData& map, std::string& error);
const MapData* find_map(uint32_t id);

}  // namespace awbw
