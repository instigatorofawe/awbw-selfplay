#pragma once

#include <cstdint>
#include <string>

#include "core/config.h"
#include "core/tables.h"

namespace awbw {

inline constexpr uint8_t kNoProp = 0xFF;
inline constexpr int16_t kOffMap = -1;
inline constexpr int kNumDirs = 4;  // north, east, south, west

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

  int tile(int x, int y) const { return y * width + x; }
  int x_of(int t) const { return t % width; }
  int y_of(int t) const { return t / width; }
  int distance(int a, int b) const;
};

bool is_capturable(int kind);
// Properties that pay income at the start of their owner's turn.
bool pays_income(int kind);

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
