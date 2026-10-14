#include "core/map.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "core/movement.h"

namespace awbw {

int MapData::distance(int a, int b) const {
  return std::abs(x_of(a) - x_of(b)) + std::abs(y_of(a) - y_of(b));
}

bool is_capturable(int kind) {
  switch (kind) {
    case kCity:
    case kBase:
    case kAirport:
    case kPort:
    case kHq:
    case kLab:
    case kComtower:
      return true;
    default:
      return false;
  }
}

bool pays_income(int kind) {
  return kind == kCity || kind == kBase || kind == kAirport || kind == kPort || kind == kHq;
}

bool factory_builds(int kind, int move_type) {
  switch (kind) {
    case kBase:
      return move_type != kMoveAir && move_type != kMoveSea && move_type != kMoveLander;
    case kAirport:
      return move_type == kMoveAir;
    case kPort:
      return move_type == kMoveSea || move_type == kMoveLander;
    default:
      return false;
  }
}

bool build_map(uint32_t id, int width, int height, const int16_t* terrain_ids, MapData& out,
               std::string& error) {
  if (width <= 0 || height <= 0 || width > kMaxWidth || height > kMaxHeight) {
    error = "map " + std::to_string(id) + " is " + std::to_string(width) + "x" +
            std::to_string(height) + "; the engine supports up to " + std::to_string(kMaxWidth) +
            "x" + std::to_string(kMaxHeight);
    return false;
  }
  out = MapData{};
  out.id = id;
  out.width = width;
  out.height = height;
  out.n_tiles = width * height;
  std::memset(out.prop, kNoProp, sizeof out.prop);
  for (int t = 0; t < out.n_tiles; ++t) {
    const int tid = terrain_ids[t];
    if (tid < 0 || tid > kMaxTerrainId || !kTerrain[tid].valid) {
      error = "map " + std::to_string(id) + ": unknown terrain id " + std::to_string(tid);
      return false;
    }
    const TerrainSpec& spec = kTerrain[tid];
    out.terrain_id[t] = static_cast<uint16_t>(tid);
    out.kind[t] = spec.kind;
    out.stars[t] = spec.defense;
    for (int w = 0; w < kNumWeathers; ++w) {
      for (int m = 0; m < kNumMoveTypes; ++m) out.move_cost[w][m][t] = kMoveCost[w][spec.kind][m];
    }
    if (is_capturable(spec.kind) || spec.kind == kSeam || spec.kind == kSilo) {
      if (out.n_props == kMaxProps) {
        error = "map " + std::to_string(id) + " has more than " + std::to_string(kMaxProps) +
                " properties";
        return false;
      }
      const int p = out.n_props++;
      out.prop[t] = static_cast<uint8_t>(p);
      out.prop_tile[p] = static_cast<uint16_t>(t);
      out.prop_kind[p] = spec.kind;
      out.prop_country[p] = spec.country;
    }
    const int x = t % width, y = t / width;
    out.neighbor[t][0] = y > 0 ? static_cast<int16_t>(t - width) : kOffMap;
    out.neighbor[t][1] = x + 1 < width ? static_cast<int16_t>(t + 1) : kOffMap;
    out.neighbor[t][2] = y + 1 < height ? static_cast<int16_t>(t + width) : kOffMap;
    out.neighbor[t][3] = x > 0 ? static_cast<int16_t>(t - 1) : kOffMap;
  }
  int k = 0;
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      const int p = out.prop[out.tile(x, y)];
      if (p != kNoProp) out.prop_by_column[k++] = static_cast<uint8_t>(p);
    }
  }
  compute_factory_turns(out);
  return true;
}

namespace {

struct Registry {
  std::shared_mutex mu;
  std::unordered_map<uint32_t, std::unique_ptr<const MapData>> maps;
};

Registry& registry() {
  static Registry* r = new Registry;  // never destroyed: maps outlive static teardown
  return *r;
}

bool same_terrain(const MapData& a, const MapData& b) {
  return a.width == b.width && a.height == b.height &&
         std::memcmp(a.terrain_id, b.terrain_id, sizeof(uint16_t) * a.n_tiles) == 0;
}

}  // namespace

const MapData* register_map(const MapData& map, std::string& error) {
  Registry& r = registry();
  std::unique_lock lock(r.mu);
  auto it = r.maps.find(map.id);
  if (it != r.maps.end()) {
    if (!same_terrain(*it->second, map)) {
      error = "map " + std::to_string(map.id) + " is already registered with different terrain";
      return nullptr;
    }
    return it->second.get();
  }
  auto copy = std::make_unique<MapData>(map);
  const MapData* ptr = copy.get();
  r.maps.emplace(map.id, std::move(copy));
  return ptr;
}

const MapData* find_map(uint32_t id) {
  Registry& r = registry();
  std::shared_lock lock(r.mu);
  auto it = r.maps.find(id);
  return it == r.maps.end() ? nullptr : it->second.get();
}

}  // namespace awbw
