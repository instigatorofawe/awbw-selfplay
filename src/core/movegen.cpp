#include "core/movegen.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

namespace awbw {

int move_points(const GameState& s, const Unit& u) {
  const Player& p = s.players[u.owner];
  return kUnits[u.type].move + kCos[p.co].state[p.power_on].move[u.type];
}

int move_budget(const GameState& s, const Unit& u) {
  if (u.has(kMoved) || u.has(kStunned) || u.has(kCarried)) return 0;
  return std::min<int>(move_points(s, u), u.fuel);
}

bool visible_to(const MapData& m, const GameState& s, int slot, int viewer) {
  const Unit& u = s.units[slot];
  if (u.owner == viewer || !u.has(kHidden)) return true;
  for (int d = 0; d < kNumDirs; ++d) {
    const int nb = m.neighbor[u.tile][d];
    if (nb != kOffMap && s.unit_at[nb] != kNoUnit && s.units[s.unit_at[nb]].owner == viewer)
      return true;
  }
  return false;
}

namespace {

// What a unit's step costs depend on, worked out once per search.
struct Mover {
  const Unit& u;
  int move_type;
  int weather;
  bool cost_one;  // Sturm, and Lash's powers: every move costs 1, except in snow

  Mover(const GameState& s, const Unit& unit)
      : u(unit), move_type(kUnits[unit.type].move_type), weather(weather_for(s, unit.owner)) {
    const Player& p = s.players[unit.owner];
    cost_one = kCos[p.co].state[p.power_on].move_cost_one && weather != kSnow;
  }
};

int enter_cost(const MapData& m, const GameState& s, const Mover& mv, int tile) {
  const int cost = move_cost(m, s, tile, mv.move_type, mv.weather);
  if (cost == 0) return 0;
  const int other = s.unit_at[tile];
  if (other != kNoUnit && s.units[other].owner != mv.u.owner && visible_to(m, s, other, mv.u.owner))
    return 0;
  return mv.cost_one ? 1 : cost;
}

}  // namespace

int step_cost(const MapData& m, const GameState& s, const Unit& u, int tile) {
  return enter_cost(m, s, Mover(s, u), tile);
}

void reachable(const MapData& m, const GameState& s, int slot, Reach& r) {
  const Unit& u = s.units[slot];
  const Mover mv(s, u);
  std::memset(r.cost, kUnreachable, m.n_tiles);
  r.n = 0;
  const int budget = move_budget(s, u);
  r.cost[u.tile] = 0;
  r.prev[u.tile] = -1;
  if (budget == 0) {
    r.tiles[r.n++] = u.tile;
    return;
  }
  // Dijkstra; entries are cost << 16 | tile. A tile is pushed only when its
  // cost strictly drops, so exactly one entry per tile is current.
  thread_local std::vector<uint32_t> heap;
  heap.clear();
  heap.push_back(u.tile);
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), std::greater<>());
    const uint32_t e = heap.back();
    heap.pop_back();
    const int t = e & 0xFFFF, c = static_cast<int>(e >> 16);
    if (c != r.cost[t]) continue;
    r.tiles[r.n++] = static_cast<uint16_t>(t);
    for (int d = 0; d < kNumDirs; ++d) {
      const int nb = m.neighbor[t][d];
      if (nb == kOffMap) continue;
      const int w = enter_cost(m, s, mv, nb);
      if (w == 0) continue;
      const int nc = c + w;
      if (nc > budget || nc >= r.cost[nb]) continue;
      r.cost[nb] = static_cast<uint8_t>(nc);
      r.prev[nb] = static_cast<int16_t>(t);
      heap.push_back(static_cast<uint32_t>(nc) << 16 | static_cast<uint32_t>(nb));
      std::push_heap(heap.begin(), heap.end(), std::greater<>());
    }
  }
}

int path_cost(const MapData& m, const GameState& s, int slot, const uint16_t* path, int n) {
  const Unit& u = s.units[slot];
  if (n < 1 || path[0] != u.tile) return -1;
  const Mover mv(s, u);
  int total = 0;
  for (int i = 1; i < n; ++i) {
    if (path[i] >= m.n_tiles || m.distance(path[i - 1], path[i]) != 1) return -1;
    const int w = enter_cost(m, s, mv, path[i]);
    if (w == 0) return -1;
    total += w;
  }
  return total <= move_budget(s, u) || n == 1 ? total : -1;
}

}  // namespace awbw
