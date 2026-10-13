#pragma once

#include <cstdint>

#include "core/map.h"
#include "core/state.h"

namespace awbw {

inline constexpr uint8_t kUnreachable = 0xFF;

// Cheapest paths from one unit's tile. Covers every tile the unit can move
// through, including ones it can't stop on (friendly units).
struct Reach {
  int n = 0;
  uint16_t tiles[kMaxTiles];  // reached tiles in order of cost, the start first
  uint8_t cost[kMaxTiles];    // path cost, or kUnreachable
  int16_t prev[kMaxTiles];    // previous tile on a cheapest path; -1 at the start
};

// Movement points with CO bonuses, before the fuel cap.
int move_points(const GameState& s, const Unit& u);
// Movement points the unit can spend now: 0 if it can't move this turn.
int move_budget(const GameState& s, const Unit& u);
// Whether `viewer` can see the unit in slot `slot`. Dived Subs and hidden
// Stealths are invisible to the enemy, even without fog, unless next to one of
// the enemy's units or on one of the enemy's properties.
bool visible_to(const MapData& m, const GameState& s, int slot, int viewer);

// Cost of entering `tile` for the unit, 0 if it can't: terrain cost (1 under
// Sturm and Lash's powers), and enemy units its owner can see block. Moves are
// planned from the mover's view; an unseen hidden enemy on the way traps the
// unit when the move is carried out (see apply).
int step_cost(const MapData& m, const GameState& s, const Unit& u, int tile);

void reachable(const MapData& m, const GameState& s, int slot, Reach& r);

// Cost of moving along `path` (path[0] is the unit's tile), or -1 if the path
// is not a legal move from the mover's view: tiles not adjacent, impassable,
// blocked, or over budget.
int path_cost(const MapData& m, const GameState& s, int slot, const uint16_t* path, int n);

}  // namespace awbw
