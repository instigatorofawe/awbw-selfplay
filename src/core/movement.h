#pragma once

// Movement counted in turns, for the network's movement planes (encode.h):
// how many turns of moving a unit needs to reach a tile on an empty board,
// where no unit blocks and fuel never runs out.

#include <cstdint>

#include "core/map.h"

namespace awbw {

// The most turns turns_from counts: a label packs (turn, points spent) into
// 16 bits.
inline constexpr int kMaxTurns = 254;

// The fewest turns of moving that a mover with `mp` points a turn (1..15)
// needs to reach each tile from any of the n sources, entering tile t for
// cost[t] points (0: never). Points don't carry over: a step that doesn't fit
// in this turn's remaining points waits for the next turn, and a tile that
// costs more than mp is never entered. Source i starts moving after delay[i]
// turns. turns[t] is 0 at a source, k if the k-th turn of moving first gets
// there, and kUnreachable (movegen.h) past max_turns or never.
void turns_from(const MapData& m, const uint8_t* cost, int mp, const uint16_t* sources,
                const uint8_t* delay, int n, int max_turns, uint8_t* turns);

// Fill m.factory_row and m.factory_turns; build_map does, once the terrain
// and neighbours are in.
void compute_factory_turns(MapData& m);

}  // namespace awbw
