#pragma once

// 64-bit keys for transposition tables (search, PLAN §8).
//
// Two states get the same key when they play the same from here on, whatever
// order of actions reached them. The key covers every unit's tile, type,
// owner, HP, fuel, ammo, flags and cargo (by cargo slot, since Unload1 and
// Unload2 differ), the HP beliefs, the properties, the players, the turn and
// the settings. It leaves out what no rule reads: which slot a unit is in,
// unit ids (a rule that ever orders units by id must bring them in), the next
// id, the RNG, unit_at (it follows from the units) and the bytes of free
// slots. Units are hashed one by one and summed, so their order doesn't count
// either.
//
// It is computed from scratch rather than kept up to date move by move (a
// Zobrist hash): about 0.1 us for 59 units, against 2 us to encode the state
// for the network. Unlike absl::Hash, which is seeded per process, the value
// is the same on every run, so it can be stored and compared across
// processes.

#include <cstdint>

#include "core/legal.h"
#include "core/state.h"

namespace awbw {

uint64_t state_hash(const GameState& s);

// A decision in progress: the state's key with the selection's (its unit by
// tile, not slot). An idle selection gives state_hash(s).
uint64_t state_hash(const GameState& s, const Selection& sel);

}  // namespace awbw
