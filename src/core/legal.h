#pragma once

#include <cstdint>
#include <vector>

#include "core/actions.h"

namespace awbw {

// Every complete action the active player can take now, Resign aside. Each
// one applies: with kOk, or with kTrapped when an enemy the player can't see
// is in the way. Unit actions name the destination (Action::to), and apply
// takes the cheapest path there. Action::target is the destination for kinds
// that act on no other tile. Empty once the game is over.
void legal_actions(const MapData& m, const GameState& s, std::vector<Action>& out);

// --- Micro-steps (PLAN.md §3) ----------------------------------------------
//
// The policy plays an action as a few steps it can score tile by tile: pick a
// unit (or factory), then the unit's destination, then what it does there
// (or the unit type to build). Nothing changes in the state until the last
// step, and a pick can't be undone. A step's id is channel * kMaxTiles + tile
// for the spatial channels, then the non-spatial ids below. Tiles are the
// map's (y * width + x).

enum Phase : uint8_t {
  kPhaseIdle,      // pick a unit or factory, scrap a unit, use a power or end the turn
  kPhaseSelected,  // pick where `unit` moves (its own tile to stay)
  kPhaseMoved,     // pick what `unit` does on `tile`
  kPhaseFactory,   // pick the unit type to build on `tile`
};

// The active player's action in progress.
struct Selection {
  uint8_t phase = kPhaseIdle;  // Phase
  uint8_t unit = kNoUnit;      // kPhaseSelected, kPhaseMoved
  uint16_t tile = 0;           // the unit's tile, its destination, or the factory
  bool operator==(const Selection&) const = default;
};
static_assert(sizeof(Selection) == 4);

enum Channel : uint8_t {
  // Idle.
  kChSelect,  // the unit on the tile, or the factory if there is none
  kChDelete,  // scrap the unit on the tile
  // Unit picked.
  kChMove,  // its destination
  // Unit moved: these act where it stands, which is the tile.
  kChWait,
  kChCapture,
  kChLoad,
  kChJoin,
  kChSupply,
  kChHide,
  kChUnhide,
  kChExplode,
  // Unit moved: the tile acted on.
  kChFire,
  kChAttackSeam,
  kChRepair,
  kChLaunch,
  kChUnload1,  // where the unit in cargo slot 1 goes
  kChUnload2,
  kNumChannels,
};
inline constexpr int kIdEndTurn = kNumChannels * kMaxTiles;
inline constexpr int kIdCop = kIdEndTurn + 1;
inline constexpr int kIdScop = kIdEndTurn + 2;
inline constexpr int kIdBuild = kIdEndTurn + 3;  // + UnitType, once a factory is picked
inline constexpr int kNumActionIds = kIdBuild + kNumUnitTypes;
static_assert(kNumActionIds <= 65536, "ids are stored as uint16_t");

constexpr int action_id(int channel, int tile) { return channel * kMaxTiles + tile; }

// Legal step ids from `sel`, in increasing order. Each leads on to at least
// one legal action, so the list is empty only once the game is over (or for
// a selection no legal step produced).
void legal_ids(const MapData& m, const GameState& s, const Selection& sel,
               std::vector<uint16_t>& out);

// If step `id` from `sel` completes an action, sets `out` to it and returns
// true (whether or not the step is legal). Picks return false.
bool step_action(const MapData& m, const GameState& s, const Selection& sel, int id, Action& out);

// Take step `id` from `sel`. A pick advances `sel` and returns kOk; a step
// that completes an action applies it with `d`, sets `sel` back to idle and
// returns apply's status. An id that isn't legal returns an error and
// changes nothing.
Status step(const MapData& m, GameState& s, Selection& sel, int id, const Detail& d = {});

// The steps that play a complete action: writes up to 3 ids to `out` and
// returns how many (0 for Resign, which has none).
int action_steps(const MapData& m, const GameState& s, const Action& a, uint16_t* out);

}  // namespace awbw
