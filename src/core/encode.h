#pragma once

#include <cstdint>
#include <string>

#include "core/legal.h"
#include "core/map.h"
#include "core/state.h"
#include "core/tables.h"

namespace awbw {

// Network input for one position, seen by the player to move ("me"): uint8
// planes over the board plus a float vector of global features. Owners are
// relative (me / opponent).
//
// What `me` knows is what a player who watched every turn knows. Enemy units
// `me` can't see (hidden Subs and Stealths, see visible_to) are left off the
// board, since where they went is unknown, but the unit count and army value
// include them, as AWBW's game page does: both players saw them built. Cargo
// shows with its transport: AWBW hides an enemy transport's cargo, but every
// load happened in plain sight. HP is what players see, the displayed HP, and
// the posterior over internal HP that watching every fight gives
// (GameState::hp_belief). Sonja's opponent doesn't see her units' displayed
// HP: one they can't tell shows as HP 0 with the probability planes over its
// display (GameState::hp_hidden), and the army value counts the expected
// display.
//
// Bump kEncodeVersion whenever a plane or global changes meaning.
inline constexpr int kEncodeVersion = 4;

enum Plane : uint8_t {
  kPlaneTerrain,  // 0 off the board; else 1 + terrain kind now + kNumTerrainKinds * side,
                  // side 0 for terrain and unowned properties, 1 mine, 2 the opponent's
  kPlanePoints,   // capture points left on a capturable property (20 untouched), pipe seam HP
  kPlaneUnit,     // 0 none; 1 + type for mine, 1 + kNumUnitTypes + type for the opponent's
  kPlaneHp,       // displayed HP 1..10, or 0 if I can't tell it (Sonja's units)
  kPlaneHpProb,   // kHpBins planes: 255 x the probability that the internal HP is
                  // 10 * (displayed - 1) + 1 + k, for plane kPlaneHpProb + k; for
                  // an HP of 0, that the displayed HP is k + 1
  kPlaneFuel = kPlaneHpProb + kHpBins,
  kPlaneAmmo,
  kPlaneFlags,   // UnitFlag bits: moved, hidden, stunned, capturing
  kPlaneCargo1,  // 0 empty; 1 + type of the unit in cargo slot 1
  kPlaneCargo2,
  kPlaneCargo1Hp,  // expected internal HP of the unit in cargo slot 1, rounded
  kPlaneCargo2Hp,
  kNumPlanes
};

struct PlaneSpec {
  const char* name;
  int values;        // values run 0 .. values - 1
  bool categorical;  // an index to embed; otherwise a quantity
};

inline constexpr PlaneSpec kPlanes[kNumPlanes] = {
    {"terrain", 1 + 3 * kNumTerrainKinds, true},
    {"points", 100, false},
    {"unit", 1 + 2 * kNumUnitTypes, true},
    {"hp", 11, false},
    {"hp_p0", 256, false},
    {"hp_p1", 256, false},
    {"hp_p2", 256, false},
    {"hp_p3", 256, false},
    {"hp_p4", 256, false},
    {"hp_p5", 256, false},
    {"hp_p6", 256, false},
    {"hp_p7", 256, false},
    {"hp_p8", 256, false},
    {"hp_p9", 256, false},
    {"fuel", 100, false},
    {"ammo", 10, false},
    {"flags", 32, true},
    {"cargo1", 1 + kNumUnitTypes, true},
    {"cargo2", 1 + kNumUnitTypes, true},
    {"cargo1_hp", 101, false},
    {"cargo2_hp", 101, false},
};

// Global features: kSideGlobals for me, the same again for the opponent, then
// the shared ones. Funds and costs are scaled to roughly 0..1.
enum SideGlobal : uint8_t {
  kGFunds,       // / 10,000
  kGIncome,      // funds paid at each turn start, / 10,000
  kGCopCharge,   // power meter / COP cost (0 if the CO has none)
  kGScopCharge,  // power meter / SCOP cost
  kGCopCost,     // / 100,000
  kGScopCost,
  kGCopOn,
  kGScopOn,
  kGPowerUses,       // / 10
  kGProps,           // capturable properties owned, / 30
  kGComtowers,       // / 10
  kGUnits,           // all units, hidden and carried ones included, / 50
  kGArmyValue,       // cost x displayed HP / 10 of those units (expected, if hidden), / 100,000
  kGHiddenSubs,      // Subs the other player can't see, / 10
  kGHiddenStealths,  // Stealths the other player can't see, / 10
  kGHiddenValue,     // army value of the units the other player can't see, / 100,000
  kGCo,              // one-hot AWBW CO id, kMaxCoId + 1 entries
  kSideGlobals = kGCo + kMaxCoId + 1
};

enum SharedGlobal : uint8_t {
  kGDay = 2 * kSideGlobals,                  // / 30
  kGWeather,                                 // one-hot Weather, kNumWeathers entries
  kGWeatherMine = kGWeather + kNumWeathers,  // the weather came from my power
  kGWeatherTheirs,
  kGCaptureWinOn,
  kGCaptureWin,        // properties needed to win, / 50; 0 when off
  kGFundsPerProperty,  // / 1,000
  kGMovedFirst,        // me is the player who moves first each day
  kNumGlobals
};

// Feature names, "me.funds", "opp.co.Andy", "weather.rain", ...
std::string global_name(int i);

// Encode `s` into planes [kNumPlanes][h][w] and globals [kNumGlobals]. The
// board sits at the top-left of the h x w window and every other cell is 0.
// Needs m.height <= h, m.width <= w, and a state that passes check_state.
void encode(const MapData& m, const GameState& s, uint8_t* planes, int h, int w, float* globals);

// --- The decision in progress (policy input) --------------------------------
//
// The policy network reads encode()'s output plus the micro-step decision
// being made (legal.h's Selection): the picked unit's or factory's tile, the
// tile the unit moves to, the phase, and where each channel's step is legal.
// The non-spatial ids' legality comes back on its own, for the mask over the
// policy logits; the spatial ids' mask is the legal planes.
//
// Bump kSelectionEncodeVersion whenever one of these changes meaning.
inline constexpr int kSelectionEncodeVersion = 1;

enum SelectionPlane : uint8_t {
  kSelPick,   // 0; 1 the picked unit's tile, 2 where it moves, 3 both (it stays), 4 the factory
  kSelLegal,  // kNumChannels planes: 1 where the channel's step is legal (legal.h's Channel)
  kNumSelectionPlanes = kSelLegal + kNumChannels,
};

inline constexpr PlaneSpec kSelectionPlanes[kNumSelectionPlanes] = {
    {"pick", 5, true},          {"legal.select", 2, false},      {"legal.delete", 2, false},
    {"legal.move", 2, false},   {"legal.wait", 2, false},        {"legal.capture", 2, false},
    {"legal.load", 2, false},   {"legal.join", 2, false},        {"legal.supply", 2, false},
    {"legal.hide", 2, false},   {"legal.unhide", 2, false},      {"legal.explode", 2, false},
    {"legal.fire", 2, false},   {"legal.attack_seam", 2, false}, {"legal.repair", 2, false},
    {"legal.launch", 2, false}, {"legal.unload1", 2, false},     {"legal.unload2", 2, false},
};

enum SelectionGlobal : uint8_t {
  kSelGPhase,                      // one-hot Phase
  kSelGUnitType = kSelGPhase + 4,  // one-hot type of the picked unit
  kNumSelectionGlobals = kSelGUnitType + kNumUnitTypes,
};

// The ids after the spatial ones: End Turn, COP, SCOP, then Build + type.
inline constexpr int kNumNonSpatialIds = kNumActionIds - kIdEndTurn;

// "phase.idle", "picked.Infantry", ...
std::string selection_global_name(int i);

// Encode the decision `sel`, from which legal_ids() gave the n ids `ids`,
// into planes [kNumSelectionPlanes][h][w] and globals [kNumSelectionGlobals],
// laid out as encode()'s, and the non-spatial ids' legality
// [kNumNonSpatialIds] as 0 / 1.
void encode_selection(const MapData& m, const GameState& s, const Selection& sel,
                      const uint16_t* ids, int n, uint8_t* planes, int h, int w, float* globals,
                      uint8_t* nonspatial);

// --- Movement (optional policy input) ----------------------------------------
//
// How many turns units need to get places, counted on an empty board where no
// unit blocks and fuel never runs out (movement.h). The network could work
// these out from the terrain and units, but only over many layers. Laid out
// as encode()'s.
//
// Factory planes: on each unit's tile, the turns its type needs from the
// nearest factory its owner holds that builds it, and from the nearest of the
// other side's, as precomputed per map (MapData::factory_turns): how far
// reinforcements are, and how overextended the unit is.
//
// Reach planes: per side and movement group, the fewest turns any of the
// side's units in the group needs to get to each tile. Turns count from the
// side's next chance to move: mine from this one (a unit that has moved
// starts next turn, a stunned one the turn after), the opponent's from their
// next turn (a stunned one the turn after). Each unit moves with its CO's
// bonuses and in the weather it will have then: mine as they are now, the
// opponent's once the powers and power weather that end at their turn start
// are over; Sturm's and Lash's moves cost 1, except in snow. Carried units and
// enemies I can't see are left out.
//
// Bump kMovementEncodeVersion whenever one of these changes meaning.
inline constexpr int kMovementEncodeVersion = 1;
inline constexpr int kFactoryTurnsCap = 15;
inline constexpr int kReachTurns = 3;

enum MoveGroup : uint8_t {
  kGroupFoot,     // Foot, Boot
  kGroupVehicle,  // Tread, Tire, Pipe
  kGroupAir,
  kGroupNaval,  // Sea, Lander
  kNumMoveGroups
};

enum MovementPlane : uint8_t {
  kMovHome,   // 0 none or out of reach; else 1 + the turns from its owner's nearest
              // factory, capped at kFactoryTurnsCap
  kMovAway,   // the same from the other side's factories
  kMovReach,  // 2 x kNumMoveGroups planes, mine then the opponent's, by MoveGroup: 1 + the
              // fewest turns (0 where one of the units stands), or 0 past kReachTurns
  kNumMovementPlanes = kMovReach + 2 * kNumMoveGroups,
};

inline constexpr PlaneSpec kMovementPlanes[kNumMovementPlanes] = {
    {"home", kFactoryTurnsCap + 2, true},      {"away", kFactoryTurnsCap + 2, true},
    {"reach.me.foot", kReachTurns + 2, true},  {"reach.me.vehicle", kReachTurns + 2, true},
    {"reach.me.air", kReachTurns + 2, true},   {"reach.me.naval", kReachTurns + 2, true},
    {"reach.opp.foot", kReachTurns + 2, true}, {"reach.opp.vehicle", kReachTurns + 2, true},
    {"reach.opp.air", kReachTurns + 2, true},  {"reach.opp.naval", kReachTurns + 2, true},
};

int move_group(int move_type);

// Encode the movement planes of `s` into planes [kNumMovementPlanes][h][w].
void encode_movement(const MapData& m, const GameState& s, uint8_t* planes, int h, int w);

// A policy's logits over an h x w window: the spatial ids at
// channel * h * w + y * w + x, then the non-spatial ones in order.
inline constexpr int policy_size(int h, int w) { return kNumChannels * h * w + kNumNonSpatialIds; }

// Where step `id` is among those logits.
inline int policy_index(const MapData& m, int id, int h, int w) {
  if (id >= kIdEndTurn) return kNumChannels * h * w + (id - kIdEndTurn);
  const int tile = id % kMaxTiles;
  return (id / kMaxTiles * h + m.y_of(tile)) * w + m.x_of(tile);
}

}  // namespace awbw
