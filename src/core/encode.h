#pragma once

#include <cstdint>
#include <string>

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
// load happened in plain sight.
//
// Bump kEncodeVersion whenever a plane or global changes meaning.
inline constexpr int kEncodeVersion = 2;

enum Plane : uint8_t {
  kPlaneTerrain,  // 0 off the board; else 1 + terrain kind now + kNumTerrainKinds * side,
                  // side 0 for terrain and unowned properties, 1 mine, 2 the opponent's
  kPlanePoints,   // capture points left on a capturable property (20 untouched), pipe seam HP
  kPlaneUnit,     // 0 none; 1 + type for mine, 1 + kNumUnitTypes + type for the opponent's
  kPlaneHp,       // internal HP 1..100
  kPlaneFuel,
  kPlaneAmmo,
  kPlaneFlags,   // UnitFlag bits: moved, hidden, stunned, capturing
  kPlaneCargo1,  // 0 empty; 1 + type of the unit in cargo slot 1
  kPlaneCargo2,
  kPlaneCargo1Hp,
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
    {"hp", 101, false},
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
  kGArmyValue,       // cost x displayed HP / 10 of those units, / 100,000
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

}  // namespace awbw
