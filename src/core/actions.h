#pragma once

#include <cstdint>

#include "core/combat.h"
#include "core/map.h"
#include "core/state.h"

namespace awbw {

// Power meter per star. AWBW's meter counts 10 per funds of damage value.
inline constexpr int kStarValue = 90000;
// Each power use raises both power costs by a fifth of their base, up to this
// many uses.
inline constexpr int kMaxCostedPowerUses = 10;
// Longest path a move can take (tiles, the start included).
inline constexpr int kMaxPath = 32;

enum ActionKind : uint8_t {
  // Unit actions: `unit` moves to `to` (its own tile to stay put), then acts.
  kActWait,
  kActFire,        // attack the unit on `target`
  kActCapture,     // capture the property on `to`
  kActLoad,        // board the transport on `to`
  kActJoin,        // merge into the damaged unit of the same type on `to`
  kActUnload,      // drop cargo `arg` (0 or 1) on `target`; in place, allowed even after acting
  kActSupply,      // APC: resupply adjacent units
  kActRepair,      // Black Boat: repair the unit on `target`
  kActHide,        // Sub dives, Stealth hides
  kActUnhide,      // Sub surfaces, Stealth appears
  kActAttackSeam,  // attack the pipe seam on `target`
  kActExplode,     // Black Bomb: 5 HP to every other unit within 3 tiles
  kActLaunch,      // fire the loaded missile silo on `to` at `target`: 3 HP within 2 tiles
  // Player actions.
  kActBuild,   // build UnitType `arg` on the property at `target`
  kActPower,   // arg 1: COP, 2: SCOP; immediate effects (heals, damage, weather, ...) included
  kActDelete,  // scrap `unit`, which hasn't moved
  kActEndTurn,
  kActResign,
  kNumActionKinds,
};

struct Action {
  uint8_t kind = kActEndTurn;  // ActionKind
  uint8_t unit = kNoUnit;      // acting unit's slot
  uint16_t to = 0;             // tile the unit moves to
  uint16_t target = 0;         // tile acted on
  uint8_t arg = 0;
  uint8_t _pad = 0;
};
static_assert(sizeof(Action) == 8);

// What the caller decides instead of the engine. By default a unit takes the
// cheapest path its owner can see, and combat luck is drawn from the state's
// RNG.
struct Detail {
  const uint16_t* path = nullptr;  // path[0] = the unit's tile, path[n - 1] = Action::to
  int path_len = 0;
  int att_hp = -1;  // Fire: attacker and defender HP after the fight; -1 = roll
  int def_hp = -1;
  // Power: missile centres (Rachel's three, Sturm's and Von Bolt's one);
  // nullptr = aim them as AWBW does.
  const uint16_t* missiles = nullptr;
  int n_missiles = 0;
};

enum Status : uint8_t {
  kOk,
  kTrapped,  // applied, but the unit ran into an unseen enemy and stopped; its action is lost
  kErrGameOver,
  kErrBadAction,    // unknown kind or argument
  kErrNotYourUnit,  // no unit in that slot, or not the active player's
  kErrCannotAct,    // the unit has moved, is stunned or is carried
  kErrBadPath,      // path not legal or not ending on `to`
  kErrUnreachable,  // `to` is out of reach
  kErrOccupied,     // `to` is taken
  kErrBadTarget,    // nothing suitable on `target`
  kErrCannotFire,   // out of range, no weapon, can't see the target, or moved an indirect unit
  kErrCannotCapture,
  kErrCannotLoad,
  kErrCannotJoin,
  kErrCannotUnload,
  kErrCannotSupply,
  kErrCannotRepair,
  kErrCannotHide,
  kErrCannotBuild,
  kErrFunds,
  kErrUnitLimit,
  kErrPower,       // power unavailable, already on, or meter short
  kErrBadOutcome,  // forced combat result the fight can't produce
  kNumStatuses,
};
const char* status_name(Status s);
inline bool applied(Status s) { return s == kOk || s == kTrapped; }

// Carry out one action for the active player. On an error the state is left
// unchanged. Ending the turn runs the next player's start of turn: power
// ends, income, repairs and resupply on properties, APC resupply, daily fuel
// and crashes.
Status apply(const MapData& m, GameState& s, const Action& a, const Detail& d = {});

// The fight as the damage formula sees it, for the unit in `slot` standing on `tile`.
Combatant combatant(const MapData& m, const GameState& s, int slot, int tile);

// The results Fire action `a` can have (combat_outcomes for the unit at its
// destination), for search's chance nodes; Detail::att_hp / def_hp force one.
// False, with `out` empty, if `a` isn't a Fire, its move runs into an enemy
// the mover can't see (nothing is fought), or there is nothing to fight.
bool fire_outcomes(const MapData& m, const GameState& s, const Action& a,
                   std::vector<Outcome>& out);

// Where the active player's power `level` (1 COP, 2 SCOP) would aim its
// missiles now, as AWBW does: writes up to 3 tiles (kNoTile if there is no
// target) and returns how many missiles the power fires.
inline constexpr uint16_t kNoTile = 0xFFFF;
int missile_targets(const MapData& m, const GameState& s, int level, uint16_t* out);

// Whether the active player can build a unit of `type` on `tile` now: an
// empty factory of theirs that makes it, the type not banned, room under the
// unit limit, and the funds.
bool can_build(const MapData& m, const GameState& s, int tile, int type);
// Whether the active player can use power `level` (1 COP, 2 SCOP) now.
bool can_use_power(const GameState& s, int level);
// Whether the terrain on `tile` lets the unit in `slot` move there (weather
// included). Cargo is unloaded only onto such a tile, from a transport
// standing on one.
bool passable(const MapData& m, const GameState& s, int slot, int tile);

// Power costs with `uses` powers used so far (AWBW co_max_power / co_max_spower).
// A CO without a COP reports the SCOP cost for both, as AWBW does.
int cop_cost(int co, int uses);
int scop_cost(int co, int uses);

}  // namespace awbw
