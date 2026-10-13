#pragma once

#include <cstdint>
#include <string>
#include <type_traits>

#include "core/config.h"
#include "core/map.h"
#include "core/tables.h"

namespace awbw {

inline constexpr uint8_t kNoUnit = 0xFF;
inline constexpr uint8_t kNoOwner = 0xFF;
inline constexpr uint8_t kDraw = 2;  // GameState::winner
inline constexpr int kFullCapture = 20;
inline constexpr int kFullSeamHp = 99;
// Internal HPs that share a displayed HP (see GameState::hp_belief).
inline constexpr int kHpBins = 10;

enum UnitFlag : uint8_t {
  kMoved = 1,       // has acted this turn
  kHidden = 2,      // dived Sub or hidden Stealth
  kCarried = 4,     // inside a transport
  kStunned = 8,     // can't act this turn (Von Bolt)
  kCapturing = 16,  // captured this turn (AWBW's flag; progress lives on the property)
};

struct Unit {
  uint32_t id;    // creation order: AWBW unit id in replay states; 0 = free slot
  uint16_t tile;  // carried units: their transport's tile
  uint8_t type;   // UnitType
  uint8_t owner;  // player index
  uint8_t hp;     // internal HP 1..100; displayed is (hp + 9) / 10
  uint8_t fuel;
  uint8_t ammo;
  uint8_t flags;     // UnitFlag
  uint8_t cargo[2];  // slots of loaded units, or kNoUnit
  uint8_t carrier;   // slot of the transport carrying this unit, or kNoUnit
  uint8_t _pad;

  bool alive() const { return id != 0; }
  bool has(uint8_t f) const { return (flags & f) != 0; }
};
static_assert(sizeof(Unit) == 16);

struct Player {
  int32_t funds;
  int32_t power;      // power meter, in AWBW's units (co_power)
  int32_t cop_cost;   // meter needed for the COP (co_max_power); 0 = none
  int32_t scop_cost;  // meter needed for the SCOP (co_max_spower); 0 = none
  uint8_t co;         // AWBW CO id
  uint8_t country;    // AWBW country id
  uint8_t power_on;   // 0 none, 1 COP, 2 SCOP
  uint8_t power_uses;
  uint8_t eliminated;
  uint8_t _pad;
  int16_t power_atk;  // attack % fixed when a power starts (Colin's Power of Money)
};
static_assert(sizeof(Player) == 24);

// Game options that vary between AWBW games.
struct Settings {
  uint16_t funds_per_property = 1000;
  uint16_t capture_win = 1000;  // properties needed to win; AWBW's 1000 = off
  uint16_t day_limit = 0;       // 0 = none
  uint8_t unit_limit = 0;       // per player; 0 = none
  uint8_t use_powers = 1;
  uint32_t banned_units = 0;  // bit per UnitType
};

// Everything that changes during a game. A plain value: copy it with memcpy,
// store it as raw bytes. The map is named by id (see find_map).
struct GameState {
  uint32_t map_id;
  Settings settings;
  uint32_t next_id;  // id of the next unit built
  uint16_t day;
  uint8_t active;         // player index to move
  uint8_t weather;        // Weather
  uint8_t weather_owner;  // player whose power set the weather (it ends at their next turn), or
                          // kNoOwner
  uint8_t winner;         // kNoOwner while the game runs; kDraw
  uint8_t _pad[2];
  Player players[kMaxPlayers];
  Unit units[kMaxUnits];
  uint8_t unit_at[kMaxTiles];  // slot of the uncarried unit on each tile, or kNoUnit
  // Per map property (MapData::prop_*): owner player index or kNoOwner, and
  // points: capture points left for properties, HP for pipe seams (0 = broken
  // into rubble), 1/0 for a loaded/fired silo.
  uint8_t prop_owner[kMaxProps];
  uint8_t prop_points[kMaxProps];
  // What a player who watched every turn knows of each unit's internal HP:
  // players see only displayed HP, so hp_belief[slot][k] (0..255, the most
  // likely 255) weighs internal HP 10 * (displayed - 1) + 1 + k. Never 0 at
  // the unit's own HP. The engine keeps it up to date (core/belief.h).
  uint8_t hp_belief[kMaxUnits][kHpBins];
  uint64_t rng[2];
};
static_assert(std::is_trivially_copyable_v<GameState>);
static_assert(std::is_standard_layout_v<GameState>);

// An empty state on `map`: no units, properties owned by nobody.
void init_state(const MapData& map, GameState& s);

// Lowest free slot, or -1 if the state holds kMaxUnits units.
int alloc_unit(const GameState& s);
// Put a new uncarried unit on an empty tile, its HP known to both players.
// Returns its slot, or -1.
int add_unit(GameState& s, const Unit& u);
// Remove a unit and everything it carries.
void remove_unit(GameState& s, int slot);
// Move an uncarried unit (and its cargo) to an empty tile, or to its own.
void relocate_unit(GameState& s, int slot, int tile);
// Put an uncarried unit into a transport's first free cargo slot. Cargo
// slots don't shift when a unit unloads, as in AWBW.
void load_unit(GameState& s, int slot, int transport);
// Take cargo out of its transport onto an empty tile.
void unload_unit(GameState& s, int slot, int tile);

// Terrain kind / defense stars / move cost in `weather` of a tile now: pipe
// seams break into rubble and silos empty when fired.
int terrain_kind(const MapData& m, const GameState& s, int tile);
int terrain_stars(const MapData& m, const GameState& s, int tile);
int move_cost(const MapData& m, const GameState& s, int tile, int move_type, int weather);
// The weather `player`'s units move in (Olaf is unaffected by snow and moves
// in rain as in snow; Drake is unaffected by rain).
int weather_for(const GameState& s, int player);

int count_units(const GameState& s, int owner);
int count_props(const MapData& m, const GameState& s, int owner, int kind = -1);

// Consistency checks (occupancy grid, cargo links, ranges). Returns "" if
// the state is sound, else what is wrong.
std::string check_state(const MapData& m, const GameState& s);

}  // namespace awbw
