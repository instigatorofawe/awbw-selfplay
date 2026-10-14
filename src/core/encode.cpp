#include "core/encode.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/belief.h"
#include "core/combat.h"
#include "core/movegen.h"
#include "core/movement.h"

namespace awbw {

std::string global_name(int i) {
  static const char* const kSide[kGCo] = {
      "funds",      "income",      "cop_charge",      "scop_charge", "cop_cost",  "scop_cost",
      "cop_on",     "scop_on",     "power_uses",      "props",       "comtowers", "units",
      "army_value", "hidden_subs", "hidden_stealths", "hidden_value"};
  static const char* const kWeatherNames[kNumWeathers] = {"clear", "rain", "snow"};
  if (i < 0 || i >= kNumGlobals) return "";
  if (i < 2 * kSideGlobals) {
    const std::string who = i < kSideGlobals ? "me." : "opp.";
    const int j = i % kSideGlobals;
    if (j < kGCo) return who + kSide[j];
    const int co = j - kGCo;
    return who + "co." + (kCos[co].valid ? kCos[co].name : std::to_string(co));
  }
  if (i >= kGWeather && i < kGWeather + kNumWeathers)
    return std::string("weather.") + kWeatherNames[i - kGWeather];
  switch (i) {
    case kGDay:
      return "day";
    case kGWeatherMine:
      return "weather_mine";
    case kGWeatherTheirs:
      return "weather_theirs";
    case kGCaptureWinOn:
      return "capture_win_on";
    case kGCaptureWin:
      return "capture_win";
    case kGFundsPerProperty:
      return "funds_per_property";
    case kGMovedFirst:
      return "moved_first";
    default:
      return "";
  }
}

namespace {

uint8_t clamp_u8(int v, int hi) { return static_cast<uint8_t>(std::clamp(v, 0, hi)); }

}  // namespace

void encode(const MapData& m, const GameState& s, uint8_t* planes, int h, int w, float* g) {
  std::memset(planes, 0, sizeof(uint8_t) * kNumPlanes * h * w);
  std::fill(g, g + kNumGlobals, 0.0f);
  const int me = s.active;
  const int opp = 1 - me;
  auto cell = [&](int plane, int tile) -> uint8_t& {
    return planes[(static_cast<size_t>(plane) * h + m.y_of(tile)) * w + m.x_of(tile)];
  };
  auto side_of = [&](int owner) { return owner == me ? 0 : owner == opp ? 1 : -1; };

  // Board.
  int props[2] = {}, towers[2] = {}, paying[2] = {};
  for (int t = 0; t < m.n_tiles; ++t) {
    const int prop = m.prop[t];
    int side = -1;
    if (prop != kNoProp) {
      const int kind = m.prop_kind[prop];
      if (is_capturable(kind)) {
        side = side_of(s.prop_owner[prop]);
        cell(kPlanePoints, t) = clamp_u8(s.prop_points[prop], kPlanes[kPlanePoints].values - 1);
        if (side >= 0) {
          ++props[side];
          towers[side] += kind == kComtower;
          paying[side] += pays_income(kind);
        }
      } else if (kind == kSeam) {
        cell(kPlanePoints, t) = clamp_u8(s.prop_points[prop], kPlanes[kPlanePoints].values - 1);
      }
    }
    cell(kPlaneTerrain, t) =
        static_cast<uint8_t>(1 + terrain_kind(m, s, t) + kNumTerrainKinds * (side + 1));
  }

  // Units. The board shows mine and the opponent's that I can see, cargo with
  // its transport; the totals count every unit.
  int units[2] = {}, hidden_subs[2] = {}, hidden_stealths[2] = {};
  double value[2] = {}, hidden_value[2] = {};
  for (int i = 0; i < kMaxUnits; ++i) {
    const Unit& u = s.units[i];
    if (!u.alive()) continue;
    const bool carried = u.has(kCarried);
    const int side = side_of(u.owner);
    const double worth = kUnits[u.type].cost * display_mean_for(s, i, me) / 10;
    ++units[side];
    value[side] += worth;
    const bool seen = visible_to(m, s, carried ? u.carrier : i, 1 - u.owner);
    if (!seen) {
      hidden_subs[side] += u.type == kSub;
      hidden_stealths[side] += u.type == kStealth;
      hidden_value[side] += worth;
    }
    if (carried || (!seen && side == 1)) continue;
    cell(kPlaneUnit, u.tile) = static_cast<uint8_t>(1 + u.type + kNumUnitTypes * side);
    // A display I don't know shows as 0, with its distribution in the
    // probability planes.
    const bool shown = display_known_to(s, i, me);
    cell(kPlaneHp, u.tile) = shown ? static_cast<uint8_t>(displayed_hp(u.hp)) : 0;
    const uint8_t* w = shown ? s.hp_belief[i] : s.hp_hidden[i];
    int total = 0;
    for (int k = 0; k < kHpBins; ++k) total += w[k];
    for (int k = 0; k < kHpBins; ++k)
      cell(kPlaneHpProb + k, u.tile) = static_cast<uint8_t>((w[k] * 255 + total / 2) / total);
    cell(kPlaneFuel, u.tile) = clamp_u8(u.fuel, kPlanes[kPlaneFuel].values - 1);
    cell(kPlaneAmmo, u.tile) = clamp_u8(u.ammo, kPlanes[kPlaneAmmo].values - 1);
    cell(kPlaneFlags, u.tile) =
        static_cast<uint8_t>(u.flags & (kMoved | kHidden | kStunned | kCapturing));
    for (int c = 0; c < 2; ++c) {
      if (u.cargo[c] == kNoUnit) continue;
      const Unit& k = s.units[u.cargo[c]];
      cell(kPlaneCargo1 + c, u.tile) = static_cast<uint8_t>(1 + k.type);
      cell(kPlaneCargo1Hp + c, u.tile) =
          static_cast<uint8_t>(std::lround(hp_mean_for(s, u.cargo[c], me)));
    }
  }

  // Per-player globals.
  for (int side = 0; side < 2; ++side) {
    const Player& p = s.players[side == 0 ? me : opp];
    float* f = g + side * kSideGlobals;
    f[kGFunds] = p.funds / 10000.0f;
    // Powers are over by the time income is paid.
    f[kGIncome] = paying[side] *
                  (s.settings.funds_per_property + kCos[p.co].state[0].income_bonus) / 10000.0f;
    f[kGCopCharge] = p.cop_cost > 0 ? static_cast<float>(p.power) / p.cop_cost : 0.0f;
    f[kGScopCharge] = p.scop_cost > 0 ? static_cast<float>(p.power) / p.scop_cost : 0.0f;
    f[kGCopCost] = p.cop_cost / 100000.0f;
    f[kGScopCost] = p.scop_cost / 100000.0f;
    f[kGCopOn] = p.power_on == 1;
    f[kGScopOn] = p.power_on == 2;
    f[kGPowerUses] = p.power_uses / 10.0f;
    f[kGProps] = props[side] / 30.0f;
    f[kGComtowers] = towers[side] / 10.0f;
    f[kGUnits] = units[side] / 50.0f;
    f[kGArmyValue] = static_cast<float>(value[side] / 100000.0);
    f[kGHiddenSubs] = hidden_subs[side] / 10.0f;
    f[kGHiddenStealths] = hidden_stealths[side] / 10.0f;
    f[kGHiddenValue] = static_cast<float>(hidden_value[side] / 100000.0);
    f[kGCo + p.co] = 1.0f;
  }

  // Shared globals.
  g[kGDay] = s.day / 30.0f;
  g[kGWeather + s.weather] = 1.0f;
  g[kGWeatherMine] = s.weather_owner == me;
  g[kGWeatherTheirs] = s.weather_owner == opp;
  const bool capture_win = s.settings.capture_win < 1000;
  g[kGCaptureWinOn] = capture_win;
  g[kGCaptureWin] = capture_win ? s.settings.capture_win / 50.0f : 0.0f;
  g[kGFundsPerProperty] = s.settings.funds_per_property / 1000.0f;
  g[kGMovedFirst] = me == 0;
}

std::string selection_global_name(int i) {
  static const char* const kPhaseNames[4] = {"idle", "selected", "moved", "factory"};
  if (i >= kSelGPhase && i < kSelGPhase + 4) return std::string("phase.") + kPhaseNames[i];
  if (i >= kSelGUnitType && i < kNumSelectionGlobals)
    return std::string("picked.") + kUnits[i - kSelGUnitType].name;
  return "";
}

void encode_selection(const MapData& m, const GameState& s, const Selection& sel,
                      const uint16_t* ids, int n, uint8_t* planes, int h, int w, float* g,
                      uint8_t* nonspatial) {
  std::memset(planes, 0, sizeof(uint8_t) * kNumSelectionPlanes * h * w);
  std::fill(g, g + kNumSelectionGlobals, 0.0f);
  std::memset(nonspatial, 0, kNumNonSpatialIds);
  auto cell = [&](int plane, int tile) -> uint8_t& {
    return planes[(static_cast<size_t>(plane) * h + m.y_of(tile)) * w + m.x_of(tile)];
  };
  g[kSelGPhase + sel.phase] = 1.0f;
  switch (sel.phase) {
    case kPhaseSelected:
      cell(kSelPick, sel.tile) = 1;
      g[kSelGUnitType + s.units[sel.unit].type] = 1.0f;
      break;
    case kPhaseMoved:
      cell(kSelPick, s.units[sel.unit].tile) = 1;
      cell(kSelPick, sel.tile) |= 2;
      g[kSelGUnitType + s.units[sel.unit].type] = 1.0f;
      break;
    case kPhaseFactory:
      cell(kSelPick, sel.tile) = 4;
      break;
    default:
      break;
  }
  for (int i = 0; i < n; ++i) {
    const int id = ids[i];
    if (id >= kIdEndTurn) {
      nonspatial[id - kIdEndTurn] = 1;
    } else {
      cell(kSelLegal + id / kMaxTiles, id % kMaxTiles) = 1;
    }
  }
}

int move_group(int move_type) {
  switch (move_type) {
    case kMoveFoot:
    case kMoveBoot:
      return kGroupFoot;
    case kMoveAir:
      return kGroupAir;
    case kMoveSea:
    case kMoveLander:
      return kGroupNaval;
    default:
      return kGroupVehicle;
  }
}

void encode_movement(const MapData& m, const GameState& s, uint8_t* planes, int h, int w) {
  std::memset(planes, 0, sizeof(uint8_t) * kNumMovementPlanes * h * w);
  const int me = s.active;
  const int opp = 1 - me;
  auto cell = [&](int plane, int tile) -> uint8_t& {
    return planes[(static_cast<size_t>(plane) * h + m.y_of(tile)) * w + m.x_of(tile)];
  };

  // Each player's factories, and the nearest one's turns for a mover class.
  uint8_t factories[kMaxPlayers][kMaxProps];
  int n_factories[kMaxPlayers] = {};
  for (int p = 0; p < m.n_props; ++p) {
    const int owner = s.prop_owner[p];
    if (owner < kMaxPlayers &&
        (m.prop_kind[p] == kBase || m.prop_kind[p] == kAirport || m.prop_kind[p] == kPort))
      factories[owner][n_factories[owner]++] = static_cast<uint8_t>(p);
  }
  auto nearest = [&](int owner, int cls, int tile) {
    int best = kUnreachable;
    for (int j = 0; j < n_factories[owner]; ++j) {
      const int row = m.factory_row[factories[owner][j]][cls];
      if (row != kNoRow) best = std::min<int>(best, m.turns_row(row)[tile]);
    }
    return best;
  };

  // How each side moves at its next chance: mine as now, the opponent's once
  // their power and power weather are over.
  int weather[2];
  bool cost_one[2];
  for (int side = 0; side < 2; ++side) {
    const Player& p = s.players[side == 0 ? me : opp];
    const CoStateSpec& co = kCos[p.co].state[side == 0 ? p.power_on : 0];
    const int now = side == 1 && s.weather_owner == opp ? int{kClear} : s.weather;
    weather[side] = co.weather_as[now];
    cost_one[side] = co.move_cost_one && weather[side] != kSnow;
  }

  // Factory planes, and the units grouped by (side, movement type, points).
  struct Source {
    uint16_t key, tile;
    uint8_t delay;
  };
  Source src[kMaxUnits];
  int n = 0;
  for (int i = 0; i < kMaxUnits; ++i) {
    const Unit& u = s.units[i];
    if (!u.alive() || u.has(kCarried)) continue;
    const int side = u.owner == me ? 0 : 1;
    if (side == 1 && !visible_to(m, s, i, me)) continue;
    const int cls = kMovers.of_type[u.type];
    const int home = nearest(u.owner, cls, u.tile), away = nearest(1 - u.owner, cls, u.tile);
    if (home != kUnreachable) cell(kMovHome, u.tile) = clamp_u8(1 + home, 1 + kFactoryTurnsCap);
    if (away != kUnreachable) cell(kMovAway, u.tile) = clamp_u8(1 + away, 1 + kFactoryTurnsCap);
    const Player& p = s.players[u.owner];
    const int mp =
        side == 0 ? move_points(s, u) : kUnits[u.type].move + kCos[p.co].state[0].move[u.type];
    const bool stunned = u.has(kStunned);
    const int delay = side == 0 ? (u.has(kMoved) || stunned) + stunned : stunned;
    src[n++] = {
        static_cast<uint16_t>(side << 8 | kUnits[u.type].move_type << 4 | std::clamp(mp, 1, 15)),
        u.tile, static_cast<uint8_t>(delay)};
  }
  std::sort(src, src + n, [](const Source& a, const Source& b) { return a.key < b.key; });

  // Reach planes: one search per group.
  uint8_t cost[kMaxTiles], turns[kMaxTiles];
  uint16_t tiles[kMaxUnits];
  uint8_t delays[kMaxUnits];
  int cost_key = -1;
  for (int a = 0, b = 0; a < n; a = b) {
    const int key = src[a].key;
    for (b = a; b < n && src[b].key == key; ++b) {
      tiles[b - a] = src[b].tile;
      delays[b - a] = src[b].delay;
    }
    const int side = key >> 8, move_type = key >> 4 & 15, mp = key & 15;
    if ((key >> 4) != cost_key) {  // move_cost's, from the map's row and the broken seams
      cost_key = key >> 4;
      std::memcpy(cost, m.move_cost[weather[side]][move_type], m.n_tiles);
      for (int p = 0; p < m.n_props; ++p) {
        if (m.prop_kind[p] == kSeam && s.prop_points[p] == 0)
          cost[m.prop_tile[p]] = kMoveCost[weather[side]][kRubble][move_type];
      }
      if (cost_one[side]) {
        for (int t = 0; t < m.n_tiles; ++t) cost[t] = cost[t] != 0;
      }
    }
    turns_from(m, cost, mp, tiles, delays, b - a, kReachTurns, turns);
    const int plane = kMovReach + side * kNumMoveGroups + move_group(move_type);
    for (int t = 0; t < m.n_tiles; ++t) {
      if (turns[t] == kUnreachable) continue;
      uint8_t& v = cell(plane, t);
      if (v == 0 || turns[t] + 1 < v) v = static_cast<uint8_t>(turns[t] + 1);
    }
  }
}

void encode_decision(const MapData& m, const GameState& s, const Selection& sel,
                     const uint16_t* ids, int n, uint8_t* planes, int h, int w, float* globals,
                     uint8_t* nonspatial, bool movement) {
  uint8_t scratch[kNumNonSpatialIds];
  const size_t area = static_cast<size_t>(h) * w;
  encode(m, s, planes, h, w, globals);
  encode_selection(m, s, sel, ids, n, planes + kNumPlanes * area, h, w, globals + kNumGlobals,
                   nonspatial ? nonspatial : scratch);
  if (movement) encode_movement(m, s, planes + policy_planes(false) * area, h, w);
}

}  // namespace awbw
