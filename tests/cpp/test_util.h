#pragma once

// Helpers for building small maps and states in C++ tests and benchmarks.

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/actions.h"
#include "core/map.h"
#include "core/state.h"

namespace awbw::testing {

inline constexpr int kOrangeStar = 1;
inline constexpr int kBlueMoon = 2;

// One glyph per tile:
//   . plain   M mountain  F wood   R river  = road  ^ bridge
//   ~ sea     _ shoal     % reef   | pipe   S pipe seam  X missile silo
//   # neutral city  $ neutral base  & neutral comtower
//   Orange Star: C city  B base  A airport  P port  H HQ  T comtower  L lab
//   Blue Moon:   c city  b base  a airport  p port  h HQ  t comtower  l lab
inline int16_t terrain_for(char g) {
  switch (g) {
    case '.':
      return 1;
    case 'M':
      return 2;
    case 'F':
      return 3;
    case 'R':
      return 4;
    case '=':
      return 15;
    case '^':
      return 26;
    case '~':
      return 28;
    case '_':
      return 29;
    case '%':
      return 33;
    case '|':
      return 101;
    case 'S':
      return 113;
    case 'X':
      return 111;
    case '#':
      return 34;
    case '$':
      return 35;
    case '&':
      return 133;
    case 'C':
      return 38;
    case 'B':
      return 39;
    case 'A':
      return 40;
    case 'P':
      return 41;
    case 'H':
      return 42;
    case 'T':
      return 134;
    case 'c':
      return 43;
    case 'b':
      return 44;
    case 'a':
      return 45;
    case 'p':
      return 46;
    case 'h':
      return 47;
    case 't':
      return 129;
    case 'L':
      return 146;
    case 'l':
      return 140;
    default:
      throw std::invalid_argument(std::string("unknown map glyph ") + g);
  }
}

inline MapData make_map(const std::vector<std::string>& rows, uint32_t id = 1) {
  const int h = static_cast<int>(rows.size());
  const int w = h ? static_cast<int>(rows[0].size()) : 0;
  std::vector<int16_t> ids;
  for (const std::string& row : rows) {
    if (static_cast<int>(row.size()) != w) throw std::invalid_argument("ragged map");
    for (char g : row) ids.push_back(terrain_for(g));
  }
  MapData m;
  std::string error;
  if (!build_map(id, w, h, ids.data(), m, error)) throw std::invalid_argument(error);
  return m;
}

// A state on `m` with Orange Star (player 0) against Blue Moon (player 1),
// owning the properties the map gives their countries.
inline GameState make_state(const MapData& m, int co0 = 1, int co1 = 1) {
  GameState s;
  init_state(m, s);
  const int cos[2] = {co0, co1};
  const int countries[2] = {kOrangeStar, kBlueMoon};
  for (int p = 0; p < kMaxPlayers; ++p) {
    s.players[p].co = static_cast<uint8_t>(cos[p]);
    s.players[p].country = static_cast<uint8_t>(countries[p]);
  }
  for (int p = 0; p < m.n_props; ++p) {
    for (int pl = 0; pl < kMaxPlayers; ++pl) {
      if (m.prop_country[p] == countries[pl]) s.prop_owner[p] = static_cast<uint8_t>(pl);
    }
  }
  return s;
}

// Add a full-health, fully supplied unit; returns its slot.
inline int put(GameState& s, const MapData& m, int type, int owner, int x, int y, int hp = 100) {
  Unit u{};
  u.tile = static_cast<uint16_t>(m.tile(x, y));
  u.type = static_cast<uint8_t>(type);
  u.owner = static_cast<uint8_t>(owner);
  u.hp = static_cast<uint8_t>(hp);
  u.fuel = kUnits[type].fuel;
  u.ammo = kUnits[type].ammo;
  const int slot = add_unit(s, u);
  if (slot < 0) throw std::runtime_error("put: tile taken or no free slot");
  return slot;
}

// --- Random games ------------------------------------------------------------

// Glyphs as in test_util.h: Orange Star's properties at the top left, Blue
// Moon's at the bottom right, with sea, shoals, pipes and seams, and silos.
inline const std::vector<std::string> kArena = {
    "HB.#..F..~~~",  //
    "BA=====.._~~",  //
    ".C.M.X.F._P~",  //
    "..F.|S|..__~",  //
    "#..R^R..#.%~",  //
    "~%..R...F.~~",  //
    "~_..#.X.M..c",  //
    "~p_..|S|.F..",  //
    "~~_..=====ab",  //
    "~~~..F..#.bh",  //
};

struct Placed {
  int type, owner, x, y;
  int cargo = -1;  // UnitType loaded into it
  bool hidden = false;
};

inline const Placed kArmies[] = {
    {kInfantry, 0, 2, 2},
    {kInfantry, 0, 4, 6},
    {kMech, 0, 4, 2},
    {kTank, 0, 7, 3},
    {kArtillery, 0, 3, 3},
    {kApc, 0, 2, 4, kInfantry},
    {kLander, 0, 9, 2, kTank},
    {kBlackBoat, 0, 10, 3, kInfantry},
    {kSub, 0, 11, 3, -1, true},
    {kCruiser, 0, 11, 4},
    {kTCopter, 0, 8, 1, kMech},
    {kFighter, 0, 7, 5},
    {kStealth, 0, 5, 5, -1, true},
    {kBCopter, 0, 9, 4},
    {kBlackBomb, 0, 8, 3},
    {kPiperunner, 0, 4, 3},
    {kInfantry, 1, 9, 6},
    {kInfantry, 1, 8, 7},
    {kMech, 1, 6, 5},
    {kTank, 1, 5, 6},
    {kArtillery, 1, 7, 6},
    {kMdTank, 1, 9, 9},
    {kApc, 1, 3, 7, kInfantry},
    {kAntiAir, 1, 4, 8},
    {kMissile, 1, 7, 8},
    {kLander, 1, 2, 7, kApc},
    {kBlackBoat, 1, 2, 8},
    {kSub, 1, 0, 6, -1, true},
    {kBattleship, 1, 0, 8},
    {kCarrier, 1, 1, 9, kFighter},
    {kTCopter, 1, 3, 9, kInfantry},
    {kStealth, 1, 10, 6, -1, true},
    {kBCopter, 1, 10, 7},
    {kPiperunner, 1, 5, 7},
};

inline GameState arena_start(const MapData& m, std::mt19937& rng) {
  std::vector<int> cos;
  for (int co = 0; co <= kMaxCoId; ++co) {
    if (kCos[co].valid) cos.push_back(co);
  }
  GameState s = make_state(m, cos[rng() % cos.size()], cos[rng() % cos.size()]);
  s.settings.day_limit = 5;
  if (rng() % 2) s.settings.unit_limit = 22;
  s.rng[0] = rng();
  s.rng[1] = rng() | 1;
  for (int p = 0; p < kMaxPlayers; ++p) {
    Player& pl = s.players[p];
    pl.funds = static_cast<int32_t>(rng() % 40000);
    pl.cop_cost = cop_cost(pl.co, 0);
    pl.scop_cost = scop_cost(pl.co, 0);
    pl.power = static_cast<int32_t>(rng() % (std::max(pl.cop_cost, pl.scop_cost) + 1));
  }
  for (const Placed& p : kArmies) {
    const int slot = put(s, m, p.type, p.owner, p.x, p.y, 10 + static_cast<int>(rng() % 91));
    if (p.hidden) s.units[slot].flags |= kHidden;
    if (p.cargo >= 0) {
      const int c = put(s, m, p.cargo, p.owner, 0, 2, 10 + static_cast<int>(rng() % 91));
      load_unit(s, c, slot);
    }
  }
  return s;
}

}  // namespace awbw::testing
