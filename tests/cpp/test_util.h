#pragma once

// Helpers for building small maps and states in C++ tests and benchmarks.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/map.h"
#include "core/state.h"

namespace awbw::testing {

inline constexpr int kOrangeStar = 1;
inline constexpr int kBlueMoon = 2;

// One glyph per tile:
//   . plain   M mountain  F wood   R river  = road  ^ bridge
//   ~ sea     _ shoal     % reef   | pipe   S pipe seam  X missile silo
//   # neutral city  $ neutral base  & neutral comtower
//   Orange Star: C city  B base  A airport  P port  H HQ  T comtower
//   Blue Moon:   c city  b base  a airport  p port  h HQ  t comtower
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

}  // namespace awbw::testing
