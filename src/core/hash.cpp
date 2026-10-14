#include "core/hash.h"

#include <cstddef>
#include <cstring>

#include "core/belief.h"

namespace awbw {
namespace {

__extension__ typedef unsigned __int128 u128;

// wyhash's constants and mixer: the 128-bit product of two words, folded.
constexpr uint64_t kP0 = 0xa0761d6478bd642full;
constexpr uint64_t kP1 = 0xe7037ed1a0b428dbull;
constexpr uint64_t kP2 = 0x8ebc6af09c88c6e3ull;
constexpr uint64_t kP3 = 0x589965cc75374cc3ull;

inline uint64_t mum(uint64_t a, uint64_t b) {
  const u128 r = static_cast<u128>(a) * b;
  return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
}

// h with two more words. A chain's seed must not be kP1, which would cancel
// and let a zero `b` wipe out `a`.
inline uint64_t absorb(uint64_t h, uint64_t a, uint64_t b) { return mum(a ^ kP0, b ^ h ^ kP1); }

inline uint64_t load8(const uint8_t* p) {
  uint64_t w;
  std::memcpy(&w, p, sizeof w);
  return w;
}

static_assert(kHpBins == 10);
static_assert(offsetof(Unit, owner) == offsetof(Unit, type) + 1 &&
              offsetof(Unit, hp) == offsetof(Unit, type) + 2 &&
              offsetof(Unit, fuel) == offsetof(Unit, type) + 3 &&
              offsetof(Unit, ammo) == offsetof(Unit, type) + 4 &&
              offsetof(Unit, flags) == offsetof(Unit, type) + 5 &&
              offsetof(Unit, type) + 8 <= sizeof(Unit));
static_assert(kMaxProps % 16 == 0);

// A unit at `place` (its tile, or kCargo + its cargo slot): what it is and
// what both players know of its HP.
constexpr uint64_t kCargo = 0xF000;  // above every tile
static_assert(kMaxTiles < kCargo);

uint64_t unit_hash(const GameState& s, int slot, uint64_t place, bool hidden_hp, uint64_t h) {
  // type, owner, hp, fuel, ammo and flags in one load (the cargo slots after
  // them masked off).
  const uint64_t word = (load8(&s.units[slot].type) & 0xFFFF'FFFF'FFFFull) | place << 48;
  const uint8_t* belief = s.hp_belief[slot];
  uint64_t rest = uint64_t{belief[8]} | uint64_t{belief[9]} << 8, hidden = 0;
  if (hidden_hp) {  // otherwise hp_hidden isn't kept up to date
    const uint8_t* d = s.hp_hidden[slot];
    hidden = load8(d);
    rest |= uint64_t{d[8]} << 16 | uint64_t{d[9]} << 24;
  }
  h = absorb(h, word, load8(belief));
  return absorb(h, rest, hidden);
}

}  // namespace

uint64_t state_hash(const GameState& s) {
  // Three chains that don't wait on each other: the units, the players and
  // turn, the properties (two lanes).
  const bool hides[kMaxPlayers] = {hides_hp(s, 0), hides_hp(s, 1)};
  uint64_t units = 0;  // each uncarried unit with its cargo (by cargo slot), summed
  for (int i = 0; i < kMaxUnits; ++i) {
    const Unit& u = s.units[i];
    if (!u.alive() || u.has(kCarried)) continue;
    uint64_t h = unit_hash(s, i, u.tile, hides[u.owner], kP3);
    for (int k = 0; k < 2; ++k) {
      const int c = u.cargo[k];
      if (c != kNoUnit) h = unit_hash(s, c, kCargo + k, hides[s.units[c].owner], h);
    }
    units += h;
  }

  const Settings& st = s.settings;
  uint64_t h =
      absorb(kP2,
             uint64_t{s.map_id} | uint64_t{s.day} << 32 | uint64_t{s.active} << 48 |
                 uint64_t{s.weather} << 56,
             uint64_t{s.weather_owner} | uint64_t{s.winner} << 8 | uint64_t{st.unit_limit} << 16 |
                 uint64_t{st.use_powers} << 24 | uint64_t{st.banned_units} << 32);
  h = absorb(h,
             uint64_t{st.funds_per_property} | uint64_t{st.capture_win} << 16 |
                 uint64_t{st.day_limit} << 32,
             0);
  for (const Player& p : s.players) {
    h = absorb(
        h,
        uint64_t{static_cast<uint32_t>(p.funds)} | uint64_t{static_cast<uint32_t>(p.power)} << 32,
        uint64_t{static_cast<uint32_t>(p.cop_cost)} | uint64_t{static_cast<uint32_t>(p.scop_cost)}
                                                          << 32);
    h = absorb(h,
               uint64_t{p.co} | uint64_t{p.country} << 8 | uint64_t{p.power_on} << 16 |
                   uint64_t{p.power_uses} << 24 | uint64_t{p.eliminated} << 32 |
                   uint64_t{static_cast<uint16_t>(p.power_atk)} << 40,
               0);
  }
  uint64_t owners = kP3, points = kP0;
  for (int i = 0; i < kMaxProps; i += 16) {
    owners = absorb(owners, load8(s.prop_owner + i), load8(s.prop_owner + i + 8));
    points = absorb(points, load8(s.prop_points + i), load8(s.prop_points + i + 8));
  }
  return absorb(absorb(h, owners, points), units, kP2);
}

uint64_t state_hash(const GameState& s, const Selection& sel) {
  const uint64_t h = state_hash(s);
  if (sel.phase == kPhaseIdle) return h;
  const uint64_t unit_tile = sel.unit < kMaxUnits ? s.units[sel.unit].tile : 0xFFFF;
  return absorb(h, uint64_t{sel.phase} | unit_tile << 8 | uint64_t{sel.tile} << 24, kP3);
}

}  // namespace awbw
