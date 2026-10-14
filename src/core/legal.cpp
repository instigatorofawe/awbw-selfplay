#include "core/legal.h"

#include <algorithm>
#include <cstdlib>

#include "core/combat.h"
#include "core/movegen.h"

namespace awbw {

namespace {

bool is_indirect(int type) { return kUnits[type].max_range > 1; }
int cargo_count(const Unit& t) { return (t.cargo[0] != kNoUnit) + (t.cargo[1] != kNoUnit); }

bool can_load(const Unit& u, const Unit& t) {
  return t.owner == u.owner && ((kUnits[t.type].carry_mask >> u.type) & 1) &&
         cargo_count(t) < kUnits[t.type].carry_slots && cargo_count(u) == 0;
}

bool can_join(const Unit& u, const Unit& t) {
  return t.owner == u.owner && t.type == u.type && displayed_hp(t.hp) < 10 && cargo_count(u) == 0;
}

// Call f(tile, dist) for every tile within `r` of `centre`, `centre` aside.
template <typename F>
void for_tiles_within(const MapData& m, int centre, int r, F&& f) {
  const int cx = m.x_of(centre), cy = m.y_of(centre);
  for (int dy = -r; dy <= r; ++dy) {
    const int y = cy + dy;
    if (y < 0 || y >= m.height) continue;
    const int reach = r - std::abs(dy);
    for (int dx = -reach; dx <= reach; ++dx) {
      const int x = cx + dx;
      if (x < 0 || x >= m.width || (dx == 0 && dy == 0)) continue;
      f(m.tile(x, y), std::abs(dx) + std::abs(dy));
    }
  }
}

// Every action the unit in `slot` can take standing on `dest`, a tile it can
// reach, passed to emit(action) until emit returns true. Returns whether it
// did. As in apply: the move is planned from the mover's view, and what it
// can attack is what it sees from `dest`.
template <typename F>
bool for_each_option(const MapData& m, const GameState& s, int slot, int dest, F&& emit) {
  const Unit& u = s.units[slot];
  const UnitSpec& spec = kUnits[u.type];
  const bool in_place = dest == u.tile;
  Action a;
  a.unit = static_cast<uint8_t>(slot);
  a.to = static_cast<uint16_t>(dest);
  auto put = [&](int kind, int target, int arg = 0) {
    a.kind = static_cast<uint8_t>(kind);
    a.target = static_cast<uint16_t>(target);
    a.arg = static_cast<uint8_t>(arg);
    return emit(static_cast<const Action&>(a));
  };

  // The mover sees one of its own units on `dest`, or nothing: an enemy it
  // sees blocks the way, and one it doesn't looks like an empty tile (moving
  // there traps it).
  int occ = s.unit_at[dest];
  if (occ == slot || (occ != kNoUnit && s.units[occ].owner != u.owner)) occ = kNoUnit;
  if (occ != kNoUnit) {
    if (u.has(kMoved)) return false;
    const Unit& t = s.units[occ];
    if (can_load(u, t) && put(kActLoad, dest)) return true;
    return can_join(u, t) && put(kActJoin, dest);
  }

  if (!u.has(kMoved)) {
    if (put(kActWait, dest)) return true;
    const int prop = m.prop[dest];
    if (spec.capture && prop != kNoProp && is_capturable(m.prop_kind[prop]) &&
        s.prop_owner[prop] != u.owner && put(kActCapture, dest))
      return true;

    // Direct units fire after moving, at what is next to them (which they
    // see); indirect units only from where they stand, at what they see.
    const bool direct = !is_indirect(u.type);
    if (direct || in_place) {
      const Combatant att = combatant(m, s, slot, dest);
      bool stop = false;
      for_tiles_within(m, dest, direct ? 1 : max_range(att), [&](int t, int dist) {
        if (stop) return;
        const int ti = s.unit_at[t];
        if (ti != kNoUnit && s.units[ti].owner != u.owner &&
            (dist == 1 || visible_to(m, s, ti, u.owner)) &&
            can_attack(att, combatant(m, s, ti, t), dist) && put(kActFire, t)) {
          stop = true;
          return;
        }
        // An air unit over a seam leaves the seam open to attack as well.
        const int p = m.prop[t];
        if (p != kNoProp && m.prop_kind[p] == kSeam && s.prop_points[p] > 0 &&
            can_attack(att, seam_target(), dist))
          stop = put(kActAttackSeam, t);
      });
      if (stop) return true;
    }

    if (spec.supplies || spec.repairs) {
      for (int dir = 0; dir < kNumDirs; ++dir) {
        const int nb = m.neighbor[dest][dir];
        if (nb == kOffMap) continue;
        const int k = s.unit_at[nb];
        if (k == kNoUnit || k == slot || s.units[k].owner != u.owner) continue;
        if (spec.repairs && put(kActRepair, nb)) return true;
        if (spec.supplies) {
          if (put(kActSupply, dest)) return true;
          break;  // one Supply for all neighbours
        }
      }
    }
    if (spec.hides && put(u.has(kHidden) ? kActUnhide : kActHide, dest)) return true;
    if (spec.explodes && put(kActExplode, dest)) return true;
    if (spec.capture && prop != kNoProp && m.prop_kind[prop] == kSilo && s.prop_points[prop] > 0) {
      for (int t = 0; t < m.n_tiles; ++t) {
        if (put(kActLaunch, t)) return true;
      }
    }
  }

  // Unloading in place is open even after the transport has acted.
  for (int i = 0; i < 2; ++i) {
    const int c = u.cargo[i];
    if (c == kNoUnit || !passable(m, s, c, dest)) continue;
    for (int dir = 0; dir < kNumDirs; ++dir) {
      const int nb = m.neighbor[dest][dir];
      if (nb == kOffMap) continue;
      // The tile the transport is leaving counts as free.
      const int k = s.unit_at[nb];
      if ((k == kNoUnit || (k == slot && !in_place)) && passable(m, s, c, nb) &&
          put(kActUnload, nb, i))
        return true;
    }
  }
  return false;
}

bool has_option(const MapData& m, const GameState& s, int slot, int dest) {
  return for_each_option(m, s, slot, dest, [](const Action&) { return true; });
}

// Whether the active player can pick the unit in `slot`: one of theirs, not
// carried or stunned, that can still act, or a transport that can unload
// where it stands.
bool selectable(const MapData& m, const GameState& s, int slot) {
  if (slot >= kMaxUnits) return false;
  const Unit& u = s.units[slot];
  if (!u.alive() || u.owner != s.active || u.has(kStunned) || u.has(kCarried)) return false;
  if (!u.has(kMoved)) return true;  // it can always wait where it is
  return cargo_count(u) > 0 && has_option(m, s, slot, u.tile);
}

// Whether `tile` is an empty factory of the active player's that can build
// something now.
bool factory_open(const MapData& m, const GameState& s, int tile) {
  if (m.prop[tile] == kNoProp || s.unit_at[tile] != kNoUnit) return false;
  for (int type = 0; type < kNumUnitTypes; ++type) {
    if (can_build(m, s, tile, type)) return true;
  }
  return false;
}

// Whether `dest` is a legal destination for the unit in `slot`, picked.
bool dest_ok(const MapData& m, const GameState& s, int slot, int dest) {
  if (dest < 0 || dest >= m.n_tiles) return false;
  const Unit& u = s.units[slot];
  if (dest != u.tile) {
    thread_local Reach r;
    reachable(m, s, slot, r);
    if (r.cost[dest] == kUnreachable) return false;
  }
  return has_option(m, s, slot, dest);
}

int channel_of(const Action& a) {
  switch (a.kind) {
    case kActWait:
      return kChWait;
    case kActFire:
      return kChFire;
    case kActCapture:
      return kChCapture;
    case kActLoad:
      return kChLoad;
    case kActJoin:
      return kChJoin;
    case kActUnload:
      return a.arg ? kChUnload2 : kChUnload1;
    case kActSupply:
      return kChSupply;
    case kActRepair:
      return kChRepair;
    case kActHide:
      return kChHide;
    case kActUnhide:
      return kChUnhide;
    case kActAttackSeam:
      return kChAttackSeam;
    case kActExplode:
      return kChExplode;
    case kActLaunch:
      return kChLaunch;
    default:
      return -1;
  }
}

bool targets_other_tile(int kind) {
  return kind == kActFire || kind == kActUnload || kind == kActRepair || kind == kActAttackSeam ||
         kind == kActLaunch;
}

// The action kind of a step after the unit moved, and its unload slot.
bool moved_kind(int channel, int& kind, int& arg) {
  static constexpr int8_t kKinds[kNumChannels] = {
      -1,         -1,         -1,         kActWait,    kActCapture, kActLoad,       kActJoin,
      kActSupply, kActHide,   kActUnhide, kActExplode, kActFire,    kActAttackSeam, kActRepair,
      kActLaunch, kActUnload, kActUnload,
  };
  if (channel < 0 || channel >= kNumChannels || kKinds[channel] < 0) return false;
  kind = kKinds[channel];
  arg = channel == kChUnload2;
  return true;
}

void idle_ids(const MapData& m, const GameState& s, std::vector<uint16_t>& out) {
  for (int i = 0; i < kMaxUnits; ++i) {
    const Unit& u = s.units[i];
    if (!u.alive() || u.owner != s.active || u.has(kCarried)) continue;
    if (selectable(m, s, i)) out.push_back(static_cast<uint16_t>(action_id(kChSelect, u.tile)));
    if (!u.has(kMoved)) out.push_back(static_cast<uint16_t>(action_id(kChDelete, u.tile)));
  }
  for (int p = 0; p < m.n_props; ++p) {
    const int tile = m.prop_tile[p];
    if (s.prop_owner[p] == s.active && factory_open(m, s, tile))
      out.push_back(static_cast<uint16_t>(action_id(kChSelect, tile)));
  }
  out.push_back(kIdEndTurn);
  if (can_use_power(s, 1)) out.push_back(kIdCop);
  if (can_use_power(s, 2)) out.push_back(kIdScop);
}

// Whether `sel` is a selection the steps could have produced now.
bool selection_ok(const MapData& m, const GameState& s, const Selection& sel) {
  switch (sel.phase) {
    case kPhaseIdle:
      return true;
    case kPhaseSelected:
      return selectable(m, s, sel.unit) && s.units[sel.unit].tile == sel.tile;
    case kPhaseMoved:
      return selectable(m, s, sel.unit) && dest_ok(m, s, sel.unit, sel.tile);
    case kPhaseFactory:
      return sel.tile < m.n_tiles && factory_open(m, s, sel.tile);
    default:
      return false;
  }
}

}  // namespace

void legal_actions(const MapData& m, const GameState& s, std::vector<Action>& out) {
  out.clear();
  if (s.winner != kNoOwner) return;
  auto push = [&](const Action& a) {
    out.push_back(a);
    return false;
  };
  thread_local Reach r;
  for (int i = 0; i < kMaxUnits; ++i) {
    if (!selectable(m, s, i)) continue;
    reachable(m, s, i, r);
    for (int k = 0; k < r.n; ++k) for_each_option(m, s, i, r.tiles[k], push);
  }
  Action a;
  for (int p = 0; p < m.n_props; ++p) {
    if (s.prop_owner[p] != s.active) continue;
    for (int type = 0; type < kNumUnitTypes; ++type) {
      if (!can_build(m, s, m.prop_tile[p], type)) continue;
      a.kind = kActBuild;
      a.target = m.prop_tile[p];
      a.arg = static_cast<uint8_t>(type);
      out.push_back(a);
    }
  }
  for (int level = 1; level <= 2; ++level) {
    if (!can_use_power(s, level)) continue;
    a = Action{};
    a.kind = kActPower;
    a.arg = static_cast<uint8_t>(level);
    out.push_back(a);
  }
  for (int i = 0; i < kMaxUnits; ++i) {
    const Unit& u = s.units[i];
    if (!u.alive() || u.owner != s.active || u.has(kCarried) || u.has(kMoved)) continue;
    a = Action{};
    a.kind = kActDelete;
    a.unit = static_cast<uint8_t>(i);
    a.to = u.tile;
    out.push_back(a);
  }
  out.push_back(Action{});  // EndTurn
}

void legal_ids(const MapData& m, const GameState& s, const Selection& sel,
               std::vector<uint16_t>& out) {
  out.clear();
  if (s.winner != kNoOwner || !selection_ok(m, s, sel)) return;
  switch (sel.phase) {
    case kPhaseIdle:
      idle_ids(m, s, out);
      break;
    case kPhaseSelected: {
      thread_local Reach r;
      reachable(m, s, sel.unit, r);
      for (int k = 0; k < r.n; ++k) {
        if (has_option(m, s, sel.unit, r.tiles[k]))
          out.push_back(static_cast<uint16_t>(action_id(kChMove, r.tiles[k])));
      }
      break;
    }
    case kPhaseMoved:
      for_each_option(m, s, sel.unit, sel.tile, [&](const Action& a) {
        out.push_back(static_cast<uint16_t>(action_id(channel_of(a), a.target)));
        return false;
      });
      break;
    case kPhaseFactory:
      for (int type = 0; type < kNumUnitTypes; ++type) {
        if (can_build(m, s, sel.tile, type)) out.push_back(static_cast<uint16_t>(kIdBuild + type));
      }
      break;
  }
  std::sort(out.begin(), out.end());
}

bool step_action(const MapData& m, const GameState& s, const Selection& sel, int id, Action& out) {
  out = Action{};
  if (id < 0 || id >= kNumActionIds) return false;
  const int ch = id / kMaxTiles, tile = id % kMaxTiles;
  switch (sel.phase) {
    case kPhaseIdle:
      if (id == kIdEndTurn) return true;
      if (id == kIdCop || id == kIdScop) {
        out.kind = kActPower;
        out.arg = id == kIdCop ? 1 : 2;
        return true;
      }
      if (ch == kChDelete && tile < m.n_tiles) {
        out.kind = kActDelete;
        out.unit = s.unit_at[tile];
        out.to = static_cast<uint16_t>(tile);
        return true;
      }
      return false;
    case kPhaseMoved: {
      int kind, arg;
      if (id >= kIdEndTurn || !moved_kind(ch, kind, arg)) return false;
      out.kind = static_cast<uint8_t>(kind);
      out.unit = sel.unit;
      out.to = sel.tile;
      out.target = static_cast<uint16_t>(tile);
      out.arg = static_cast<uint8_t>(arg);
      return true;
    }
    case kPhaseFactory:
      if (id < kIdBuild) return false;
      out.kind = kActBuild;
      out.target = sel.tile;
      out.arg = static_cast<uint8_t>(id - kIdBuild);
      return true;
    default:
      return false;
  }
}

Status step(const MapData& m, GameState& s, Selection& sel, int id, const Detail& d) {
  if (s.winner != kNoOwner) return kErrGameOver;
  if (!selection_ok(m, s, sel)) return kErrBadAction;
  Action a;
  if (step_action(m, s, sel, id, a)) {
    if (sel.phase == kPhaseMoved) {
      // apply alone would take any action where the move is trapped.
      const bool listed = for_each_option(m, s, sel.unit, sel.tile, [&](const Action& o) {
        return o.kind == a.kind && o.target == a.target && o.arg == a.arg;
      });
      if (!listed) return kErrBadAction;
    }
    const Status st = apply(m, s, a, d);
    if (applied(st)) sel = Selection{};
    return st;
  }
  if (id >= kIdEndTurn) return kErrBadAction;
  const int ch = id / kMaxTiles, tile = id % kMaxTiles;
  if (tile >= m.n_tiles) return kErrBadAction;
  if (sel.phase == kPhaseIdle && ch == kChSelect) {
    const int occ = s.unit_at[tile];
    if (occ != kNoUnit && s.units[occ].owner == s.active) {
      if (!selectable(m, s, occ)) return kErrCannotAct;
      sel = Selection{kPhaseSelected, static_cast<uint8_t>(occ), static_cast<uint16_t>(tile)};
      return kOk;
    }
    if (!factory_open(m, s, tile)) return kErrCannotBuild;
    sel = Selection{kPhaseFactory, kNoUnit, static_cast<uint16_t>(tile)};
    return kOk;
  }
  if (sel.phase == kPhaseSelected && ch == kChMove) {
    if (!dest_ok(m, s, sel.unit, tile)) return kErrUnreachable;
    sel = Selection{kPhaseMoved, sel.unit, static_cast<uint16_t>(tile)};
    return kOk;
  }
  return kErrBadAction;
}

int action_steps(const MapData& m, const GameState& s, const Action& a, uint16_t* out) {
  switch (a.kind) {
    case kActBuild:
      out[0] = static_cast<uint16_t>(action_id(kChSelect, a.target));
      out[1] = static_cast<uint16_t>(kIdBuild + a.arg);
      return 2;
    case kActPower:
      out[0] = static_cast<uint16_t>(a.arg == 2 ? kIdScop : kIdCop);
      return 1;
    case kActDelete:
      out[0] = static_cast<uint16_t>(action_id(kChDelete, s.units[a.unit].tile));
      return 1;
    case kActEndTurn:
      out[0] = kIdEndTurn;
      return 1;
    case kActResign:
      return 0;
    default: {
      (void)m;
      out[0] = static_cast<uint16_t>(action_id(kChSelect, s.units[a.unit].tile));
      out[1] = static_cast<uint16_t>(action_id(kChMove, a.to));
      out[2] = static_cast<uint16_t>(
          action_id(channel_of(a), targets_other_tile(a.kind) ? a.target : a.to));
      return 3;
    }
  }
}

int selection_steps(const GameState& s, const Selection& sel, uint16_t* out) {
  switch (sel.phase) {
    case kPhaseSelected:
      out[0] = static_cast<uint16_t>(action_id(kChSelect, sel.tile));
      return 1;
    case kPhaseMoved:
      // Picks change nothing, so the unit is still where it was picked.
      out[0] = static_cast<uint16_t>(action_id(kChSelect, s.units[sel.unit].tile));
      out[1] = static_cast<uint16_t>(action_id(kChMove, sel.tile));
      return 2;
    case kPhaseFactory:
      out[0] = static_cast<uint16_t>(action_id(kChSelect, sel.tile));
      return 1;
    default:
      return 0;
  }
}

}  // namespace awbw
