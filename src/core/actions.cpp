#include "core/actions.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "core/movegen.h"

namespace awbw {

namespace {

constexpr int kCoAndy = 1;
constexpr int kCoDrake = 5;
constexpr int kCoSami = 8;
constexpr int kCoOlaf = 9;
constexpr int kCoEagle = 10;
constexpr int kCoHawke = 12;
constexpr int kCoSensei = 13;
constexpr int kCoJess = 14;
constexpr int kCoColin = 15;
constexpr int kCoHachi = 17;
constexpr int kCoSasha = 19;
constexpr int kCoKindle = 23;
constexpr int kCoRachel = 28;
constexpr int kCoSturm = 29;
constexpr int kCoVonBolt = 30;
constexpr int kMissileRadius = 2;

const CoStateSpec& co_state(const GameState& s, int player) {
  const Player& p = s.players[player];
  return kCos[p.co].state[p.power_on];
}

bool is_air(int type) {
  const int cls = kUnits[type].cls;
  return cls == kCopter || cls == kPlane;
}
bool is_naval(int type) { return kUnits[type].cls == kNaval; }
bool is_ground(int type) { return !is_air(type) && !is_naval(type); }
bool is_indirect(int type) { return kUnits[type].max_range > 1; }
int cargo_count(const Unit& t) { return (t.cargo[0] != kNoUnit) + (t.cargo[1] != kNoUnit); }

// Whether a property of `kind` repairs and resupplies units of `type`.
bool services(int kind, int type) {
  switch (kind) {
    case kCity:
    case kHq:
    case kBase:
      return is_ground(type);
    case kAirport:
      return is_air(type);
    case kPort:
      return is_naval(type);
    default:
      return false;
  }
}

bool builds(int kind, int type, const Player& p) {
  switch (kind) {
    case kBase:
      return is_ground(type);
    case kAirport:
      return is_air(type);
    case kPort:
      return is_naval(type);
    case kCity:  // Hachi's Merchant Union
      return p.co == kCoHachi && p.power_on == 2 && is_ground(type);
    default:
      return false;
  }
}

void resupply(Unit& u) {
  u.fuel = kUnits[u.type].fuel;
  u.ammo = kUnits[u.type].ammo;
}

// APC supply reaches a transport's cargo too.
void resupply_with_cargo(GameState& s, Unit& u) {
  resupply(u);
  for (uint8_t c : u.cargo) {
    if (c != kNoUnit) resupply(s.units[c]);
  }
}

int scaled_cost(int stars, int uses) {
  return stars * kStarValue * (5 + std::min(uses, kMaxCostedPowerUses)) / 5;
}

// Power meter gains are capped at the larger power cost and stop while a power is on.
void charge(Player& p, int64_t amount) {
  if (p.power_on || amount <= 0) return;
  const int64_t cap = std::max(p.cop_cost, p.scop_cost);
  p.power = static_cast<int32_t>(std::max<int64_t>(p.power, std::min(p.power + amount, cap)));
}

void eliminate(GameState& s, int player) {
  s.players[player].eliminated = 1;
  if (s.winner == kNoOwner) s.winner = static_cast<uint8_t>(1 - player);  // two players
}

// Rout: a player who loses their last unit is out.
void check_rout(GameState& s, int player) {
  if (!s.players[player].eliminated && count_units(s, player) == 0) eliminate(s, player);
}

// A capture in progress belongs to the unit standing on the property: it
// resets when that unit leaves or dies.
void stop_capturing(const MapData& m, GameState& s, const Unit& u) {
  const int prop = m.prop[u.tile];
  if (prop != kNoProp && is_capturable(m.prop_kind[prop]) && s.prop_owner[prop] != u.owner)
    s.prop_points[prop] = kFullCapture;
}

void destroy(const MapData& m, GameState& s, int slot) {
  Unit& u = s.units[slot];
  if (!u.has(kCarried)) stop_capturing(m, s, u);
  remove_unit(s, slot);
}

// Damage from powers, bombs and silos never destroys: units keep at least 1 HP.
void hurt(Unit& u, int amount) { u.hp = static_cast<uint8_t>(std::max(1, u.hp - amount)); }

struct MovePlan {
  int n = 0;  // tiles on the path, the start included
  uint16_t tiles[kMaxPath];
  int stop = 0;  // index of the tile the unit ends on
  int fuel = 0;  // movement cost up to `stop`
  bool trapped = false;

  int dest() const { return tiles[stop]; }
};

Status plan_move(const MapData& m, const GameState& s, int slot, int to, const Detail& d,
                 MovePlan& mp) {
  const Unit& u = s.units[slot];
  if (to < 0 || to >= m.n_tiles) return kErrBadPath;
  if (d.path) {
    if (d.path_len < 1 || d.path_len > kMaxPath || d.path[0] != u.tile ||
        d.path[d.path_len - 1] != to || path_cost(m, s, slot, d.path, d.path_len) < 0)
      return kErrBadPath;
    mp.n = d.path_len;
    std::copy(d.path, d.path + d.path_len, mp.tiles);
  } else if (to == u.tile) {
    mp.n = 1;
    mp.tiles[0] = u.tile;
  } else {
    thread_local Reach r;
    reachable(m, s, slot, r);
    if (r.cost[to] == kUnreachable) return kErrUnreachable;
    int n = 0;
    for (int t = to; t != -1; t = r.prev[t]) {
      if (n == kMaxPath) return kErrBadPath;
      mp.tiles[n++] = static_cast<uint16_t>(t);
    }
    std::reverse(mp.tiles, mp.tiles + n);
    mp.n = n;
  }
  // The move was planned from the mover's view; on the real board an unseen
  // enemy on the way stops it on the last tile before, or further back if a
  // friendly unit is in the way there.
  mp.stop = mp.n - 1;
  for (int i = 1; i < mp.n; ++i) {
    const int occ = s.unit_at[mp.tiles[i]];
    if (occ != kNoUnit && s.units[occ].owner != u.owner) {
      mp.stop = i - 1;
      mp.trapped = true;
      break;
    }
  }
  if (mp.trapped) {
    while (mp.stop > 0 && s.unit_at[mp.tiles[mp.stop]] != kNoUnit) --mp.stop;
  }
  mp.fuel = 0;
  for (int i = 1; i <= mp.stop; ++i) mp.fuel += step_cost(m, s, u, mp.tiles[i]);
  return kOk;
}

// Spend the fuel and leave the old tile. `place` puts the unit on its
// destination (Load and Join take it off the board instead).
void execute_move(const MapData& m, GameState& s, int slot, const MovePlan& mp, bool place) {
  Unit& u = s.units[slot];
  if (mp.dest() != u.tile) stop_capturing(m, s, u);
  u.fuel = static_cast<uint8_t>(u.fuel - mp.fuel);
  if (place) relocate_unit(s, slot, mp.dest());
}

bool free_for(const GameState& s, int tile, int slot) {
  return s.unit_at[tile] == kNoUnit || s.unit_at[tile] == slot;
}

// Whether the active player sees the unit on `tile` once `mover` stands on `dest`.
bool sees(const MapData& m, const GameState& s, int target_slot, int viewer, int dest) {
  return visible_to(m, s, target_slot, viewer) || m.distance(s.units[target_slot].tile, dest) == 1;
}

Status fire(const MapData& m, GameState& s, const Action& a, const Detail& d, const MovePlan& mp) {
  Unit& u = s.units[a.unit];
  const int dest = mp.dest();
  if (!free_for(s, dest, a.unit)) return kErrOccupied;
  if (is_indirect(u.type) && dest != u.tile) return kErrCannotFire;
  if (a.target >= m.n_tiles) return kErrBadTarget;
  const int ti = s.unit_at[a.target];
  if (ti == kNoUnit || s.units[ti].owner == u.owner || !sees(m, s, ti, u.owner, dest))
    return kErrBadTarget;
  const Combatant att = combatant(m, s, a.unit, dest);
  const Combatant def = combatant(m, s, ti, a.target);
  const int dist = m.distance(dest, a.target);
  if (!can_attack(att, def, dist)) return kErrCannotFire;
  Outcome o{};
  if (d.att_hp >= 0 || d.def_hp >= 0) {
    thread_local std::vector<Outcome> outs;
    combat_outcomes(att, def, dist, outs);
    auto it = std::find_if(outs.begin(), outs.end(), [&](const Outcome& x) {
      return x.att_hp == d.att_hp && x.def_hp == d.def_hp;
    });
    if (it == outs.end()) return kErrBadOutcome;
    o = *it;
  } else {
    sample_fight(att, def, dist, s.rng, o);
  }

  execute_move(m, s, a.unit, mp, true);
  Unit& t = s.units[ti];
  const bool def_first = strikes_first(def) && can_counter(def, att, dist);
  const bool att_struck = !def_first || o.att_hp > 0;
  // A unit that can't see a hidden attacker deals it nothing back but still
  // spends the ammo (AWBW).
  Combatant att_seen = att;
  att_seen.hidden = 0;
  const bool def_struck = can_counter(def, att_seen, dist) && (def_first || o.def_hp > 0);
  if (att_struck && choose_weapon(att, def.type) == kPrimaryWeapon) --u.ammo;
  if (def_struck && choose_weapon(def, att.type) == kPrimaryWeapon) --t.ammo;

  const int64_t lost_att =
      int64_t{displayed_hp(u.hp) - displayed_hp(o.att_hp)} * kUnits[u.type].cost;
  const int64_t lost_def =
      int64_t{displayed_hp(t.hp) - displayed_hp(o.def_hp)} * kUnits[t.type].cost;
  charge(s.players[u.owner], lost_att + lost_def / 2);
  charge(s.players[t.owner], lost_def + lost_att / 2);
  // War Bonds: Sasha earns half the value of the damage her units deal, in
  // her attacks and in their counterattacks.
  Player& ap = s.players[u.owner];
  Player& dp = s.players[t.owner];
  if (ap.co == kCoSasha && ap.power_on == 2) ap.funds += static_cast<int32_t>(lost_def / 20);
  if (dp.co == kCoSasha && dp.power_on == 2) dp.funds += static_cast<int32_t>(lost_att / 20);

  u.hp = o.att_hp;
  t.hp = o.def_hp;
  u.flags |= kMoved;
  const int att_owner = u.owner, def_owner = t.owner;
  if (t.hp == 0) destroy(m, s, ti);
  if (u.hp == 0) destroy(m, s, a.unit);
  check_rout(s, def_owner);
  check_rout(s, att_owner);
  return kOk;
}

Status capture(const MapData& m, GameState& s, const Action& a, const MovePlan& mp) {
  Unit& u = s.units[a.unit];
  const int dest = mp.dest();
  if (!free_for(s, dest, a.unit)) return kErrOccupied;
  const int prop = m.prop[dest];
  if (!kUnits[u.type].capture || prop == kNoProp || !is_capturable(m.prop_kind[prop]) ||
      s.prop_owner[prop] == u.owner)
    return kErrCannotCapture;

  execute_move(m, s, a.unit, mp, true);
  const Player& p = s.players[u.owner];
  int amount = displayed_hp(u.hp) * co_state(s, u.owner).capture / 100;
  if (p.co == kCoSami && p.power_on == 2) amount = kFullCapture;  // Victory March
  u.flags |= kMoved | kCapturing;
  const int left = s.prop_points[prop] - amount;
  if (left > 0) {
    s.prop_points[prop] = static_cast<uint8_t>(left);
    return kOk;
  }
  const int loser = s.prop_owner[prop];
  const int kind = m.prop_kind[prop];
  s.prop_owner[prop] = u.owner;
  s.prop_points[prop] = kFullCapture;
  if (loser != kNoOwner && (kind == kHq || (kind == kLab && count_props(m, s, loser, kHq) == 0 &&
                                            count_props(m, s, loser, kLab) == 0))) {
    eliminate(s, loser);  // HQ taken, or the last lab of a player without an HQ
  } else if (count_props(m, s, u.owner) - count_props(m, s, u.owner, kComtower) -
                 count_props(m, s, u.owner, kLab) >=
             s.settings.capture_win) {
    // Comtowers and labs don't count toward the capture limit (every logged
    // capture win, and 335 captures past the limit counting them).
    eliminate(s, 1 - u.owner);
  }
  return kOk;
}

Status load(const MapData& m, GameState& s, const Action& a, const MovePlan& mp) {
  const Unit& u = s.units[a.unit];
  const int ti = s.unit_at[mp.dest()];
  if (ti == kNoUnit || ti == a.unit) return kErrCannotLoad;
  const Unit& t = s.units[ti];
  if (t.owner != u.owner || !((kUnits[t.type].carry_mask >> u.type) & 1) ||
      cargo_count(t) >= kUnits[t.type].carry_slots)
    return kErrCannotLoad;
  execute_move(m, s, a.unit, mp, false);
  load_unit(s, a.unit, ti);
  s.units[a.unit].flags |= kMoved;
  return kOk;
}

Status join(const MapData& m, GameState& s, const Action& a, const MovePlan& mp) {
  const Unit& u = s.units[a.unit];
  const int ti = s.unit_at[mp.dest()];
  if (ti == kNoUnit || ti == a.unit) return kErrCannotJoin;
  const Unit& t = s.units[ti];
  if (t.owner != u.owner || t.type != u.type || displayed_hp(t.hp) >= 10 || cargo_count(u) > 0)
    return kErrCannotJoin;
  execute_move(m, s, a.unit, mp, false);
  Unit& mover = s.units[a.unit];
  Unit& into = s.units[ti];
  const int shown = displayed_hp(mover.hp) + displayed_hp(into.hp);
  if (shown > 10) s.players[into.owner].funds += (shown - 10) * kUnits[into.type].cost / 10;
  into.hp = static_cast<uint8_t>(10 * std::min(10, shown));  // displayed HPs add up
  into.fuel = static_cast<uint8_t>(std::min<int>(kUnits[into.type].fuel, mover.fuel + into.fuel));
  into.ammo = static_cast<uint8_t>(std::min<int>(kUnits[into.type].ammo, mover.ammo + into.ammo));
  into.flags |= kMoved;
  remove_unit(s, a.unit);
  return kOk;
}

Status unload(const MapData& m, GameState& s, const Action& a, const MovePlan& mp) {
  const Unit& u = s.units[a.unit];
  const int dest = mp.dest();
  if (!free_for(s, dest, a.unit)) return kErrOccupied;
  if (a.arg > 1 || u.cargo[a.arg] == kNoUnit) return kErrCannotUnload;
  const int c = u.cargo[a.arg];
  // The tile the transport is leaving counts as free.
  const bool free =
      s.unit_at[a.target] == kNoUnit || (s.unit_at[a.target] == a.unit && a.target != dest);
  if (a.target >= m.n_tiles || m.distance(dest, a.target) != 1 || !free ||
      move_cost(m, s, a.target, kUnits[s.units[c].type].move_type, weather_for(s, u.owner)) == 0)
    return kErrCannotUnload;
  const bool moves = dest != u.tile;
  execute_move(m, s, a.unit, mp, true);
  unload_unit(s, c, a.target);
  s.units[c].flags |= kMoved;
  // AWBW: unloading in place doesn't use the transport's action; moving does.
  if (moves) s.units[a.unit].flags |= kMoved;
  return kOk;
}

Status supply(const MapData& m, GameState& s, const Action& a, const MovePlan& mp) {
  const Unit& u = s.units[a.unit];
  const int dest = mp.dest();
  if (!kUnits[u.type].supplies) return kErrCannotSupply;
  if (!free_for(s, dest, a.unit)) return kErrOccupied;
  int targets[kNumDirs];
  int n = 0;
  for (int dir = 0; dir < kNumDirs; ++dir) {
    const int nb = m.neighbor[dest][dir];
    if (nb == kOffMap) continue;
    const int k = s.unit_at[nb];
    if (k != kNoUnit && k != a.unit && s.units[k].owner == u.owner) targets[n++] = k;
  }
  if (n == 0) return kErrCannotSupply;
  execute_move(m, s, a.unit, mp, true);
  for (int i = 0; i < n; ++i) resupply_with_cargo(s, s.units[targets[i]]);
  s.units[a.unit].flags |= kMoved;
  return kOk;
}

Status repair(const MapData& m, GameState& s, const Action& a, const MovePlan& mp) {
  const Unit& u = s.units[a.unit];
  const int dest = mp.dest();
  if (!kUnits[u.type].repairs) return kErrCannotRepair;
  if (!free_for(s, dest, a.unit)) return kErrOccupied;
  if (a.target >= m.n_tiles || m.distance(dest, a.target) != 1) return kErrBadTarget;
  const int ti = s.unit_at[a.target];
  if (ti == kNoUnit || ti == a.unit || s.units[ti].owner != u.owner) return kErrBadTarget;
  execute_move(m, s, a.unit, mp, true);
  Unit& t = s.units[ti];
  Player& p = s.players[u.owner];
  // +1 HP, paid per displayed HP gained (topping up 9.6 to 10 is free).
  const int hp = std::min(100, t.hp + 10);
  const int price = (displayed_hp(hp) - displayed_hp(t.hp)) * kUnits[t.type].cost / 10;
  if (price <= p.funds) {
    t.hp = static_cast<uint8_t>(hp);
    p.funds -= price;
  }
  resupply(t);
  s.units[a.unit].flags |= kMoved;
  return kOk;
}

Status hide(const MapData& m, GameState& s, const Action& a, const MovePlan& mp, bool on) {
  const Unit& u = s.units[a.unit];
  if (!kUnits[u.type].hides || u.has(kHidden) == on) return kErrCannotHide;
  if (!free_for(s, mp.dest(), a.unit)) return kErrOccupied;
  execute_move(m, s, a.unit, mp, true);
  Unit& v = s.units[a.unit];
  v.flags = static_cast<uint8_t>(on ? v.flags | kHidden : v.flags & ~kHidden);
  v.flags |= kMoved;
  return kOk;
}

Status attack_seam(const MapData& m, GameState& s, const Action& a, const MovePlan& mp) {
  const Unit& u = s.units[a.unit];
  const int dest = mp.dest();
  if (!free_for(s, dest, a.unit)) return kErrOccupied;
  if (is_indirect(u.type) && dest != u.tile) return kErrCannotFire;
  if (a.target >= m.n_tiles) return kErrBadTarget;
  const int prop = m.prop[a.target];
  if (prop == kNoProp || m.prop_kind[prop] != kSeam || s.prop_points[prop] == 0)
    return kErrBadTarget;
  const Combatant att = combatant(m, s, a.unit, dest);
  if (!can_attack(att, seam_target(), m.distance(dest, a.target))) return kErrCannotFire;
  execute_move(m, s, a.unit, mp, true);
  Unit& v = s.units[a.unit];
  if (choose_weapon(att, kNeotank) == kPrimaryWeapon) --v.ammo;
  s.prop_points[prop] = static_cast<uint8_t>(std::max(0, s.prop_points[prop] - seam_damage(att)));
  v.flags |= kMoved;
  return kOk;
}

Status explode(const MapData& m, GameState& s, const Action& a, const MovePlan& mp) {
  const Unit& u = s.units[a.unit];
  const int dest = mp.dest();
  if (!kUnits[u.type].explodes) return kErrBadAction;
  if (!free_for(s, dest, a.unit)) return kErrOccupied;
  execute_move(m, s, a.unit, mp, true);
  const int owner = u.owner;
  for (int i = 0; i < kMaxUnits; ++i) {
    Unit& v = s.units[i];
    if (i != a.unit && v.alive() && !v.has(kCarried) && m.distance(v.tile, dest) <= 3) hurt(v, 50);
  }
  destroy(m, s, a.unit);
  check_rout(s, owner);
  return kOk;
}

Status launch(const MapData& m, GameState& s, const Action& a, const MovePlan& mp) {
  const Unit& u = s.units[a.unit];
  const int dest = mp.dest();
  const int prop = m.prop[dest];
  if (!kUnits[u.type].capture || prop == kNoProp || m.prop_kind[prop] != kSilo ||
      s.prop_points[prop] == 0 || a.target >= m.n_tiles)
    return kErrBadAction;
  if (!free_for(s, dest, a.unit)) return kErrOccupied;
  execute_move(m, s, a.unit, mp, true);
  for (Unit& v : s.units) {
    if (v.alive() && !v.has(kCarried) && m.distance(v.tile, a.target) <= 2) hurt(v, 30);
  }
  s.prop_points[prop] = 0;
  s.units[a.unit].flags |= kMoved;
  return kOk;
}

Status unit_action(const MapData& m, GameState& s, const Action& a, const Detail& d) {
  if (a.unit >= kMaxUnits) return kErrNotYourUnit;
  Unit& u = s.units[a.unit];
  if (!u.alive() || u.owner != s.active) return kErrNotYourUnit;
  // AWBW lets a transport unload where it stands at any time in its turn,
  // even after it has moved or acted.
  const bool in_place = a.kind == kActUnload && a.to == u.tile && (!d.path || d.path_len <= 1);
  if (u.has(kStunned) || u.has(kCarried) || (u.has(kMoved) && !in_place)) return kErrCannotAct;
  MovePlan mp;
  const Status planned = plan_move(m, s, a.unit, a.to, d, mp);
  if (planned != kOk) return planned;
  if (mp.trapped) {
    execute_move(m, s, a.unit, mp, true);
    s.units[a.unit].flags |= kMoved;
    return kTrapped;
  }
  Status st = kErrBadAction;
  switch (a.kind) {
    case kActWait:
      if (!free_for(s, mp.dest(), a.unit)) {
        st = kErrOccupied;
        break;
      }
      execute_move(m, s, a.unit, mp, true);
      s.units[a.unit].flags |= kMoved;
      st = kOk;
      break;
    case kActFire:
      st = fire(m, s, a, d, mp);
      break;
    case kActCapture:
      st = capture(m, s, a, mp);
      break;
    case kActLoad:
      st = load(m, s, a, mp);
      break;
    case kActJoin:
      st = join(m, s, a, mp);
      break;
    case kActUnload:
      st = unload(m, s, a, mp);
      break;
    case kActSupply:
      st = supply(m, s, a, mp);
      break;
    case kActRepair:
      st = repair(m, s, a, mp);
      break;
    case kActHide:
    case kActUnhide:
      st = hide(m, s, a, mp, a.kind == kActHide);
      break;
    case kActAttackSeam:
      st = attack_seam(m, s, a, mp);
      break;
    case kActExplode:
      st = explode(m, s, a, mp);
      break;
    case kActLaunch:
      st = launch(m, s, a, mp);
      break;
    default:
      break;
  }
  return st;
}

Status build(const MapData& m, GameState& s, const Action& a) {
  const int me = s.active;
  Player& p = s.players[me];
  if (a.target >= m.n_tiles || a.arg >= kNumUnitTypes) return kErrBadAction;
  const int prop = m.prop[a.target];
  if (prop == kNoProp || s.prop_owner[prop] != me || !builds(m.prop_kind[prop], a.arg, p) ||
      ((s.settings.banned_units >> a.arg) & 1))
    return kErrCannotBuild;
  if (s.unit_at[a.target] != kNoUnit) return kErrOccupied;
  if ((s.settings.unit_limit && count_units(s, me) >= s.settings.unit_limit) || alloc_unit(s) < 0)
    return kErrUnitLimit;
  const int price = kUnits[a.arg].cost * co_state(s, me).cost / 100;
  if (p.funds < price) return kErrFunds;
  p.funds -= price;
  Unit n{};
  n.tile = a.target;
  n.type = a.arg;
  n.owner = static_cast<uint8_t>(me);
  n.hp = 100;
  resupply(n);
  n.flags = kMoved;
  add_unit(s, n);
  return kOk;
}

bool is_urban(int kind) {
  switch (kind) {
    case kCity:
    case kBase:
    case kAirport:
    case kPort:
    case kHq:
    case kLab:
    case kComtower:
      return true;
    default:
      return false;
  }
}

void heal(Unit& u, int amount) { u.hp = static_cast<uint8_t>(std::min(100, u.hp + amount)); }

// Apply `f` to every unit of `player`, or only the uncarried ones. Replays:
// Andy's heals reach units in transports; Hawke's heals and all power damage
// don't.
template <typename F>
void for_units_of(GameState& s, int player, bool carried, F&& f) {
  for (Unit& u : s.units) {
    if (u.alive() && u.owner == player && (carried || !u.has(kCarried))) f(u);
  }
}

int missile_count(int co, bool super) {
  if (co == kCoRachel && super) return 3;
  if (co == kCoSturm || (co == kCoVonBolt && super)) return 1;
  return 0;
}

// Internal HP a missile takes from each unit it hits.
int missile_damage(int co, bool super) {
  if (co == kCoSturm) return super ? 80 : 40;
  return 30;  // Rachel's Covering Fire, Von Bolt's Ex Machina
}

enum MissileAim { kAimFoot, kAimCost, kAimHp, kAimVonBolt, kAimSturm };

// Whether a footsoldier stands on a property it is partway through capturing.
bool mid_capture(const MapData& m, const GameState& s, const Unit& u) {
  const int prop = m.prop[u.tile];
  return prop != kNoProp && is_capturable(m.prop_kind[prop]) && s.prop_owner[prop] != u.owner &&
         s.prop_points[prop] < kFullCapture;
}

// Where AWBW aims a missile, per awbw.fandom.com/wiki/Rachel, /Von_Bolt and
// /Sturm. Every uncarried unit within reach of a tile adds a value, negative
// for the user's own; the tile with the highest total wins, then the highest
// tie score, then the first tile in row order. Values:
//   Rachel, in her missiles' order, with HP clamped to 0.1 below 1.0 and to
//   3.0 above it: HP x4 for an Infantry or Mech over 1.0 HP (x8 mid-capture);
//   HP x cost (2G below 1.0 HP), ties to the enemy's unclamped HP x cost; HP,
//   ties to the enemy's unclamped HP.
//   Von Bolt: displayed HP lost (at most 3) x cost, ties as Rachel's second.
//   Sturm: displayed HP x cost, centred only on an enemy unit.
// Against logged centres (check-turns): 4,520 of 4,701 Rachel missiles,
// 1,016 of 1,049 Von Bolt's, all 165 of Sturm's.
int aim_missile(const MapData& m, const GameState& s, MissileAim aim, int me) {
  thread_local int64_t score[kMaxTiles], tie[kMaxTiles];
  std::fill(score, score + m.n_tiles, 0);
  std::fill(tie, tie + m.n_tiles, 0);
  for (const Unit& u : s.units) {
    if (!u.alive() || u.has(kCarried)) continue;
    const int sign = u.owner == me ? -1 : 1;
    const int64_t cost = kUnits[u.type].cost;
    const int hp = u.hp > 30 ? 30 : u.hp < 10 ? 1 : u.hp;
    int64_t v = 0, t = 0;
    switch (aim) {
      case kAimFoot:
        v = hp;
        if ((u.type == kInfantry || u.type == kMech) && u.hp > 10)
          v *= mid_capture(m, s, u) ? 8 : 4;
        break;
      case kAimCost:
        v = u.hp < 10 ? 20 : hp * cost;
        t = u.hp * cost;
        break;
      case kAimHp:
        v = hp;
        t = u.hp;
        break;
      case kAimVonBolt:
        v = (u.hp > 30 ? 3 : displayed_hp(u.hp) - 1) * cost;
        t = u.hp * cost;
        break;
      case kAimSturm:
        v = displayed_hp(u.hp) * cost;
        break;
    }
    v *= sign;
    t = sign > 0 ? t : 0;  // tie scores count the enemy's units only
    const int ux = m.x_of(u.tile), uy = m.y_of(u.tile);
    for (int dy = -kMissileRadius; dy <= kMissileRadius; ++dy) {
      const int reach = kMissileRadius - std::abs(dy);
      for (int dx = -reach; dx <= reach; ++dx) {
        const int x = ux + dx, y = uy + dy;
        if (x < 0 || y < 0 || x >= m.width || y >= m.height) continue;
        score[m.tile(x, y)] += v;
        tie[m.tile(x, y)] += t;
      }
    }
  }
  int best = -1;
  for (int t = 0; t < m.n_tiles; ++t) {
    if (aim == kAimSturm && (s.unit_at[t] == kNoUnit || s.units[s.unit_at[t]].owner == me))
      continue;
    if (best < 0 || score[t] > score[best] || (score[t] == score[best] && tie[t] > tie[best]))
      best = t;
  }
  return best;
}

Status power(const MapData& m, GameState& s, const Action& a, const Detail& d) {
  const int me = s.active, enemy = 1 - me;
  Player& p = s.players[me];
  const CoSpec& co = kCos[p.co];
  if (!s.settings.use_powers || p.power_on || (a.arg != 1 && a.arg != 2)) return kErrPower;
  const bool super = a.arg == 2;
  if ((super ? co.scop_stars : co.cop_stars) == 0) return kErrPower;
  const int price = super ? p.scop_cost : p.cop_cost;
  if (p.power < price) return kErrPower;
  const int n_missiles = missile_count(p.co, super);
  if (d.missiles && d.n_missiles != n_missiles) return kErrBadTarget;
  for (int i = 0; d.missiles && i < n_missiles; ++i) {
    if (d.missiles[i] >= m.n_tiles) return kErrBadTarget;
  }

  p.power -= price;
  p.power_on = a.arg;
  if (p.power_uses < 255) ++p.power_uses;
  p.cop_cost = cop_cost(p.co, p.power_uses);
  p.scop_cost = scop_cost(p.co, p.power_uses);
  Player& foe = s.players[enemy];
  switch (p.co) {
    case kCoAndy:  // Hyper Repair, Hyper Upgrade
      for_units_of(s, me, true, [&](Unit& u) { heal(u, super ? 50 : 20); });
      break;
    case kCoDrake:  // Tsunami, Typhoon
      for_units_of(s, enemy, false, [&](Unit& u) {
        hurt(u, super ? 20 : 10);
        u.fuel = static_cast<uint8_t>((u.fuel + 1) / 2);
      });
      if (super) {
        s.weather = kRain;
        s.weather_owner = static_cast<uint8_t>(me);
      }
      break;
    case kCoOlaf:  // Blizzard, Winter Fury
      if (super) for_units_of(s, enemy, false, [&](Unit& u) { hurt(u, 20); });
      s.weather = kSnow;
      s.weather_owner = static_cast<uint8_t>(me);
      break;
    case kCoEagle:  // Lightning Strike: everything but footsoldiers acts again
      if (super) {
        for (Unit& u : s.units) {
          if (u.alive() && u.owner == me && kUnits[u.type].cls != kFoot)
            u.flags &= static_cast<uint8_t>(~kMoved);
        }
      }
      break;
    case kCoHawke:  // Black Wave, Black Storm
      for_units_of(s, me, false, [&](Unit& u) { heal(u, super ? 20 : 10); });
      for_units_of(s, enemy, false, [&](Unit& u) { hurt(u, super ? 20 : 10); });
      break;
    case kCoSensei: {  // Copter Command, Airborne Assault: 9 HP footsoldiers on empty cities
      for (int prop = 0; prop < m.n_props; ++prop) {
        const int tile = m.prop_tile[prop];
        if (m.prop_kind[prop] != kCity || s.prop_owner[prop] != me || s.unit_at[tile] != kNoUnit)
          continue;
        if ((s.settings.unit_limit && count_units(s, me) >= s.settings.unit_limit) ||
            alloc_unit(s) < 0)
          break;
        Unit n{};
        n.tile = static_cast<uint16_t>(tile);
        n.type = super ? kMech : kInfantry;
        n.owner = static_cast<uint8_t>(me);
        n.hp = 90;
        resupply(n);
        add_unit(s, n);
      }
      break;
    }
    case kCoJess:  // Turbo Charge, Overdrive
      for (Unit& u : s.units) {
        if (u.alive() && u.owner == me) resupply(u);
      }
      break;
    case kCoColin:
      if (super) {
        p.power_atk = static_cast<int16_t>(3 * p.funds / 1000);  // Power of Money
      } else {
        p.funds = p.funds * 3 / 2;  // Gold Rush
      }
      break;
    case kCoSasha:  // Market Crash: the enemy meter loses (funds / 50000) of its maximum
      if (!super) {
        const int64_t drain = int64_t{foe.scop_cost} * p.funds / 50000;
        foe.power = static_cast<int32_t>(std::max<int64_t>(0, foe.power - drain));
      }
      break;
    case kCoKindle:  // Urban Blight
      if (!super) {
        for_units_of(s, enemy, false, [&](Unit& u) {
          if (is_urban(terrain_kind(m, s, u.tile))) hurt(u, 30);
        });
      }
      break;
    default:
      break;
  }
  if (n_missiles) {
    // Every missile is aimed before any lands.
    const int damage = missile_damage(p.co, super);
    uint16_t aimed[3];
    if (!d.missiles) missile_targets(m, s, a.arg, aimed);
    const uint16_t* centres = d.missiles ? d.missiles : aimed;
    for (int i = 0; i < n_missiles; ++i) {
      const int centre = centres[i];
      if (centre == kNoTile) continue;
      // Missiles hit every unit in reach, the user's own included.
      for (Unit& u : s.units) {
        if (!u.alive() || u.has(kCarried) || m.distance(u.tile, centre) > kMissileRadius) continue;
        hurt(u, damage);
        if (p.co == kCoVonBolt) u.flags |= kStunned;
      }
    }
  }
  return kOk;
}

// Start of the active player's turn.
void start_turn(const MapData& m, GameState& s) {
  const int me = s.active;
  Player& p = s.players[me];
  p.power_on = 0;
  p.power_atk = 0;
  const CoStateSpec& co = kCos[p.co].state[0];
  if (s.weather_owner == me) {  // power weather lasts until its owner's next turn
    s.weather = kClear;
    s.weather_owner = kNoOwner;
  }

  for (Unit& u : s.units) {
    if (!u.alive() || u.owner != me) continue;
    // A stunned unit sits out this turn; AWBW leaves its flags alone, so a
    // capture flag stays up.
    if (u.has(kStunned)) {
      u.flags = static_cast<uint8_t>((u.flags & ~kStunned) | kMoved);
    } else {
      u.flags &= static_cast<uint8_t>(~kCapturing);
    }
    if (u.has(kCarried) || s.day == 1) continue;  // no daily fuel on day 1
    int burn = u.has(kHidden) ? kUnits[u.type].hidden_fuel_per_day : kUnits[u.type].fuel_per_day;
    if (is_air(u.type)) burn = std::max(0, burn + co.air_fuel_per_day);
    u.fuel = static_cast<uint8_t>(std::max(0, u.fuel - burn));
  }

  int paying = 0;
  for (int i = 0; i < m.n_props; ++i)
    paying += s.prop_owner[i] == me && pays_income(m.prop_kind[i]);
  p.funds += paying * (s.settings.funds_per_property + co.income_bonus);

  // Repairs on properties, column by column (x, then y) as AWBW orders its
  // buildings. A unit whose whole repair costs more than the funds left gets
  // none and the next one is tried (the only two such turns in replays).
  const int gain = 20 + co.repair_bonus;
  for (int i = 0; i < m.n_props; ++i) {
    const int prop = m.prop_by_column[i];
    const int slot = s.unit_at[m.prop_tile[prop]];
    if (slot == kNoUnit) continue;
    Unit& u = s.units[slot];
    if (u.owner != me || s.prop_owner[prop] != me || !services(m.prop_kind[prop], u.type)) continue;
    resupply(u);
    const int hp = std::min(100, u.hp + gain);
    const int price = (displayed_hp(hp) - displayed_hp(u.hp)) * kUnits[u.type].cost / 10;
    if (price <= p.funds) {
      p.funds -= price;
      u.hp = static_cast<uint8_t>(hp);
    }
  }

  // APCs resupply the units next to them.
  for (const Unit& t : s.units) {
    if (!t.alive() || t.owner != me || t.has(kCarried) || !kUnits[t.type].supplies) continue;
    for (int dir = 0; dir < kNumDirs; ++dir) {
      const int nb = m.neighbor[t.tile][dir];
      if (nb == kOffMap || s.unit_at[nb] == kNoUnit) continue;
      Unit& v = s.units[s.unit_at[nb]];
      if (v.owner == me) resupply_with_cargo(s, v);
    }
  }

  // Air units out of fuel crash; ships out of fuel sink.
  bool lost = false;
  for (int i = 0; i < kMaxUnits; ++i) {
    const Unit& u = s.units[i];
    if (u.alive() && u.owner == me && !u.has(kCarried) && u.fuel == 0 && !is_ground(u.type)) {
      destroy(m, s, i);
      lost = true;
    }
  }
  if (lost) check_rout(s, me);
}

void end_turn(const MapData& m, GameState& s) {
  const int me = s.active;
  for (Unit& u : s.units) {
    if (u.alive() && u.owner == me) u.flags &= static_cast<uint8_t>(~kMoved);
  }
  const int next = (me + 1) % kMaxPlayers;
  if (next == 0) ++s.day;
  s.active = static_cast<uint8_t>(next);
  start_turn(m, s);
}

}  // namespace

const char* status_name(Status s) {
  static constexpr const char* kNames[kNumStatuses] = {
      "ok",
      "trapped",
      "game over",
      "bad action",
      "not your unit",
      "cannot act",
      "bad path",
      "unreachable",
      "occupied",
      "bad target",
      "cannot fire",
      "cannot capture",
      "cannot load",
      "cannot join",
      "cannot unload",
      "cannot supply",
      "cannot repair",
      "cannot hide",
      "cannot build",
      "not enough funds",
      "unit limit",
      "power unavailable",
      "impossible combat result",
  };
  return s < kNumStatuses ? kNames[s] : "?";
}

Combatant combatant(const MapData& m, const GameState& s, int slot, int tile) {
  const Unit& u = s.units[slot];
  const Player& p = s.players[u.owner];
  Combatant c;
  c.type = u.type;
  c.hp = u.hp;
  c.ammo = u.ammo;
  c.co = p.co;
  c.power = p.power_on;
  c.terrain = static_cast<uint8_t>(terrain_kind(m, s, tile));
  c.stars = static_cast<uint8_t>(terrain_stars(m, s, tile));
  c.towers = static_cast<uint8_t>(count_props(m, s, u.owner, kComtower));
  if (p.power_on == 2 && p.co == kCoColin) c.extra_atk = p.power_atk;
  if (p.power_on == 2 && p.co == kCoKindle)
    c.extra_atk = static_cast<int16_t>(3 * count_props(m, s, u.owner));  // every property is urban
  c.hidden = u.has(kHidden);
  return c;
}

int missile_targets(const MapData& m, const GameState& s, int level, uint16_t* out) {
  const Player& p = s.players[s.active];
  const int n = missile_count(p.co, level == 2);
  static constexpr MissileAim kRachel[3] = {kAimFoot, kAimCost, kAimHp};
  for (int i = 0; i < n; ++i) {
    const MissileAim aim = p.co == kCoRachel  ? kRachel[i]
                           : p.co == kCoSturm ? kAimSturm
                                              : kAimVonBolt;
    const int t = aim_missile(m, s, aim, s.active);
    out[i] = t < 0 ? kNoTile : static_cast<uint16_t>(t);
  }
  return n;
}

int scop_cost(int co, int uses) { return scaled_cost(kCos[co].scop_stars, uses); }

int cop_cost(int co, int uses) {
  return kCos[co].cop_stars ? scaled_cost(kCos[co].cop_stars, uses) : scop_cost(co, uses);
}

Status apply(const MapData& m, GameState& s, const Action& a, const Detail& d) {
  if (s.winner != kNoOwner) return kErrGameOver;
  switch (a.kind) {
    case kActBuild:
      return build(m, s, a);
    case kActPower:
      return power(m, s, a, d);
    case kActDelete: {
      if (a.unit >= kMaxUnits || !s.units[a.unit].alive() || s.units[a.unit].owner != s.active ||
          s.units[a.unit].has(kCarried))
        return kErrNotYourUnit;
      destroy(m, s, a.unit);
      check_rout(s, s.active);
      return kOk;
    }
    case kActEndTurn:
      end_turn(m, s);
      return kOk;
    case kActResign:
      eliminate(s, s.active);
      return kOk;
    default:
      return a.kind < kActBuild ? unit_action(m, s, a, d) : kErrBadAction;
  }
}

}  // namespace awbw
