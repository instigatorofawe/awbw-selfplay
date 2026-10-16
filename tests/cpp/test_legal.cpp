#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "core/actions.h"
#include "core/legal.h"
#include "core/movegen.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::arena_start;
using testing::kArena;
using testing::make_map;
using testing::make_state;
using testing::put;

using Key = std::tuple<int, int, int, int, int>;  // kind, unit, to, target, arg

bool uses_target(int kind) {
  return kind == kActFire || kind == kActUnload || kind == kActRepair || kind == kActAttackSeam ||
         kind == kActLaunch || kind == kActBuild;
}

// An action as apply reads it: fields a kind ignores are left out.
Key key(const Action& a) {
  const bool unit_kind = a.kind < kActBuild;
  return {a.kind, unit_kind || a.kind == kActDelete ? a.unit : -1, unit_kind ? a.to : -1,
          uses_target(a.kind) ? a.target : -1,
          a.kind == kActUnload || a.kind == kActBuild || a.kind == kActPower ? a.arg : 0};
}

Action make(int kind, int unit, int to, int target = 0, int arg = 0) {
  Action a;
  a.kind = static_cast<uint8_t>(kind);
  a.unit = static_cast<uint8_t>(unit);
  a.to = static_cast<uint16_t>(to);
  a.target = static_cast<uint16_t>(target);
  a.arg = static_cast<uint8_t>(arg);
  return a;
}

std::set<Key> legal_keys(const MapData& m, const GameState& s) {
  std::vector<Action> acts;
  legal_actions(m, s, acts);
  std::set<Key> keys;
  for (const Action& a : acts) EXPECT_TRUE(keys.insert(key(a)).second) << "listed twice";
  return keys;
}

bool same(const GameState& a, const GameState& b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// Every action apply accepts, found by trying them all on a copy of `s`.
// Moves into an enemy the mover can't see apply with kTrapped whatever the
// action, so those are kept apart.
void brute_force(const MapData& m, const GameState& s, std::set<Key>& ok, std::set<Key>& trapped) {
  GameState t = s;
  auto attempt = [&](const Action& a, const Detail& d = {}) {
    const Status st = apply(m, t, a, d);
    if (st == kOk) ok.insert(key(a));
    if (st == kTrapped) trapped.insert(key(a));
    if (applied(st)) {
      t = s;
    } else {
      EXPECT_TRUE(same(t, s)) << "an error changed the state";
    }
  };
  static constexpr int kPlain[] = {kActWait,   kActCapture, kActLoad,   kActJoin,
                                   kActSupply, kActHide,    kActUnhide, kActExplode};
  static constexpr int kTargeted[] = {kActFire, kActRepair, kActAttackSeam, kActLaunch};
  Reach r;
  for (int slot = 0; slot < kMaxUnits; ++slot) {
    const Unit& u = s.units[slot];
    if (!u.alive() || u.owner != s.active) continue;
    std::vector<int> tos;
    for (int to = 0; to < m.n_tiles; ++to) {
      for (int kind : kPlain) attempt(make(kind, slot, to));
      GameState probe = s;
      const Status st = apply(m, probe, make(kActWait, slot, to));
      if (to == u.tile || (st != kErrUnreachable && st != kErrBadPath && st != kErrCannotAct))
        tos.push_back(to);
    }
    // Acting on another tile: every target from every destination, along
    // the cheapest path (so apply needn't search again for each one).
    reachable(m, s, slot, r);
    for (int to : tos) {
      std::vector<uint16_t> path;
      if (r.cost[to] != kUnreachable) {
        for (int x = to; x != -1; x = r.prev[x]) path.push_back(static_cast<uint16_t>(x));
        std::reverse(path.begin(), path.end());
      }
      Detail d;
      if (!path.empty()) {
        d.path = path.data();
        d.path_len = static_cast<int>(path.size());
      }
      for (int target = 0; target < m.n_tiles; ++target) {
        for (int kind : kTargeted) attempt(make(kind, slot, to, target), d);
        for (int arg = 0; arg < 2; ++arg) attempt(make(kActUnload, slot, to, target, arg), d);
      }
    }
  }
  for (int tile = 0; tile < m.n_tiles; ++tile) {
    for (int type = 0; type < kNumUnitTypes; ++type)
      attempt(make(kActBuild, kNoUnit, 0, tile, type));
  }
  for (int level = 1; level <= 2; ++level) attempt(make(kActPower, kNoUnit, 0, 0, level));
  for (int slot = 0; slot < kMaxUnits; ++slot) attempt(make(kActDelete, slot, 0));
  attempt(make(kActEndTurn, kNoUnit, 0));
}

void expect_matches_apply(const MapData& m, const GameState& s) {
  std::set<Key> ok, trapped;
  brute_force(m, s, ok, trapped);
  const std::set<Key> legal = legal_keys(m, s);
  for (const Key& k : ok) {
    EXPECT_TRUE(legal.count(k)) << "apply takes kind " << std::get<0>(k) << " unit "
                                << std::get<1>(k) << " to " << std::get<2>(k) << " target "
                                << std::get<3>(k) << " arg " << std::get<4>(k) << ", not listed";
  }
  for (const Key& k : legal) {
    EXPECT_TRUE(ok.count(k) || trapped.count(k))
        << "listed kind " << std::get<0>(k) << " unit " << std::get<1>(k) << " to "
        << std::get<2>(k) << " target " << std::get<3>(k) << " arg " << std::get<4>(k)
        << ", apply refuses it";
  }
}

// Every action the micro-steps from `sel` can complete, and checks along the
// way: each pick changes nothing but the selection, and no pick is a dead end.
void expand(const MapData& m, const GameState& s, const Selection& sel, std::vector<Action>& out) {
  std::vector<uint16_t> ids;
  legal_ids(m, s, sel, ids);
  ASSERT_FALSE(ids.empty()) << "dead end in phase " << int(sel.phase);
  EXPECT_TRUE(std::is_sorted(ids.begin(), ids.end()));
  EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end());
  for (int id : ids) {
    Action a;
    if (step_action(m, s, sel, id, a)) {
      out.push_back(a);
      continue;
    }
    GameState t = s;
    Selection next = sel;
    ASSERT_EQ(step(m, t, next, id), kOk);
    EXPECT_TRUE(same(t, s));
    EXPECT_NE(next, sel);
    expand(m, s, next, out);
  }
}

void expect_steps_match_actions(const MapData& m, const GameState& s) {
  std::vector<Action> stepped;
  expand(m, s, Selection{}, stepped);
  std::set<Key> keys;
  for (const Action& a : stepped) EXPECT_TRUE(keys.insert(key(a)).second) << "reached twice";
  EXPECT_EQ(keys, legal_keys(m, s));

  // Each action's own steps play it exactly as apply does.
  std::vector<Action> acts;
  legal_actions(m, s, acts);
  for (const Action& a : acts) {
    uint16_t ids[3];
    const int n = action_steps(m, s, a, ids);
    ASSERT_GE(n, 1);
    GameState t = s, u = s;
    Selection sel;
    for (int i = 0; i < n - 1; ++i) ASSERT_EQ(step(m, t, sel, ids[i]), kOk);
    const Status st = step(m, t, sel, ids[n - 1]);
    EXPECT_EQ(st, apply(m, u, a));
    EXPECT_TRUE(same(t, u));
    EXPECT_EQ(sel, Selection{});
  }
}

std::vector<uint16_t> ids_of(const MapData& m, const GameState& s, const Selection& sel) {
  std::vector<uint16_t> ids;
  legal_ids(m, s, sel, ids);
  return ids;
}

bool has_id(const MapData& m, const GameState& s, const Selection& sel, int id) {
  const std::vector<uint16_t> ids = ids_of(m, s, sel);
  return std::binary_search(ids.begin(), ids.end(), id);
}

// --- Rules -------------------------------------------------------------------

TEST(Legal, AFreshUnitCanAlwaysWaitWhereItIs) {
  const MapData m = make_map({"...", "..."});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 1, 0);
  put(s, m, kInfantry, 1, 2, 1);
  EXPECT_TRUE(has_id(m, s, Selection{}, action_id(kChSelect, m.tile(1, 0))));
  const Selection picked{kPhaseSelected, static_cast<uint8_t>(inf),
                         static_cast<uint16_t>(m.tile(1, 0))};
  EXPECT_TRUE(has_id(m, s, picked, action_id(kChMove, m.tile(1, 0))));
  const Selection moved{kPhaseMoved, static_cast<uint8_t>(inf),
                        static_cast<uint16_t>(m.tile(1, 0))};
  EXPECT_TRUE(has_id(m, s, moved, action_id(kChWait, m.tile(1, 0))));
  expect_matches_apply(m, s);
  expect_steps_match_actions(m, s);
}

TEST(Legal, UnitsAreDeletedOnlyBeforeTheyMove) {
  const MapData m = make_map({"....", "...."});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 3, 1);
  const int del = action_id(kChDelete, m.tile(0, 0));
  EXPECT_TRUE(has_id(m, s, Selection{}, del));
  ASSERT_EQ(apply(m, s, make(kActWait, inf, m.tile(1, 0))), kOk);
  EXPECT_FALSE(has_id(m, s, Selection{}, action_id(kChDelete, m.tile(1, 0))));
  EXPECT_EQ(apply(m, s, make(kActDelete, inf, 0)), kErrCannotAct);
  expect_matches_apply(m, s);
}

TEST(Legal, CargoComesOffOnlyFromTerrainItCouldMoveOn) {
  // A Lander at sea can't drop its Tank on the beach next to it; on the shoal it can.
  const MapData m = make_map({"~~~~", "~_..", "~~.."});
  GameState s = make_state(m);
  const int lander = put(s, m, kLander, 0, 0, 1);
  const int tank = put(s, m, kTank, 0, 2, 1);
  load_unit(s, tank, lander);
  put(s, m, kInfantry, 1, 3, 2);
  EXPECT_EQ(apply(m, s, make(kActUnload, lander, m.tile(0, 1), m.tile(1, 1), 0)),
            kErrCannotUnload);  // shoal next to it, but the Lander is at sea
  const Selection at_sea{kPhaseMoved, static_cast<uint8_t>(lander),
                         static_cast<uint16_t>(m.tile(0, 1))};
  EXPECT_FALSE(has_id(m, s, at_sea, action_id(kChUnload1, m.tile(1, 1))));
  const Selection on_shoal{kPhaseMoved, static_cast<uint8_t>(lander),
                           static_cast<uint16_t>(m.tile(1, 1))};
  EXPECT_TRUE(has_id(m, s, on_shoal, action_id(kChUnload1, m.tile(2, 1))));
  expect_matches_apply(m, s);
  expect_steps_match_actions(m, s);
}

TEST(Legal, AUnitLoadedThisTurnCanBeDroppedFurtherOn) {
  // AWBW's boosting: Infantry rides an APC and gets off further along.
  const MapData m = make_map({"........"});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  const int apc = put(s, m, kApc, 0, 1, 0);
  put(s, m, kInfantry, 1, 7, 0);
  ASSERT_EQ(apply(m, s, make(kActLoad, inf, m.tile(1, 0))), kOk);
  const Selection moved{kPhaseMoved, static_cast<uint8_t>(apc),
                        static_cast<uint16_t>(m.tile(5, 0))};
  EXPECT_TRUE(has_id(m, s, moved, action_id(kChUnload1, m.tile(6, 0))));
  expect_matches_apply(m, s);
  expect_steps_match_actions(m, s);
}

TEST(Legal, AMovedTransportCanOnlyUnloadWhereItStands) {
  const MapData m = make_map({"......"});
  GameState s = make_state(m);
  const int apc = put(s, m, kApc, 0, 1, 0);
  const int inf = put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 5, 0);
  ASSERT_EQ(apply(m, s, make(kActLoad, inf, m.tile(1, 0))), kOk);
  ASSERT_EQ(apply(m, s, make(kActWait, apc, m.tile(3, 0))), kOk);
  const Selection picked{kPhaseSelected, static_cast<uint8_t>(apc),
                         static_cast<uint16_t>(m.tile(3, 0))};
  EXPECT_EQ(ids_of(m, s, picked),
            std::vector<uint16_t>{static_cast<uint16_t>(action_id(kChMove, m.tile(3, 0)))});
  const Selection moved{kPhaseMoved, static_cast<uint8_t>(apc),
                        static_cast<uint16_t>(m.tile(3, 0))};
  EXPECT_EQ(ids_of(m, s, moved),
            (std::vector<uint16_t>{static_cast<uint16_t>(action_id(kChUnload1, m.tile(2, 0))),
                                   static_cast<uint16_t>(action_id(kChUnload1, m.tile(4, 0)))}));
  expect_matches_apply(m, s);
  expect_steps_match_actions(m, s);
}

// AWBW lets a loaded APC board a Lander (52 games of the archive).
TEST(Legal, ALoadedTransportCanBoardAnother) {
  const MapData m = make_map({"~~~~", "_..."});
  GameState s = make_state(m);
  const int lander = put(s, m, kLander, 0, 0, 1);
  const int apc = put(s, m, kApc, 0, 1, 1);
  const int inf = put(s, m, kInfantry, 0, 2, 1);
  put(s, m, kInfantry, 1, 3, 1);
  ASSERT_EQ(apply(m, s, make(kActLoad, inf, m.tile(1, 1))), kOk);
  expect_matches_apply(m, s);
  ASSERT_EQ(apply(m, s, make(kActLoad, apc, m.tile(0, 1))), kOk);
  EXPECT_EQ(s.units[lander].cargo[0], apc);
  EXPECT_EQ(s.units[inf].tile, m.tile(0, 1));
  EXPECT_EQ(check_state(m, s), "");
}

TEST(Legal, MovingIntoAnUnseenEnemyIsLegalAndTraps) {
  const MapData m = make_map({".....", "....."});
  GameState s = make_state(m);
  const int recon = put(s, m, kRecon, 0, 0, 0);
  const int stealth = put(s, m, kStealth, 1, 3, 0);
  s.units[stealth].flags |= kHidden;
  put(s, m, kInfantry, 1, 4, 1);
  const Selection picked{kPhaseSelected, static_cast<uint8_t>(recon),
                         static_cast<uint16_t>(m.tile(0, 0))};
  EXPECT_TRUE(has_id(m, s, picked, action_id(kChMove, m.tile(4, 0))));  // through it
  EXPECT_TRUE(has_id(m, s, picked, action_id(kChMove, m.tile(3, 0))));  // onto it
  expect_matches_apply(m, s);
  expect_steps_match_actions(m, s);
  Selection sel = picked;
  ASSERT_EQ(step(m, s, sel, action_id(kChMove, m.tile(4, 0))), kOk);
  EXPECT_EQ(step(m, s, sel, action_id(kChWait, m.tile(4, 0))), kTrapped);
  EXPECT_EQ(s.units[recon].tile, m.tile(2, 0));
  EXPECT_EQ(sel, Selection{});
}

TEST(Legal, AHiddenEnemyIsATargetOnceTheAttackerIsNextToIt) {
  const MapData m = make_map({"......", "......"});
  GameState s = make_state(m);
  const int fighter = put(s, m, kFighter, 0, 0, 0);
  const int stealth = put(s, m, kStealth, 1, 4, 0);
  s.units[stealth].flags |= kHidden;
  put(s, m, kInfantry, 1, 5, 1);
  const Selection next_to{kPhaseMoved, static_cast<uint8_t>(fighter),
                          static_cast<uint16_t>(m.tile(3, 0))};
  EXPECT_TRUE(has_id(m, s, next_to, action_id(kChFire, m.tile(4, 0))));
  expect_matches_apply(m, s);
  expect_steps_match_actions(m, s);
}

TEST(Legal, FactoriesListWhatTheFundsBuy) {
  const MapData m = make_map({"B..", "..b"});
  GameState s = make_state(m);
  put(s, m, kInfantry, 1, 2, 0);
  s.players[0].funds = 7000;
  const Selection idle;
  EXPECT_TRUE(has_id(m, s, idle, action_id(kChSelect, m.tile(0, 0))));
  const Selection base{kPhaseFactory, kNoUnit, static_cast<uint16_t>(m.tile(0, 0))};
  EXPECT_TRUE(has_id(m, s, base, kIdBuild + kTank));
  EXPECT_FALSE(has_id(m, s, base, kIdBuild + kMdTank));
  EXPECT_FALSE(has_id(m, s, base, kIdBuild + kFighter));
  s.players[0].funds = 500;
  EXPECT_FALSE(has_id(m, s, idle, action_id(kChSelect, m.tile(0, 0))));
}

TEST(Legal, StepsRejectWhatIsNotLegalAndChangeNothing) {
  const MapData m = make_map({"B...", "...."});
  GameState s = make_state(m);
  const int inf = put(s, m, kInfantry, 0, 1, 0);
  put(s, m, kInfantry, 1, 3, 1);
  s.players[0].funds = 0;
  s.players[0].cop_cost = cop_cost(s.players[0].co, 0);
  s.players[0].scop_cost = scop_cost(s.players[0].co, 0);
  const GameState before = s;
  Selection sel;
  EXPECT_NE(step(m, s, sel, action_id(kChSelect, m.tile(0, 0))), kOk);  // nothing affordable
  EXPECT_NE(step(m, s, sel, action_id(kChSelect, m.tile(3, 1))), kOk);  // the enemy's
  EXPECT_NE(step(m, s, sel, action_id(kChWait, m.tile(1, 0))), kOk);    // nothing picked
  EXPECT_NE(step(m, s, sel, kIdCop), kOk);
  EXPECT_TRUE(same(s, before));
  EXPECT_EQ(sel, Selection{});
  ASSERT_EQ(step(m, s, sel, action_id(kChSelect, m.tile(1, 0))), kOk);
  EXPECT_NE(step(m, s, sel, action_id(kChMove, m.tile(3, 1))), kOk);  // occupied by the enemy
  EXPECT_NE(step(m, s, sel, kIdEndTurn), kOk);                        // a unit is picked
  ASSERT_EQ(step(m, s, sel, action_id(kChMove, m.tile(2, 1))), kOk);
  EXPECT_NE(step(m, s, sel, action_id(kChWait, m.tile(1, 0))), kOk);  // not where it went
  EXPECT_NE(step(m, s, sel, action_id(kChCapture, m.tile(2, 1))), kOk);
  EXPECT_TRUE(same(s, before));
  EXPECT_EQ(step(m, s, sel, action_id(kChWait, m.tile(2, 1))), kOk);
  EXPECT_EQ(s.units[inf].tile, m.tile(2, 1));
  EXPECT_EQ(sel, Selection{});
}

TEST(Legal, NothingIsLegalOnceTheGameIsOver) {
  const MapData m = make_map({"..."});
  GameState s = make_state(m);
  put(s, m, kInfantry, 0, 0, 0);
  put(s, m, kInfantry, 1, 2, 0);
  ASSERT_EQ(apply(m, s, make(kActResign, kNoUnit, 0)), kOk);
  EXPECT_TRUE(ids_of(m, s, Selection{}).empty());
  EXPECT_TRUE(legal_keys(m, s).empty());
}

// --- Random games ------------------------------------------------------------

// Random micro-steps until the game ends. Scrapping units is rare, and so are
// steps that end the turn early, so turns get played out.
TEST(Legal, RandomGamesKeepTheStateSoundAndAlwaysHaveAMove) {
  const MapData m = make_map(kArena);
  std::mt19937 rng(7);
  int games = 0, steps = 0, checked = 0;
  int played[kNumActionKinds + 1] = {};  // by kind, then trapped moves
  for (int game = 0; game < 12; ++game) {
    GameState s = arena_start(m, rng);
    ASSERT_EQ(check_state(m, s), "");
    Selection sel;
    int idle_steps = 0;
    for (int n = 0; n < 4000 && s.winner == kNoOwner; ++n) {
      const std::vector<uint16_t> ids = ids_of(m, s, sel);
      ASSERT_FALSE(ids.empty()) << "no move in phase " << int(sel.phase);
      if (sel.phase == kPhaseIdle && idle_steps++ % 25 == 3 && checked < 24) {
        expect_matches_apply(m, s);
        expect_steps_match_actions(m, s);
        ++checked;
        if (HasFailure()) return;
      }
      // An id that isn't legal changes nothing.
      const int bad = static_cast<int>(rng() % kNumActionIds);
      if (!std::binary_search(ids.begin(), ids.end(), bad)) {
        GameState t = s;
        Selection tsel = sel;
        EXPECT_FALSE(applied(step(m, t, tsel, bad)));
        EXPECT_TRUE(same(t, s));
        EXPECT_EQ(tsel, sel);
      }
      int id;
      do {
        id = ids[rng() % ids.size()];
      } while (ids.size() > 1 && ((id / kMaxTiles == kChDelete && id < kIdEndTurn && rng() % 20) ||
                                  (id == kIdEndTurn && rng() % 4)));
      Action a;
      const bool completes = step_action(m, s, sel, id, a);
      const Status st = step(m, s, sel, id);
      ASSERT_TRUE(applied(st)) << status_name(st);
      if (completes) ++played[st == kTrapped ? kNumActionKinds : a.kind];
      ASSERT_EQ(check_state(m, s), "");
      ++steps;
    }
    EXPECT_NE(s.winner, kNoOwner) << "game " << game << " didn't end";
    EXPECT_TRUE(ids_of(m, s, sel).empty());
    ++games;
  }
  EXPECT_EQ(games, 12);
  EXPECT_GT(steps, 2000);
  EXPECT_EQ(checked, 24);
  // Every kind of action came up (Resign has no step), and so did traps.
  for (int k = 0; k < kNumActionKinds; ++k) {
    if (k != kActResign) EXPECT_GT(played[k], 0) << "kind " << k;
  }
  EXPECT_GT(played[kNumActionKinds], 0);
}

}  // namespace
}  // namespace awbw
