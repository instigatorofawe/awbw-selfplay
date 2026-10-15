#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include "core/belief.h"
#include "core/encode.h"
#include "core/legal.h"
#include "core/search.h"
#include "test_util.h"

namespace awbw {
namespace {

using testing::arena_start;
using testing::kArena;
using testing::make_map;
using testing::make_state;
using testing::put;

// A stand-in network: a uniform prior, and as the value either 0 or, with
// `material`, how far the mover leads in army value (from encode()'s
// globals: tanh of the lead over 20,000 funds).
struct FakeNet {
  bool material = false;
  float picked = 0;  // added to the value of a decision with an action in progress
  float lean = 0;    // taken from player 0's value and added to player 1's
  int evals = 0;

  // One collect / submit round; false once there was nothing to search.
  bool round(BatchedSearch& bs) {
    const int n = bs.collect();
    if (n == 0) return false;
    const int size = bs.size(), width = bs.max_legal();
    const size_t plane_stride =
        static_cast<size_t>(policy_planes(bs.config().movement)) * size * size;
    std::vector<uint8_t> planes(n * plane_stride), movers(n);
    std::vector<float> globals(static_cast<size_t>(n) * kNumPolicyGlobals);
    std::vector<int64_t> legal(static_cast<size_t>(n) * width);
    bs.write_batch(planes.data(), globals.data(), legal.data(), width, movers.data());
    std::vector<float> logits(legal.size(), 0.0f), values(n, 0.0f);
    for (int j = 0; j < n; ++j) {
      const float* g = globals.data() + static_cast<size_t>(j) * kNumPolicyGlobals;
      if (material) values[j] = std::tanh(5.0f * (g[kGArmyValue] - g[kSideGlobals + kGArmyValue]));
      if (g[kNumGlobals + kSelGPhase + kPhaseIdle] == 0) values[j] += picked;
      values[j] += movers[j] == 0 ? -lean : lean;
    }
    bs.submit(logits.data(), width, values.data());
    evals += n;
    return true;
  }

  void run(BatchedSearch& bs) {
    while (round(bs)) {
    }
  }
};

// Orange Star's HQ and base at the top left, Blue Moon's HQ at the bottom right.
const std::vector<std::string> kField = {
    "HB...",  //
    "....h",  //
};

int slot_at(const GameState& s, const MapData& m, int x, int y) { return s.unit_at[m.tile(x, y)]; }

// make_state with the COs' power costs, so an empty meter can't use a power.
GameState start(const MapData& m) {
  GameState s = make_state(m);
  for (Player& p : s.players) {
    p.cop_cost = cop_cost(p.co, 0);
    p.scop_cost = scop_cost(p.co, 0);
  }
  return s;
}

TEST(Search, SequentialHalvingSpreadsTheVisits) {
  // Four steps: build, pick the Infantry, scrap it, end the turn. Scrapping
  // the last unit loses, so it's the one dropped first.
  const MapData m = make_map(kField);
  GameState s = start(m);
  put(s, m, kInfantry, 0, 2, 0);
  put(s, m, kInfantry, 1, 3, 1);
  s.players[0].funds = 1000;
  SearchConfig cfg;
  cfg.sims = 8;
  cfg.gumbel_scale = 0;
  BatchedSearch bs(cfg, 8);
  const int slot = bs.add_root(m, s);
  FakeNet net;
  net.run(bs);
  ASSERT_TRUE(bs.done(slot));
  const SearchResult& r = bs.result(slot);
  ASSERT_EQ(r.ids.size(), 4u);
  std::vector<uint32_t> visits = r.visits;
  std::sort(visits.begin(), visits.end());
  EXPECT_EQ(visits, (std::vector<uint32_t>{1, 1, 3, 3}));
  const int del = action_id(kChDelete, m.tile(2, 0));
  const auto at = std::find(r.ids.begin(), r.ids.end(), del) - r.ids.begin();
  EXPECT_EQ(r.visits[at], 1u);
  EXPECT_FLOAT_EQ(r.q[at], -1.0f);
  EXPECT_NE(r.id, del);
  EXPECT_EQ(r.sims, 8);
  EXPECT_EQ(net.evals, 1 + 8 - 1);  // the root, then a leaf per simulation but the loss
  EXPECT_NEAR(std::accumulate(r.policy.begin(), r.policy.end(), 0.0), 1.0, 1e-5);
  EXPECT_LT(r.policy[at], 0.05f);
}

TEST(Search, ValuesAreReadFromTheMoversSide) {
  // As above with Blue Moon to move: scrapping their last unit loses for them.
  const MapData m = make_map(kField);
  GameState s = start(m);
  put(s, m, kInfantry, 0, 2, 0);
  put(s, m, kInfantry, 1, 3, 1);
  s.active = 1;
  SearchConfig cfg;
  cfg.sims = 6;
  cfg.gumbel_scale = 0;
  BatchedSearch bs(cfg, 8);
  const int slot = bs.add_root(m, s);
  FakeNet().run(bs);
  const SearchResult& r = bs.result(slot);
  ASSERT_EQ(r.ids.size(), 3u);  // pick, scrap, end the turn
  const int del = action_id(kChDelete, m.tile(3, 1));
  const auto at = std::find(r.ids.begin(), r.ids.end(), del) - r.ids.begin();
  EXPECT_EQ(r.mover, 1);
  EXPECT_FLOAT_EQ(r.q[at], -1.0f);
  EXPECT_NE(r.id, del);
}

TEST(Search, SelfPlayTakesAWinningCapture) {
  // Orange Star's Infantry stands on Blue Moon's HQ with 10 points left: one
  // more capture wins, three steps down (pick it, stay, capture).
  const std::vector<std::string> rows = {"H...h"};
  const MapData m = make_map(rows);
  GameState s = start(m);
  put(s, m, kInfantry, 0, 4, 0);
  put(s, m, kInfantry, 1, 2, 0);
  s.prop_points[m.prop[m.tile(4, 0)]] = 10;
  SearchConfig cfg;
  cfg.sims = 16;
  cfg.gumbel_scale = 0;
  cfg.max_days = 5;
  BatchedSearch bs(cfg, 8);
  bs.add_game(m, s);
  FakeNet().run(bs);
  const std::vector<GameRecord> recs = bs.take_records();
  ASSERT_EQ(recs.size(), 1u);
  EXPECT_EQ(recs[0].winner, 0);
  EXPECT_EQ(recs[0].actions, 1);
  EXPECT_EQ(recs[0].decisions, 3);
  EXPECT_FALSE(recs[0].capped);
  const std::vector<SearchSample> samples = bs.take_samples();
  ASSERT_EQ(samples.size(), 3u);
  // Sequential halving visits Wait as often as Capture, so the root's mean
  // value is about half; the capture's Q and the improved policy show the win.
  const SearchResult& last = samples[2].result;
  ASSERT_EQ(last.ids,
            (std::vector<uint16_t>{static_cast<uint16_t>(action_id(kChWait, m.tile(4, 0))),
                                   static_cast<uint16_t>(action_id(kChCapture, m.tile(4, 0)))}));
  EXPECT_EQ(last.id, last.ids[1]);
  EXPECT_FLOAT_EQ(last.q[1], 1.0f);
  EXPECT_GT(last.policy[1], 0.9f);
}

TEST(Search, ChanceNodesVisitResultsByTheirProbability) {
  const std::vector<std::string> rows = {"H...h"};
  const MapData m = make_map(rows);
  GameState s = start(m);
  const int tank = put(s, m, kTank, 0, 1, 0);
  put(s, m, kTank, 1, 2, 0, 75);
  put(s, m, kInfantry, 1, 4, 0);
  const Selection moved{kPhaseMoved, static_cast<uint8_t>(tank),
                        static_cast<uint16_t>(m.tile(1, 0))};
  SearchConfig cfg;
  cfg.sims = 120;
  BatchedSearch bs(cfg, 8);
  const int slot = bs.add_root(m, s, moved);
  FakeNet net;
  net.material = true;
  while (net.round(bs)) {
  }
  const SearchResult& r = bs.result(slot);
  ASSERT_EQ(r.ids.size(), 2u);  // Wait, Fire
  EXPECT_EQ(r.visits[0] + r.visits[1], 120u);
  const std::vector<SearchNode>& nodes = bs.nodes(slot);
  const std::vector<SearchEdge>& edges = bs.edges(slot);
  const SearchEdge& fire = edges[nodes[0].first + 1];
  ASSERT_EQ(fire.id, action_id(kChFire, m.tile(2, 0)));
  const SearchNode& chance = nodes[fire.child];
  ASSERT_EQ(chance.kind, kNodeChance);
  ASSERT_GE(chance.n, 3);
  double total = 0;
  for (int i = 0; i < chance.n; ++i) {
    const SearchEdge& o = edges[chance.first + i];
    total += o.prior;
    const double seen = o.child == kNoNode ? 0 : nodes[o.child].visits;
    EXPECT_LT(std::abs(seen - o.prior * chance.visits), 1.5) << "result " << i;
  }
  EXPECT_NEAR(total, 1.0, 1e-6);
  // Firing on a damaged Tank gains army value, so it gets more visits and wins.
  EXPECT_EQ(r.id, fire.id);
  EXPECT_GT(r.q[1], r.q[0]);
}

// A Tank at (1, 0) firing on an enemy Tank of `def_hp` at (2, 0); with
// `unknown`, both units' beliefs allow every internal HP of their displays.
struct Fight {
  MapData m = make_map({"H...h"});
  GameState s;
  int att = 0, def = 0;
  Action a;

  Fight(int att_hp, int def_hp, bool unknown) : s(start(m)) {
    att = put(s, m, kTank, 0, 1, 0, att_hp);
    def = put(s, m, kTank, 1, 2, 0, def_hp);
    put(s, m, kInfantry, 1, 4, 0);
    if (unknown) {
      belief_unknown(s, att);
      belief_unknown(s, def);
    }
    a.kind = kActFire;
    a.unit = static_cast<uint8_t>(att);
    a.to = static_cast<uint16_t>(m.tile(1, 0));
    a.target = static_cast<uint16_t>(m.tile(2, 0));
  }
};

// combat_outcomes at the units' own HPs, merged by displayed HP: (att, def) -> p.
std::map<std::pair<int, int>, double> by_display(const Fight& f, int att_hp, int def_hp) {
  Combatant att = combatant(f.m, f.s, f.att, f.m.tile(1, 0));
  Combatant def = combatant(f.m, f.s, f.def, f.m.tile(2, 0));
  att.hp = static_cast<uint8_t>(att_hp);
  def.hp = static_cast<uint8_t>(def_hp);
  std::vector<Outcome> outs;
  EXPECT_TRUE(combat_outcomes(att, def, 1, outs));
  std::map<std::pair<int, int>, double> p;
  for (const Outcome& o : outs) p[{displayed_hp(o.att_hp), displayed_hp(o.def_hp)}] += o.prob;
  return p;
}

TEST(Search, FireOutcomesAreWhatThePlayersSee) {
  // A known HP: combat_outcomes, merged by display.
  {
    Fight f(100, 75, false);
    std::vector<SeenOutcome> out;
    ASSERT_TRUE(fire_outcomes(f.m, f.s, f.a, out));
    const auto want = by_display(f, 100, 75);
    ASSERT_EQ(out.size(), want.size());
    for (const SeenOutcome& o : out) {
      EXPECT_EQ(o.att_from, 100);
      EXPECT_EQ(o.def_from, 75);
      EXPECT_NEAR(o.prob, want.at({displayed_hp(o.att_hp), displayed_hp(o.def_hp)}), 1e-12);
    }
  }
  // Unknown HPs: the average over every pair of internal HPs the displays
  // allow, whatever the units' own HPs are.
  std::map<std::pair<int, int>, double> want;
  for (int i = 51; i <= 60; ++i)
    for (int j = 71; j <= 80; ++j)
      for (const auto& [key, p] : by_display(Fight(i, j, true), i, j)) want[key] += p / 100;
  std::vector<SeenOutcome> first;
  for (const auto& [att_hp, def_hp] : {std::pair{55, 75}, {51, 80}, {60, 71}}) {
    Fight f(att_hp, def_hp, true);
    std::vector<SeenOutcome> out;
    ASSERT_TRUE(fire_outcomes(f.m, f.s, f.a, out));
    ASSERT_EQ(out.size(), want.size());
    double total = 0;
    for (size_t k = 0; k < out.size(); ++k) {
      const SeenOutcome& o = out[k];
      total += o.prob;
      EXPECT_NEAR(o.prob, want.at({displayed_hp(o.att_hp), displayed_hp(o.def_hp)}), 1e-12);
      // The world it goes on in gives this result.
      EXPECT_EQ(displayed_hp(o.att_from), 6);
      EXPECT_EQ(displayed_hp(o.def_from), 8);
      const std::pair<int, int> seen{displayed_hp(o.att_hp), displayed_hp(o.def_hp)};
      EXPECT_GT(by_display(f, o.att_from, o.def_from)[seen], 0);
      if (!first.empty()) {
        EXPECT_EQ(o.att_from, first[k].att_from);
        EXPECT_EQ(o.def_from, first[k].def_from);
        EXPECT_EQ(o.att_hp, first[k].att_hp);
        EXPECT_EQ(o.def_hp, first[k].def_hp);
      }
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
    if (first.empty()) first = out;
  }
}

TEST(Search, TheTreeCantTellInternalHpsApart) {
  // Searches of two states that differ only in internal HPs the players
  // can't see (the displays and beliefs are the same) come out the same.
  SearchResult results[2];
  for (int v = 0; v < 2; ++v) {
    Fight f(v ? 51 : 60, v ? 80 : 71, true);
    const Selection moved{kPhaseMoved, static_cast<uint8_t>(f.att), f.a.to};
    SearchConfig cfg;
    cfg.sims = 200;
    cfg.seed = 5;
    BatchedSearch bs(cfg, 8);
    const int slot = bs.add_root(f.m, f.s, moved);
    FakeNet net;
    net.material = true;
    net.run(bs);
    results[v] = bs.result(slot);
    const std::vector<SearchNode>& nodes = bs.nodes(slot);
    const std::vector<SearchEdge>& edges = bs.edges(slot);
    const SearchNode& chance = nodes[edges[nodes[0].first + 1].child];
    ASSERT_EQ(chance.kind, kNodeChance);
    std::vector<SeenOutcome> out;
    ASSERT_TRUE(fire_outcomes(f.m, f.s, f.a, out));
    ASSERT_EQ(chance.n, out.size());
    for (int k = 0; k < chance.n; ++k)
      EXPECT_FLOAT_EQ(edges[chance.first + k].prior, static_cast<float>(out[k].prob));
  }
  EXPECT_EQ(results[0].id, results[1].id);
  EXPECT_EQ(results[0].visits, results[1].visits);
  EXPECT_EQ(results[0].q, results[1].q);
  EXPECT_EQ(results[0].policy, results[1].policy);
}

TEST(Search, PicksTakeTheValueOfTheirState) {
  const MapData m = make_map(kField);
  GameState s = start(m);
  put(s, m, kInfantry, 0, 2, 0);
  put(s, m, kInfantry, 1, 3, 1);
  s.players[0].funds = 1000;
  for (const bool pick_values : {false, true}) {
    SearchConfig cfg;
    cfg.sims = 8;
    cfg.pick_values = pick_values;
    BatchedSearch bs(cfg, 8);
    const int slot = bs.add_root(m, s);
    FakeNet net;
    net.picked = 0.5f;  // the network values the same state differently once a unit is picked
    net.run(bs);
    int picks = 0;
    for (const SearchNode& n : bs.nodes(slot)) {
      if (n.kind != kNodeDecision || n.sel.phase == kPhaseIdle) continue;
      const float mine = n.mover == 0 ? n.raw : -n.raw;  // raw is from player 0's side
      EXPECT_FLOAT_EQ(mine, pick_values ? 0.5f : 0.0f);
      ++picks;
    }
    EXPECT_GT(picks, 0);
  }
}

TEST(Search, ASingleLegalStepNeedsNoSearch) {
  const std::vector<std::string> rows = {"H...h"};
  const MapData m = make_map(rows);
  GameState s = start(m);
  const int inf = put(s, m, kInfantry, 0, 1, 0);
  s.units[inf].flags |= kMoved;  // only End Turn is left
  put(s, m, kInfantry, 1, 3, 0);
  BatchedSearch bs(SearchConfig{}, 8);
  const int slot = bs.add_root(m, s);
  EXPECT_TRUE(bs.done(slot));
  EXPECT_EQ(bs.collect(), 0);
  EXPECT_EQ(bs.result(slot).id, kIdEndTurn);
}

// The tree's invariants: decisions with a choice, results that add up, and
// visits that agree with their children's.
void expect_sound_tree(const BatchedSearch& bs, int slot) {
  const std::vector<SearchNode>& nodes = bs.nodes(slot);
  const std::vector<SearchEdge>& edges = bs.edges(slot);
  for (size_t i = 0; i < nodes.size(); ++i) {
    const SearchNode& n = nodes[i];
    uint32_t below = 0;
    for (int k = 0; k < n.n; ++k) {
      const SearchEdge& e = edges[n.first + k];
      if (e.child != kNoNode) below += nodes[e.child].visits;
    }
    if (n.kind == kNodeDecision) {
      EXPECT_GE(n.n, 2) << "a forced step got a node";
      EXPECT_EQ(n.visits, below + (n.evaluated ? 1 : 0)) << "node " << i;
    } else if (n.kind == kNodeChance) {
      double total = 0;
      for (int k = 0; k < n.n; ++k) total += edges[n.first + k].prior;
      EXPECT_NEAR(total, 1.0, 1e-5);
      EXPECT_EQ(n.visits, below);
    } else {
      EXPECT_EQ(n.n, 0);
    }
  }
}

struct Played {
  std::vector<GameRecord> records;
  std::vector<SearchSample> samples;
  int evals = 0;
};

Played self_play(const MapData& m, const SearchConfig& cfg, int games, bool check_trees,
                 float lean = 0) {
  std::mt19937 rng(11);
  BatchedSearch bs(cfg, 16);
  for (int g = 0; g < games; ++g) bs.add_game(m, arena_start(m, rng));
  FakeNet net;
  net.material = lean == 0;
  net.lean = lean;
  Played out;
  while (net.round(bs)) {
    if (check_trees) {
      for (int i = 0; i < bs.slots(); ++i) {
        if (!bs.nodes(i).empty()) expect_sound_tree(bs, i);
      }
      if (::testing::Test::HasFailure()) break;
    }
    for (SearchSample& smp : bs.take_samples()) out.samples.push_back(std::move(smp));
  }
  for (SearchSample& smp : bs.take_samples()) out.samples.push_back(std::move(smp));
  out.records = bs.take_records();
  out.evals = net.evals;
  return out;
}

TEST(Search, SelfPlayGamesEndWithSoundSamples) {
  const MapData m = make_map(kArena);
  SearchConfig cfg;
  cfg.sims = 8;
  cfg.considered = 4;
  cfg.max_days = 3;
  cfg.seed = 5;
  const Played p = self_play(m, cfg, 4, true);
  ASSERT_EQ(p.records.size(), 4u);
  int decisions = 0;
  for (const GameRecord& r : p.records) {
    EXPECT_EQ(check_state(m, r.final_state), "");
    EXPECT_TRUE(r.capped || r.final_state.winner != kNoOwner);
    EXPECT_LE(r.days, 4);
    decisions += r.decisions;
  }
  EXPECT_EQ(static_cast<int>(p.samples.size()), decisions);  // every search is full
  EXPECT_GT(decisions, 100);
  std::vector<uint16_t> ids;
  for (const SearchSample& smp : p.samples) {
    const SearchResult& r = smp.result;
    ASSERT_EQ(check_state(m, smp.state), "");
    legal_ids(m, smp.state, smp.sel, ids);
    ASSERT_EQ(ids, r.ids);
    EXPECT_TRUE(std::binary_search(ids.begin(), ids.end(), static_cast<uint16_t>(r.id)));
    EXPECT_EQ(r.mover, smp.state.active);
    EXPECT_EQ(r.sims, cfg.sims);
    EXPECT_NEAR(std::accumulate(r.policy.begin(), r.policy.end(), 0.0), 1.0, 1e-5);
    EXPECT_NEAR(std::accumulate(r.prior.begin(), r.prior.end(), 0.0), 1.0, 1e-5);
    EXPECT_LE(std::abs(r.value), 1.0f);
    // The picks reach the selection.
    uint16_t picks[2];
    const int k = selection_steps(smp.state, smp.sel, picks);
    GameState t = smp.state;
    Selection sel;
    for (int j = 0; j < k; ++j) EXPECT_EQ(step(m, t, sel, picks[j]), kOk);
    EXPECT_EQ(sel, smp.sel);
    if (::testing::Test::HasFailure()) return;
  }

  // The same seed plays the same games.
  const Played again = self_play(m, cfg, 4, false);
  ASSERT_EQ(again.records.size(), p.records.size());
  for (size_t i = 0; i < p.records.size(); ++i) {
    EXPECT_EQ(again.records[i].game, p.records[i].game);
    EXPECT_EQ(again.records[i].winner, p.records[i].winner);
    EXPECT_EQ(again.records[i].actions, p.records[i].actions);
  }
  EXPECT_EQ(again.evals, p.evals);
}

TEST(Search, FastSearchesArePlayedButNotKept) {
  const MapData m = make_map(kArena);
  SearchConfig cfg;
  cfg.sims = 8;
  cfg.fast_sims = 2;
  cfg.full_fraction = 0.25f;
  cfg.considered = 4;
  cfg.max_days = 2;
  const Played p = self_play(m, cfg, 2, false);
  int decisions = 0;
  for (const GameRecord& r : p.records) decisions += r.decisions;
  const double kept = static_cast<double>(p.samples.size()) / decisions;
  EXPECT_GT(kept, 0.15);
  EXPECT_LT(kept, 0.35);
  for (const SearchSample& smp : p.samples) EXPECT_EQ(smp.result.sims, 8);
}

TEST(Search, ALosingPlayerResignsAtItsTurnStart) {
  const MapData m = make_map(kArena);
  SearchConfig cfg;
  cfg.sims = 4;
  cfg.considered = 4;
  cfg.max_days = 6;
  cfg.resign_turns = 2;
  cfg.resign_off = 0;
  // Player 0's every value is -1: it resigns at the start of its second turn.
  Played p = self_play(m, cfg, 4, false, 1.0f);
  ASSERT_EQ(p.records.size(), 4u);
  for (const GameRecord& r : p.records) {
    EXPECT_TRUE(r.resigned);
    EXPECT_FALSE(r.capped);
    EXPECT_EQ(r.winner, 1);
    EXPECT_EQ(r.days, 2);
    EXPECT_EQ(r.final_state.winner, kNoOwner);
    EXPECT_EQ(r.would_resign, -1);
  }
  // The resigning decision is still a sample.
  ASSERT_FALSE(p.samples.empty());
  const SearchSample& last = p.samples.back();
  EXPECT_EQ(last.result.mover, 0);
  EXPECT_EQ(last.state.day, 2);
  EXPECT_LE(last.result.value, cfg.resign_value);

  // Games that play on note when the rule would have resigned.
  cfg.resign_off = 1;
  p = self_play(m, cfg, 4, false, 1.0f);
  for (const GameRecord& r : p.records) {
    EXPECT_FALSE(r.resigned);
    EXPECT_GT(r.days, 2);
    EXPECT_EQ(r.would_resign, 0);
    EXPECT_EQ(r.would_resign_day, 2);
  }

  // A line below every value never resigns, and neither does the default.
  cfg.resign_off = 0;
  cfg.resign_value = -1.5f;
  p = self_play(m, cfg, 2, false, 1.0f);
  for (const GameRecord& r : p.records) EXPECT_FALSE(r.resigned);
  cfg = SearchConfig{};
  cfg.sims = 4;
  cfg.considered = 4;
  cfg.max_days = 3;
  p = self_play(m, cfg, 2, false, 1.0f);
  for (const GameRecord& r : p.records) {
    EXPECT_FALSE(r.resigned);
    EXPECT_EQ(r.would_resign, -1);
  }
}

TEST(Search, ADayCapDecidesByPropertiesThenArmy) {
  const std::vector<std::string> rows = {"H.#.h"};
  const MapData m = make_map(rows);
  GameState s = start(m);
  put(s, m, kInfantry, 0, 1, 0);
  put(s, m, kInfantry, 1, 3, 0);
  EXPECT_EQ(day_limit_winner(m, s), kDraw);
  s.units[slot_at(s, m, 3, 0)].hp = 55;
  EXPECT_EQ(day_limit_winner(m, s), 0);  // the larger army
  s.prop_owner[m.prop[m.tile(2, 0)]] = 1;
  EXPECT_EQ(day_limit_winner(m, s), 1);  // more properties first
}

}  // namespace
}  // namespace awbw
