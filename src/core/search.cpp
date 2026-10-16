#include "core/search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/actions.h"
#include "core/encode.h"
#include "core/rng.h"

namespace awbw {

namespace {

constexpr uint8_t kDecision = kNodeDecision, kChance = kNodeChance, kTerminal = kNodeTerminal;
enum SlotMode : uint8_t { kFree, kOneSearch, kGame };

float side(int player) { return player == 0 ? 1.0f : -1.0f; }
// A result from player 0's side.
float result_value(int winner) { return winner == kDraw ? 0.0f : side(winner); }

double uniform(uint64_t (&rng)[2]) {
  return (static_cast<double>(rng_next(rng) >> 11) + 0.5) * 0x1.0p-53;
}

// Sequential halving's schedule, as mctx's get_sequence_of_considered_visits:
// at simulation i the root visits a considered step with exactly out[i]
// visits so far, so `considered` steps get a visit each, then the better half
// more, and so on, spreading `sims` simulations over log2(considered) rounds.
void halving_schedule(int considered, int sims, std::vector<int>& out) {
  out.clear();
  if (considered <= 1) {
    for (int i = 0; i < sims; ++i) out.push_back(i);
    return;
  }
  const int log2max = static_cast<int>(std::ceil(std::log2(considered)));
  std::vector<int> visits(considered, 0);
  int n = considered;
  while (static_cast<int>(out.size()) < sims) {
    const int extra = std::max(1, sims / (log2max * n));
    for (int r = 0; r < extra; ++r) {
      out.insert(out.end(), visits.begin(), visits.begin() + n);
      for (int i = 0; i < n; ++i) ++visits[i];
    }
    n = std::max(2, n / 2);
  }
  out.resize(sims);
}

void softmax(const std::vector<float>& x, std::vector<float>& out) {
  out.resize(x.size());
  const float hi = *std::max_element(x.begin(), x.end());
  double total = 0;
  for (size_t i = 0; i < x.size(); ++i) total += out[i] = std::exp(x[i] - hi);
  for (float& p : out) p = static_cast<float>(p / total);
}

int argmax(const std::vector<float>& x) {
  return static_cast<int>(std::max_element(x.begin(), x.end()) - x.begin());
}

}  // namespace

int day_limit_winner(const MapData& m, const GameState& s) {
  const int props[2] = {count_props(m, s, 0), count_props(m, s, 1)};
  if (props[0] != props[1]) return props[0] > props[1] ? 0 : 1;
  int64_t army[2] = {};
  for (const Unit& u : s.units) {
    if (u.alive() && u.owner < kMaxPlayers)
      army[u.owner] += int64_t{kUnits[u.type].cost} * displayed_hp(u.hp);
  }
  if (army[0] != army[1]) return army[0] > army[1] ? 0 : 1;
  return kDraw;
}

BatchedSearch::BatchedSearch(const SearchConfig& cfg, int size) : cfg_(cfg), size_(size) {
  if (size < 1 || size > kMaxWidth || size > kMaxHeight)
    throw std::invalid_argument("size must be 1.." + std::to_string(kMaxWidth));
  if (cfg.sims < 0 || cfg.fast_sims < 0 || cfg.considered < 1)
    throw std::invalid_argument("sims and fast_sims must be >= 0 and considered >= 1");
}

int BatchedSearch::new_slot() {
  for (size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].mode == kFree) return static_cast<int>(i);
  }
  slots_.emplace_back();
  return static_cast<int>(slots_.size()) - 1;
}

int BatchedSearch::add_root(const MapData& m, const GameState& s, const Selection& sel) {
  if (m.width > size_ || m.height > size_)
    throw std::invalid_argument("the map doesn't fit in the search's window");
  if (s.map_id != m.id) throw std::invalid_argument("the state is for another map");
  if (s.winner != kNoOwner) throw std::invalid_argument("the game is over");
  thread_local std::vector<uint16_t> ids;
  legal_ids(m, s, sel, ids);
  if (ids.empty()) throw std::invalid_argument("nothing is legal from this selection");
  const int i = new_slot();
  Slot& sl = slots_[i];
  sl.ids = ids;
  sl.mode = kOneSearch;
  sl.finished = false;
  sl.map = &m;
  sl.game = s;
  sl.sel = sel;
  sl.full = true;
  rng_seed(sl.rng, cfg_.seed + 0x9E3779B97F4A7C15ull * ++searches_);
  if (ids.size() == 1) {
    // Nothing to search.
    SearchResult& r = sl.result;
    r = SearchResult{};
    r.id = ids[0];
    r.mover = s.active;
    r.ids = ids;
    r.prior = r.policy = {1.0f};
    r.visits = {0};
    r.q = {0.0f};
    sl.nodes.clear();
    sl.finished = true;
    return i;
  }
  start_root(sl);
  return i;
}

uint64_t BatchedSearch::add_game(const MapData& m, const GameState& s) {
  if (m.width > size_ || m.height > size_)
    throw std::invalid_argument("the map doesn't fit in the search's window");
  if (s.map_id != m.id) throw std::invalid_argument("the state is for another map");
  const int i = new_slot();
  Slot& sl = slots_[i];
  sl.mode = kGame;
  sl.finished = false;
  sl.map = &m;
  sl.game = s;
  sl.sel = Selection{};
  sl.game_no = next_game_++;
  sl.actions = sl.decisions = sl.skipped = 0;
  rng_seed(sl.rng, cfg_.seed ^ (0xD1B54A32D192ED03ull * (sl.game_no + 1)));
  rng_seed(sl.game.rng, rng_next(sl.rng));  // games from one start draw their own luck
  sl.start = sl.game;
  sl.steps.clear();
  sl.searched.clear();
  sl.can_resign = cfg_.resign_turns > 0 && !(uniform(sl.rng) < cfg_.resign_off);
  sl.turn = -1;
  sl.low[0] = sl.low[1] = 0;
  sl.would_resign = sl.would_resign_day = -1;
  begin_decision(sl);
  return sl.game_no;
}

// Take the game's forced steps, then search the next decision, or end the game.
void BatchedSearch::begin_decision(Slot& sl) {
  const MapData& m = *sl.map;
  GameState& g = sl.game;
  Action a;
  for (;;) {
    if (g.winner != kNoOwner) return finish_game(sl, false);
    if ((cfg_.max_days > 0 && g.day > cfg_.max_days) ||
        (cfg_.max_actions > 0 && sl.actions >= cfg_.max_actions))
      return finish_game(sl, true);
    legal_ids(m, g, sl.sel, sl.ids);
    if (sl.ids.size() != 1) break;
    const bool completes = step_action(m, g, sl.sel, sl.ids[0], a);
    if (!applied(step(m, g, sl.sel, sl.ids[0]))) throw std::logic_error("a legal step was refused");
    sl.steps.push_back(sl.ids[0]);
    sl.actions += completes;
  }
  if (sl.ids.empty()) return finish_game(sl, true);  // can't happen: no step only once it's over
  sl.full = cfg_.full_fraction >= 1.0f || uniform(sl.rng) < cfg_.full_fraction;
  start_root(sl);
}

void BatchedSearch::start_root(Slot& sl) {
  sl.target = sl.full ? cfg_.sims : cfg_.fast_sims;
  sl.sims = 0;
  sl.nodes.clear();
  sl.edges.clear();
  sl.states.clear();
  sl.nodes.reserve(2 * static_cast<size_t>(sl.target) + 1);
  sl.states.reserve(static_cast<size_t>(sl.target) + 1);
  sl.states.push_back(sl.game);
  add_decision(sl, sl.game, sl.sel, 0);
  const SearchNode& root = sl.nodes[0];
  sl.gumbel.resize(root.n);
  for (float& g : sl.gumbel) {
    g = cfg_.gumbel_scale > 0
            ? cfg_.gumbel_scale * static_cast<float>(-std::log(-std::log(uniform(sl.rng))))
            : 0.0f;
  }
  halving_schedule(std::min<int>(cfg_.considered, root.n), sl.target, sl.schedule);
}

uint32_t BatchedSearch::add_decision(Slot& sl, const GameState& s, const Selection& sel,
                                     int32_t state_index) {
  SearchNode n;
  n.kind = kDecision;
  n.mover = s.active;
  n.state = state_index;
  n.sel = sel;
  n.first = static_cast<uint32_t>(sl.edges.size());
  n.n = static_cast<uint16_t>(sl.ids.size());
  for (uint16_t id : sl.ids) {
    SearchEdge e;
    e.id = id;
    sl.edges.push_back(e);
  }
  sl.nodes.push_back(n);
  return static_cast<uint32_t>(sl.nodes.size()) - 1;
}

// A Fire step `id` from (s, sel), with a child per result in sl.outs.
uint32_t BatchedSearch::add_chance(Slot& sl, const GameState& s, const Selection& sel, int id,
                                   int32_t state_index) {
  SearchNode n;
  n.kind = kChance;
  n.mover = s.active;
  n.state = state_index;
  n.sel = sel;
  n.step = static_cast<uint16_t>(id);
  n.first = static_cast<uint32_t>(sl.edges.size());
  n.n = static_cast<uint16_t>(sl.outs.size());
  for (const SeenOutcome& o : sl.outs) {
    SearchEdge e;
    e.att_from = o.att_from;
    e.def_from = o.def_from;
    e.att_hp = o.att_hp;
    e.def_hp = o.def_hp;
    e.prior = static_cast<float>(o.prob);
    sl.edges.push_back(e);
  }
  sl.nodes.push_back(n);
  return static_cast<uint32_t>(sl.nodes.size()) - 1;
}

// The node for (s, sel), after the steps with a single legal id: a decision,
// a chance node or the end. `state_index` is where `s` is stored already, or
// -1 if it isn't.
uint32_t BatchedSearch::settle(Slot& sl, GameState& s, Selection& sel, int32_t state_index) {
  const MapData& m = *sl.map;
  Action a;
  auto store = [&] {
    if (state_index < 0) {
      sl.states.push_back(s);
      state_index = static_cast<int32_t>(sl.states.size()) - 1;
    }
  };
  for (;;) {
    const bool over = s.winner != kNoOwner;
    if (over || (cfg_.max_days > 0 && s.day > cfg_.max_days)) {
      SearchNode n;
      n.kind = kTerminal;
      n.mover = s.active;
      n.raw = result_value(over ? s.winner : day_limit_winner(m, s));
      sl.nodes.push_back(n);
      return static_cast<uint32_t>(sl.nodes.size()) - 1;
    }
    legal_ids(m, s, sel, sl.ids);
    if (sl.ids.size() != 1) break;
    const int id = sl.ids[0];
    const bool completes = step_action(m, s, sel, id, a);
    if (completes && fire_outcomes(m, s, a, sl.outs) && sl.outs.size() > 1) {
      store();
      return add_chance(sl, s, sel, id, state_index);
    }
    if (!applied(step(m, s, sel, id))) throw std::logic_error("a legal step was refused");
    if (completes) state_index = -1;
  }
  if (sl.ids.empty()) throw std::logic_error("no legal step in a game that is going");
  store();
  return add_decision(sl, s, sel, state_index);
}

uint32_t BatchedSearch::expand_step(Slot& sl, uint32_t node, int edge) {
  const MapData& m = *sl.map;
  const SearchNode p = sl.nodes[node];
  const int id = sl.edges[edge].id;
  GameState s = sl.states[p.state];
  Selection sel = p.sel;
  Action a;
  const bool completes = step_action(m, s, sel, id, a);
  if (completes && fire_outcomes(m, s, a, sl.outs) && sl.outs.size() > 1)
    return add_chance(sl, s, sel, id, p.state);  // the state before the Fire
  if (!applied(step(m, s, sel, id))) throw std::logic_error("a legal step was refused");
  return settle(sl, s, sel, completes ? -1 : p.state);
}

uint32_t BatchedSearch::expand_outcome(Slot& sl, uint32_t node, int edge) {
  const MapData& m = *sl.map;
  const SearchNode c = sl.nodes[node];
  const SearchEdge e = sl.edges[edge];
  GameState s = sl.states[c.state];
  Selection sel = c.sel;
  Action a;
  step_action(m, s, sel, c.step, a);
  // The world this result is likeliest in: the internal HPs before the fight.
  s.units[s.unit_at[a.target]].hp = e.def_from;
  s.units[a.unit].hp = e.att_from;
  Detail d;
  d.att_hp = e.att_hp;
  d.def_hp = e.def_hp;
  if (!applied(step(m, s, sel, c.step, d))) throw std::logic_error("a fight result was refused");
  return settle(sl, s, sel, -1);
}

// For each of a decision's steps: its logit, sigma(completed Q) and visits.
// An unvisited step's Q is the node's value mixed with its visited steps'
// (mctx's qtransform_completed_by_mix_value); the Qs are then mapped to 0..1
// (SearchConfig::rescale).
void BatchedSearch::completed_q(const Slot& sl, uint32_t node, std::vector<float>& logits,
                                std::vector<float>& sigma, std::vector<uint32_t>& visits) const {
  const SearchNode& n = sl.nodes[node];
  const float sgn = side(n.mover);
  logits.resize(n.n);
  sigma.resize(n.n);
  visits.resize(n.n);
  uint32_t total = 0, most = 0;
  for (int i = 0; i < n.n; ++i) {
    const SearchEdge& e = sl.edges[n.first + i];
    logits[i] = e.prior;
    visits[i] = e.child == kNoNode ? 0 : sl.nodes[e.child].visits;
    total += visits[i];
    most = std::max(most, visits[i]);
  }
  thread_local std::vector<float> prob;
  softmax(logits, prob);
  // sigma holds the Qs first.
  double visited_prob = 0, weighted_q = 0;
  for (int i = 0; i < n.n; ++i) {
    if (!visits[i]) continue;
    const SearchNode& c = sl.nodes[sl.edges[n.first + i].child];
    sigma[i] = sgn * c.value_sum / static_cast<float>(c.visits);
    const double p = std::max(prob[i], std::numeric_limits<float>::min());
    visited_prob += p;
    weighted_q += p * sigma[i];
  }
  const float raw = sgn * n.raw;
  const float mixed =
      total ? static_cast<float>((raw + total * (weighted_q / visited_prob)) / (total + 1)) : raw;
  float lo = -1.0f, hi = 1.0f;
  if (cfg_.rescale) {
    lo = std::numeric_limits<float>::infinity();
    hi = -lo;
  }
  for (int i = 0; i < n.n; ++i) {
    if (!visits[i]) sigma[i] = mixed;
    if (cfg_.rescale) {
      lo = std::min(lo, sigma[i]);
      hi = std::max(hi, sigma[i]);
    }
  }
  const float scale = (cfg_.visit_scale + most) * cfg_.value_scale;
  const float range = std::max(hi - lo, 1e-8f);
  for (float& x : sigma) x = scale * (x - lo) / range;
}

// Below the root: the step whose share of the visits is furthest below the
// improved policy's.
int BatchedSearch::select_decision(const Slot& sl, uint32_t node) const {
  thread_local std::vector<float> logits, sigma, pi;
  thread_local std::vector<uint32_t> visits;
  completed_q(sl, node, logits, sigma, visits);
  uint32_t total = 0;
  for (size_t i = 0; i < logits.size(); ++i) {
    logits[i] += sigma[i];
    total += visits[i];
  }
  softmax(logits, pi);
  for (size_t i = 0; i < pi.size(); ++i) pi[i] -= static_cast<float>(visits[i]) / (1 + total);
  return sl.nodes[node].first + argmax(pi);
}

// At the root: sequential halving among the steps Gumbel-top-k considers.
int BatchedSearch::select_root(const Slot& sl) const {
  thread_local std::vector<float> logits, sigma, score;
  thread_local std::vector<uint32_t> visits;
  completed_q(sl, 0, logits, sigma, visits);
  uint32_t sim = 0;
  for (uint32_t v : visits) sim += v;
  const int want = sl.schedule[std::min<size_t>(sim, sl.schedule.size() - 1)];
  const float top = *std::max_element(logits.begin(), logits.end());
  score.resize(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    score[i] = static_cast<int>(visits[i]) == want
                   ? std::max(-1e9f, sl.gumbel[i] + logits[i] - top + sigma[i])
                   : -std::numeric_limits<float>::infinity();
  }
  const int best = argmax(score);
  if (score[best] == -std::numeric_limits<float>::infinity()) return select_decision(sl, 0);
  return sl.nodes[0].first + best;
}

// The result whose share of the visits is furthest below its probability.
int BatchedSearch::select_chance(const Slot& sl, uint32_t node) const {
  const SearchNode& n = sl.nodes[node];
  uint32_t total = 0;
  for (int i = 0; i < n.n; ++i) {
    const SearchEdge& e = sl.edges[n.first + i];
    total += e.child == kNoNode ? 0 : sl.nodes[e.child].visits;
  }
  int best = 0;
  float best_score = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n.n; ++i) {
    const SearchEdge& e = sl.edges[n.first + i];
    const uint32_t v = e.child == kNoNode ? 0 : sl.nodes[e.child].visits;
    const float score = e.prior - static_cast<float>(v) / (1 + total);
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return n.first + best;
}

void BatchedSearch::backup(Slot& sl, float value) {
  for (uint32_t i : sl.path) {
    ++sl.nodes[i].visits;
    sl.nodes[i].value_sum += value;
  }
}

bool BatchedSearch::simulate(Slot& sl) {
  sl.path.clear();
  uint32_t node = 0;
  sl.path.push_back(node);
  for (;;) {
    const uint8_t kind = sl.nodes[node].kind;
    if (kind == kTerminal) {
      backup(sl, sl.nodes[node].raw);
      ++sl.sims;
      return false;
    }
    int e;
    if (kind == kDecision) {
      if (!sl.nodes[node].evaluated) {
        sl.leaf = node;
        return true;
      }
      e = node == 0 ? select_root(sl) : select_decision(sl, node);
    } else {
      e = select_chance(sl, node);
    }
    uint32_t child = sl.edges[e].child;
    if (child == kNoNode) {
      child = kind == kDecision ? expand_step(sl, node, e) : expand_outcome(sl, node, e);
      sl.edges[e].child = child;
    }
    node = child;
    sl.path.push_back(node);
  }
}

void BatchedSearch::finish_search(Slot& sl) {
  std::vector<float> logits, sigma;
  std::vector<uint32_t> visits;
  completed_q(sl, 0, logits, sigma, visits);
  const SearchNode& root = sl.nodes[0];
  const float sgn = side(root.mover);
  SearchResult& r = sl.result;
  r.mover = root.mover;
  r.sims = sl.sims;
  r.value = sgn * root.value_sum / static_cast<float>(root.visits);
  r.net_value = sgn * root.raw;
  r.ids.resize(root.n);
  r.visits = visits;
  r.q.assign(root.n, 0.0f);
  const uint32_t most = *std::max_element(visits.begin(), visits.end());
  const float top = *std::max_element(logits.begin(), logits.end());
  int best = -1;
  float best_score = -std::numeric_limits<float>::infinity();
  std::vector<float> z(root.n);
  for (int i = 0; i < root.n; ++i) {
    const SearchEdge& e = sl.edges[root.first + i];
    r.ids[i] = e.id;
    if (visits[i]) {
      const SearchNode& c = sl.nodes[e.child];
      r.q[i] = sgn * c.value_sum / static_cast<float>(c.visits);
    }
    z[i] = logits[i] + sigma[i];
    const float score = sl.gumbel[i] + logits[i] - top + sigma[i];
    if (visits[i] == most && score > best_score) {
      best_score = score;
      best = i;
    }
  }
  if (best < 0)
    best = static_cast<int>(std::max_element(visits.begin(), visits.end()) - visits.begin());
  r.id = r.ids[best];
  softmax(logits, r.prior);
  softmax(z, r.policy);
}

// At a game's first searched decision of a turn, whether the mover resigns
// (SearchConfig::resign_turns); a game that plays on notes the first time the
// rule would have.
bool BatchedSearch::resigns(Slot& sl) {
  const int mover = sl.result.mover;
  const int turn = sl.game.day * 2 + mover;
  if (cfg_.resign_turns <= 0 || turn == sl.turn) return false;
  sl.turn = turn;
  int& low = sl.low[mover];
  low = sl.result.value <= cfg_.resign_value ? low + 1 : 0;
  if (low < cfg_.resign_turns) return false;
  if (sl.can_resign) return true;
  if (sl.would_resign < 0) {
    sl.would_resign = mover;
    sl.would_resign_day = sl.game.day;
  }
  return false;
}

void BatchedSearch::finish_game(Slot& sl, bool capped, int resigner) {
  GameRecord rec;
  rec.game = sl.game_no;
  rec.winner = resigner >= 0 ? 1 - resigner
               : capped      ? day_limit_winner(*sl.map, sl.game)
                             : sl.game.winner;
  rec.days = sl.game.day;
  rec.actions = sl.actions;
  rec.decisions = sl.decisions;
  rec.skipped = sl.skipped;
  rec.capped = capped;
  rec.resigned = resigner >= 0;
  rec.would_resign = sl.would_resign;
  rec.would_resign_day = sl.would_resign_day;
  rec.final_state = sl.game;
  rec.start_state = sl.start;
  rec.steps = std::exchange(sl.steps, {});
  rec.searched = std::exchange(sl.searched, {});
  records_.push_back(std::move(rec));
  sl.mode = kFree;
  sl.nodes.clear();
}

int BatchedSearch::collect() {
  pending_.clear();
  for (size_t i = 0; i < slots_.size(); ++i) {
    Slot& sl = slots_[i];
    while (sl.mode == kGame || (sl.mode == kOneSearch && !sl.finished)) {
      if (sl.nodes[0].evaluated && sl.sims >= sl.target) {
        finish_search(sl);
        if (sl.mode == kOneSearch) {
          sl.finished = true;
          break;
        }
        // Self-play: keep a full search as a sample, take the step, go on.
        if (sl.full) {
          SearchSample smp;
          smp.game = sl.game_no;
          smp.decision = sl.decisions;
          smp.state = sl.game;
          smp.sel = sl.sel;
          smp.result = sl.result;
          samples_.push_back(std::move(smp));
        }
        const SearchResult& r = sl.result;
        sl.searched.push_back({static_cast<int32_t>(sl.steps.size()), r.value, r.net_value, r.sims,
                               static_cast<uint8_t>(r.mover), sl.full});
        ++sl.decisions;
        if (resigns(sl)) {
          finish_game(sl, false, sl.result.mover);
          break;
        }
        Action a;
        const MapData& m = *sl.map;
        const bool completes = step_action(m, sl.game, sl.sel, sl.result.id, a);
        if (!applied(step(m, sl.game, sl.sel, sl.result.id)))
          throw std::logic_error("the searched step was refused");
        sl.steps.push_back(static_cast<uint16_t>(sl.result.id));
        sl.actions += completes;
        begin_decision(sl);
        continue;
      }
      if (simulate(sl)) {
        pending_.push_back(static_cast<int>(i));
        break;
      }
    }
  }
  return static_cast<int>(pending_.size());
}

int BatchedSearch::max_legal() const {
  int most = 0;
  for (int i : pending_) {
    const Slot& sl = slots_[i];
    most = std::max<int>(most, sl.nodes[sl.leaf].n);
  }
  return most;
}

void BatchedSearch::write_batch(uint8_t* planes, float* globals, int64_t* legal, int legal_stride,
                                uint8_t* movers) const {
  if (legal_stride < max_legal()) throw std::invalid_argument("legal_stride < max_legal()");
  const int n_planes = policy_planes(cfg_.movement);
  const size_t plane_stride = static_cast<size_t>(n_planes) * size_ * size_;
  std::vector<uint16_t> ids;
  for (size_t j = 0; j < pending_.size(); ++j) {
    const Slot& sl = slots_[pending_[j]];
    const SearchNode& n = sl.nodes[sl.leaf];
    const MapData& m = *sl.map;
    ids.resize(n.n);
    int64_t* row = legal + j * legal_stride;
    for (int k = 0; k < n.n; ++k) {
      ids[k] = sl.edges[n.first + k].id;
      row[k] = policy_index(m, ids[k], size_, size_);
    }
    std::fill(row + n.n, row + legal_stride, -1);
    encode_decision(m, sl.states[n.state], n.sel, ids.data(), n.n, planes + j * plane_stride, size_,
                    size_, globals + j * kNumPolicyGlobals, nullptr, cfg_.movement);
    if (movers) movers[j] = n.mover;
  }
}

void BatchedSearch::fast_leaves(uint8_t* fast) const {
  for (size_t j = 0; j < pending_.size(); ++j) {
    const Slot& sl = slots_[pending_[j]];
    fast[j] = sl.mode == kGame && !sl.full;
  }
}

void BatchedSearch::submit(const float* logits, int stride, const float* values) {
  for (size_t j = 0; j < pending_.size(); ++j) {
    Slot& sl = slots_[pending_[j]];
    SearchNode& n = sl.nodes[sl.leaf];
    const float* row = logits + j * stride;
    for (int k = 0; k < n.n; ++k) sl.edges[n.first + k].prior = row[k];
    n.evaluated = true;
    // A pick's parent is the decision it was picked from, in the same state
    // (SearchConfig::pick_values).
    const bool picked = n.sel.phase != kPhaseIdle && sl.leaf != 0;
    n.raw = picked && !cfg_.pick_values ? sl.nodes[sl.path[sl.path.size() - 2]].raw
                                        : side(n.mover) * values[j];
    backup(sl, n.raw);
    if (sl.leaf != 0) {
      ++sl.sims;
    } else if (cfg_.skip_prior > 0 && sl.target > 0 && !(sl.mode == kGame && sl.full)) {
      // A confident root ends the search (SearchConfig::skip_prior): the top
      // step's probability is 1 / sum(exp(logit - top)).
      float top = row[0];
      for (int k = 1; k < n.n; ++k) top = std::max(top, row[k]);
      double total = 0;
      for (int k = 0; k < n.n; ++k) total += std::exp(row[k] - top);
      if (1.0 / total >= cfg_.skip_prior) {
        sl.target = 0;
        sl.skipped += sl.mode == kGame;
      }
    }
  }
  pending_.clear();
}

int BatchedSearch::active() const {
  int n = 0;
  for (const Slot& sl : slots_) n += sl.mode == kGame || (sl.mode == kOneSearch && !sl.finished);
  return n;
}

bool BatchedSearch::done(int slot) const {
  return slot >= 0 && slot < static_cast<int>(slots_.size()) && slots_[slot].mode == kOneSearch &&
         slots_[slot].finished;
}

const SearchResult& BatchedSearch::result(int slot) const {
  if (!done(slot)) throw std::invalid_argument("no finished search in that slot");
  return slots_[slot].result;
}

std::vector<SearchSample> BatchedSearch::take_samples() { return std::exchange(samples_, {}); }

std::vector<GameRecord> BatchedSearch::take_records() { return std::exchange(records_, {}); }

void BatchedSearch::clear() {
  for (Slot& sl : slots_) {
    sl.mode = kFree;
    sl.nodes.clear();
  }
  pending_.clear();
}

}  // namespace awbw
