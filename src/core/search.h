#pragma once

// Batched Gumbel MCTS over micro-steps (PLAN.md §8).
//
// A BatchedSearch runs many searches at once, one per slot, and asks for the
// network's view of all their new leaves in one batch: collect() walks each
// slot's tree down to a leaf, write_batch() encodes the leaves as the policy
// network reads them (encode_decision), and submit() takes the network's
// logits over each leaf's legal steps and its value, expands the leaves and
// backs the values up. A slot either searches one decision and keeps the
// result (add_root), or plays a whole game, searching every decision with a
// choice (add_game, self-play).
//
// The tree is over micro-steps (legal.h). A node is a decision (a state and
// the selection in progress), a chance node (a Fire whose luck is still to be
// drawn, with a child per result fire_outcomes gives as the players see it)
// or the end of the game.
// Steps with a single legal id are taken on the way, without a node. Values
// are kept from player 0's side and read from the mover's, so a value changes
// sign only where the mover changes, not at every step.
//
// At the root, Gumbel-top-k sampling and sequential halving choose the steps
// to visit; below it, each decision visits the step whose share of the visits
// is furthest below the improved policy's (Danihelka et al., "Policy
// improvement by planning with Gumbel", ICLR 2022; the same choices as
// DeepMind's mctx). A chance node visits the result whose share of the visits
// is furthest below its probability. The improved policy at the root,
// softmax(logits + sigma(completed Q)), is the training target.
//
// Players see only displayed HP, so a chance node's results are the displays
// the fight can leave, with probabilities over the units' HP beliefs, and the
// tree goes on from the likeliest internal HPs that show each: nothing in it
// depends on the internal HPs it can't see (Rachel's and Von Bolt's missiles
// aside). Hidden units are another matter: a move into one traps in the tree
// as in the game, so the search knows where they are, while the network's
// input holds only what the mover knows (encode.h).

#include <cstdint>
#include <vector>

#include "core/combat.h"
#include "core/legal.h"
#include "core/map.h"
#include "core/state.h"

namespace awbw {

struct SearchConfig {
  int sims = 64;  // simulations of a full search, after the root's own evaluation
  // Playout cap randomization (KataGo): in self-play a share `full_fraction` of
  // the searches are full and recorded as samples; the rest run `fast_sims`
  // simulations and only choose the step.
  int fast_sims = 0;
  float full_fraction = 1.0f;
  // A search whose root the network gives one step at least this probability
  // ends at the root's evaluation and plays as a search of 0 simulations: the
  // step Gumbel-max sampling picks from the prior (its likeliest without
  // noise). Self-play's full searches, which become samples, never do; 0:
  // none do.
  float skip_prior = 0.0f;
  int considered = 16;        // steps the root samples to consider (Gumbel top-k)
  float gumbel_scale = 1.0f;  // 0: no noise; the root considers the likeliest steps
  // sigma(q) = (visit_scale + the most visits of a step) * value_scale * q, with q
  // in 0..1: (Q + 1) / 2, or with `rescale` Q rescaled to 0..1 among the
  // node's steps (mctx's default). Rescaling stretches whatever spread the Qs
  // have, the network's noise between steps of equal worth included.
  float visit_scale = 50.0f;
  float value_scale = 0.1f;
  bool rescale = false;
  // Whether a decision with an action in progress (a unit or factory picked)
  // takes the network's value, or else the value of the decision it was
  // picked from: picks change no state, but the value head reads the
  // selection too. Neither played better in the arena (PLAN.md §8).
  bool pick_values = true;
  // A game still going after this day ends as a day limit decides it
  // (day_limit_winner), in the tree and in self-play; 0 = no cap.
  int max_days = 0;
  int max_actions = 0;  // self-play: a game ends after this many actions, decided the same way
  // Self-play resignation (AlphaGo Zero's): a player resigns at its first
  // searched decision of a turn when the searched value there has been at most
  // `resign_value` at the start of each of its last `resign_turns` turns, this
  // one included; 0 turns: never. A share `resign_off` of the games play on
  // to the end and record when the rule would first have resigned, to check
  // how often that player went on to win.
  float resign_value = -0.9f;
  int resign_turns = 0;
  float resign_off = 0.1f;
  bool movement = false;  // the network reads the movement planes
  uint64_t seed = 0;
};

// The winner of a game stopped at a day limit: more properties (comtowers and
// labs included) wins, as AWBW decides a day limit; then the larger army value
// (cost x displayed HP / 10, carried units included); else kDraw.
int day_limit_winner(const MapData& m, const GameState& s);

// One searched decision.
struct SearchResult {
  int id = -1;                // the step chosen
  int mover = 0;              // the player deciding
  float value = 0;            // the root's searched value, from the mover's side
  float net_value = 0;        // the network's value of the root, from the mover's side
  int sims = 0;               // simulations run
  std::vector<uint16_t> ids;  // the legal steps, ascending
  std::vector<float> prior;   // the network's policy over them
  std::vector<float> policy;  // the improved policy
  std::vector<uint32_t> visits;
  std::vector<float> q;  // each step's value from the mover's side (0 if unvisited)
};

// A searched decision of a self-play game, for training.
struct SearchSample {
  uint64_t game = 0;  // the game's number (add_game's return value)
  int decision = 0;   // its place among the game's searched decisions
  GameState state;    // the state (the action in progress changes no state)
  Selection sel;      // the decision in progress
  SearchResult result;
};

// How a self-play game ended.
struct GameRecord {
  uint64_t game = 0;
  int winner = kDraw;  // player index, or kDraw
  int days = 0;
  int actions = 0;
  int decisions = 0;      // searched (full and fast)
  int skipped = 0;        // of those, fast searches that ended at the root (skip_prior)
  bool capped = false;    // stopped by max_days or max_actions and decided by day_limit_winner
  bool resigned = false;  // the loser resigned (SearchConfig::resign_turns)
  // A game that played on (resign_off): the player the rule would first have
  // had resign, and the day, else -1.
  int would_resign = -1;
  int would_resign_day = -1;
  GameState final_state;
};

// The search tree of one slot. Node 0 is the root.
enum SearchNodeKind : uint8_t { kNodeDecision, kNodeChance, kNodeTerminal };
inline constexpr uint32_t kNoNode = 0xFFFFFFFF;

struct SearchNode {
  uint32_t first = 0;  // its edges: [first, first + n)
  uint16_t n = 0;
  uint8_t kind = 0;        // SearchNodeKind
  uint8_t mover = 0;       // the active player
  int32_t state = 0;       // index into SearchTree::states
  Selection sel;           // decision: the selection; chance: the one the Fire step completes
  uint16_t step = 0;       // chance: the Fire step
  bool evaluated = false;  // decision: the network has seen it
  uint32_t visits = 0;
  float value_sum = 0;  // player 0's side
  float raw = 0;        // decision: the network's value; terminal: the result (player 0's side)
};

struct SearchEdge {
  uint32_t child = kNoNode;  // not expanded yet
  uint16_t id = 0;           // decision: the step
  uint8_t att_from = 0;      // chance: the result (SeenOutcome)
  uint8_t def_from = 0;
  uint8_t att_hp = 0;
  uint8_t def_hp = 0;
  float prior = 0;  // decision: the network's logit; chance: the result's probability
};

class BatchedSearch {
 public:
  // Leaves are encoded in a size x size window; every map searched must fit.
  BatchedSearch(const SearchConfig& cfg, int size);

  const SearchConfig& config() const { return cfg_; }
  int size() const { return size_; }

  // Search the decision `sel` in `s` on map `m` (which must outlive the
  // search) and keep the result; returns the slot. The game must be going.
  int add_root(const MapData& m, const GameState& s, const Selection& sel = {});
  // Play a game from `s` to its end, searching each decision with a choice;
  // returns the game's number. Finished games come out of take_records(). The
  // game's luck is seeded from the config's seed and its number.
  uint64_t add_game(const MapData& m, const GameState& s);

  // Walk each searching slot down to a leaf for the network, finishing the
  // searches that are done (and, in self-play, taking their steps and
  // starting the next). Returns the number of leaves; 0 once nothing is left
  // to search.
  int collect();
  int leaves() const { return static_cast<int>(pending_.size()); }
  // The most legal steps of a leaf in this batch.
  int max_legal() const;
  // Encode the leaves: planes [n][policy_planes(movement)][size][size],
  // globals [n][kNumPolicyGlobals], and each leaf's legal steps as policy
  // indices (policy_index) in rows of `legal_stride` (>= max_legal()), padded
  // with -1. `movers` [n] (may be null): the player deciding at each leaf.
  void write_batch(uint8_t* planes, float* globals, int64_t* legal, int legal_stride,
                   uint8_t* movers) const;
  // fast [n]: whether each leaf is a self-play fast search's (not kept as a
  // sample), so that a cheaper network can score it.
  void fast_leaves(uint8_t* fast) const;
  // The network's view of each leaf: logits over its legal steps, in the
  // order write_batch gave them, in rows of `stride`, and the value [n] for
  // the player deciding there (P(win) - P(loss)).
  void submit(const float* logits, int stride, const float* values);

  // Slots in use: searches not finished and games still going.
  int active() const;
  int slots() const { return static_cast<int>(slots_.size()); }
  bool done(int slot) const;
  const SearchResult& result(int slot) const;
  std::vector<SearchSample> take_samples();
  std::vector<GameRecord> take_records();
  // Free every slot.
  void clear();

  // A slot's current tree, for tests and debugging: its nodes (the root
  // first; empty if nothing was searched) and their edges.
  const std::vector<SearchNode>& nodes(int slot) const { return slots_.at(slot).nodes; }
  const std::vector<SearchEdge>& edges(int slot) const { return slots_.at(slot).edges; }

 private:
  struct Slot {
    uint8_t mode = 0;       // free, one search, or a game (search.cpp)
    bool finished = false;  // one search: the result is ready
    const MapData* map = nullptr;
    GameState game;  // the root's state, or the game being played
    Selection sel;
    uint64_t game_no = 0;
    int actions = 0, decisions = 0, skipped = 0;
    // Resignation: whether the game may resign, the turn (day * 2 + mover)
    // whose start was last checked, each player's turns in a row that started
    // at or below resign_value, and the rule's first verdict in a game that
    // plays on.
    bool can_resign = false;
    int turn = -1;
    int low[2] = {};
    int would_resign = -1, would_resign_day = -1;
    bool full = true;
    int target = 0;             // simulations of this search
    int sims = 0;               // simulations done
    std::vector<int> schedule;  // sequential halving: the visits a step needs, by simulation
    std::vector<float> gumbel;  // per root edge
    uint64_t rng[2] = {};
    std::vector<SearchNode> nodes;
    std::vector<SearchEdge> edges;
    std::vector<GameState> states;
    std::vector<uint32_t> path;  // root to the current leaf
    uint32_t leaf = 0;
    SearchResult result;
    std::vector<uint16_t> ids;  // scratch
    std::vector<SeenOutcome> outs;
  };

  int new_slot();
  void start_root(Slot& sl);
  void begin_decision(Slot& sl);
  void finish_search(Slot& sl);
  void finish_game(Slot& sl, bool capped, int resigner = -1);
  bool resigns(Slot& sl);
  bool simulate(Slot& sl);  // true: a leaf awaits the network
  int select_root(const Slot& sl) const;
  int select_decision(const Slot& sl, uint32_t node) const;
  int select_chance(const Slot& sl, uint32_t node) const;
  uint32_t expand_step(Slot& sl, uint32_t node, int edge);
  uint32_t expand_outcome(Slot& sl, uint32_t node, int edge);
  uint32_t settle(Slot& sl, GameState& s, Selection& sel, int32_t state_index);
  uint32_t add_decision(Slot& sl, const GameState& s, const Selection& sel, int32_t state_index);
  uint32_t add_chance(Slot& sl, const GameState& s, const Selection& sel, int id,
                      int32_t state_index);
  void completed_q(const Slot& sl, uint32_t node, std::vector<float>& logits,
                   std::vector<float>& sigma, std::vector<uint32_t>& visits) const;
  void backup(Slot& sl, float value);

  SearchConfig cfg_;
  int size_;
  std::vector<Slot> slots_;
  std::vector<int> pending_;  // slots with a leaf in this batch
  std::vector<SearchSample> samples_;
  std::vector<GameRecord> records_;
  uint64_t next_game_ = 0;
  uint64_t searches_ = 0;
};

}  // namespace awbw
