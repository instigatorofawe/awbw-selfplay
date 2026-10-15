#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/actions.h"
#include "core/belief.h"
#include "core/combat.h"
#include "core/config.h"
#include "core/encode.h"
#include "core/hash.h"
#include "core/legal.h"
#include "core/map.h"
#include "core/movegen.h"
#include "core/movement.h"
#include "core/search.h"
#include "core/snapshot.h"
#include "core/state.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename T, size_t Cols>
using Rows = nb::ndarray<const T, nb::shape<-1, Cols>, nb::c_contig, nb::device::cpu>;

const awbw::MapData& map_for(const awbw::GameState& s) {
  const awbw::MapData* m = awbw::find_map(s.map_id);
  if (!m) throw nb::value_error(("map " + std::to_string(s.map_id) + " is not registered").c_str());
  return *m;
}

void check_slot(const awbw::GameState& s, int slot) {
  if (slot < 0 || slot >= awbw::kMaxUnits || !s.units[slot].alive())
    throw nb::index_error(("no unit in slot " + std::to_string(slot)).c_str());
}

using XY = std::tuple<int, int>;

uint16_t tile_of(const awbw::MapData& md, XY xy) {
  auto [x, y] = xy;
  if (x < 0 || y < 0 || x >= md.width || y >= md.height)
    throw nb::index_error(
        ("(" + std::to_string(x) + ", " + std::to_string(y) + ") is off the map").c_str());
  return static_cast<uint16_t>(md.tile(x, y));
}

awbw::Action make_action(const awbw::MapData& md, const awbw::GameState& s, int kind, int unit,
                         std::optional<XY> to, std::optional<XY> target, int arg) {
  if (kind < 0 || kind >= awbw::kNumActionKinds) throw nb::value_error("unknown action kind");
  awbw::Action a;
  a.kind = static_cast<uint8_t>(kind);
  a.unit = unit < 0 || unit >= awbw::kMaxUnits ? awbw::kNoUnit : static_cast<uint8_t>(unit);
  if (to) {
    a.to = tile_of(md, *to);
  } else if (a.unit != awbw::kNoUnit && s.units[a.unit].alive()) {
    a.to = s.units[a.unit].tile;
  }
  if (target) a.target = tile_of(md, *target);
  a.arg = static_cast<uint8_t>(arg);
  return a;
}

// What apply() and step() take besides the action, and the buffers it points into.
struct DetailArgs {
  awbw::Detail d;
  std::vector<uint16_t> tiles, centres;

  DetailArgs(const awbw::MapData& md, const std::optional<std::vector<XY>>& path, int att_hp,
             int def_hp, const std::optional<std::vector<XY>>& missiles) {
    if (path) {
      for (const XY& xy : *path) tiles.push_back(tile_of(md, xy));
      d.path = tiles.data();
      d.path_len = static_cast<int>(tiles.size());
    }
    d.att_hp = att_hp;
    d.def_hp = def_hp;
    if (missiles) {
      for (const XY& xy : *missiles) centres.push_back(tile_of(md, xy));
      d.missiles = centres.data();
      d.n_missiles = static_cast<int>(centres.size());
    }
  }
};

int apply_action(awbw::GameState& s, int kind, int unit, std::optional<XY> to,
                 std::optional<XY> target, int arg, std::optional<std::vector<XY>> path, int att_hp,
                 int def_hp, std::optional<std::vector<XY>> missiles) {
  const awbw::MapData& md = map_for(s);
  const awbw::Action a = make_action(md, s, kind, unit, to, target, arg);
  const DetailArgs d(md, path, att_hp, def_hp, missiles);
  return awbw::apply(md, s, a, d.d);
}

// An action as apply()'s arguments: (kind, unit, to, target, arg), with -1 /
// None / 0 where the kind takes none.
using ActionTuple = std::tuple<int, int, std::optional<XY>, std::optional<XY>, int>;

ActionTuple action_tuple(const awbw::MapData& md, const awbw::Action& a) {
  using namespace awbw;
  const bool unit_kind = a.kind < kActBuild;
  const bool targeted = a.kind == kActFire || a.kind == kActUnload || a.kind == kActRepair ||
                        a.kind == kActAttackSeam || a.kind == kActLaunch || a.kind == kActBuild;
  auto xy = [&](int t) { return std::optional<XY>(XY{md.x_of(t), md.y_of(t)}); };
  return {a.kind, unit_kind || a.kind == kActDelete ? a.unit : -1,
          unit_kind ? xy(a.to) : std::nullopt, targeted ? xy(a.target) : std::nullopt,
          a.kind == kActUnload || a.kind == kActBuild || a.kind == kActPower ? a.arg : 0};
}

using IdArray = nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>;

IdArray legal_id_array(const awbw::GameState& s, const awbw::Selection& sel) {
  std::vector<uint16_t> ids;
  awbw::legal_ids(map_for(s), s, sel, ids);
  auto* data = new int64_t[ids.size()];
  std::copy(ids.begin(), ids.end(), data);
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
  return IdArray(data, {ids.size()}, owner);
}

awbw::GameState load_snapshot(const awbw::MapData& m, const awbw::Settings& settings, int day,
                              int active, Rows<int64_t, 8> players, Rows<int64_t, 4> buildings,
                              Rows<int64_t, 11> units, int weather, int weather_owner) {
  if (players.shape(0) != awbw::kMaxPlayers) throw nb::value_error("players must have 2 rows");
  awbw::Snapshot snap{};
  snap.settings = settings;
  snap.day = day;
  snap.active = active;
  snap.weather = weather;
  snap.weather_owner = weather_owner;
  auto pv = players.view();
  for (int i = 0; i < awbw::kMaxPlayers; ++i) {
    auto c = [&](int j) { return static_cast<int>(pv(i, j)); };
    snap.players[i] = awbw::SnapshotPlayer{c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7)};
  }
  std::vector<awbw::SnapshotBuilding> b(buildings.shape(0));
  auto bv = buildings.view();
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = {static_cast<int>(bv(i, 0)), static_cast<int>(bv(i, 1)), static_cast<int>(bv(i, 2)),
            static_cast<int>(bv(i, 3))};
  }
  std::vector<awbw::SnapshotUnit> u(units.shape(0));
  auto uv = units.view();
  for (size_t i = 0; i < u.size(); ++i) {
    auto c = [&](int j) { return static_cast<int>(uv(i, j)); };
    u[i] = {static_cast<uint32_t>(uv(i, 0)),
            c(1),
            c(2),
            c(3),
            c(4),
            c(5),
            c(6),
            c(7),
            c(8),
            {static_cast<uint32_t>(uv(i, 9)), static_cast<uint32_t>(uv(i, 10))}};
  }
  snap.buildings = b.data();
  snap.n_buildings = static_cast<int>(b.size());
  snap.units = u.data();
  snap.n_units = static_cast<int>(u.size());
  awbw::GameState s;
  std::string error;
  if (!awbw::load_snapshot(m, snap, s, error)) throw nb::value_error(error.c_str());
  return s;
}

using StateRows = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using PlaneOut = nb::ndarray<uint8_t, nb::ndim<4>, nb::c_contig, nb::device::cpu>;
using GlobalOut = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

nb::ndarray<nb::numpy, uint64_t, nb::ndim<1>> hash_batch(StateRows states) {
  const size_t n = states.shape(0);
  if (states.shape(1) != sizeof(awbw::GameState))
    throw nb::value_error(
        ("states must have STATE_BYTES = " + std::to_string(sizeof(awbw::GameState)) + " columns")
            .c_str());
  auto* keys = new uint64_t[n];
  nb::capsule owner(keys, [](void* p) noexcept { delete[] static_cast<uint64_t*>(p); });
  std::string error;
  {
    nb::gil_scoped_release release;
    awbw::GameState s;
    const awbw::MapData* md = nullptr;
    for (size_t i = 0; i < n && error.empty(); ++i) {
      std::memcpy(&s, states.data() + i * sizeof s, sizeof s);  // rows needn't be aligned
      if (!md || md->id != s.map_id) md = awbw::find_map(s.map_id);
      if (!md) {
        error = "map " + std::to_string(s.map_id) + " is not registered";
      } else if (std::string bad = awbw::check_state(*md, s); !bad.empty()) {
        error = bad;
      } else {
        keys[i] = awbw::state_hash(s);
        continue;
      }
      error = "state " + std::to_string(i) + ": " + error;
    }
  }
  if (!error.empty()) throw nb::value_error(error.c_str());
  return nb::ndarray<nb::numpy, uint64_t, nb::ndim<1>>(keys, {n}, owner);
}

void encode_batch(StateRows states, PlaneOut planes, GlobalOut globals) {
  const size_t n = states.shape(0);
  if (states.shape(1) != sizeof(awbw::GameState))
    throw nb::value_error(
        ("states must have STATE_BYTES = " + std::to_string(sizeof(awbw::GameState)) + " columns")
            .c_str());
  if (planes.shape(0) != n || planes.shape(1) != awbw::kNumPlanes)
    throw nb::value_error("planes must be [N, NUM_PLANES, H, W]");
  if (globals.shape(0) != n || globals.shape(1) != awbw::kNumGlobals)
    throw nb::value_error("globals must be [N, NUM_GLOBALS]");
  const int h = static_cast<int>(planes.shape(2)), w = static_cast<int>(planes.shape(3));
  const size_t plane_stride = size_t{awbw::kNumPlanes} * h * w;
  std::string error;
  {
    nb::gil_scoped_release release;
    awbw::GameState s;
    const awbw::MapData* md = nullptr;
    for (size_t i = 0; i < n && error.empty(); ++i) {
      std::memcpy(&s, states.data() + i * sizeof s, sizeof s);  // rows needn't be aligned
      if (!md || md->id != s.map_id) md = awbw::find_map(s.map_id);
      if (!md) {
        error = "map " + std::to_string(s.map_id) + " is not registered";
      } else if (md->height > h || md->width > w) {
        error = "map " + std::to_string(s.map_id) + " is " + std::to_string(md->width) + "x" +
                std::to_string(md->height) + ", larger than the planes";
      } else if (std::string bad = awbw::check_state(*md, s); !bad.empty()) {
        error = bad;
      } else {
        awbw::encode(*md, s, planes.data() + i * plane_stride, h, w,
                     globals.data() + i * awbw::kNumGlobals);
        continue;
      }
      error = "state " + std::to_string(i) + ": " + error;
    }
  }
  if (!error.empty()) throw nb::value_error(error.c_str());
}

using PickRows = nb::ndarray<const uint16_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using CountIn = nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using MaskOut = nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using TargetOut = nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

void encode_policy_batch(StateRows states, PickRows picks, CountIn n_picks, PlaneOut planes,
                         GlobalOut globals, MaskOut nonspatial, std::optional<TargetOut> targets,
                         bool movement) {
  using namespace awbw;
  const size_t n = states.shape(0);
  const size_t all_planes = policy_planes(movement);
  if (states.shape(1) != sizeof(GameState))
    throw nb::value_error(
        ("states must have STATE_BYTES = " + std::to_string(sizeof(GameState)) + " columns")
            .c_str());
  if (picks.shape(0) != n || n_picks.shape(0) != n)
    throw nb::value_error("picks must be [N, 3] and n_picks [N]");
  if (planes.shape(0) != n || planes.shape(1) != all_planes)
    throw nb::value_error(
        movement
            ? "planes must be [N, NUM_PLANES + NUM_SELECTION_PLANES + NUM_MOVEMENT_PLANES, H, W]"
            : "planes must be [N, NUM_PLANES + NUM_SELECTION_PLANES, H, W]");
  if (globals.shape(0) != n || globals.shape(1) != kNumPolicyGlobals)
    throw nb::value_error("globals must be [N, NUM_GLOBALS + NUM_SELECTION_GLOBALS]");
  if (nonspatial.shape(0) != n || nonspatial.shape(1) != kNumNonSpatialIds)
    throw nb::value_error("nonspatial must be [N, NUM_NONSPATIAL_IDS]");
  if (targets && targets->shape(0) != n) throw nb::value_error("targets must be [N]");
  const int h = static_cast<int>(planes.shape(2)), w = static_cast<int>(planes.shape(3));
  const size_t plane_stride = all_planes * h * w;
  std::string error;
  {
    nb::gil_scoped_release release;
    GameState s;
    const MapData* md = nullptr;
    std::vector<uint16_t> ids;
    for (size_t i = 0; i < n && error.empty(); ++i) {
      std::memcpy(&s, states.data() + i * sizeof s, sizeof s);
      if (!md || md->id != s.map_id) md = find_map(s.map_id);
      const int k = n_picks.data()[i];
      if (!md) {
        error = "map " + std::to_string(s.map_id) + " is not registered";
      } else if (md->height > h || md->width > w) {
        error = "map " + std::to_string(s.map_id) + " is " + std::to_string(md->width) + "x" +
                std::to_string(md->height) + ", larger than the planes";
      } else if (std::string bad = check_state(*md, s); !bad.empty()) {
        error = bad;
      } else if (k > (targets ? 2 : 3)) {
        error = "too many picks";
      } else {
        // The picks only move the selection; the state stays as it is.
        Selection sel;
        bool ok = true;
        for (int j = 0; j < k && ok; ++j) {
          Action a;
          const int id = picks.data()[i * 3 + j];
          ok = !step_action(*md, s, sel, id, a) && step(*md, s, sel, id) == kOk;
        }
        if (!ok && !targets) {
          error = "a pick isn't legal";
        } else {
          legal_ids(*md, s, sel, ids);
          encode_decision(*md, s, sel, ids.data(), static_cast<int>(ids.size()),
                          planes.data() + i * plane_stride, h, w,
                          globals.data() + i * kNumPolicyGlobals,
                          nonspatial.data() + i * kNumNonSpatialIds, movement);
          if (targets) {
            const uint16_t want = picks.data()[i * 3 + k];
            const bool legal = ok && std::binary_search(ids.begin(), ids.end(), want);
            targets->data()[i] = legal ? policy_index(*md, want, h, w) : -1;
          }
          continue;
        }
      }
      error = "state " + std::to_string(i) + ": " + error;
    }
  }
  if (!error.empty()) throw nb::value_error(error.c_str());
}

using StatsOut = nb::ndarray<nb::numpy, int32_t, nb::ndim<3>>;

// Per player of each state: what the auxiliary training targets read.
enum PlayerStat {
  kStatArmyValue,  // cost x displayed HP / 10, carried and hidden units included
  kStatUnits,
  kStatPower,
  kStatCopCost,
  kStatScopCost,
  kStatPowerUses,
  kStatPowerOn,
  kStatUsePowers,  // the game's setting
  kNumPlayerStats
};

StatsOut player_stats_batch(StateRows states) {
  using namespace awbw;
  const size_t n = states.shape(0);
  if (states.shape(1) != sizeof(GameState))
    throw nb::value_error(
        ("states must have STATE_BYTES = " + std::to_string(sizeof(GameState)) + " columns")
            .c_str());
  auto* out = new int32_t[n * kMaxPlayers * kNumPlayerStats]();
  nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
  {
    nb::gil_scoped_release release;
    GameState s;
    for (size_t i = 0; i < n; ++i) {
      std::memcpy(&s, states.data() + i * sizeof s, sizeof s);
      int32_t* row = out + i * kMaxPlayers * kNumPlayerStats;
      for (const Unit& u : s.units) {
        if (!u.alive() || u.owner >= kMaxPlayers) continue;
        int32_t* f = row + u.owner * kNumPlayerStats;
        f[kStatArmyValue] += kUnits[u.type].cost * displayed_hp(u.hp) / 10;
        f[kStatUnits] += 1;
      }
      for (int p = 0; p < kMaxPlayers; ++p) {
        int32_t* f = row + p * kNumPlayerStats;
        const Player& pl = s.players[p];
        f[kStatPower] = pl.power;
        f[kStatCopCost] = pl.cop_cost;
        f[kStatScopCost] = pl.scop_cost;
        f[kStatPowerUses] = pl.power_uses;
        f[kStatPowerOn] = pl.power_on;
        f[kStatUsePowers] = s.settings.use_powers;
      }
    }
  }
  return StatsOut(out, {n, size_t{kMaxPlayers}, size_t{kNumPlayerStats}}, owner);
}

// --- Search -------------------------------------------------------------------

template <typename T>
using Array = nb::ndarray<nb::numpy, T>;

// A new zeroed numpy array that owns its memory; `data` points into it.
template <typename T>
Array<T> new_array(std::initializer_list<size_t> shape, T*& data) {
  size_t n = 1;
  for (size_t d : shape) n *= d;
  data = new T[std::max<size_t>(n, 1)]();
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return Array<T>(data, shape, owner);
}

template <typename T, typename U>
Array<T> array_of(const std::vector<U>& v) {
  T* data;
  Array<T> a = new_array<T>({v.size()}, data);
  std::copy(v.begin(), v.end(), data);
  return a;
}

nb::tuple search_batch(const awbw::BatchedSearch& bs) {
  const size_t n = bs.leaves(), size = bs.size();
  const size_t width = std::max(1, bs.max_legal());
  uint8_t *planes, *movers;
  float* globals;
  int64_t* legal;
  auto p = new_array<uint8_t>(
      {n, static_cast<size_t>(awbw::policy_planes(bs.config().movement)), size, size}, planes);
  auto g = new_array<float>({n, static_cast<size_t>(awbw::kNumPolicyGlobals)}, globals);
  auto l = new_array<int64_t>({n, width}, legal);
  auto mv = new_array<uint8_t>({n}, movers);
  {
    nb::gil_scoped_release release;
    bs.write_batch(planes, globals, legal, static_cast<int>(width), movers);
  }
  return nb::make_tuple(p, g, l, mv);
}

using LogitIn = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ValueIn = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

void search_submit(awbw::BatchedSearch& bs, LogitIn logits, ValueIn values) {
  const size_t n = bs.leaves();
  if (logits.shape(0) != n || logits.shape(1) < static_cast<size_t>(bs.max_legal()))
    throw nb::value_error("logits must be [leaves, >= max_legal()], as batch() gave them");
  if (values.shape(0) != n) throw nb::value_error("values must be [leaves]");
  nb::gil_scoped_release release;
  bs.submit(logits.data(), static_cast<int>(logits.shape(1)), values.data());
}

// Samples as arrays, their steps flattened: sample i's are offsets[i]..offsets[i + 1].
nb::dict search_samples(awbw::BatchedSearch& bs) {
  const std::vector<awbw::SearchSample> v = bs.take_samples();
  const size_t n = v.size();
  size_t steps = 0;
  for (const auto& smp : v) steps += smp.result.ids.size();
  uint64_t* game;
  int32_t *decision, *chosen, *sims;
  uint8_t *states, *n_picks, *mover;
  uint16_t* picks;
  float *value, *net_value, *prior, *policy, *q;
  int64_t *offsets, *ids, *index;
  uint32_t* visits;
  nb::dict d;
  d["game"] = new_array<uint64_t>({n}, game);
  d["decision"] = new_array<int32_t>({n}, decision);
  d["states"] = new_array<uint8_t>({n, sizeof(awbw::GameState)}, states);
  d["picks"] = new_array<uint16_t>({n, 3}, picks);
  d["n_picks"] = new_array<uint8_t>({n}, n_picks);
  d["mover"] = new_array<uint8_t>({n}, mover);
  d["chosen"] = new_array<int32_t>({n}, chosen);
  d["sims"] = new_array<int32_t>({n}, sims);
  d["value"] = new_array<float>({n}, value);
  d["net_value"] = new_array<float>({n}, net_value);
  d["offsets"] = new_array<int64_t>({n + 1}, offsets);
  d["ids"] = new_array<int64_t>({steps}, ids);
  d["index"] = new_array<int64_t>({steps}, index);
  d["prior"] = new_array<float>({steps}, prior);
  d["policy"] = new_array<float>({steps}, policy);
  d["visits"] = new_array<uint32_t>({steps}, visits);
  d["q"] = new_array<float>({steps}, q);
  size_t at = 0;
  for (size_t i = 0; i < n; ++i) {
    const awbw::SearchSample& smp = v[i];
    const awbw::SearchResult& r = smp.result;
    game[i] = smp.game;
    decision[i] = smp.decision;
    std::memcpy(states + i * sizeof(awbw::GameState), &smp.state, sizeof(awbw::GameState));
    n_picks[i] = static_cast<uint8_t>(awbw::selection_steps(smp.state, smp.sel, picks + i * 3));
    mover[i] = static_cast<uint8_t>(r.mover);
    chosen[i] = r.id;
    sims[i] = r.sims;
    value[i] = r.value;
    net_value[i] = r.net_value;
    offsets[i] = static_cast<int64_t>(at);
    const awbw::MapData& md = map_for(smp.state);
    for (size_t k = 0; k < r.ids.size(); ++k, ++at) {
      ids[at] = r.ids[k];
      index[at] = awbw::policy_index(md, r.ids[k], bs.size(), bs.size());
      prior[at] = r.prior[k];
      policy[at] = r.policy[k];
      visits[at] = r.visits[k];
      q[at] = r.q[k];
    }
  }
  offsets[n] = static_cast<int64_t>(at);
  return d;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "AWBW engine core (C++)";
  m.attr("MAX_WIDTH") = awbw::kMaxWidth;
  m.attr("MAX_HEIGHT") = awbw::kMaxHeight;
  m.attr("MAX_TILES") = awbw::kMaxTiles;
  m.attr("MAX_PLAYERS") = awbw::kMaxPlayers;
  m.attr("MAX_UNITS") = awbw::kMaxUnits;
  m.attr("MAX_PROPS") = awbw::kMaxProps;
  m.attr("HP_BINS") = awbw::kHpBins;
  m.attr("STATE_BYTES") = sizeof(awbw::GameState);
  m.attr("NO_UNIT") = awbw::kNoUnit;
  m.attr("NO_OWNER") = awbw::kNoOwner;
  m.attr("UNIT_MOVED") = static_cast<int>(awbw::kMoved);
  m.attr("UNIT_HIDDEN") = static_cast<int>(awbw::kHidden);
  m.attr("UNIT_CARRIED") = static_cast<int>(awbw::kCarried);
  m.attr("UNIT_STUNNED") = static_cast<int>(awbw::kStunned);
  m.attr("UNIT_CAPTURING") = static_cast<int>(awbw::kCapturing);
  m.attr("STAR_VALUE") = awbw::kStarValue;
  m.attr("WEATHERS") = nb::make_tuple("clear", "rain", "snow");
  static_assert(awbw::kNumWeathers == 3 && awbw::kClear == 0 && awbw::kRain == 1 &&
                awbw::kSnow == 2);

  {
    nb::dict kinds;
    const char* names[awbw::kNumActionKinds] = {
        "Wait",   "Fire",   "Capture", "Load",   "Join",       "Unload",
        "Supply", "Repair", "Hide",    "Unhide", "AttackSeam", "Explode",
        "Launch", "Build",  "Power",   "Delete", "EndTurn",    "Resign"};
    for (int k = 0; k < awbw::kNumActionKinds; ++k) kinds[names[k]] = k;
    m.attr("ACTIONS") = kinds;
  }
  m.attr("ENCODE_VERSION") = awbw::kEncodeVersion;
  m.attr("NUM_PLANES") = static_cast<int>(awbw::kNumPlanes);
  m.attr("NUM_GLOBALS") = static_cast<int>(awbw::kNumGlobals);
  {
    nb::list planes, globals;
    for (const awbw::PlaneSpec& p : awbw::kPlanes)
      planes.append(nb::make_tuple(p.name, p.values, p.categorical));
    for (int i = 0; i < awbw::kNumGlobals; ++i) globals.append(awbw::global_name(i));
    m.attr("PLANES") = nb::tuple(planes);
    m.attr("GLOBAL_NAMES") = nb::tuple(globals);
  }
  m.def("encode_batch", &encode_batch, "states"_a, "planes"_a, "globals"_a,
        "Encode raw states [N, STATE_BYTES] for the network, each from its player to move's\n"
        "side, into planes [N, NUM_PLANES, H, W] (uint8) and globals [N, NUM_GLOBALS]\n"
        "(float32). Each board sits at the top-left of its H x W window, the rest 0. Every\n"
        "state's map must be registered and fit; states are checked before encoding.\n"
        "PLANES lists each plane's (name, number of values, categorical).");

  m.attr("SELECTION_ENCODE_VERSION") = awbw::kSelectionEncodeVersion;
  m.attr("NUM_SELECTION_PLANES") = static_cast<int>(awbw::kNumSelectionPlanes);
  m.attr("NUM_SELECTION_GLOBALS") = static_cast<int>(awbw::kNumSelectionGlobals);
  m.attr("NUM_NONSPATIAL_IDS") = awbw::kNumNonSpatialIds;
  {
    nb::list planes, globals;
    for (const awbw::PlaneSpec& p : awbw::kSelectionPlanes)
      planes.append(nb::make_tuple(p.name, p.values, p.categorical));
    for (int i = 0; i < awbw::kNumSelectionGlobals; ++i)
      globals.append(awbw::selection_global_name(i));
    m.attr("SELECTION_PLANES") = nb::tuple(planes);
    m.attr("SELECTION_GLOBAL_NAMES") = nb::tuple(globals);
  }
  m.def("policy_size", &awbw::policy_size, "h"_a, "w"_a,
        "Number of policy logits over an h x w window: NUM_CHANNELS * h * w spatial ones,\n"
        "then NUM_NONSPATIAL_IDS.");
  m.def(
      "policy_index",
      [](const awbw::GameState& s, int id, int h, int w) {
        const awbw::MapData& md = map_for(s);
        if (id < 0 || id >= awbw::kNumActionIds) throw nb::value_error("not an action id");
        if (md.height > h || md.width > w) throw nb::value_error("the map is larger than h x w");
        return awbw::policy_index(md, id, h, w);
      },
      "state"_a, "id"_a, "h"_a, "w"_a,
      "Where micro-step `id` is among the policy logits over an h x w window: the channel\n"
      "times h * w plus y * w + x for a spatial id, NUM_CHANNELS * h * w plus its place\n"
      "after ID_END_TURN for the others.");
  m.attr("MOVEMENT_ENCODE_VERSION") = awbw::kMovementEncodeVersion;
  m.attr("NUM_MOVEMENT_PLANES") = static_cast<int>(awbw::kNumMovementPlanes);
  {
    nb::list planes;
    for (const awbw::PlaneSpec& p : awbw::kMovementPlanes)
      planes.append(nb::make_tuple(p.name, p.values, p.categorical));
    m.attr("MOVEMENT_PLANES") = nb::tuple(planes);
  }
  m.def("encode_policy_batch", &encode_policy_batch, "states"_a, "picks"_a, "n_picks"_a, "planes"_a,
        "globals"_a, "nonspatial"_a, "targets"_a = nb::none(), "movement"_a = false,
        "Encode decisions for the policy network. Decision i is in state i after its first\n"
        "n_picks[i] picks (micro-steps that don't complete an action) of picks[i] [N, 3]:\n"
        "planes [N, NUM_PLANES + NUM_SELECTION_PLANES, H, W] (encode_batch's, then\n"
        "SELECTION_PLANES), globals [N, NUM_GLOBALS + NUM_SELECTION_GLOBALS] and the\n"
        "legality of the non-spatial ids [N, NUM_NONSPATIAL_IDS] (uint8). With `targets`\n"
        "[N] (int64), the step taken next, picks[i][n_picks[i]], goes there as its\n"
        "policy_index, or -1 if it (or a pick before it) isn't legal; without, a pick\n"
        "that isn't legal raises. With `movement`, MOVEMENT_PLANES follow in planes (turns\n"
        "from factories and to reach tiles; core/encode.h).");
  m.attr("PLAYER_STATS") = nb::make_tuple("army_value", "units", "power", "cop_cost", "scop_cost",
                                          "power_uses", "power_on", "use_powers");
  m.def("player_stats_batch", &player_stats_batch, "states"_a,
        "Per player of raw states [N, STATE_BYTES], by player index: PLAYER_STATS as int32\n"
        "[N, 2, len(PLAYER_STATS)]. The army value is cost x displayed HP / 10 over every\n"
        "unit, carried and hidden ones included.");

  m.attr("OK") = static_cast<int>(awbw::kOk);
  m.attr("TRAPPED") = static_cast<int>(awbw::kTrapped);
  m.def(
      "status_name",
      [](int st) { return std::string(awbw::status_name(static_cast<awbw::Status>(st))); },
      "status"_a, "What an apply() status code means.");

  nb::class_<awbw::MapData>(m, "MapData", "A map's static data, shared by every game on it.")
      .def_ro("id", &awbw::MapData::id)
      .def_ro("width", &awbw::MapData::width)
      .def_ro("height", &awbw::MapData::height)
      .def_ro("n_props", &awbw::MapData::n_props)
      .def(
          "props",
          [](const awbw::MapData& md) {
            std::vector<std::tuple<int, int, int, int>> out;
            for (int p = 0; p < md.n_props; ++p) {
              out.emplace_back(md.x_of(md.prop_tile[p]), md.y_of(md.prop_tile[p]), md.prop_kind[p],
                               md.prop_country[p]);
            }
            return out;
          },
          "(x, y, terrain kind, starting country) for each tile with per-game state.")
      .def(
          "factory_turns",
          [](const awbw::MapData& md, int prop, int unit_type) {
            if (prop < 0 || prop >= md.n_props) throw nb::index_error("no such property");
            if (unit_type < 0 || unit_type >= awbw::kNumUnitTypes)
              throw nb::value_error("not a unit type");
            const int row = md.factory_row[prop][awbw::kMovers.of_type[unit_type]];
            if (row == awbw::kNoRow) throw nb::value_error("the property doesn't build that type");
            auto* data = new uint8_t[md.n_tiles];
            std::memcpy(data, md.turns_row(row), md.n_tiles);
            nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>>(
                data, {static_cast<size_t>(md.height), static_cast<size_t>(md.width)}, owner);
          },
          "prop"_a, "unit_type"_a,
          "Turns a unit of `unit_type` needs from factory `prop` (an index into props()) to\n"
          "each tile [height, width], on the starting terrain in clear weather without CO\n"
          "bonuses; 255 where it can't go.")
      .def("tile", &awbw::MapData::tile, "x"_a, "y"_a);

  m.def(
      "register_map",
      [](uint32_t id,
         nb::ndarray<const int16_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> terrain) {
        auto md = std::make_unique<awbw::MapData>();
        std::string error;
        if (!awbw::build_map(id, static_cast<int>(terrain.shape(1)),
                             static_cast<int>(terrain.shape(0)), terrain.data(), *md, error))
          throw nb::value_error(error.c_str());
        const awbw::MapData* reg = awbw::register_map(*md, error);
        if (!reg) throw nb::value_error(error.c_str());
        return reg;
      },
      "map_id"_a, "terrain"_a, nb::rv_policy::reference,
      "Build a map from AWBW terrain ids [height, width] and register it under `map_id`.");
  m.def("find_map", &awbw::find_map, "map_id"_a, nb::rv_policy::reference,
        "The registered map with this id, or None.");

  nb::class_<awbw::Settings>(m, "Settings", "Game options that vary between AWBW games.")
      .def(nb::init<>())
      .def_rw("funds_per_property", &awbw::Settings::funds_per_property)
      .def_rw("capture_win", &awbw::Settings::capture_win)
      .def_rw("day_limit", &awbw::Settings::day_limit)
      .def_rw("unit_limit", &awbw::Settings::unit_limit)
      .def_rw("use_powers", &awbw::Settings::use_powers)
      .def_rw("banned_units", &awbw::Settings::banned_units);

  nb::class_<awbw::Unit>(m, "Unit")
      .def_ro("id", &awbw::Unit::id)
      .def_ro("tile", &awbw::Unit::tile)
      .def_ro("type", &awbw::Unit::type)
      .def_ro("owner", &awbw::Unit::owner)
      .def_ro("hp", &awbw::Unit::hp)
      .def_ro("fuel", &awbw::Unit::fuel)
      .def_ro("ammo", &awbw::Unit::ammo)
      .def_ro("flags", &awbw::Unit::flags)
      .def_ro("carrier", &awbw::Unit::carrier)
      .def_prop_ro(
          "cargo",
          [](const awbw::Unit& u) {
            auto slot = [](uint8_t c) { return c == awbw::kNoUnit ? -1 : static_cast<int>(c); };
            return std::make_tuple(slot(u.cargo[0]), slot(u.cargo[1]));
          },
          "Slots of the units in cargo slots 1 and 2; -1 = empty (slots don't shift).");

  nb::class_<awbw::Player>(m, "Player")
      .def_ro("funds", &awbw::Player::funds)
      .def_ro("power", &awbw::Player::power)
      .def_ro("cop_cost", &awbw::Player::cop_cost)
      .def_ro("scop_cost", &awbw::Player::scop_cost)
      .def_ro("co", &awbw::Player::co)
      .def_ro("country", &awbw::Player::country)
      .def_ro("power_on", &awbw::Player::power_on)
      .def_ro("power_uses", &awbw::Player::power_uses)
      .def_ro("eliminated", &awbw::Player::eliminated)
      .def_ro("power_atk", &awbw::Player::power_atk);

  nb::class_<awbw::GameState>(m, "GameState", "A game position: a fixed-size plain value.")
      .def("copy", [](const awbw::GameState& s) { return s; })
      .def(
          "assign", [](awbw::GameState& s, const awbw::GameState& other) { s = other; }, "other"_a,
          "Make this state equal to `other`, in place.")
      .def("to_bytes",
           [](const awbw::GameState& s) {
             return nb::bytes(reinterpret_cast<const char*>(&s), sizeof s);
           })
      .def_static(
          "from_bytes",
          [](nb::bytes b) {
            if (b.size() != sizeof(awbw::GameState))
              throw nb::value_error("wrong size for a GameState");
            awbw::GameState s;
            std::memcpy(&s, b.c_str(), sizeof s);
            return s;
          },
          "data"_a)
      .def_ro("map_id", &awbw::GameState::map_id)
      .def_ro("settings", &awbw::GameState::settings)
      .def_ro("day", &awbw::GameState::day)
      .def_ro("active", &awbw::GameState::active)
      .def_ro("weather", &awbw::GameState::weather, "Index into WEATHERS.")
      .def_ro("weather_owner", &awbw::GameState::weather_owner,
              "Player whose power set the weather, or 255 for the game's own weather.")
      .def_ro("winner", &awbw::GameState::winner)
      .def(
          "seed", [](awbw::GameState& s, uint64_t seed) { awbw::rng_seed(s.rng, seed); }, "seed"_a,
          "Reseed the state's RNG (combat luck).")
      .def_prop_ro("players",
                   [](const awbw::GameState& s) {
                     return std::vector<awbw::Player>(s.players, s.players + awbw::kMaxPlayers);
                   })
      .def(
          "unit",
          [](const awbw::GameState& s, int slot) {
            check_slot(s, slot);
            return s.units[slot];
          },
          "slot"_a)
      .def(
          "hp_belief",
          [](const awbw::GameState& s, int slot) {
            check_slot(s, slot);
            return std::vector<int>(s.hp_belief[slot], s.hp_belief[slot] + awbw::kHpBins);
          },
          "slot"_a,
          "What a player who watched every turn knows of the unit's internal HP: HP_BINS\n"
          "weights (0..255, the largest 255), the k-th for internal HP 10 * (displayed - 1) + 1 + "
          "k.")
      .def(
          "set_hp_belief",
          [](awbw::GameState& s, int slot, const std::vector<int>& w) {
            check_slot(s, slot);
            if (w.size() != awbw::kHpBins) throw nb::value_error("need HP_BINS weights");
            const int hp = s.units[slot].hp;
            for (int v : w) {
              if (v < 0 || v > awbw::kBeliefMax) throw nb::value_error("weights run 0..255");
            }
            if (w[hp - awbw::hp_bin_base(hp)] == 0)
              throw nb::value_error("the belief rules out the unit's own HP");
            for (int k = 0; k < awbw::kHpBins; ++k)
              s.hp_belief[slot][k] = static_cast<uint8_t>(w[k]);
          },
          "slot"_a, "weights"_a, "Replace the unit's HP belief (see hp_belief).")
      .def(
          "set_hp_unknown",
          [](awbw::GameState& s, int slot) {
            check_slot(s, slot);
            awbw::belief_unknown(s, slot);
          },
          "slot"_a, "Make every internal HP the unit's display allows equally likely.")
      .def(
          "set_hp",
          [](awbw::GameState& s, int slot, int hp) {
            check_slot(s, slot);
            if (hp < 1 || hp > 100) throw nb::value_error("HP runs 1..100");
            awbw::Unit& u = s.units[slot];
            const bool same_display = awbw::displayed_hp(hp) == awbw::displayed_hp(u.hp);
            if (same_display && s.hp_belief[slot][hp - awbw::hp_bin_base(hp)] == 0)
              throw nb::value_error("the HP belief rules out that HP");
            if (awbw::hides_hp(s, u.owner) && s.hp_hidden[slot][awbw::displayed_hp(hp) - 1] == 0)
              throw nb::value_error("the hidden-HP belief rules out that display");
            u.hp = static_cast<uint8_t>(hp);
            if (!same_display) awbw::belief_unknown(s, slot);
          },
          "slot"_a, "hp"_a,
          "Set the unit's internal HP, keeping what players know: within its display the HP\n"
          "belief must allow it, and a new display makes every HP behind it equally likely.\n"
          "For a unit whose owner hides HP, the hidden-HP belief must allow its display.")
      .def(
          "set_supplies",
          [](awbw::GameState& s, int slot, int fuel, int ammo) {
            check_slot(s, slot);
            const awbw::UnitSpec& t = awbw::kUnits[s.units[slot].type];
            if (fuel < 0 || fuel > t.fuel) throw nb::value_error("fuel out of range");
            if (ammo < 0 || ammo > t.ammo) throw nb::value_error("ammo out of range");
            s.units[slot].fuel = static_cast<uint8_t>(fuel);
            s.units[slot].ammo = static_cast<uint8_t>(ammo);
          },
          "slot"_a, "fuel"_a, "ammo"_a, "Set the unit's fuel and ammo.")
      .def(
          "hides_hp",
          [](const awbw::GameState& s, int player) {
            if (player < 0 || player >= awbw::kMaxPlayers) throw nb::index_error("bad player");
            return awbw::hides_hp(s, player);
          },
          "player"_a, "Whether the player's units show no HP to the other player (Sonja).")
      .def(
          "hidden_belief",
          [](const awbw::GameState& s, int slot) {
            check_slot(s, slot);
            return std::vector<int>(s.hp_hidden[slot], s.hp_hidden[slot] + awbw::kHpBins);
          },
          "slot"_a,
          "What the opponent of a player who hides HP knows of the unit's displayed HP:\n"
          "HP_BINS weights (0..255, the largest 255), the k-th for displayed HP k + 1. Only\n"
          "kept for units of a player who hides HP (hides_hp).")
      .def(
          "set_hidden_belief",
          [](awbw::GameState& s, int slot, const std::vector<int>& w) {
            check_slot(s, slot);
            if (w.size() != awbw::kHpBins) throw nb::value_error("need HP_BINS weights");
            for (int v : w) {
              if (v < 0 || v > awbw::kBeliefMax) throw nb::value_error("weights run 0..255");
            }
            if (w[awbw::displayed_hp(s.units[slot].hp) - 1] == 0)
              throw nb::value_error("the belief rules out the unit's own displayed HP");
            for (int k = 0; k < awbw::kHpBins; ++k)
              s.hp_hidden[slot][k] = static_cast<uint8_t>(w[k]);
          },
          "slot"_a, "weights"_a, "Replace the unit's hidden-HP belief (see hidden_belief).")
      .def(
          "set_hidden_unknown",
          [](awbw::GameState& s, int slot) {
            check_slot(s, slot);
            awbw::hidden_unknown(s, slot);
          },
          "slot"_a, "Make every displayed HP equally likely to the unit's opponent.")
      .def(
          "unit_slots",
          [](const awbw::GameState& s) {
            std::vector<int> out;
            for (int i = 0; i < awbw::kMaxUnits; ++i) {
              if (s.units[i].alive()) out.push_back(i);
            }
            return out;
          },
          "Slots holding a unit, in slot order.")
      .def(
          "find_unit",
          [](const awbw::GameState& s, uint32_t id) {
            for (int i = 0; i < awbw::kMaxUnits; ++i) {
              if (s.units[i].id == id && id != 0) return i;
            }
            return -1;
          },
          "id"_a, "Slot of the unit with this id, or -1.")
      .def(
          "unit_rows",
          [](const awbw::GameState& s) {
            const awbw::MapData& md = map_for(s);
            std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int>> out;
            for (int i = 0; i < awbw::kMaxUnits; ++i) {
              const awbw::Unit& u = s.units[i];
              if (!u.alive()) continue;
              auto id_at = [&](uint8_t slot) {
                return slot == awbw::kNoUnit ? 0 : static_cast<int>(s.units[slot].id);
              };
              out.emplace_back(static_cast<int>(u.id), u.type, u.owner, md.x_of(u.tile),
                               md.y_of(u.tile), u.hp, u.fuel, u.ammo, u.flags, id_at(u.cargo[0]),
                               id_at(u.cargo[1]));
            }
            return out;
          },
          "(id, type, owner, x, y, hp, fuel, ammo, flags, cargo1_id, cargo2_id) for every unit, "
          "in slot order; cargo ids are 0 when empty.")
      .def(
          "unit_at",
          [](const awbw::GameState& s, int x, int y) {
            const awbw::MapData& md = map_for(s);
            if (x < 0 || y < 0 || x >= md.width || y >= md.height)
              throw nb::index_error("off the map");
            const int slot = s.unit_at[md.tile(x, y)];
            return slot == awbw::kNoUnit ? -1 : slot;
          },
          "x"_a, "y"_a, "Slot of the uncarried unit on a tile, or -1.")
      .def(
          "prop_owner", [](const awbw::GameState& s, int p) { return s.prop_owner[p]; }, "prop"_a)
      .def(
          "prop_points", [](const awbw::GameState& s, int p) { return s.prop_points[p]; }, "prop"_a)
      .def(
          "check", [](const awbw::GameState& s) { return awbw::check_state(map_for(s), s); },
          "Consistency problems, or '' if the state is sound.")
      .def("__getstate__",
           [](const awbw::GameState& s) {
             return nb::bytes(reinterpret_cast<const char*>(&s), sizeof s);
           })
      .def("__setstate__", [](awbw::GameState& s, nb::bytes b) {
        if (b.size() != sizeof(awbw::GameState))
          throw nb::value_error("wrong size for a GameState");
        new (&s) awbw::GameState;
        std::memcpy(&s, b.c_str(), sizeof s);
      });

  m.def("load_snapshot", &load_snapshot, "map"_a, "settings"_a, "day"_a, "active"_a, "players"_a,
        "buildings"_a, "units"_a, "weather"_a = 0, "weather_owner"_a = -1,
        "GameState from a flattened AWBW snapshot. players: [2, 8] (co, country, funds, power, "
        "cop_cost, scop_cost, power_on, eliminated); buildings: [n, 4] (x, y, terrain_id, "
        "capture); units: [n, 11] (id, type, owner, x, y, hp, fuel, ammo, flags, cargo1_id, "
        "cargo2_id); weather: WEATHERS index; weather_owner: player index whose power set "
        "it, or -1.");

  m.def(
      "reachable",
      [](const awbw::GameState& s, int slot) {
        check_slot(s, slot);
        const awbw::MapData& md = map_for(s);
        auto r = std::make_unique<awbw::Reach>();
        awbw::reachable(md, s, slot, *r);
        std::vector<std::tuple<int, int, int>> out;
        out.reserve(r->n);
        for (int i = 0; i < r->n; ++i) {
          const int t = r->tiles[i];
          out.emplace_back(md.x_of(t), md.y_of(t), r->cost[t]);
        }
        return out;
      },
      "state"_a, "slot"_a,
      "(x, y, cost) of every tile the unit can move through this turn, cheapest first.");
  m.def(
      "path_cost",
      [](const awbw::GameState& s, int slot, const std::vector<std::tuple<int, int>>& path) {
        check_slot(s, slot);
        const awbw::MapData& md = map_for(s);
        std::vector<uint16_t> tiles;
        for (auto [x, y] : path) {
          if (x < 0 || y < 0 || x >= md.width || y >= md.height) return -1;
          tiles.push_back(static_cast<uint16_t>(md.tile(x, y)));
        }
        return awbw::path_cost(md, s, slot, tiles.data(), static_cast<int>(tiles.size()));
      },
      "state"_a, "slot"_a, "path"_a,
      "Movement cost of a path of (x, y) starting on the unit's tile, or -1 if it is not a legal "
      "move.");
  m.def(
      "move_points",
      [](const awbw::GameState& s, int slot) {
        check_slot(s, slot);
        return awbw::move_points(s, s.units[slot]);
      },
      "state"_a, "slot"_a, "Movement points with CO bonuses, before the fuel cap.");

  nb::class_<awbw::Combatant>(
      m, "Combatant",
      "One side of a fight: unit, tile and CO state as the damage formula sees them.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](awbw::Combatant* c, int type, int hp, int ammo, int co, int power, int terrain,
             int stars, int towers, int extra_atk, bool hidden) {
            new (c) awbw::Combatant{static_cast<uint8_t>(type),      static_cast<uint8_t>(hp),
                                    static_cast<uint8_t>(ammo),      static_cast<uint8_t>(co),
                                    static_cast<uint8_t>(power),     static_cast<uint8_t>(terrain),
                                    static_cast<uint8_t>(stars),     static_cast<uint8_t>(towers),
                                    static_cast<int16_t>(extra_atk), static_cast<uint8_t>(hidden)};
          },
          "type"_a, "hp"_a, "ammo"_a = 0, "co"_a = 0, "power"_a = 0, "terrain"_a = 0, "stars"_a = 0,
          "towers"_a = 0, "extra_atk"_a = 0, "hidden"_a = false)
      .def_rw("type", &awbw::Combatant::type, "engine unit index")
      .def_rw("hp", &awbw::Combatant::hp, "internal HP, 1..100")
      .def_rw("ammo", &awbw::Combatant::ammo)
      .def_rw("co", &awbw::Combatant::co, "AWBW CO id")
      .def_rw("power", &awbw::Combatant::power, "0 none, 1 COP, 2 SCOP")
      .def_rw("terrain", &awbw::Combatant::terrain, "terrain kind index of the unit's tile")
      .def_rw("stars", &awbw::Combatant::stars, "defense stars of the unit's tile")
      .def_rw("towers", &awbw::Combatant::towers, "comtowers owned by the unit's player")
      .def_rw("extra_atk", &awbw::Combatant::extra_atk, "attack % from state-dependent CO effects")
      .def_rw("hidden", &awbw::Combatant::hidden, "dived Sub or hidden Stealth");

  m.def("attack_value", &awbw::attack_value, "att"_a, "defender"_a);
  m.def("defense_value", &awbw::defense_value, "defender"_a, "att"_a);
  m.def("damage", &awbw::damage, "att"_a, "defender"_a, "luck"_a, "counter"_a = false,
        "Damage in internal HP for a given net luck roll.");
  m.def("can_attack", &awbw::can_attack, "att"_a, "defender"_a, "dist"_a);
  m.def(
      "weapon",
      [](const awbw::Combatant& att, int def_type) {
        return static_cast<int>(awbw::choose_weapon(att, def_type));
      },
      "att"_a, "def_type"_a, "-1 none, 0 ammo weapon, 1 weapon without ammo");
  m.def(
      "combat_outcomes",
      [](const awbw::Combatant& att, const awbw::Combatant& def, int dist) {
        std::vector<awbw::Outcome> out;
        awbw::combat_outcomes(att, def, dist, out);
        std::vector<std::tuple<int, int, double>> res;
        res.reserve(out.size());
        for (const auto& o : out) res.emplace_back(o.att_hp, o.def_hp, o.prob);
        return res;
      },
      "att"_a, "defender"_a, "dist"_a,
      "Exact (attacker_hp, defender_hp, probability) results of an attack, counterattack "
      "included.");

  m.def("apply", &apply_action, "state"_a, "kind"_a, "unit"_a = -1, "to"_a = nb::none(),
        "target"_a = nb::none(), "arg"_a = 0, "path"_a = nb::none(), "att_hp"_a = -1,
        "def_hp"_a = -1, "missiles"_a = nb::none(),
        "Carry out one action for the active player, in place. kind: ACTIONS[name]; unit: slot; "
        "to / target: (x, y); path: [(x, y), ...] from the unit's tile to `to` (default: the "
        "cheapest path its owner can see); att_hp / def_hp force a fight's result; missiles: "
        "a power's missile centres [(x, y), ...] (default: aimed as AWBW does). Returns OK, "
        "TRAPPED or an error status (state unchanged); see status_name.");
  {
    nb::list channels;
    const char* names[awbw::kNumChannels] = {
        "Select", "Delete",  "Move", "Wait",       "Capture", "Load",   "Join",    "Supply", "Hide",
        "Unhide", "Explode", "Fire", "AttackSeam", "Repair",  "Launch", "Unload1", "Unload2"};
    for (const char* n : names) channels.append(n);
    m.attr("CHANNELS") = nb::tuple(channels);
    m.attr("NUM_CHANNELS") = static_cast<int>(awbw::kNumChannels);
    m.attr("PHASES") = nb::make_tuple("idle", "selected", "moved", "factory");
  }
  m.attr("ID_END_TURN") = awbw::kIdEndTurn;
  m.attr("ID_COP") = awbw::kIdCop;
  m.attr("ID_SCOP") = awbw::kIdScop;
  m.attr("ID_BUILD") = awbw::kIdBuild;
  m.attr("NUM_ACTION_IDS") = awbw::kNumActionIds;

  nb::class_<awbw::Selection>(
      m, "Selection",
      "The active player's action in progress, between micro-steps: phase (PHASES index), the\n"
      "picked unit's slot (-1 if none) and tile: the unit's own, its destination, or the\n"
      "factory's (a map tile index, y * width + x).")
      .def(nb::init<>())
      .def_rw("phase", &awbw::Selection::phase)
      .def_prop_rw(
          "unit",
          [](const awbw::Selection& s) { return s.unit == awbw::kNoUnit ? -1 : int(s.unit); },
          [](awbw::Selection& s, int v) {
            s.unit = v < 0 || v >= awbw::kMaxUnits ? awbw::kNoUnit : static_cast<uint8_t>(v);
          })
      .def_rw("tile", &awbw::Selection::tile)
      .def("copy", [](const awbw::Selection& s) { return s; })
      .def("__eq__",
           [](const awbw::Selection& a, nb::handle b) {
             return nb::isinstance<awbw::Selection>(b) && a == nb::cast<awbw::Selection>(b);
           })
      .def("__repr__", [](const awbw::Selection& s) {
        static const char* kPhases[] = {"idle", "selected", "moved", "factory"};
        return std::string("Selection(") + (s.phase < 4 ? kPhases[s.phase] : "?") +
               ", unit=" + std::to_string(s.unit == awbw::kNoUnit ? -1 : int(s.unit)) +
               ", tile=" + std::to_string(s.tile) + ")";
      });

  m.def(
      "legal_actions",
      [](const awbw::GameState& s) {
        const awbw::MapData& md = map_for(s);
        std::vector<awbw::Action> acts;
        awbw::legal_actions(md, s, acts);
        std::vector<ActionTuple> out;
        out.reserve(acts.size());
        for (const awbw::Action& a : acts) out.push_back(action_tuple(md, a));
        return out;
      },
      "state"_a,
      "Every complete action the player to move can take, Resign aside, as apply()'s\n"
      "arguments (kind, unit, to, target, arg): apply(state, *a) takes each one, with OK, or\n"
      "TRAPPED when an enemy the player can't see is in the way. [] once the game is over.");
  m.def("legal_ids", &legal_id_array, "state"_a, "selection"_a = awbw::Selection{},
        "Legal micro-step ids from `selection`, ascending (int64). An id is\n"
        "CHANNELS index * MAX_TILES + tile, or ID_END_TURN, ID_COP, ID_SCOP, ID_BUILD + unit\n"
        "type. Every id leads on to a legal action; empty once the game is over.");
  m.def(
      "state_hash",
      [](const awbw::GameState& s, const awbw::Selection& sel) { return awbw::state_hash(s, sel); },
      "state"_a, "selection"_a = awbw::Selection{},
      "64-bit transposition-table key (core/hash.h): equal for states that play the same\n"
      "whatever order of actions reached them, since unit slots and ids, the RNG and free\n"
      "slots don't count. With a selection, the decision in progress (an idle one gives the\n"
      "state's key). The same on every run.");
  m.def("hash_batch", &hash_batch, "states"_a,
        "state_hash of raw states [N, STATE_BYTES] (uint64 [N]); their maps must be registered,\n"
        "and states are checked first.");
  m.def(
      "legal_mask",
      [](const awbw::GameState& s, const awbw::Selection& sel) {
        std::vector<uint16_t> ids;
        awbw::legal_ids(map_for(s), s, sel, ids);
        auto* data = new bool[awbw::kNumActionIds]();
        for (uint16_t id : ids) data[id] = true;
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<bool*>(p); });
        return nb::ndarray<nb::numpy, bool, nb::ndim<1>>(
            data, {static_cast<size_t>(awbw::kNumActionIds)}, owner);
      },
      "state"_a, "selection"_a = awbw::Selection{},
      "legal_ids() as a mask over all NUM_ACTION_IDS ids.");
  m.def(
      "step",
      [](awbw::GameState& s, awbw::Selection& sel, int id, std::optional<std::vector<XY>> path,
         int att_hp, int def_hp, std::optional<std::vector<XY>> missiles) {
        const awbw::MapData& md = map_for(s);
        const DetailArgs d(md, path, att_hp, def_hp, missiles);
        return static_cast<int>(awbw::step(md, s, sel, id, d.d));
      },
      "state"_a, "selection"_a, "id"_a, "path"_a = nb::none(), "att_hp"_a = -1, "def_hp"_a = -1,
      "missiles"_a = nb::none(),
      "Take micro-step `id`, in place: a pick advances `selection` and returns OK; a step\n"
      "that completes an action applies it (path / att_hp / def_hp / missiles as in apply),\n"
      "resets `selection` and returns apply's status. An id that isn't legal returns an\n"
      "error and changes nothing.");
  m.def(
      "step_action",
      [](const awbw::GameState& s, const awbw::Selection& sel,
         int id) -> std::optional<ActionTuple> {
        const awbw::MapData& md = map_for(s);
        awbw::Action a;
        if (!awbw::step_action(md, s, sel, id, a)) return std::nullopt;
        return action_tuple(md, a);
      },
      "state"_a, "selection"_a, "id"_a,
      "The action (as in legal_actions) that step `id` from `selection` would complete, or\n"
      "None for a pick. Legality isn't checked.");
  m.def(
      "action_steps",
      [](const awbw::GameState& s, int kind, int unit, std::optional<XY> to,
         std::optional<XY> target, int arg) {
        const awbw::MapData& md = map_for(s);
        const awbw::Action a = make_action(md, s, kind, unit, to, target, arg);
        if ((a.kind < awbw::kActBuild || a.kind == awbw::kActDelete) &&
            (a.unit == awbw::kNoUnit || !s.units[a.unit].alive()))
          throw nb::index_error("no unit in that slot");
        uint16_t ids[3];
        const int n = awbw::action_steps(md, s, a, ids);
        return std::vector<int>(ids, ids + n);
      },
      "state"_a, "kind"_a, "unit"_a = -1, "to"_a = nb::none(), "target"_a = nb::none(), "arg"_a = 0,
      "The micro-step ids that play an action given as apply()'s arguments (so\n"
      "action_steps(state, *a) for a in legal_actions(state)); [] for Resign.");
  m.def(
      "missile_targets",
      [](const awbw::GameState& s, int level) {
        const awbw::MapData& md = map_for(s);
        uint16_t out[3];
        const int n = awbw::missile_targets(md, s, level, out);
        std::vector<std::optional<XY>> xy;
        for (int i = 0; i < n; ++i) {
          if (out[i] == awbw::kNoTile) {
            xy.emplace_back(std::nullopt);
          } else {
            xy.emplace_back(XY{md.x_of(out[i]), md.y_of(out[i])});
          }
        }
        return xy;
      },
      "state"_a, "level"_a,
      "Where the active player's power `level` (1 COP, 2 SCOP) would aim its missiles now, as "
      "AWBW does: [(x, y), ...] (None for a missile with no target); [] for a power without "
      "missiles.");
  m.def(
      "combatant",
      [](const awbw::GameState& s, int slot, std::optional<XY> at) {
        check_slot(s, slot);
        const awbw::MapData& md = map_for(s);
        return awbw::combatant(md, s, slot, at ? tile_of(md, *at) : s.units[slot].tile);
      },
      "state"_a, "slot"_a, "at"_a = nb::none(),
      "The unit as the damage formula sees it, standing on `at` (default: its tile).");
  m.def(
      "fight_outcomes",
      [](const awbw::GameState& s, int slot, XY from, XY target) {
        check_slot(s, slot);
        const awbw::MapData& md = map_for(s);
        const int from_t = tile_of(md, from), target_t = tile_of(md, target);
        const int def = s.unit_at[target_t];
        std::vector<std::tuple<int, int, double>> res;
        if (def == awbw::kNoUnit) return res;
        std::vector<awbw::Outcome> out;
        awbw::combat_outcomes(awbw::combatant(md, s, slot, from_t),
                              awbw::combatant(md, s, def, target_t), md.distance(from_t, target_t),
                              out);
        for (const auto& o : out) res.emplace_back(o.att_hp, o.def_hp, o.prob);
        return res;
      },
      "state"_a, "slot"_a, "from_"_a, "target"_a,
      "(attacker_hp, defender_hp, probability) results of the unit attacking from `from_` the "
      "unit on `target`.");
  nb::class_<awbw::SearchConfig>(m, "SearchConfig", "Settings of a BatchedSearch (core/search.h).")
      .def(
          "__init__",
          [](awbw::SearchConfig* c, int sims, int fast_sims, float full_fraction, int considered,
             float gumbel_scale, float visit_scale, float value_scale, bool rescale,
             bool pick_values, int max_days, int max_actions, float resign_value, int resign_turns,
             float resign_off, bool movement, uint64_t seed) {
            new (c) awbw::SearchConfig{.sims = sims,
                                       .fast_sims = fast_sims,
                                       .full_fraction = full_fraction,
                                       .considered = considered,
                                       .gumbel_scale = gumbel_scale,
                                       .visit_scale = visit_scale,
                                       .value_scale = value_scale,
                                       .rescale = rescale,
                                       .pick_values = pick_values,
                                       .max_days = max_days,
                                       .max_actions = max_actions,
                                       .resign_value = resign_value,
                                       .resign_turns = resign_turns,
                                       .resign_off = resign_off,
                                       .movement = movement,
                                       .seed = seed};
          },
          nb::kw_only(), "sims"_a = 64, "fast_sims"_a = 0, "full_fraction"_a = 1.0f,
          "considered"_a = 16, "gumbel_scale"_a = 1.0f, "visit_scale"_a = 50.0f,
          "value_scale"_a = 0.1f, "rescale"_a = false, "pick_values"_a = true, "max_days"_a = 0,
          "max_actions"_a = 0, "resign_value"_a = -0.9f, "resign_turns"_a = 0,
          "resign_off"_a = 0.1f, "movement"_a = false, "seed"_a = 0)
      .def_rw("sims", &awbw::SearchConfig::sims,
              "simulations of a full search, after the root's own evaluation")
      .def_rw("fast_sims", &awbw::SearchConfig::fast_sims,
              "self-play: simulations of a fast search, played but not kept as a sample")
      .def_rw("full_fraction", &awbw::SearchConfig::full_fraction,
              "self-play: share of the searches that are full")
      .def_rw("considered", &awbw::SearchConfig::considered,
              "steps the root samples to consider (Gumbel top-k)")
      .def_rw("gumbel_scale", &awbw::SearchConfig::gumbel_scale,
              "0: no noise, the root considers the likeliest steps")
      .def_rw("visit_scale", &awbw::SearchConfig::visit_scale,
              "sigma(q) = (visit_scale + most visits) * value_scale * q")
      .def_rw("value_scale", &awbw::SearchConfig::value_scale)
      .def_rw("rescale", &awbw::SearchConfig::rescale,
              "map Q to 0..1 by its spread among a node's steps (mctx's default), not (Q + 1) / 2")
      .def_rw("pick_values", &awbw::SearchConfig::pick_values,
              "decisions with a unit or factory picked take the network's value; else that of the\n"
              "decision they were picked from (the same state)")
      .def_rw("max_days", &awbw::SearchConfig::max_days,
              "a game still going after this day is decided as a day limit decides it; 0: no cap")
      .def_rw("max_actions", &awbw::SearchConfig::max_actions,
              "self-play: a game ends after this many actions, decided the same way; 0: no cap")
      .def_rw("resign_value", &awbw::SearchConfig::resign_value,
              "self-play: a turn whose first searched decision has at most this value counts "
              "toward resigning")
      .def_rw("resign_turns", &awbw::SearchConfig::resign_turns,
              "self-play: a player resigns after this many such turns in a row; 0: never")
      .def_rw("resign_off", &awbw::SearchConfig::resign_off,
              "self-play: share of the games that play on to the end, to check the rule")
      .def_rw("movement", &awbw::SearchConfig::movement, "the network reads the movement planes")
      .def_rw("seed", &awbw::SearchConfig::seed);

  nb::class_<awbw::SearchResult>(m, "SearchResult", "One searched decision.")
      .def_ro("id", &awbw::SearchResult::id, "the step chosen")
      .def_ro("mover", &awbw::SearchResult::mover)
      .def_ro("value", &awbw::SearchResult::value,
              "the root's searched value (mean over its visits), from the mover's side")
      .def_ro("net_value", &awbw::SearchResult::net_value, "the network's value of the root")
      .def_ro("sims", &awbw::SearchResult::sims)
      .def_prop_ro(
          "ids", [](const awbw::SearchResult& r) { return array_of<int64_t>(r.ids); },
          nb::rv_policy::move, "the legal steps, ascending")
      .def_prop_ro(
          "prior", [](const awbw::SearchResult& r) { return array_of<float>(r.prior); },
          nb::rv_policy::move)
      .def_prop_ro(
          "policy", [](const awbw::SearchResult& r) { return array_of<float>(r.policy); },
          nb::rv_policy::move, "the improved policy, softmax(logits + sigma(completed Q))")
      .def_prop_ro(
          "visits", [](const awbw::SearchResult& r) { return array_of<uint32_t>(r.visits); },
          nb::rv_policy::move)
      .def_prop_ro(
          "q", [](const awbw::SearchResult& r) { return array_of<float>(r.q); },
          nb::rv_policy::move, "each step's value from the mover's side (0 if unvisited)");

  nb::class_<awbw::GameRecord>(m, "GameRecord", "How a self-play game ended.")
      .def_ro("game", &awbw::GameRecord::game)
      .def_ro("winner", &awbw::GameRecord::winner, "player index, or DRAW")
      .def_ro("days", &awbw::GameRecord::days)
      .def_ro("actions", &awbw::GameRecord::actions)
      .def_ro("decisions", &awbw::GameRecord::decisions, "searched, full and fast")
      .def_ro("capped", &awbw::GameRecord::capped,
              "stopped by max_days or max_actions and decided as a day limit decides")
      .def_ro("resigned", &awbw::GameRecord::resigned, "the loser resigned")
      .def_ro("would_resign", &awbw::GameRecord::would_resign,
              "a game that played on (resign_off): the player the rule would first have had "
              "resign, else -1")
      .def_ro("would_resign_day", &awbw::GameRecord::would_resign_day, "and the day, else -1")
      .def_ro("final_state", &awbw::GameRecord::final_state);

  nb::class_<awbw::BatchedSearch>(
      m, "BatchedSearch",
      "Many Gumbel MCTS searches at once, with the network called in batches (core/search.h):\n"
      "    while search.collect():\n"
      "        planes, globals, legal, movers = search.batch()\n"
      "        search.submit(logits, values)\n"
      "Not thread-safe.")
      .def(nb::init<const awbw::SearchConfig&, int>(), "config"_a, "size"_a,
           "Leaves are encoded in a size x size window, which every map searched must fit.")
      .def_prop_ro("config", &awbw::BatchedSearch::config)
      .def_prop_ro("size", &awbw::BatchedSearch::size)
      .def(
          "add_root",
          [](awbw::BatchedSearch& bs, const awbw::GameState& s, const awbw::Selection& sel) {
            return bs.add_root(map_for(s), s, sel);
          },
          "state"_a, "selection"_a = awbw::Selection{},
          "Search the decision `selection` in `state` and keep the result; returns its slot.")
      .def(
          "add_game",
          [](awbw::BatchedSearch& bs, const awbw::GameState& s) {
            return bs.add_game(map_for(s), s);
          },
          "state"_a,
          "Play a game from `state` to its end, searching each decision with a choice; returns\n"
          "its number. Samples and records come out of take_samples() and take_records().")
      .def(
          "collect",
          [](awbw::BatchedSearch& bs) {
            nb::gil_scoped_release release;
            return bs.collect();
          },
          "Walk each search down to a leaf for the network, finishing the searches that are\n"
          "done; returns the number of leaves, 0 once nothing is left to search.")
      .def_prop_ro("leaves", &awbw::BatchedSearch::leaves)
      .def("max_legal", &awbw::BatchedSearch::max_legal)
      .def("batch", &search_batch,
           "The leaves as the policy network reads them: planes [n, P, size, size] uint8 and\n"
           "globals [n, NUM_GLOBALS + NUM_SELECTION_GLOBALS] float32 (as encode_policy_batch),\n"
           "each leaf's legal steps as policy indices [n, max_legal()] int64 padded with -1,\n"
           "and the player deciding at each [n] uint8.")
      .def("submit", &search_submit, "logits"_a, "values"_a,
           "The network's logits over each leaf's legal steps [n, >= max_legal()] float32 (in\n"
           "batch()'s order), and its value [n] float32 for the player deciding there,\n"
           "P(win) - P(loss).")
      .def_prop_ro("active", &awbw::BatchedSearch::active,
                   "searches not finished and games still going")
      .def_prop_ro("slots", &awbw::BatchedSearch::slots)
      .def("done", &awbw::BatchedSearch::done, "slot"_a)
      .def(
          "result",
          [](const awbw::BatchedSearch& bs, int slot) {
            if (!bs.done(slot)) throw nb::index_error("no finished search in that slot");
            return bs.result(slot);
          },
          "slot"_a)
      .def(
          "take_samples", &search_samples,
          "Self-play's full searches since the last call, as arrays: game, decision, states\n"
          "[N, STATE_BYTES], picks [N, 3] and n_picks (as encode_policy_batch takes them), mover,\n"
          "chosen, sims, value, net_value; then each sample's legal steps flattened, sample i's\n"
          "from offsets[i] to offsets[i + 1]: ids, index (each id's policy_index in the\n"
          "search's size x size window), prior, policy (the improved policy, the training\n"
          "target), visits and q.")
      .def("take_records", &awbw::BatchedSearch::take_records,
           "Self-play games finished since the last call.")
      .def("clear", &awbw::BatchedSearch::clear, "Free every slot.");
  m.attr("DRAW") = static_cast<int>(awbw::kDraw);
  m.def(
      "day_limit_winner",
      [](const awbw::GameState& s) { return awbw::day_limit_winner(map_for(s), s); }, "state"_a,
      "The winner of a game stopped at a day limit: more properties (comtowers and labs\n"
      "included), then the larger army value; else DRAW.");
  m.def("cop_cost", &awbw::cop_cost, "co"_a, "uses"_a, "COP cost after `uses` power uses.");
  m.def("scop_cost", &awbw::scop_cost, "co"_a, "uses"_a, "SCOP cost after `uses` power uses.");
}
