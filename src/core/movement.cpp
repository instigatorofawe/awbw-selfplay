#include "core/movement.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "core/movegen.h"

namespace awbw {

void turns_from(const MapData& m, const uint8_t* cost, int mp, const uint16_t* sources,
                const uint8_t* delay, int n, int max_turns, uint8_t* turns) {
  // Dijkstra over labels turn << 4 | points spent in that turn, compared as
  // numbers. A step costing w goes from (k, c) to (k, c + w) if that fits in
  // mp, else to (k + 1, w); either way the label grows by 1..30, so a ring
  // of 32 buckets holds every pending label (Dial's algorithm). A source
  // starts at (delay, mp): its first step opens turn delay + 1.
  constexpr uint16_t kNone = 0xFFFF;
  constexpr int kRing = 32;
  thread_local uint16_t best[kMaxTiles];
  thread_local std::vector<uint16_t> ring[kRing];
  thread_local std::vector<int> order;
  mp = std::clamp(mp, 1, 15);
  max_turns = std::min(max_turns, kMaxTurns);
  std::fill(best, best + m.n_tiles, kNone);
  std::memset(turns, kUnreachable, m.n_tiles);
  // Sources go in when the scan reaches their label, so the ring never holds
  // labels a lap apart.
  order.resize(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return delay[a] < delay[b]; });
  size_t next = 0;
  int pending = 0;
  for (int label = 0; label <= (max_turns << 4 | 15); ++label) {
    for (; next < order.size() && (delay[order[next]] << 4 | mp) == label; ++next) {
      const int t = sources[order[next]];
      if (label < best[t]) {
        best[t] = static_cast<uint16_t>(label);
        ring[label % kRing].push_back(static_cast<uint16_t>(t));
        ++pending;
      }
    }
    if (pending == 0) {
      if (next == order.size() || delay[order[next]] > max_turns) break;
      continue;
    }
    std::vector<uint16_t>& bucket = ring[label % kRing];
    const int k = label >> 4, spent = label & 15;
    for (size_t j = 0; j < bucket.size(); ++j) {  // steps add to other buckets only
      const int t = bucket[j];
      if (best[t] != label) continue;  // a cheaper label came later
      for (int d = 0; d < kNumDirs; ++d) {
        const int nb = m.neighbor[t][d];
        if (nb == kOffMap) continue;
        const int w = cost[nb];
        if (w == 0 || w > mp) continue;
        const int to = spent + w <= mp ? label + w : ((k + 1) << 4) + w;
        if ((to >> 4) > max_turns || to >= best[nb]) continue;
        best[nb] = static_cast<uint16_t>(to);
        ring[to % kRing].push_back(static_cast<uint16_t>(nb));
        ++pending;
      }
    }
    pending -= static_cast<int>(bucket.size());
    bucket.clear();
  }
  for (int t = 0; t < m.n_tiles; ++t) {
    if (best[t] != kNone) turns[t] = static_cast<uint8_t>(best[t] >> 4);
  }
  for (int i = 0; i < n; ++i) turns[sources[i]] = 0;
}

void compute_factory_turns(MapData& m) {
  std::fill(&m.factory_row[0][0], &m.factory_row[0][0] + kMaxProps * kNumMoverClasses, kNoRow);
  int rows = 0;
  for (int p = 0; p < m.n_props; ++p) {
    for (int c = 0; c < kNumMoverClasses; ++c) {
      if (factory_builds(m.prop_kind[p], kMovers.move_type[c]))
        m.factory_row[p][c] = static_cast<uint16_t>(rows++);
    }
  }
  m.factory_turns.assign(static_cast<size_t>(rows) * m.n_tiles, kUnreachable);
  const uint8_t no_delay = 0;
  for (int p = 0; p < m.n_props; ++p) {
    for (int c = 0; c < kNumMoverClasses; ++c) {
      const int row = m.factory_row[p][c];
      if (row == kNoRow) continue;
      turns_from(m, m.move_cost[kClear][kMovers.move_type[c]], kMovers.move[c], &m.prop_tile[p],
                 &no_delay, 1, kMaxTurns,
                 m.factory_turns.data() + static_cast<size_t>(row) * m.n_tiles);
    }
  }
}

}  // namespace awbw
