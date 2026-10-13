#pragma once

#include <cstdint>

namespace awbw {

// xoroshiro128+ (Blackman and Vigna). The state lives in GameState::rng, so a
// game replays exactly from any saved position.
inline uint64_t rng_next(uint64_t (&s)[2]) {
  const uint64_t s0 = s[0];
  uint64_t s1 = s[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  s[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
  s[1] = (s1 << 37) | (s1 >> 27);
  return result;
}

// Uniform integer in [0, n), n >= 1 (Lemire's multiply-shift; the bias is
// below 2^-56 for the small n the engine uses).
inline int rng_below(uint64_t (&s)[2], int n) {
  return static_cast<int>(((rng_next(s) >> 32) * static_cast<uint64_t>(n)) >> 32);
}

// Seed the state from one 64-bit value (splitmix64).
inline void rng_seed(uint64_t (&s)[2], uint64_t seed) {
  for (uint64_t& w : s) {
    uint64_t z = (seed += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    w = z ^ (z >> 31);
  }
}

}  // namespace awbw
