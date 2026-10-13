#pragma once

// Compile-time limits for the engine. GameState is a fixed-size POD sized by
// these, so changing them changes the on-disk state format.
namespace awbw {

inline constexpr int kMaxWidth = 40;
inline constexpr int kMaxHeight = 40;
inline constexpr int kMaxTiles = kMaxWidth * kMaxHeight;
inline constexpr int kMaxPlayers = 2;

}  // namespace awbw
