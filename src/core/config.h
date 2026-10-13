#pragma once

// Compile-time limits for the engine. GameState is a fixed-size POD sized by
// these, so changing them changes the on-disk state format.
namespace awbw {

inline constexpr int kMaxWidth = 40;
inline constexpr int kMaxHeight = 40;
inline constexpr int kMaxTiles = kMaxWidth * kMaxHeight;
inline constexpr int kMaxPlayers = 2;
// Units on the board or carried, both players. Global League caps each
// player at 50.
inline constexpr int kMaxUnits = 128;
// Tiles with per-game state: capturable properties, pipe seams, loaded silos.
// The largest map in the cached Global League pool has 72.
inline constexpr int kMaxProps = 128;

}  // namespace awbw
