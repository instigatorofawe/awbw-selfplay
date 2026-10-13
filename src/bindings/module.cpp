#include <nanobind/nanobind.h>

#include "core/config.h"

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "AWBW engine core (C++)";
  m.attr("MAX_WIDTH") = awbw::kMaxWidth;
  m.attr("MAX_HEIGHT") = awbw::kMaxHeight;
  m.attr("MAX_TILES") = awbw::kMaxTiles;
  m.attr("MAX_PLAYERS") = awbw::kMaxPlayers;
}
