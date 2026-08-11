#ifndef GPUSIMD_COVERAGE_SCAN_H
#define GPUSIMD_COVERAGE_SCAN_H

#include "lane_program.h"

#include <cstdint>
#include <span>
#include <vector>

namespace gpusimd {

struct CoverageDeltas {
  std::vector<int32_t> values;
  uint32_t scale = 0;
  uint32_t verticalSamples = 0;
  uint32_t horizontalUnits = 0;
};

CoverageDeltas buildCoverageDeltas(
  uint32_t width,
  uint32_t height,
  std::span<const Edge> edges,
  uint32_t verticalSamples = 64,
  uint32_t horizontalUnits = 256);

void scanCoverageScalar(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale);

void scanCoverageAvx2(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale);

} // namespace gpusimd

#endif
