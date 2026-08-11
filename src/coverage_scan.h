#ifndef GPUSIMD_COVERAGE_SCAN_H
#define GPUSIMD_COVERAGE_SCAN_H

#include "lane_program.h"

#include <cstdint>
#include <span>
#include <vector>

namespace gpusimd {

enum class CoverageResolveMode : uint32_t {
  kDirect = 0,
  kNonZero = 1,
  kEvenOdd = 2
};

struct CoverageDeltas {
  std::vector<int32_t> values;
  uint32_t scale = 0;
  uint32_t verticalSamples = 0;
  uint32_t horizontalUnits = 0;
};

struct AnalyticTileCells {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t tileWidth = 64;
  uint32_t tileHeight = 16;
  uint32_t tilesX = 0;
  uint32_t tilesY = 0;
  uint32_t scale = 0;
  uint64_t binnedEdgeReferences = 0;
  std::vector<uint32_t> tileIds;
  std::vector<uint32_t> tileLookup;
  std::vector<int32_t> values;
};

CoverageDeltas buildCoverageDeltas(
  uint32_t width,
  uint32_t height,
  std::span<const Edge> edges,
  uint32_t verticalSamples = 64,
  uint32_t horizontalUnits = 256);

CoverageDeltas buildAnalyticCoverageDeltas(
  uint32_t width,
  uint32_t height,
  std::span<const Edge> edges);

AnalyticTileCells buildTiledAnalyticCells(
  uint32_t width,
  uint32_t height,
  std::span<const Edge> edges,
  uint32_t tileWidth = 64,
  uint32_t tileHeight = 16);

CoverageDeltas materializeAnalyticTiles(const AnalyticTileCells& tiles);

void scanAnalyticTilesScalar(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  const AnalyticTileCells& tiles,
  CoverageResolveMode mode = CoverageResolveMode::kEvenOdd);

void scanAnalyticTilesAvx2(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  const AnalyticTileCells& tiles,
  CoverageResolveMode mode = CoverageResolveMode::kEvenOdd);

void scanAnalyticTilesAvx2Rows(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  const AnalyticTileCells& tiles,
  uint32_t yBegin,
  uint32_t yEnd,
  CoverageResolveMode mode = CoverageResolveMode::kEvenOdd);

void scanAnalyticTilesAvx2Threaded(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  const AnalyticTileCells& tiles,
  uint32_t threadCount,
  CoverageResolveMode mode = CoverageResolveMode::kEvenOdd);

uint32_t resolveCoverageValue(
  int32_t accumulated,
  uint32_t coverageScale,
  CoverageResolveMode mode) noexcept;

void validateAnalyticCoverageReference();

void scanCoverageScalar(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale,
  CoverageResolveMode mode = CoverageResolveMode::kDirect);

void scanCoverageAvx2(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale,
  CoverageResolveMode mode = CoverageResolveMode::kDirect);

void scanCoverageAvx2Threaded(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale,
  uint32_t threadCount,
  CoverageResolveMode mode = CoverageResolveMode::kDirect);

} // namespace gpusimd

#endif
