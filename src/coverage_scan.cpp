#include "coverage_scan.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace gpusimd {
namespace {

void addSpan(
  std::span<uint32_t> row,
  int64_t x0,
  int64_t x1,
  uint32_t horizontalUnits) {

  const int64_t limit = int64_t(row.size()) * horizontalUnits;
  x0 = std::clamp<int64_t>(x0, 0, limit);
  x1 = std::clamp<int64_t>(x1, 0, limit);
  if (x1 <= x0)
    return;

  const size_t first = size_t(x0 / horizontalUnits);
  const size_t last = size_t((x1 - 1) / horizontalUnits);
  if (first == last) {
    row[first] += uint32_t(x1 - x0);
    return;
  }

  row[first] += horizontalUnits - uint32_t(x0 % horizontalUnits);
  for (size_t x = first + 1u; x < last; ++x)
    row[x] += horizontalUnits;
  row[last] += uint32_t((x1 - 1) % horizontalUnits) + 1u;
}

void validateBuffers(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height) {

  const size_t count = size_t(width) * height;
  if (coverage.size() != count || deltas.size() != count ||
      (!pixels.empty() && pixels.size() != count))
    throw std::runtime_error("coverage scan buffer size mismatch");
}

void paintLanes(
  std::span<uint32_t> pixels,
  size_t offset,
  uint32_t x,
  uint32_t y,
  uint32_t width,
  uint32_t height,
  const uint32_t* coverage,
  uint32_t laneCount,
  uint32_t coverageScale) {

  if (pixels.empty())
    return;
  for (uint32_t lane = 0; lane < laneCount; ++lane)
    pixels[offset + lane] = shadePixel(
      x + lane, y, width, height, coverage[lane], coverageScale);
}

void scanCoverageAvx2Rows(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale,
  CoverageResolveMode mode,
  uint32_t yBegin,
  uint32_t yEnd) {

  alignas(32) int32_t rawLanes[8];
  alignas(32) uint32_t resolvedLanes[8];
  const uint64_t evenOddPeriod64 = uint64_t(coverageScale) * 2u;
  const bool vectorEvenOdd =
    mode == CoverageResolveMode::kEvenOdd &&
    evenOddPeriod64 <= std::numeric_limits<uint32_t>::max() &&
    (evenOddPeriod64 & (evenOddPeriod64 - 1u)) == 0u;
  const __m256i coverageLimit = _mm256_set1_epi32(int32_t(coverageScale));
  const __m256i evenOddMask = _mm256_set1_epi32(
    vectorEvenOdd ? int32_t(evenOddPeriod64 - 1u) : 0);
  const __m256i evenOddPeriod = _mm256_set1_epi32(
    vectorEvenOdd ? int32_t(evenOddPeriod64) : 0);
  for (uint32_t y = yBegin; y < yEnd; ++y) {
    int32_t carry = 0;
    uint32_t x = 0;
    for (; x + 8u <= width; x += 8u) {
      const size_t index = size_t(y) * width + x;
      __m256i values = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(deltas.data() + index));
      values = _mm256_add_epi32(values, _mm256_slli_si256(values, 4));
      values = _mm256_add_epi32(values, _mm256_slli_si256(values, 8));
      const int32_t lowerHalf = _mm256_extract_epi32(values, 3);
      values = _mm256_add_epi32(values, _mm256_setr_epi32(
        carry, carry, carry, carry,
        carry + lowerHalf, carry + lowerHalf, carry + lowerHalf, carry + lowerHalf));
      _mm256_store_si256(reinterpret_cast<__m256i*>(rawLanes), values);
      carry = rawLanes[7];
      if (mode == CoverageResolveMode::kDirect) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(coverage.data() + index), values);
        paintLanes(
          pixels, index, x, y, width, height,
          reinterpret_cast<const uint32_t*>(rawLanes), 8, coverageScale);
      }
      else if (mode == CoverageResolveMode::kNonZero || vectorEvenOdd) {
        __m256i resolved = _mm256_abs_epi32(values);
        if (vectorEvenOdd) {
          resolved = _mm256_and_si256(resolved, evenOddMask);
          resolved = _mm256_min_epu32(
            resolved, _mm256_sub_epi32(evenOddPeriod, resolved));
        }
        resolved = _mm256_min_epu32(resolved, coverageLimit);
        _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(coverage.data() + index), resolved);
        if (!pixels.empty()) {
          _mm256_store_si256(
            reinterpret_cast<__m256i*>(resolvedLanes), resolved);
          paintLanes(
            pixels, index, x, y, width, height,
            resolvedLanes, 8, coverageScale);
        }
      }
      else {
        for (uint32_t lane = 0; lane < 8u; ++lane) {
          resolvedLanes[lane] = resolveCoverageValue(
            rawLanes[lane], coverageScale, mode);
          coverage[index + lane] = resolvedLanes[lane];
        }
        paintLanes(
          pixels, index, x, y, width, height,
          resolvedLanes, 8, coverageScale);
      }
    }
    for (; x < width; ++x) {
      const size_t index = size_t(y) * width + x;
      carry += deltas[index];
      const uint32_t resolved = resolveCoverageValue(carry, coverageScale, mode);
      coverage[index] = resolved;
      if (!pixels.empty())
        pixels[index] = shadePixel(x, y, width, height, resolved, coverageScale);
    }
  }
}

constexpr int64_t kAnalyticSubpixelScale = 256;
constexpr int64_t kAnalyticAreaScale = kAnalyticSubpixelScale * 2;
constexpr uint32_t kAnalyticCoverageScale =
  uint32_t(kAnalyticSubpixelScale * kAnalyticAreaScale);

template<class AddCell>
void rasterizeAnalyticEdge(
  const Edge& edge,
  uint32_t yBegin,
  uint32_t yEnd,
  AddCell&& addCell) {

  int64_t x0 = std::llround(double(edge.x0) * kAnalyticSubpixelScale);
  int64_t y0 = std::llround(double(edge.y0) * kAnalyticSubpixelScale);
  int64_t y1 = std::llround(double(edge.y1) * kAnalyticSubpixelScale);
  int64_t x1 = std::llround(
    (double(edge.x0) + double(edge.y1 - edge.y0) * double(edge.dxOverDy)) *
    kAnalyticSubpixelScale);
  if (y0 == y1)
    return;

  int64_t sign = 1;
  if (y0 > y1) {
    std::swap(x0, x1);
    std::swap(y0, y1);
    sign = -1;
  }

  const int64_t clippedY0 = std::max<int64_t>(
    y0, int64_t(yBegin) * kAnalyticSubpixelScale);
  const int64_t clippedY1 = std::min<int64_t>(
    y1, int64_t(yEnd) * kAnalyticSubpixelScale);
  if (clippedY0 >= clippedY1)
    return;

  const double inverseDy = 1.0 / double(y1 - y0);
  const auto xAtY = [&](int64_t y) {
    return double(x0) + double(y - y0) * double(x1 - x0) * inverseDy;
  };

  const uint32_t rowBegin = uint32_t(clippedY0 / kAnalyticSubpixelScale);
  const uint32_t rowEnd = uint32_t(
    (clippedY1 - 1) / kAnalyticSubpixelScale);
  for (uint32_t y = rowBegin; y <= rowEnd; ++y) {
    const int64_t segmentY0 = std::max<int64_t>(
      clippedY0, int64_t(y) * kAnalyticSubpixelScale);
    const int64_t segmentY1 = std::min<int64_t>(
      clippedY1, int64_t(y + 1u) * kAnalyticSubpixelScale);
    const double segmentX0 = xAtY(segmentY0);
    const double segmentX1 = xAtY(segmentY1);

    std::vector<double> cuts{0.0, 1.0};
    if (segmentX0 != segmentX1) {
      const double minX = std::min(segmentX0, segmentX1);
      const double maxX = std::max(segmentX0, segmentX1);
      int64_t boundary =
        (int64_t(std::floor(minX / double(kAnalyticSubpixelScale))) + 1) *
        kAnalyticSubpixelScale;
      for (; double(boundary) < maxX; boundary += kAnalyticSubpixelScale) {
        const double t = (double(boundary) - segmentX0) /
                         (segmentX1 - segmentX0);
        if (t > 0.0 && t < 1.0)
          cuts.push_back(t);
      }
      std::sort(cuts.begin(), cuts.end());
    }

    for (size_t i = 0; i + 1u < cuts.size(); ++i) {
      const double t0 = cuts[i];
      const double t1 = cuts[i + 1u];
      const double xa = segmentX0 + (segmentX1 - segmentX0) * t0;
      const double xb = segmentX0 + (segmentX1 - segmentX0) * t1;
      const int64_t ya = std::llround(
        double(segmentY0) + double(segmentY1 - segmentY0) * t0);
      const int64_t yb = std::llround(
        double(segmentY0) + double(segmentY1 - segmentY0) * t1);
      const int64_t deltaY = yb - ya;
      if (!deltaY)
        continue;

      const int64_t cellX = int64_t(std::floor(
        ((xa + xb) * 0.5) / double(kAnalyticSubpixelScale)));
      const int64_t cellOrigin = cellX * kAnalyticSubpixelScale;
      const int64_t fx0 = std::clamp<int64_t>(
        std::llround(xa) - cellOrigin, 0, kAnalyticSubpixelScale);
      const int64_t fx1 = std::clamp<int64_t>(
        std::llround(xb) - cellOrigin, 0, kAnalyticSubpixelScale);
      const int64_t cover = sign * deltaY;
      const int64_t area = cover * (fx0 + fx1);
      addCell(y, cellX, cover * kAnalyticAreaScale - area);
      addCell(y, cellX + 1, area);
    }
  }
}

} // namespace

CoverageDeltas buildCoverageDeltas(
  uint32_t width,
  uint32_t height,
  std::span<const Edge> edges,
  uint32_t verticalSamples,
  uint32_t horizontalUnits) {

  if (!width || !height || !verticalSamples || !horizontalUnits)
    throw std::runtime_error("coverage delta dimensions and precision must be non-zero");
  const uint64_t scale64 = uint64_t(verticalSamples) * horizontalUnits;
  if (scale64 > uint64_t(std::numeric_limits<int32_t>::max()))
    throw std::runtime_error("coverage delta precision exceeds int32 range");

  CoverageDeltas result;
  result.values.resize(size_t(width) * height);
  result.scale = uint32_t(scale64);
  result.verticalSamples = verticalSamples;
  result.horizontalUnits = horizontalUnits;

  std::vector<float> intersections;
  intersections.reserve(edges.size());
  std::vector<uint32_t> rowCoverage(width);
  for (uint32_t y = 0; y < height; ++y) {
    std::fill(rowCoverage.begin(), rowCoverage.end(), 0u);
    for (uint32_t sample = 0; sample < verticalSamples; ++sample) {
      const float sy = float(y) + (float(sample) + 0.5f) / float(verticalSamples);
      intersections.clear();
      for (const Edge& edge : edges) {
        if ((edge.y0 > sy) == (edge.y1 > sy))
          continue;
        intersections.push_back(edge.x0 + (sy - edge.y0) * edge.dxOverDy);
      }
      std::sort(intersections.begin(), intersections.end());
      for (size_t i = 0; i + 1u < intersections.size(); i += 2u) {
        const int64_t x0 = std::llround(double(intersections[i]) * horizontalUnits);
        const int64_t x1 = std::llround(double(intersections[i + 1u]) * horizontalUnits);
        addSpan(rowCoverage, x0, x1, horizontalUnits);
      }
    }

    int32_t previous = 0;
    for (uint32_t x = 0; x < width; ++x) {
      const int32_t current = int32_t(rowCoverage[x]);
      result.values[size_t(y) * width + x] = current - previous;
      previous = current;
    }
  }
  return result;
}

CoverageDeltas buildAnalyticCoverageDeltas(
  uint32_t width,
  uint32_t height,
  std::span<const Edge> edges) {

  if (!width || !height)
    throw std::runtime_error("analytic coverage dimensions must be non-zero");

  CoverageDeltas result;
  result.values.resize(size_t(width) * height);
  result.scale = kAnalyticCoverageScale;
  result.horizontalUnits = uint32_t(kAnalyticSubpixelScale);
  std::vector<int64_t> cells(result.values.size());

  const auto addCell = [&](uint32_t y, int64_t x, int64_t value) {
    if (!value || x >= int64_t(width))
      return;
    const uint32_t visibleX = x < 0 ? 0u : uint32_t(x);
    cells[size_t(y) * width + visibleX] += value;
  };

  for (const Edge& edge : edges)
    rasterizeAnalyticEdge(edge, 0, height, addCell);

  for (size_t i = 0; i < cells.size(); ++i) {
    if (cells[i] < std::numeric_limits<int32_t>::min() ||
        cells[i] > std::numeric_limits<int32_t>::max())
      throw std::runtime_error("analytic coverage cell exceeds int32 range");
    result.values[i] = int32_t(cells[i]);
  }
  return result;
}

AnalyticTileCells buildTiledAnalyticCells(
  uint32_t width,
  uint32_t height,
  std::span<const Edge> edges,
  uint32_t tileWidth,
  uint32_t tileHeight) {

  if (!width || !height || !tileWidth || !tileHeight)
    throw std::runtime_error("analytic tile dimensions must be non-zero");

  AnalyticTileCells result;
  result.width = width;
  result.height = height;
  result.tileWidth = tileWidth;
  result.tileHeight = tileHeight;
  result.tilesX = (width + tileWidth - 1u) / tileWidth;
  result.tilesY = (height + tileHeight - 1u) / tileHeight;
  result.scale = kAnalyticCoverageScale;

  std::vector<std::vector<uint32_t>> edgeBins(result.tilesY);
  if (edges.size() > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("too many edges for analytic tile bins");
  const int64_t imageY1 = int64_t(height) * kAnalyticSubpixelScale;
  const int64_t bandHeight = int64_t(tileHeight) * kAnalyticSubpixelScale;
  for (uint32_t edgeIndex = 0; edgeIndex < uint32_t(edges.size()); ++edgeIndex) {
    int64_t y0 = std::llround(
      double(edges[edgeIndex].y0) * kAnalyticSubpixelScale);
    int64_t y1 = std::llround(
      double(edges[edgeIndex].y1) * kAnalyticSubpixelScale);
    if (y0 > y1)
      std::swap(y0, y1);
    y0 = std::max<int64_t>(y0, 0);
    y1 = std::min<int64_t>(y1, imageY1);
    if (y0 >= y1)
      continue;
    const uint32_t firstBand = uint32_t(y0 / bandHeight);
    const uint32_t lastBand = uint32_t((y1 - 1) / bandHeight);
    for (uint32_t band = firstBand; band <= lastBand; ++band) {
      edgeBins[band].push_back(edgeIndex);
      ++result.binnedEdgeReferences;
    }
  }

  constexpr uint32_t kNoTile = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> tileLookup(
    size_t(result.tilesX) * result.tilesY, kNoTile);
  std::vector<int64_t> tileValues;
  const size_t cellsPerTile = size_t(tileWidth) * tileHeight;
  const auto addCell = [&](uint32_t y, int64_t x, int64_t value) {
    if (!value || x >= int64_t(width))
      return;
    const uint32_t visibleX = x < 0 ? 0u : uint32_t(x);
    const uint32_t tileX = visibleX / tileWidth;
    const uint32_t tileY = y / tileHeight;
    const uint32_t tileId = tileY * result.tilesX + tileX;
    uint32_t compactIndex = tileLookup[tileId];
    if (compactIndex == kNoTile) {
      compactIndex = uint32_t(result.tileIds.size());
      tileLookup[tileId] = compactIndex;
      result.tileIds.push_back(tileId);
      tileValues.resize(tileValues.size() + cellsPerTile, 0);
    }
    const uint32_t localX = visibleX - tileX * tileWidth;
    const uint32_t localY = y - tileY * tileHeight;
    tileValues[size_t(compactIndex) * cellsPerTile +
               size_t(localY) * tileWidth + localX] += value;
  };

  for (uint32_t band = 0; band < result.tilesY; ++band) {
    const uint32_t yBegin = band * tileHeight;
    const uint32_t yEnd = std::min(height, yBegin + tileHeight);
    for (uint32_t edgeIndex : edgeBins[band])
      rasterizeAnalyticEdge(edges[edgeIndex], yBegin, yEnd, addCell);
  }

  result.values.resize(tileValues.size());
  for (size_t i = 0; i < tileValues.size(); ++i) {
    if (tileValues[i] < std::numeric_limits<int32_t>::min() ||
        tileValues[i] > std::numeric_limits<int32_t>::max())
      throw std::runtime_error("analytic tile cell exceeds int32 range");
    result.values[i] = int32_t(tileValues[i]);
  }
  return result;
}

CoverageDeltas materializeAnalyticTiles(const AnalyticTileCells& tiles) {
  if (!tiles.width || !tiles.height || !tiles.tileWidth || !tiles.tileHeight ||
      tiles.tilesX != (tiles.width + tiles.tileWidth - 1u) / tiles.tileWidth ||
      tiles.tilesY != (tiles.height + tiles.tileHeight - 1u) / tiles.tileHeight)
    throw std::runtime_error("invalid analytic tile metadata");
  const size_t cellsPerTile = size_t(tiles.tileWidth) * tiles.tileHeight;
  if (tiles.values.size() != tiles.tileIds.size() * cellsPerTile)
    throw std::runtime_error("invalid analytic tile storage size");

  CoverageDeltas result;
  result.values.resize(size_t(tiles.width) * tiles.height);
  result.scale = tiles.scale;
  result.horizontalUnits = uint32_t(kAnalyticSubpixelScale);
  for (size_t compactIndex = 0; compactIndex < tiles.tileIds.size(); ++compactIndex) {
    const uint32_t tileId = tiles.tileIds[compactIndex];
    if (tileId >= tiles.tilesX * tiles.tilesY)
      throw std::runtime_error("invalid analytic tile id");
    const uint32_t tileX = tileId % tiles.tilesX;
    const uint32_t tileY = tileId / tiles.tilesX;
    const uint32_t x0 = tileX * tiles.tileWidth;
    const uint32_t y0 = tileY * tiles.tileHeight;
    const uint32_t copyWidth = std::min(tiles.tileWidth, tiles.width - x0);
    const uint32_t copyHeight = std::min(tiles.tileHeight, tiles.height - y0);
    for (uint32_t localY = 0; localY < copyHeight; ++localY) {
      const int32_t* source = tiles.values.data() +
        compactIndex * cellsPerTile + size_t(localY) * tiles.tileWidth;
      int32_t* destination = result.values.data() +
        size_t(y0 + localY) * tiles.width + x0;
      std::copy_n(source, copyWidth, destination);
    }
  }
  return result;
}

uint32_t resolveCoverageValue(
  int32_t accumulated,
  uint32_t coverageScale,
  CoverageResolveMode mode) noexcept {

  if (mode == CoverageResolveMode::kDirect)
    return uint32_t(std::clamp<int64_t>(accumulated, 0, coverageScale));

  uint64_t magnitude = accumulated < 0
    ? uint64_t(-int64_t(accumulated)) : uint64_t(accumulated);
  if (mode == CoverageResolveMode::kEvenOdd) {
    const uint64_t period = uint64_t(coverageScale) * 2u;
    magnitude %= period;
    if (magnitude > coverageScale)
      magnitude = period - magnitude;
  }
  return uint32_t(std::min<uint64_t>(magnitude, coverageScale));
}

void validateAnalyticCoverageReference() {
  constexpr uint32_t width = 6;
  constexpr uint32_t height = 5;
  const auto verticalEdge = [](float x, float y0, float y1) {
    return Edge{x, y0, y1, 0.0f};
  };
  const auto requireCoverage = [&](std::span<const Edge> edges,
                                   CoverageResolveMode mode,
                                   auto&& expected) {
    const CoverageDeltas cells = buildAnalyticCoverageDeltas(
      width, height, edges);
    const AnalyticTileCells tiles = buildTiledAnalyticCells(
      width, height, edges, 4, 2);
    const CoverageDeltas tiledCells = materializeAnalyticTiles(tiles);
    if (tiledCells.scale != cells.scale || tiledCells.values != cells.values)
      throw std::runtime_error(
        "tiled analytic cells differ from dense reference");
    std::vector<uint32_t> coverage(size_t(width) * height);
    scanCoverageScalar(
      coverage, {}, cells.values, width, height, cells.scale, mode);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const uint32_t wanted = expected(x, y, cells.scale);
        if (coverage[size_t(y) * width + x] != wanted)
          throw std::runtime_error(
            "analytic coverage invariant failed at " +
            std::to_string(x) + "," + std::to_string(y));
      }
    }
  };

  const std::array<Edge, 2> rectangle = {
    verticalEdge(1.0f, 3.0f, 1.0f),
    verticalEdge(3.0f, 1.0f, 3.0f)
  };
  requireCoverage(rectangle, CoverageResolveMode::kNonZero,
    [](uint32_t x, uint32_t y, uint32_t scale) {
      return y >= 1u && y < 3u && x >= 1u && x < 3u ? scale : 0u;
    });

  const std::array<Edge, 2> fractionalRectangle = {
    verticalEdge(1.5f, 3.0f, 1.0f),
    verticalEdge(3.5f, 1.0f, 3.0f)
  };
  requireCoverage(fractionalRectangle, CoverageResolveMode::kNonZero,
    [](uint32_t x, uint32_t y, uint32_t scale) {
      if (y < 1u || y >= 3u)
        return 0u;
      if (x == 1u || x == 3u)
        return scale / 2u;
      return x == 2u ? scale : 0u;
    });

  const std::array<Edge, 2> clippedRectangle = {
    verticalEdge(-1.0f, 3.0f, 1.0f),
    verticalEdge(2.0f, 1.0f, 3.0f)
  };
  requireCoverage(clippedRectangle, CoverageResolveMode::kNonZero,
    [](uint32_t x, uint32_t y, uint32_t scale) {
      return y >= 1u && y < 3u && x < 2u ? scale : 0u;
    });

  std::array<Edge, 4> doubledRectangle = {
    rectangle[0], rectangle[1], rectangle[0], rectangle[1]
  };
  requireCoverage(doubledRectangle, CoverageResolveMode::kEvenOdd,
    [](uint32_t, uint32_t, uint32_t) { return 0u; });
  requireCoverage(doubledRectangle, CoverageResolveMode::kNonZero,
    [](uint32_t x, uint32_t y, uint32_t scale) {
      return y >= 1u && y < 3u && x >= 1u && x < 3u ? scale : 0u;
    });

  const std::array<Edge, 2> triangle = {
    verticalEdge(1.0f, 4.0f, 1.0f),
    Edge{5.0f, 1.0f, 4.0f, -4.0f / 3.0f}
  };
  const CoverageDeltas triangleCells = buildAnalyticCoverageDeltas(
    width, height, triangle);
  const CoverageDeltas tiledTriangleCells = materializeAnalyticTiles(
    buildTiledAnalyticCells(width, height, triangle, 4, 2));
  if (tiledTriangleCells.values != triangleCells.values)
    throw std::runtime_error(
      "tiled analytic triangle differs from dense reference");
  std::vector<uint32_t> triangleCoverage(size_t(width) * height);
  scanCoverageScalar(
    triangleCoverage, {}, triangleCells.values, width, height,
    triangleCells.scale, CoverageResolveMode::kNonZero);
  uint64_t integratedCoverage = 0;
  for (uint32_t value : triangleCoverage)
    integratedCoverage += value;
  if (integratedCoverage != uint64_t(6) * triangleCells.scale)
    throw std::runtime_error("analytic triangle area invariant failed");
}

void scanCoverageScalar(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale,
  CoverageResolveMode mode) {

  validateBuffers(coverage, pixels, deltas, width, height);
  for (uint32_t y = 0; y < height; ++y) {
    int32_t running = 0;
    for (uint32_t x = 0; x < width; ++x) {
      const size_t index = size_t(y) * width + x;
      running += deltas[index];
      const uint32_t resolved = resolveCoverageValue(
        running, coverageScale, mode);
      coverage[index] = resolved;
      if (!pixels.empty())
        pixels[index] = shadePixel(
          x, y, width, height, resolved, coverageScale);
    }
  }
}

void scanCoverageAvx2(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale,
  CoverageResolveMode mode) {

  validateBuffers(coverage, pixels, deltas, width, height);
  scanCoverageAvx2Rows(
    coverage, pixels, deltas, width, height, coverageScale, mode, 0, height);
}

void scanCoverageAvx2Threaded(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale,
  uint32_t threadCount,
  CoverageResolveMode mode) {

  validateBuffers(coverage, pixels, deltas, width, height);
  threadCount = std::max(1u, std::min(threadCount, height));
  if (threadCount == 1u) {
    scanCoverageAvx2Rows(
      coverage, pixels, deltas, width, height, coverageScale, mode, 0, height);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(threadCount);
  for (uint32_t threadId = 0; threadId < threadCount; ++threadId) {
    const uint32_t yBegin = uint32_t(
      uint64_t(height) * threadId / threadCount);
    const uint32_t yEnd = uint32_t(
      uint64_t(height) * (threadId + 1u) / threadCount);
    workers.emplace_back([=] {
      scanCoverageAvx2Rows(
        coverage, pixels, deltas, width, height, coverageScale, mode, yBegin, yEnd);
    });
  }
  for (std::thread& worker : workers)
    worker.join();
}

} // namespace gpusimd
