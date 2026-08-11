#include "coverage_scan.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
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
  uint32_t yBegin,
  uint32_t yEnd) {

  alignas(32) uint32_t lanes[8];
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
      _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), values);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(coverage.data() + index), values);
      paintLanes(pixels, index, x, y, width, height, lanes, 8, coverageScale);
      carry = int32_t(lanes[7]);
    }
    for (; x < width; ++x) {
      const size_t index = size_t(y) * width + x;
      carry += deltas[index];
      coverage[index] = uint32_t(carry);
      if (!pixels.empty())
        pixels[index] = shadePixel(x, y, width, height, uint32_t(carry), coverageScale);
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

void scanCoverageScalar(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale) {

  validateBuffers(coverage, pixels, deltas, width, height);
  for (uint32_t y = 0; y < height; ++y) {
    int32_t running = 0;
    for (uint32_t x = 0; x < width; ++x) {
      const size_t index = size_t(y) * width + x;
      running += deltas[index];
      coverage[index] = uint32_t(running);
      if (!pixels.empty())
        pixels[index] = shadePixel(x, y, width, height, uint32_t(running), coverageScale);
    }
  }
}

void scanCoverageAvx2(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale) {

  validateBuffers(coverage, pixels, deltas, width, height);
  scanCoverageAvx2Rows(
    coverage, pixels, deltas, width, height, coverageScale, 0, height);
}

void scanCoverageAvx2Threaded(
  std::span<uint32_t> coverage,
  std::span<uint32_t> pixels,
  std::span<const int32_t> deltas,
  uint32_t width,
  uint32_t height,
  uint32_t coverageScale,
  uint32_t threadCount) {

  validateBuffers(coverage, pixels, deltas, width, height);
  threadCount = std::max(1u, std::min(threadCount, height));
  if (threadCount == 1u) {
    scanCoverageAvx2Rows(
      coverage, pixels, deltas, width, height, coverageScale, 0, height);
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
        coverage, pixels, deltas, width, height, coverageScale, yBegin, yEnd);
    });
  }
  for (std::thread& worker : workers)
    worker.join();
}

} // namespace gpusimd
