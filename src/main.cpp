#define GL_GLEXT_PROTOTYPES 1

#include "coverage_scan.h"
#include "lane_program.h"

#if defined(GPUSIMD_HAS_VULKAN)
#include "vulkan_renderer.h"
#endif

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#include <unistd.h>
#endif

#ifndef GPUSIMD_BUILD_TYPE
#define GPUSIMD_BUILD_TYPE "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;
using gpusimd::Edge;

struct Options {
  uint32_t width = 512;
  uint32_t height = 512;
  uint32_t aaGrid = 4;
  uint32_t warmup = 1;
  uint32_t iterations = 5;
  uint32_t threads = std::max(1u, std::thread::hardware_concurrency());
  uint32_t curveSegments = 0;
  float flatness = 0.25f;
  uint32_t shapeCount = 1;
  uint32_t scanVerticalSamples = 64;
  std::filesystem::path outputDir = "output";
  std::filesystem::path csvPath;
  std::vector<uint32_t> sweepSizes;
  std::vector<uint32_t> sweepThreads;
  std::vector<uint32_t> sweepCurveSegments;
  std::vector<uint32_t> sweepShapeCounts;
  bool gpu = true;
  bool openGl = true;
  bool vulkan = false;
  bool writeImages = true;
  bool requireHardwareGpu = false;
  bool coverageScan = false;
};

struct Configuration {
  uint32_t width;
  uint32_t height;
  uint32_t aaGrid;
  uint32_t threads;
  uint32_t curveSegments;
  float flatness;
  uint32_t shapeCount;
};

struct Point {
  float x;
  float y;
};

struct Cubic {
  Point p0;
  Point p1;
  Point p2;
  Point p3;
};

struct Comparison {
  uint64_t differingPixels = 0;
  uint32_t maxChannelError = 0;
  double meanAbsoluteChannelError = 0.0;
};

struct Timing {
  double medianMs = 0.0;
  double minimumMs = 0.0;
  double p90Ms = 0.0;
};

struct HostInfo {
  std::string timestampUtc;
  std::string hostname;
  std::string os;
  std::string cpu;
  uint32_t logicalThreads = 0;
  std::string compiler;
  std::string buildType = GPUSIMD_BUILD_TYPE;
};

struct GpuInfo {
  std::string vendor;
  std::string renderer;
  std::string version;
  bool hardware = false;
  uint32_t subgroupSize = 0;
  uint32_t workgroupSize = 0;
  std::string subgroupOperations;
  uint32_t timestampValidBits = 0;
};

struct ResultRow {
  Configuration configuration{};
  uint32_t edgeCount = 0;
  uint32_t executionThreads = 0;
  std::string backend;
  std::string api;
  std::string timingScope;
  Timing timing;
  double speedupVsScalar = 0.0;
  Comparison comparison;
  GpuInfo gpu;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

uint32_t parseU32(std::string_view value, const char* option) {
  const std::string ownedValue(value);
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(ownedValue.c_str(), &end, 10);
  if (!end || *end != '\0' || parsed == 0 || parsed > std::numeric_limits<uint32_t>::max())
    fail(std::string("invalid value for ") + option + ": " + std::string(value));
  return uint32_t(parsed);
}

float parsePositiveFloat(std::string_view value, const char* option) {
  const std::string ownedValue(value);
  char* end = nullptr;
  const float parsed = std::strtof(ownedValue.c_str(), &end);
  if (!end || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0f)
    fail(std::string("invalid value for ") + option + ": " + std::string(value));
  return parsed;
}

std::vector<uint32_t> parseU32List(std::string_view value, const char* option) {
  std::vector<uint32_t> values;
  size_t begin = 0;
  while (begin <= value.size()) {
    const size_t comma = value.find(',', begin);
    const size_t end = comma == std::string_view::npos ? value.size() : comma;
    if (end == begin)
      fail(std::string("empty item in ") + option);
    values.push_back(parseU32(value.substr(begin, end - begin), option));
    if (comma == std::string_view::npos)
      break;
    begin = comma + 1u;
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto takeValue = [&](const char* name) -> std::string_view {
      if (++i >= argc)
        fail(std::string("missing value for ") + name);
      return argv[i];
    };

    if (arg == "--width") options.width = parseU32(takeValue("--width"), "--width");
    else if (arg == "--height") options.height = parseU32(takeValue("--height"), "--height");
    else if (arg == "--aa") options.aaGrid = parseU32(takeValue("--aa"), "--aa");
    else if (arg == "--warmup") options.warmup = parseU32(takeValue("--warmup"), "--warmup");
    else if (arg == "--iterations") options.iterations = parseU32(takeValue("--iterations"), "--iterations");
    else if (arg == "--threads") options.threads = parseU32(takeValue("--threads"), "--threads");
    else if (arg == "--curve-segments") options.curveSegments = parseU32(takeValue("--curve-segments"), "--curve-segments");
    else if (arg == "--flatness") options.flatness = parsePositiveFloat(takeValue("--flatness"), "--flatness");
    else if (arg == "--shapes") options.shapeCount = parseU32(takeValue("--shapes"), "--shapes");
    else if (arg == "--scan-y-samples") options.scanVerticalSamples = parseU32(takeValue("--scan-y-samples"), "--scan-y-samples");
    else if (arg == "--sweep-sizes") options.sweepSizes = parseU32List(takeValue("--sweep-sizes"), "--sweep-sizes");
    else if (arg == "--sweep-threads") options.sweepThreads = parseU32List(takeValue("--sweep-threads"), "--sweep-threads");
    else if (arg == "--sweep-curve-segments") options.sweepCurveSegments = parseU32List(takeValue("--sweep-curve-segments"), "--sweep-curve-segments");
    else if (arg == "--sweep-shapes") options.sweepShapeCounts = parseU32List(takeValue("--sweep-shapes"), "--sweep-shapes");
    else if (arg == "--output-dir") options.outputDir = takeValue("--output-dir");
    else if (arg == "--csv") options.csvPath = takeValue("--csv");
    else if (arg == "--no-gpu") options.gpu = false;
    else if (arg == "--vulkan") options.vulkan = true;
    else if (arg == "--no-opengl") options.openGl = false;
    else if (arg == "--no-images") options.writeImages = false;
    else if (arg == "--require-hardware-gpu") options.requireHardwareGpu = true;
    else if (arg == "--coverage-scan") options.coverageScan = true;
    else if (arg == "--help") {
      std::cout
        << "Usage: gpusimd_vector_bench [options]\n"
        << "  --width N          image width (default 512)\n"
        << "  --height N         image height (default 512)\n"
        << "  --aa N             NxN supersampling grid (default 4)\n"
        << "  --warmup N         untimed warm-up iterations (default 1)\n"
        << "  --iterations N     measured iterations (default 5)\n"
        << "  --threads N        CPU worker count (default hardware concurrency)\n"
        << "  --flatness F       adaptive curve error in pixels (default 0.25)\n"
        << "  --curve-segments N use fixed subdivision; about 6N edges/shape\n"
        << "  --shapes N         independent heart shapes in the scene (default 1)\n"
        << "  --coverage-scan    benchmark fixed-point coverage prefix scans\n"
        << "  --scan-y-samples N vertical samples for coverage-delta input (default 64)\n"
        << "  --sweep-sizes LIST comma-separated square image sizes\n"
        << "  --sweep-threads LIST comma-separated CPU worker counts\n"
        << "  --sweep-curve-segments LIST  comma-separated curve complexities\n"
        << "  --sweep-shapes LIST comma-separated shape counts\n"
        << "  --csv PATH         write self-describing benchmark rows\n"
        << "  --output-dir PATH  PPM output directory (default output)\n"
        << "  --no-images        do not write PPM images\n"
        << "  --no-gpu           run only CPU implementations\n"
        << "  --vulkan          also run the optional Vulkan compute backend\n"
        << "  --no-opengl       skip OpenGL (use with --vulkan for Vulkan-only)\n"
        << "  --require-hardware-gpu  fail if a selected GPU backend is software\n";
      std::exit(0);
    }
    else {
      fail("unknown option: " + std::string(arg));
    }
  }

  if (options.aaGrid > 8)
    fail("--aa must be between 1 and 8");
  if (options.flatness < 0.001f)
    fail("--flatness must be at least 0.001 pixel");
  return options;
}

Point cubicPoint(const Cubic& c, float t) {
  const float u = 1.0f - t;
  const float a = u * u * u;
  const float b = 3.0f * u * u * t;
  const float d = 3.0f * u * t * t;
  const float e = t * t * t;
  return Point{
    a * c.p0.x + b * c.p1.x + d * c.p2.x + e * c.p3.x,
    a * c.p0.y + b * c.p1.y + d * c.p2.y + e * c.p3.y
  };
}

void addLineEdge(std::vector<Edge>& edges, Point a, Point b) {
  if (std::abs(a.y - b.y) < 1e-7f)
    return;
  edges.push_back(Edge{a.x, a.y, b.y, (b.x - a.x) / (b.y - a.y)});
}

void flattenCubicFixed(std::vector<Edge>& edges, const Cubic& cubic, uint32_t segments) {
  Point previous = cubic.p0;
  for (uint32_t i = 1; i <= segments; ++i) {
    const Point current = cubicPoint(cubic, float(i) / float(segments));
    addLineEdge(edges, previous, current);
    previous = current;
  }
}

Point midpoint(Point a, Point b) {
  return Point{(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f};
}

float distance(Point a, Point b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}

float distanceToLine(Point point, Point lineStart, Point lineEnd) {
  const float dx = lineEnd.x - lineStart.x;
  const float dy = lineEnd.y - lineStart.y;
  const float chordLength = std::hypot(dx, dy);
  if (chordLength < 1e-7f)
    return distance(point, lineStart);
  return std::abs((point.x - lineStart.x) * dy - (point.y - lineStart.y) * dx) /
         chordLength;
}

bool cubicIsFlatEnough(const Cubic& cubic, float flatness) {
  const float controlDistance = std::max(
    distanceToLine(cubic.p1, cubic.p0, cubic.p3),
    distanceToLine(cubic.p2, cubic.p0, cubic.p3));
  const float chordLength = distance(cubic.p0, cubic.p3);
  const float controlPolygonLength =
    distance(cubic.p0, cubic.p1) + distance(cubic.p1, cubic.p2) + distance(cubic.p2, cubic.p3);
  return controlDistance <= flatness && controlPolygonLength - chordLength <= flatness * 2.0f;
}

void flattenCubicAdaptive(
  std::vector<Edge>& edges,
  const Cubic& cubic,
  float flatness,
  uint32_t depth = 0) {

  constexpr uint32_t maxDepth = 24;
  if (depth == maxDepth || cubicIsFlatEnough(cubic, flatness)) {
    addLineEdge(edges, cubic.p0, cubic.p3);
    return;
  }

  // de Casteljau subdivision at t=0.5 preserves the exact cubic geometry.
  const Point p01 = midpoint(cubic.p0, cubic.p1);
  const Point p12 = midpoint(cubic.p1, cubic.p2);
  const Point p23 = midpoint(cubic.p2, cubic.p3);
  const Point p012 = midpoint(p01, p12);
  const Point p123 = midpoint(p12, p23);
  const Point center = midpoint(p012, p123);
  flattenCubicAdaptive(edges, Cubic{cubic.p0, p01, p012, center}, flatness, depth + 1u);
  flattenCubicAdaptive(edges, Cubic{center, p123, p23, cubic.p3}, flatness, depth + 1u);
}

uint32_t adaptiveCircleSegments(float radius, float flatness) {
  if (flatness >= radius)
    return 8u;
  const double cosine = std::clamp(1.0 - double(flatness / radius), -1.0, 1.0);
  const double halfAngle = std::acos(cosine);
  if (halfAngle <= 0.0)
    fail("adaptive circle tolerance is too small");
  const double segmentCount = std::ceil(double(M_PI) / halfAngle);
  if (segmentCount > double(std::numeric_limits<uint32_t>::max()))
    fail("adaptive circle requires too many segments");
  return std::max(8u, uint32_t(segmentCount));
}

void addHeart(
  std::vector<Edge>& edges,
  float cx,
  float cy,
  float scale,
  uint32_t curveSegments,
  float flatness) {

  const auto map = [&](Point p) -> Point {
    return Point{cx + p.x * scale, cy + p.y * scale};
  };

  // Four cubic curves form a heart. They are flattened here, as a normal vector
  // renderer would do before sending edges to its rasterizer.
  const std::array<Cubic, 4> heart = {{
    {map({ 0.00f, -0.32f}), map({-0.52f, -0.94f}), map({-1.22f, -0.32f}), map({-0.94f,  0.32f})},
    {map({-0.94f,  0.32f}), map({-0.72f,  0.78f}), map({-0.28f,  1.06f}), map({ 0.00f,  1.28f})},
    {map({ 0.00f,  1.28f}), map({ 0.28f,  1.06f}), map({ 0.72f,  0.78f}), map({ 0.94f,  0.32f})},
    {map({ 0.94f,  0.32f}), map({ 1.22f, -0.32f}), map({ 0.52f, -0.94f}), map({ 0.00f, -0.32f})}
  }};

  for (const Cubic& cubic : heart) {
    if (curveSegments)
      flattenCubicFixed(edges, cubic, curveSegments);
    else
      flattenCubicAdaptive(edges, cubic, flatness);
  }

  // A reversed circular subpath makes a hole under the even-odd fill rule.
  const uint32_t holeSegments = curveSegments
    ? std::max(8u, curveSegments * 2u)
    : adaptiveCircleSegments(scale * 0.20f, flatness);
  Point previous{};
  for (uint32_t i = 0; i <= holeSegments; ++i) {
    const float angle = -2.0f * float(M_PI) * float(i) / float(holeSegments);
    const Point p{cx + std::cos(angle) * scale * 0.20f,
                  cy + std::sin(angle) * scale * 0.20f + scale * 0.18f};
    if (i != 0)
      addLineEdge(edges, previous, p);
    previous = p;
  }
}

std::vector<Edge> makeScene(
  uint32_t width,
  uint32_t height,
  uint32_t curveSegments,
  float flatness,
  uint32_t shapeCount) {

  const uint64_t estimatedEdgesPerShape = curveSegments
    ? uint64_t(curveSegments) * 4u + std::max(uint64_t(8), uint64_t(curveSegments) * 2u)
    : 128u;
  if (uint64_t(shapeCount) * estimatedEdgesPerShape > std::numeric_limits<uint32_t>::max())
    fail("requested scene has too many edges");

  std::vector<Edge> edges;
  edges.reserve(size_t(uint64_t(shapeCount) * estimatedEdgesPerShape));

  if (shapeCount == 1) {
    // Preserve the original baseline geometry exactly.
    const float scale = 0.36f * float(std::min(width, height));
    addHeart(edges, 0.50f * float(width), 0.45f * float(height), scale, curveSegments, flatness);
    return edges;
  }

  const uint32_t columns = uint32_t(std::ceil(std::sqrt(double(shapeCount))));
  const uint32_t rows = (shapeCount + columns - 1u) / columns;
  const float cellWidth = float(width) / float(columns);
  const float cellHeight = float(height) / float(rows);
  const float scale = 0.36f * std::min(cellWidth, cellHeight);
  for (uint32_t i = 0; i < shapeCount; ++i) {
    const uint32_t column = i % columns;
    const uint32_t row = i / columns;
    const float cx = (float(column) + 0.50f) * cellWidth;
    const float cy = (float(row) + 0.45f) * cellHeight;
    addHeart(edges, cx, cy, scale, curveSegments, flatness);
  }
  return edges;
}

template<class Function>
void parallelRows(uint32_t height, uint32_t threadCount, Function&& function) {
  threadCount = std::min(threadCount, height);
  if (threadCount <= 1) {
    function(0, height);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(threadCount);
  for (uint32_t threadId = 0; threadId < threadCount; ++threadId) {
    const uint32_t y0 = uint32_t((uint64_t(height) * threadId) / threadCount);
    const uint32_t y1 = uint32_t((uint64_t(height) * (threadId + 1u)) / threadCount);
    workers.emplace_back([&, y0, y1] { function(y0, y1); });
  }
  for (std::thread& worker : workers)
    worker.join();
}

void renderScalar(
  std::span<uint32_t> output,
  uint32_t width,
  uint32_t height,
  uint32_t aaGrid,
  std::span<const Edge> edges,
  uint32_t threadCount) {

  const uint32_t sampleCount = aaGrid * aaGrid;
  parallelRows(height, threadCount, [&](uint32_t y0, uint32_t y1) {
    for (uint32_t y = y0; y < y1; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const float hits = gpusimd::coverageProgram<gpusimd::ScalarOps>(float(x), float(y), edges, aaGrid);
        output[size_t(y) * width + x] = gpusimd::shadePixel(x, y, width, height, uint32_t(hits), sampleCount);
      }
    }
  });
}

void renderAvx2(
  std::span<uint32_t> output,
  uint32_t width,
  uint32_t height,
  uint32_t aaGrid,
  std::span<const Edge> edges,
  uint32_t threadCount) {

  const uint32_t sampleCount = aaGrid * aaGrid;
  parallelRows(height, threadCount, [&](uint32_t y0, uint32_t y1) {
    alignas(32) float laneHits[8];
    for (uint32_t y = y0; y < y1; ++y) {
      uint32_t x = 0;
      for (; x + 8u <= width; x += 8u) {
        const __m256 vx = _mm256_setr_ps(
          float(x + 0u), float(x + 1u), float(x + 2u), float(x + 3u),
          float(x + 4u), float(x + 5u), float(x + 6u), float(x + 7u));
        const __m256 vy = _mm256_set1_ps(float(y));
        const __m256 hits = gpusimd::coverageProgram<gpusimd::Avx2Ops>(vx, vy, edges, aaGrid);
        _mm256_store_ps(laneHits, hits);
        for (uint32_t lane = 0; lane < 8u; ++lane)
          output[size_t(y) * width + x + lane] = gpusimd::shadePixel(x + lane, y, width, height, uint32_t(laneHits[lane]), sampleCount);
      }
      for (; x < width; ++x) {
        const float hits = gpusimd::coverageProgram<gpusimd::ScalarOps>(float(x), float(y), edges, aaGrid);
        output[size_t(y) * width + x] = gpusimd::shadePixel(x, y, width, height, uint32_t(hits), sampleCount);
      }
    }
  });
}

Timing summarizeSamples(std::vector<double> samples) {
  if (samples.empty())
    fail("cannot summarize an empty timing sample set");
  std::sort(samples.begin(), samples.end());
  const size_t p90Index = size_t(std::ceil(0.90 * double(samples.size()))) - 1u;
  return Timing{samples[samples.size() / 2u], samples.front(), samples[p90Index]};
}

template<class Function>
Timing benchmark(uint32_t warmup, uint32_t iterations, Function&& function) {
  for (uint32_t i = 0; i < warmup; ++i)
    function();
  std::vector<double> samples;
  samples.reserve(iterations);
  for (uint32_t i = 0; i < iterations; ++i) {
    const auto start = Clock::now();
    function();
    const auto stop = Clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }
  return summarizeSamples(std::move(samples));
}

Comparison compareImages(std::span<const uint32_t> reference, std::span<const uint32_t> candidate) {
  if (reference.size() != candidate.size())
    fail("image sizes differ");

  Comparison result;
  uint64_t totalError = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    bool pixelDiffers = false;
    for (uint32_t shift : {0u, 8u, 16u, 24u}) {
      const uint32_t a = (reference[i] >> shift) & 0xFFu;
      const uint32_t b = (candidate[i] >> shift) & 0xFFu;
      const uint32_t error = a > b ? a - b : b - a;
      pixelDiffers |= error != 0;
      result.maxChannelError = std::max(result.maxChannelError, error);
      totalError += error;
    }
    result.differingPixels += pixelDiffers;
  }
  result.meanAbsoluteChannelError = double(totalError) / double(reference.size() * 4u);
  return result;
}

Comparison compareCoverage(
  std::span<const uint32_t> reference,
  std::span<const uint32_t> candidate) {

  if (reference.size() != candidate.size())
    fail("coverage buffer sizes differ");
  Comparison result;
  uint64_t totalError = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const uint32_t error = reference[i] > candidate[i]
      ? reference[i] - candidate[i] : candidate[i] - reference[i];
    result.differingPixels += error != 0;
    result.maxChannelError = std::max(result.maxChannelError, error);
    totalError += error;
  }
  result.meanAbsoluteChannelError = double(totalError) / double(reference.size());
  return result;
}

void writePpm(const std::filesystem::path& path, uint32_t width, uint32_t height, std::span<const uint32_t> pixels) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    fail("cannot create " + path.string());
  stream << "P6\n" << width << ' ' << height << "\n255\n";
  for (uint32_t rgba : pixels) {
    const std::array<char, 3> rgb = {
      char(rgba & 0xFFu), char((rgba >> 8u) & 0xFFu), char((rgba >> 16u) & 0xFFu)
    };
    stream.write(rgb.data(), std::streamsize(rgb.size()));
  }
}

std::string trim(std::string value) {
  const auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(), std::find_if_not(value.begin(), value.end(), isSpace));
  value.erase(std::find_if_not(value.rbegin(), value.rend(), isSpace).base(), value.end());
  if (value.size() >= 2u && value.front() == '"' && value.back() == '"')
    value = value.substr(1u, value.size() - 2u);
  return value;
}

std::string readKeyValue(const std::filesystem::path& path, std::string_view key, char delimiter) {
  std::ifstream stream(path);
  std::string line;
  while (std::getline(stream, line)) {
    const size_t split = line.find(delimiter);
    if (split != std::string::npos && trim(line.substr(0, split)) == key)
      return trim(line.substr(split + 1u));
  }
  return {};
}

std::string utcTimestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  std::ostringstream stream;
  stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return stream.str();
}

HostInfo queryHostInfo() {
  HostInfo info;
  info.timestampUtc = utcTimestamp();
  info.logicalThreads = std::max(1u, std::thread::hardware_concurrency());

#if defined(__unix__) || defined(__APPLE__)
  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size() - 1u) == 0)
    info.hostname = hostname.data();

  utsname systemInfo{};
  if (uname(&systemInfo) == 0)
    info.os = std::string(systemInfo.sysname) + " " + systemInfo.release + " " + systemInfo.machine;
#endif

#if defined(__linux__)
  const std::string prettyName = readKeyValue("/etc/os-release", "PRETTY_NAME", '=');
  if (!prettyName.empty())
    info.os = prettyName;
  info.cpu = readKeyValue("/proc/cpuinfo", "model name", ':');
#endif

  if (info.hostname.empty()) info.hostname = "unknown";
  if (info.os.empty()) info.os = "unknown";
  if (info.cpu.empty()) info.cpu = "unknown";

#if defined(__clang__)
  info.compiler = std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
  info.compiler = std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
  info.compiler = "MSVC " + std::to_string(_MSC_VER);
#else
  info.compiler = "unknown";
#endif
  return info;
}

std::vector<Configuration> makeConfigurations(const Options& options) {
  const std::vector<uint32_t> sizes = options.sweepSizes.empty()
    ? std::vector<uint32_t>{0u} : options.sweepSizes;
  const std::vector<uint32_t> threads = options.sweepThreads.empty()
    ? std::vector<uint32_t>{options.threads} : options.sweepThreads;
  const std::vector<uint32_t> curveSegments = options.sweepCurveSegments.empty()
    ? std::vector<uint32_t>{options.curveSegments} : options.sweepCurveSegments;
  const std::vector<uint32_t> shapeCounts = options.sweepShapeCounts.empty()
    ? std::vector<uint32_t>{options.shapeCount} : options.sweepShapeCounts;

  std::vector<Configuration> configurations;
  for (uint32_t size : sizes) {
    const uint32_t width = size ? size : options.width;
    const uint32_t height = size ? size : options.height;
    if (uint64_t(width) * height > uint64_t(std::numeric_limits<size_t>::max() / sizeof(uint32_t)))
      fail("image dimensions exceed addressable memory");
    for (uint32_t threadCount : threads) {
      for (uint32_t segments : curveSegments) {
        for (uint32_t shapes : shapeCounts)
          configurations.push_back(Configuration{
            width, height, options.aaGrid, threadCount, segments, options.flatness, shapes});
      }
    }
  }
  return configurations;
}

std::string configurationTag(const Configuration& config) {
  std::ostringstream stream;
  stream << config.width << 'x' << config.height
         << "_aa" << config.aaGrid
         << (config.curveSegments ? "_fixed" + std::to_string(config.curveSegments) : "_adaptive")
         << "_shapes" << config.shapeCount
         << "_t" << config.threads;
  return stream.str();
}

std::string csvEscape(std::string_view value) {
  if (value.find_first_of(",\"\r\n") == std::string_view::npos)
    return std::string(value);
  std::string escaped = "\"";
  for (char c : value) {
    escaped += c;
    if (c == '"') escaped += '"';
  }
  escaped += '"';
  return escaped;
}

void writeCsv(
  const std::filesystem::path& path,
  const HostInfo& host,
  uint32_t warmup,
  uint32_t iterations,
  std::span<const ResultRow> rows) {

  if (path.has_parent_path())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path);
  if (!stream)
    fail("cannot create " + path.string());

  stream
    << "schema_version,timestamp_utc,hostname,os,cpu,cpu_logical_threads,compiler,build_type,"
       "api,gpu_vendor,gpu_renderer,gpu_version,gpu_hardware,subgroup_size,subgroup_operations,"
       "workgroup_size,timestamp_valid_bits,"
       "width,height,aa_grid,flattening_mode,curve_segments,flatness_pixels,shape_count,edge_count,"
       "requested_cpu_threads,cpu_threads_used,warmup,iterations,"
       "backend,timing_scope,median_ms,minimum_ms,p90_ms,megapixels_per_second,speedup_vs_scalar,"
       "differing_pixels,max_channel_error,mean_absolute_channel_error\n";

  stream << std::fixed << std::setprecision(6);
  for (const ResultRow& row : rows) {
    const bool gpu = row.api != "cpu";
    const double megapixelsPerSecond =
      double(uint64_t(row.configuration.width) * row.configuration.height) /
      (row.timing.medianMs * 1000.0);
    stream
      << "3," << csvEscape(host.timestampUtc)
      << ',' << csvEscape(host.hostname)
      << ',' << csvEscape(host.os)
      << ',' << csvEscape(host.cpu)
      << ',' << host.logicalThreads
      << ',' << csvEscape(host.compiler)
      << ',' << csvEscape(host.buildType)
      << ',' << row.api
      << ',' << (gpu ? csvEscape(row.gpu.vendor) : "")
      << ',' << (gpu ? csvEscape(row.gpu.renderer) : "")
      << ',' << (gpu ? csvEscape(row.gpu.version) : "")
      << ',' << (gpu ? (row.gpu.hardware ? "1" : "0") : "")
      << ',' << (gpu && row.gpu.subgroupSize ? std::to_string(row.gpu.subgroupSize) : "")
      << ',' << (gpu ? csvEscape(row.gpu.subgroupOperations) : "")
      << ',' << (gpu ? std::to_string(row.gpu.workgroupSize) : "")
      << ',' << (gpu && row.gpu.timestampValidBits ? std::to_string(row.gpu.timestampValidBits) : "")
      << ',' << row.configuration.width
      << ',' << row.configuration.height
      << ',' << row.configuration.aaGrid
      << ',' << (row.configuration.curveSegments ? "fixed" : "adaptive")
      << ',' << (row.configuration.curveSegments ? std::to_string(row.configuration.curveSegments) : "")
      << ',' << (row.configuration.curveSegments ? "" : std::to_string(row.configuration.flatness))
      << ',' << row.configuration.shapeCount
      << ',' << row.edgeCount
      << ',' << row.configuration.threads
      << ',' << row.executionThreads
      << ',' << warmup
      << ',' << iterations
      << ',' << row.backend
      << ',' << row.timingScope
      << ',' << row.timing.medianMs
      << ',' << row.timing.minimumMs
      << ',' << row.timing.p90Ms
      << ',' << megapixelsPerSecond
      << ',' << row.speedupVsScalar
      << ',' << row.comparison.differingPixels
      << ',' << row.comparison.maxChannelError
      << ',' << row.comparison.meanAbsoluteChannelError
      << '\n';
  }
}

class EglContext {
public:
  EglContext() {
    auto getPlatformDisplay = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
      eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (!getPlatformDisplay)
      fail("EGL_EXT_platform_base is unavailable");

    display_ = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    if (display_ == EGL_NO_DISPLAY || !eglInitialize(display_, nullptr, nullptr))
      fail("cannot initialize a surfaceless EGL display");
    if (!eglBindAPI(EGL_OPENGL_API))
      fail("cannot bind the EGL OpenGL API");

    const EGLint configAttributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_NONE
    };
    EGLint configCount = 0;
    if (!eglChooseConfig(display_, configAttributes, &config_, 1, &configCount) || configCount == 0)
      fail("cannot choose an EGL OpenGL configuration");

    const EGLint contextAttributes[] = {
      EGL_CONTEXT_MAJOR_VERSION, 4,
      EGL_CONTEXT_MINOR_VERSION, 3,
      EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
      EGL_NONE
    };
    context_ = eglCreateContext(display_, config_, EGL_NO_CONTEXT, contextAttributes);
    if (context_ == EGL_NO_CONTEXT)
      fail("cannot create an OpenGL 4.3 context");

    const EGLint surfaceAttributes[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    surface_ = eglCreatePbufferSurface(display_, config_, surfaceAttributes);
    if (surface_ == EGL_NO_SURFACE)
      fail("cannot create an EGL pbuffer");
    if (!eglMakeCurrent(display_, surface_, surface_, context_))
      fail("cannot make the EGL context current");
  }

  ~EglContext() {
    if (display_ != EGL_NO_DISPLAY) {
      eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      if (surface_ != EGL_NO_SURFACE) eglDestroySurface(display_, surface_);
      if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
      eglTerminate(display_);
    }
  }

  EglContext(const EglContext&) = delete;
  EglContext& operator=(const EglContext&) = delete;

private:
  EGLDisplay display_ = EGL_NO_DISPLAY;
  EGLConfig config_ = nullptr;
  EGLContext context_ = EGL_NO_CONTEXT;
  EGLSurface surface_ = EGL_NO_SURFACE;
};

bool hasGlExtension(const char* requested) {
  GLint count = 0;
  glGetIntegerv(GL_NUM_EXTENSIONS, &count);
  for (GLint i = 0; i < count; ++i) {
    const char* extension = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, GLuint(i)));
    if (extension && std::string_view(extension) == requested)
      return true;
  }
  return false;
}

std::string shaderSource(uint32_t pixelsPerInvocation, uint32_t workgroupSize) {
  return std::string(R"GLSL(#version 430 core
#define PIXELS_PER_INVOCATION )GLSL") + std::to_string(pixelsPerInvocation) + R"GLSL(

layout(local_size_x = )GLSL" + std::to_string(workgroupSize) + R"GLSL() in;

layout(std430, binding = 0) readonly buffer EdgeBuffer {
  vec4 edgeData[];
};

layout(std430, binding = 1) writeonly buffer PixelBuffer {
  uint pixels[];
};

uniform uint uWidth;
uniform uint uHeight;
uniform uint uEdgeCount;
uniform uint uAaGrid;

uint shadePixel(uint x, uint y, uint hits) {
  uint sampleCount = uAaGrid * uAaGrid;
  uint alpha = (hits * 255u + sampleCount / 2u) / sampleCount;

  uint tile = ((x >> 5u) ^ (y >> 5u)) & 1u;
  uint bgR = tile != 0u ? 30u : 48u;
  uint bgG = tile != 0u ? 34u : 52u;
  uint bgB = tile != 0u ? 44u : 64u;

  uint srcR = 48u + (x * 190u) / (uWidth > 1u ? uWidth - 1u : 1u);
  uint srcG = 32u + (y * 172u) / (uHeight > 1u ? uHeight - 1u : 1u);
  uint srcB = 210u;
  uint invAlpha = 255u - alpha;

  uint outR = (srcR * alpha + bgR * invAlpha + 127u) / 255u;
  uint outG = (srcG * alpha + bgG * invAlpha + 127u) / 255u;
  uint outB = (srcB * alpha + bgB * invAlpha + 127u) / 255u;
  return outR | (outG << 8u) | (outB << 16u) | 0xFF000000u;
}

void renderPixel(uint pixelIndex) {
  uint pixelCount = uWidth * uHeight;
  if (pixelIndex >= pixelCount)
    return;

  uint x = pixelIndex % uWidth;
  uint y = pixelIndex / uWidth;
  uint hits = 0u;

  for (uint sampleY = 0u; sampleY < uAaGrid; ++sampleY) {
    float sy = float(y) + (float(sampleY) + 0.5) / float(uAaGrid);
    for (uint sampleX = 0u; sampleX < uAaGrid; ++sampleX) {
      float sx = float(x) + (float(sampleX) + 0.5) / float(uAaGrid);
      bool inside = false;

      for (uint edgeIndex = 0u; edgeIndex < uEdgeCount; ++edgeIndex) {
        vec4 edge = edgeData[edgeIndex];
        bool straddles = (edge.y > sy) != (edge.z > sy);
        float xIntersection = edge.x + (sy - edge.y) * edge.w;
        inside = inside != (straddles && sx < xIntersection);
      }
      hits += inside ? 1u : 0u;
    }
  }
  pixels[pixelIndex] = shadePixel(x, y, hits);
}

void main() {
  uint firstPixel = gl_GlobalInvocationID.x * uint(PIXELS_PER_INVOCATION);
  for (uint i = 0u; i < uint(PIXELS_PER_INVOCATION); ++i)
    renderPixel(firstPixel + i);
}
)GLSL";
}

GLuint compileComputeProgram(uint32_t pixelsPerInvocation, uint32_t workgroupSize) {
  const std::string source = shaderSource(pixelsPerInvocation, workgroupSize);
  const char* sourcePtr = source.c_str();
  const GLint sourceLength = GLint(source.size());

  const GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(shader, 1, &sourcePtr, &sourceLength);
  glCompileShader(shader);

  GLint status = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE) {
    GLint logLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    std::string log(size_t(std::max(logLength, 1)), '\0');
    glGetShaderInfoLog(shader, logLength, nullptr, log.data());
    glDeleteShader(shader);
    fail("compute shader compilation failed:\n" + log);
  }

  const GLuint program = glCreateProgram();
  glAttachShader(program, shader);
  glLinkProgram(program);
  glDeleteShader(shader);
  glGetProgramiv(program, GL_LINK_STATUS, &status);
  if (status != GL_TRUE) {
    GLint logLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::string log(size_t(std::max(logLength, 1)), '\0');
    glGetProgramInfoLog(program, logLength, nullptr, log.data());
    glDeleteProgram(program);
    fail("compute shader link failed:\n" + log);
  }
  return program;
}

class GpuRenderer {
public:
  struct Result {
    std::vector<uint32_t> pixels;
    Timing kernel;
    Timing withReadback;
  };

  GpuRenderer(uint32_t width, uint32_t height, uint32_t aaGrid, std::span<const Edge> edges)
    : width_(width), height_(height), aaGrid_(aaGrid), edgeCount_(uint32_t(edges.size())) {

    const char* vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    if (!vendor || !renderer || !version)
      fail("OpenGL context did not expose renderer information");
    vendor_ = vendor;
    renderer_ = renderer;
    version_ = version;

    std::string rendererLower = renderer_;
    std::transform(rendererLower.begin(), rendererLower.end(), rendererLower.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    hardware_ = rendererLower.find("llvmpipe") == std::string::npos
             && rendererLower.find("softpipe") == std::string::npos
             && rendererLower.find("software rasterizer") == std::string::npos
             && rendererLower.find("swiftshader") == std::string::npos
             && rendererLower.find("lavapipe") == std::string::npos;

    if (hasGlExtension("GL_KHR_shader_subgroup")) {
      GLint subgroupSize = 0;
      glGetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroupSize);
      if (subgroupSize > 0) {
        subgroupSize_ = uint32_t(subgroupSize);
        workgroupSize_ = subgroupSize_;
      }
    }

    glGenBuffers(1, &edgeBuffer_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, edgeBuffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(edges.size_bytes()), edges.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &pixelBuffer_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pixelBuffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(size_t(width_) * height_ * sizeof(uint32_t)), nullptr, GL_DYNAMIC_READ);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  }

  ~GpuRenderer() {
    if (pixelBuffer_) glDeleteBuffers(1, &pixelBuffer_);
    if (edgeBuffer_) glDeleteBuffers(1, &edgeBuffer_);
  }

  GpuInfo info() const {
    return GpuInfo{
      vendor_, renderer_, version_, hardware_, subgroupSize_, workgroupSize_, "", 0};
  }

  Result run(uint32_t pixelsPerInvocation, uint32_t warmup, uint32_t iterations) {
    const GLuint program = compileComputeProgram(pixelsPerInvocation, workgroupSize_);
    glUseProgram(program);
    glUniform1ui(glGetUniformLocation(program, "uWidth"), width_);
    glUniform1ui(glGetUniformLocation(program, "uHeight"), height_);
    glUniform1ui(glGetUniformLocation(program, "uEdgeCount"), edgeCount_);
    glUniform1ui(glGetUniformLocation(program, "uAaGrid"), aaGrid_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, edgeBuffer_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pixelBuffer_);

    const uint64_t pixelCount = uint64_t(width_) * height_;
    const uint64_t invocationCount = (pixelCount + pixelsPerInvocation - 1u) / pixelsPerInvocation;
    const GLuint groupCount = GLuint((invocationCount + workgroupSize_ - 1u) / workgroupSize_);

    auto dispatch = [&] {
      glDispatchCompute(groupCount, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    };

    for (uint32_t i = 0; i < warmup; ++i) {
      dispatch();
      glFinish();
    }

    // A synchronized wall-clock measurement is intentional here. Some Mesa
    // drivers expose GL_TIME_ELAPSED for compute but return zero nanoseconds.
    // glFinish() makes this slightly conservative by including submission and
    // synchronization overhead, but it is portable and never reports phantom
    // GPU speedups.
    std::vector<double> kernelSamples;
    kernelSamples.reserve(iterations);
    for (uint32_t i = 0; i < iterations; ++i) {
      const auto start = Clock::now();
      dispatch();
      glFinish();
      const auto stop = Clock::now();
      kernelSamples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    Result result;
    result.pixels.resize(size_t(pixelCount));
    std::vector<double> readbackSamples;
    for (uint32_t i = 0; i < warmup; ++i) {
      dispatch();
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, pixelBuffer_);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                         GLsizeiptr(result.pixels.size() * sizeof(uint32_t)), result.pixels.data());
    }
    readbackSamples.reserve(iterations);
    for (uint32_t i = 0; i < iterations; ++i) {
      const auto start = Clock::now();
      dispatch();
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, pixelBuffer_);
      glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                         GLsizeiptr(result.pixels.size() * sizeof(uint32_t)), result.pixels.data());
      const auto stop = Clock::now();
      readbackSamples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    result.kernel = summarizeSamples(std::move(kernelSamples));
    result.withReadback = summarizeSamples(std::move(readbackSamples));

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);
    glDeleteProgram(program);
    return result;
  }

private:
  uint32_t width_;
  uint32_t height_;
  uint32_t aaGrid_;
  uint32_t edgeCount_;
  GLuint edgeBuffer_ = 0;
  GLuint pixelBuffer_ = 0;
  std::string vendor_;
  std::string renderer_;
  std::string version_;
  bool hardware_ = false;
  uint32_t subgroupSize_ = 0;
  uint32_t workgroupSize_ = 32;
};

void printTiming(const char* name, const Timing& timing, double baselineMs) {
  std::cout << std::left << std::setw(28) << name
            << std::right << std::fixed << std::setprecision(3)
            << std::setw(10) << timing.medianMs << " ms"
            << std::setw(10) << (baselineMs / timing.medianMs) << "x"
            << "  (best " << timing.minimumMs << " ms, p90 " << timing.p90Ms << " ms)\n";
}

void printComparison(const char* name, const Comparison& comparison, size_t pixelCount) {
  std::cout << "  correctness " << std::left << std::setw(22) << name
            << comparison.differingPixels << '/' << pixelCount << " pixels differ, max channel error "
            << comparison.maxChannelError << ", mean abs channel error "
            << std::fixed << std::setprecision(6) << comparison.meanAbsoluteChannelError << '\n';
}

void printCoverageComparison(
  const char* name,
  const Comparison& comparison,
  size_t pixelCount) {

  std::cout << "  coverage " << std::left << std::setw(25) << name
            << comparison.differingPixels << '/' << pixelCount
            << " values differ, max unit error " << comparison.maxChannelError
            << ", mean abs unit error " << std::fixed << std::setprecision(6)
            << comparison.meanAbsoluteChannelError << '\n';
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc, argv);
    const HostInfo host = queryHostInfo();
    const std::vector<Configuration> configurations = makeConfigurations(options);
    std::vector<ResultRow> rows;
    const size_t rowsPerConfiguration = 3u +
      (options.gpu && options.openGl ? 4u : 0u) +
      (options.gpu && options.vulkan ? 6u : 0u);
    rows.reserve(configurations.size() * rowsPerConfiguration);

#if !defined(GPUSIMD_HAS_VULKAN)
    if (options.gpu && options.vulkan)
      fail("--vulkan requested, but this build has no Vulkan loader/headers");
#endif

    std::unique_ptr<EglContext> context;
    if (options.gpu && options.openGl)
      context = std::make_unique<EglContext>();

    std::cout << "GPU-distributed SIMD vector rasterization experiment\n"
              << "Host: " << host.hostname << " | " << host.cpu << " | " << host.os << '\n'
              << "Compiler: " << host.compiler << " | build: " << host.buildType << '\n'
              << "Configurations: " << configurations.size()
              << " | warm-up: " << options.warmup
              << " | measured iterations: " << options.iterations << "\n";

    bool printedOpenGlInfo = false;
#if defined(GPUSIMD_HAS_VULKAN)
    bool printedVulkanInfo = false;
#endif
    for (size_t configurationIndex = 0; configurationIndex < configurations.size(); ++configurationIndex) {
      const Configuration config = configurations[configurationIndex];
      const std::vector<Edge> edges = makeScene(
        config.width, config.height, config.curveSegments, config.flatness, config.shapeCount);
      const size_t pixelCount = size_t(config.width) * config.height;
      std::vector<uint32_t> scalarPixels(pixelCount);
      std::vector<uint32_t> simdPixels(pixelCount);
      std::vector<uint32_t> threadedPixels(pixelCount);
      gpusimd::CoverageDeltas coverageDeltas;
      std::vector<uint32_t> scanCoverageReference;
      std::vector<uint32_t> scanPixelsReference;
      Timing scalarScanTiming;
      Timing scalarScanPaintTiming;

      std::cout << "\n[" << (configurationIndex + 1u) << '/' << configurations.size() << "] "
                << config.width << 'x' << config.height
                << ", AA " << config.aaGrid << 'x' << config.aaGrid
                << ", flattening "
                << (config.curveSegments
                      ? "fixed " + std::to_string(config.curveSegments) + " segments/cubic"
                      : "adaptive " + std::to_string(config.flatness) + " px")
                << ", " << edges.size() << " edges in " << config.shapeCount
                << " shape(s), " << config.threads << " requested CPU threads\n";

      const Timing scalarTiming = benchmark(options.warmup, options.iterations, [&] {
        renderScalar(scalarPixels, config.width, config.height, config.aaGrid, edges, 1);
      });
      const Timing simdTiming = benchmark(options.warmup, options.iterations, [&] {
        renderAvx2(simdPixels, config.width, config.height, config.aaGrid, edges, 1);
      });
      const Timing threadedTiming = benchmark(options.warmup, options.iterations, [&] {
        renderAvx2(threadedPixels, config.width, config.height, config.aaGrid, edges, config.threads);
      });

      std::cout << "Median render time and speedup over scalar CPU:\n";
      printTiming("CPU scalar (1 thread)", scalarTiming, scalarTiming.medianMs);
      printTiming("CPU AVX2 x8 (1 thread)", simdTiming, scalarTiming.medianMs);
      printTiming("CPU AVX2 x8 (threaded)", threadedTiming, scalarTiming.medianMs);
      const Comparison simdComparison = compareImages(scalarPixels, simdPixels);
      const Comparison threadedComparison = compareImages(scalarPixels, threadedPixels);
      printComparison("CPU AVX2", simdComparison, pixelCount);
      printComparison("CPU AVX2 threaded", threadedComparison, pixelCount);
      if (simdComparison.differingPixels || threadedComparison.differingPixels)
        fail("CPU SIMD correctness check failed");

      rows.push_back(ResultRow{config, uint32_t(edges.size()), 1u, "cpu_scalar", "cpu", "render", scalarTiming, 1.0, {}, {}});
      rows.push_back(ResultRow{config, uint32_t(edges.size()), 1u, "cpu_avx2", "cpu", "render", simdTiming,
                               scalarTiming.medianMs / simdTiming.medianMs, simdComparison, {}});
      rows.push_back(ResultRow{config, uint32_t(edges.size()), std::min(config.threads, config.height),
                               "cpu_avx2_threaded", "cpu", "render", threadedTiming,
                               scalarTiming.medianMs / threadedTiming.medianMs, threadedComparison, {}});

      if (options.coverageScan) {
        const auto buildStart = Clock::now();
        coverageDeltas = gpusimd::buildCoverageDeltas(
          config.width, config.height, edges, options.scanVerticalSamples, 256);
        const auto buildStop = Clock::now();
        const double buildMs = std::chrono::duration<double, std::milli>(
          buildStop - buildStart).count();
        scanCoverageReference.resize(pixelCount);
        scanPixelsReference.resize(pixelCount);
        std::vector<uint32_t> avxCoverage(pixelCount);
        std::vector<uint32_t> avxPixels(pixelCount);
        std::vector<uint32_t> threadedScanCoverage(pixelCount);
        std::vector<uint32_t> threadedScanPixels(pixelCount);

        scalarScanTiming = benchmark(options.warmup, options.iterations, [&] {
          gpusimd::scanCoverageScalar(
            scanCoverageReference, {}, coverageDeltas.values,
            config.width, config.height, coverageDeltas.scale);
        });
        const Timing avxScanTiming = benchmark(options.warmup, options.iterations, [&] {
          gpusimd::scanCoverageAvx2(
            avxCoverage, {}, coverageDeltas.values,
            config.width, config.height, coverageDeltas.scale);
        });
        const Timing threadedScanTiming = benchmark(options.warmup, options.iterations, [&] {
          gpusimd::scanCoverageAvx2Threaded(
            threadedScanCoverage, {}, coverageDeltas.values,
            config.width, config.height, coverageDeltas.scale, config.threads);
        });
        scalarScanPaintTiming = benchmark(options.warmup, options.iterations, [&] {
          gpusimd::scanCoverageScalar(
            scanCoverageReference, scanPixelsReference, coverageDeltas.values,
            config.width, config.height, coverageDeltas.scale);
        });
        const Timing avxScanPaintTiming = benchmark(options.warmup, options.iterations, [&] {
          gpusimd::scanCoverageAvx2(
            avxCoverage, avxPixels, coverageDeltas.values,
            config.width, config.height, coverageDeltas.scale);
        });
        const Timing threadedScanPaintTiming = benchmark(options.warmup, options.iterations, [&] {
          gpusimd::scanCoverageAvx2Threaded(
            threadedScanCoverage, threadedScanPixels, coverageDeltas.values,
            config.width, config.height, coverageDeltas.scale, config.threads);
        });
        const Comparison avxCoverageComparison = compareCoverage(
          scanCoverageReference, avxCoverage);
        const Comparison avxPaintComparison = compareImages(
          scanPixelsReference, avxPixels);
        const Comparison threadedScanCoverageComparison = compareCoverage(
          scanCoverageReference, threadedScanCoverage);
        const Comparison threadedScanPaintComparison = compareImages(
          scanPixelsReference, threadedScanPixels);
        std::cout << "\nCoverage-delta input: " << coverageDeltas.verticalSamples
                  << " vertical samples, " << coverageDeltas.horizontalUnits
                  << " horizontal units, scale " << coverageDeltas.scale
                  << "; built in " << std::fixed << std::setprecision(3)
                  << buildMs << " ms\n";
        printTiming("CPU scalar coverage scan", scalarScanTiming, scalarScanTiming.medianMs);
        printTiming("CPU AVX2 coverage scan", avxScanTiming, scalarScanTiming.medianMs);
        printTiming("CPU AVX2 threaded scan", threadedScanTiming, scalarScanTiming.medianMs);
        printTiming("CPU scalar scan + paint", scalarScanPaintTiming, scalarScanPaintTiming.medianMs);
        printTiming("CPU AVX2 scan + paint", avxScanPaintTiming, scalarScanPaintTiming.medianMs);
        printTiming("CPU AVX2 threaded + paint", threadedScanPaintTiming, scalarScanPaintTiming.medianMs);
        printCoverageComparison("CPU AVX2", avxCoverageComparison, pixelCount);
        printCoverageComparison("CPU AVX2 threaded", threadedScanCoverageComparison, pixelCount);
        printComparison("CPU AVX2 scan paint", avxPaintComparison, pixelCount);
        printComparison("CPU threaded scan paint", threadedScanPaintComparison, pixelCount);
        if (avxCoverageComparison.differingPixels ||
            threadedScanCoverageComparison.differingPixels ||
            avxPaintComparison.differingPixels ||
            threadedScanPaintComparison.differingPixels)
          fail("CPU AVX2 coverage scan correctness check failed");

        rows.push_back(ResultRow{
          config, uint32_t(edges.size()), 1u, "cpu_scan_scalar", "cpu",
          "coverage_scan", scalarScanTiming, 1.0, {}, {}});
        rows.push_back(ResultRow{
          config, uint32_t(edges.size()), 1u, "cpu_scan_avx2", "cpu",
          "coverage_scan", avxScanTiming,
          scalarScanTiming.medianMs / avxScanTiming.medianMs,
          avxCoverageComparison, {}});
        rows.push_back(ResultRow{
          config, uint32_t(edges.size()), std::min(config.threads, config.height),
          "cpu_scan_avx2_threaded", "cpu", "coverage_scan", threadedScanTiming,
          scalarScanTiming.medianMs / threadedScanTiming.medianMs,
          threadedScanCoverageComparison, {}});
        rows.push_back(ResultRow{
          config, uint32_t(edges.size()), 1u, "cpu_scan_scalar", "cpu",
          "coverage_scan_paint", scalarScanPaintTiming, 1.0, {}, {}});
        rows.push_back(ResultRow{
          config, uint32_t(edges.size()), 1u, "cpu_scan_avx2", "cpu",
          "coverage_scan_paint", avxScanPaintTiming,
          scalarScanPaintTiming.medianMs / avxScanPaintTiming.medianMs,
          avxPaintComparison, {}});
        rows.push_back(ResultRow{
          config, uint32_t(edges.size()), std::min(config.threads, config.height),
          "cpu_scan_avx2_threaded", "cpu", "coverage_scan_paint",
          threadedScanPaintTiming,
          scalarScanPaintTiming.medianMs / threadedScanPaintTiming.medianMs,
          threadedScanPaintComparison, {}});
      }

      const bool multipleOutputs = configurations.size() > 1u;
      const std::filesystem::path imageDirectory = multipleOutputs
        ? options.outputDir / configurationTag(config) : options.outputDir;
      if (options.writeImages) {
        std::filesystem::create_directories(imageDirectory);
        writePpm(imageDirectory / "scalar.ppm", config.width, config.height, scalarPixels);
        writePpm(imageDirectory / "avx2.ppm", config.width, config.height, simdPixels);
      }

      if (options.gpu && options.openGl) {
        GpuRenderer gpu(config.width, config.height, config.aaGrid, edges);
        const GpuInfo gpuInfo = gpu.info();
        if (!printedOpenGlInfo) {
          std::cout << "\nOpenGL vendor:   " << gpuInfo.vendor
                    << "\nOpenGL renderer: " << gpuInfo.renderer
                    << "\nOpenGL version:  " << gpuInfo.version << '\n';
          if (gpuInfo.subgroupSize)
            std::cout << "GPU subgroup:    " << gpuInfo.subgroupSize << " lanes (used as logical SIMD width)\n";
          else
            std::cout << "GPU subgroup:    not exposed; using " << gpuInfo.workgroupSize
                      << " independent invocations\n";
          if (!gpuInfo.hardware)
            std::cout << "WARNING: software OpenGL renderer detected; GPU rows are tagged gpu_hardware=0 "
                         "and are not hardware evidence.\n";
          printedOpenGlInfo = true;
        }
        if (options.requireHardwareGpu && !gpuInfo.hardware)
          fail("--require-hardware-gpu rejected software renderer: " + gpuInfo.renderer);

        auto distributed = gpu.run(1, options.warmup, options.iterations);
        auto packed = gpu.run(8, options.warmup, options.iterations);
        const Comparison distributedComparison = compareImages(scalarPixels, distributed.pixels);
        const Comparison packedComparison = compareImages(scalarPixels, packed.pixels);

        printTiming("OpenGL distributed + sync", distributed.kernel, scalarTiming.medianMs);
        printTiming("OpenGL packed8 + sync", packed.kernel, scalarTiming.medianMs);
        printTiming("OpenGL distributed + read", distributed.withReadback, scalarTiming.medianMs);
        printTiming("OpenGL packed + readback", packed.withReadback, scalarTiming.medianMs);
        printComparison("OpenGL distributed", distributedComparison, pixelCount);
        printComparison("OpenGL packed", packedComparison, pixelCount);

        rows.push_back(ResultRow{config, uint32_t(edges.size()), 0u, "gpu_distributed", "opengl",
                                 "synchronized_dispatch", distributed.kernel,
                                 scalarTiming.medianMs / distributed.kernel.medianMs,
                                 distributedComparison, gpuInfo});
        rows.push_back(ResultRow{config, uint32_t(edges.size()), 0u, "gpu_packed8", "opengl",
                                 "synchronized_dispatch", packed.kernel,
                                 scalarTiming.medianMs / packed.kernel.medianMs,
                                 packedComparison, gpuInfo});
        rows.push_back(ResultRow{config, uint32_t(edges.size()), 0u, "gpu_distributed", "opengl",
                                 "dispatch_plus_readback", distributed.withReadback,
                                 scalarTiming.medianMs / distributed.withReadback.medianMs,
                                 distributedComparison, gpuInfo});
        rows.push_back(ResultRow{config, uint32_t(edges.size()), 0u, "gpu_packed8", "opengl",
                                 "dispatch_plus_readback", packed.withReadback,
                                 scalarTiming.medianMs / packed.withReadback.medianMs,
                                 packedComparison, gpuInfo});

        if (options.writeImages) {
          writePpm(imageDirectory / "gpu_distributed.ppm", config.width, config.height, distributed.pixels);
          writePpm(imageDirectory / "gpu_packed8.ppm", config.width, config.height, packed.pixels);
        }
      }

#if defined(GPUSIMD_HAS_VULKAN)
      if (options.gpu && options.vulkan) {
        gpusimd::VulkanRenderer renderer(
          config.width, config.height, config.aaGrid, edges);
        const gpusimd::VulkanDeviceInfo& device = renderer.deviceInfo();
        const GpuInfo gpuInfo{
          device.vendor,
          device.device,
          "Vulkan " + device.apiVersion + " | " + device.driver,
          device.hardware,
          device.subgroupSize,
          device.workgroupSize,
          device.subgroupOperations,
          device.timestampValidBits
        };
        if (!printedVulkanInfo) {
          std::cout << "\nVulkan vendor:   " << device.vendor
                    << "\nVulkan device:   " << device.device
                    << "\nVulkan API:      " << device.apiVersion
                    << "\nVulkan driver:   " << device.driver
                    << "\nVulkan subgroup: " << device.subgroupSize << " lanes; "
                    << device.subgroupOperations
                    << "\nTimestamp bits:  " << device.timestampValidBits << '\n';
          if (!device.hardware)
            std::cout << "WARNING: software Vulkan device detected; rows are tagged gpu_hardware=0 "
                         "and are not hardware evidence.\n";
          printedVulkanInfo = true;
        }
        if (options.requireHardwareGpu && !device.hardware)
          fail("--require-hardware-gpu rejected software Vulkan device: " + device.device);

        auto distributed = renderer.run(1, options.warmup, options.iterations);
        auto packed = renderer.run(8, options.warmup, options.iterations);
        const Timing distributedTimestamp = summarizeSamples(std::move(distributed.timestampMilliseconds));
        const Timing distributedSync = summarizeSamples(std::move(distributed.synchronizedMilliseconds));
        const Timing distributedReadback = summarizeSamples(std::move(distributed.readbackMilliseconds));
        const Timing packedTimestamp = summarizeSamples(std::move(packed.timestampMilliseconds));
        const Timing packedSync = summarizeSamples(std::move(packed.synchronizedMilliseconds));
        const Timing packedReadback = summarizeSamples(std::move(packed.readbackMilliseconds));
        const Comparison distributedComparison = compareImages(scalarPixels, distributed.pixels);
        const Comparison packedComparison = compareImages(scalarPixels, packed.pixels);

        printTiming("Vulkan distributed timestamp", distributedTimestamp, scalarTiming.medianMs);
        printTiming("Vulkan packed8 timestamp", packedTimestamp, scalarTiming.medianMs);
        printTiming("Vulkan distributed + sync", distributedSync, scalarTiming.medianMs);
        printTiming("Vulkan packed8 + sync", packedSync, scalarTiming.medianMs);
        printTiming("Vulkan distributed + read", distributedReadback, scalarTiming.medianMs);
        printTiming("Vulkan packed8 + readback", packedReadback, scalarTiming.medianMs);
        printComparison("Vulkan distributed", distributedComparison, pixelCount);
        printComparison("Vulkan packed", packedComparison, pixelCount);

        const auto addRow = [&](const char* backend, const char* scope,
                                const Timing& timing, const Comparison& comparison) {
          rows.push_back(ResultRow{
            config, uint32_t(edges.size()), 0u, backend, "vulkan", scope, timing,
            scalarTiming.medianMs / timing.medianMs, comparison, gpuInfo});
        };
        addRow("gpu_distributed", "device_timestamp", distributedTimestamp, distributedComparison);
        addRow("gpu_packed8", "device_timestamp", packedTimestamp, packedComparison);
        addRow("gpu_distributed", "synchronized_dispatch", distributedSync, distributedComparison);
        addRow("gpu_packed8", "synchronized_dispatch", packedSync, packedComparison);
        addRow("gpu_distributed", "dispatch_plus_readback", distributedReadback, distributedComparison);
        addRow("gpu_packed8", "dispatch_plus_readback", packedReadback, packedComparison);

        if (options.writeImages) {
          writePpm(imageDirectory / "vulkan_distributed.ppm",
                   config.width, config.height, distributed.pixels);
          writePpm(imageDirectory / "vulkan_packed8.ppm",
                   config.width, config.height, packed.pixels);
        }

        if (options.coverageScan) {
          const auto runScanAlgorithm = [&](gpusimd::CoverageScanAlgorithm algorithm,
                                            const char* backend,
                                            const char* displayName) {
            auto scan = renderer.runCoverageScan(
              coverageDeltas.values, coverageDeltas.scale, algorithm, false,
              options.warmup, options.iterations);
            auto paint = renderer.runCoverageScan(
              coverageDeltas.values, coverageDeltas.scale, algorithm, true,
              options.warmup, options.iterations);
            const Timing scanTimestamp = summarizeSamples(
              std::move(scan.timestampMilliseconds));
            const Timing scanSync = summarizeSamples(
              std::move(scan.synchronizedMilliseconds));
            const Timing paintTimestamp = summarizeSamples(
              std::move(paint.timestampMilliseconds));
            const Timing paintSync = summarizeSamples(
              std::move(paint.synchronizedMilliseconds));
            const Comparison scanComparison = compareCoverage(
              scanCoverageReference, scan.coverage);
            const Comparison paintCoverageComparison = compareCoverage(
              scanCoverageReference, paint.coverage);
            const Comparison paintComparison = compareImages(
              scanPixelsReference, paint.pixels);

            const std::string scanTimestampName =
              std::string("VK ") + displayName + " scan timestamp";
            const std::string paintTimestampName =
              std::string("VK ") + displayName + " paint timestamp";
            printTiming(scanTimestampName.c_str(), scanTimestamp, scalarScanTiming.medianMs);
            printTiming(paintTimestampName.c_str(), paintTimestamp, scalarScanPaintTiming.medianMs);
            printCoverageComparison(displayName, scanComparison, pixelCount);
            printComparison((std::string(displayName) + " paint").c_str(),
                            paintComparison, pixelCount);
            if (scanComparison.differingPixels ||
                paintCoverageComparison.differingPixels ||
                paintComparison.differingPixels) {
              uint32_t shown = 0;
              std::cerr << "  first coverage mismatches:";
              for (size_t i = 0; i < scanCoverageReference.size() && shown < 12u; ++i) {
                if (scanCoverageReference[i] == scan.coverage[i])
                  continue;
                std::cerr << " (" << (i % config.width) << ',' << (i / config.width)
                          << ": " << scanCoverageReference[i] << " != "
                          << scan.coverage[i] << ", delta="
                          << coverageDeltas.values[i] << ", prev="
                          << (i % config.width ? scanCoverageReference[i - 1u] : 0u)
                          << ')';
                ++shown;
              }
              std::cerr << '\n';
              fail(std::string("Vulkan coverage scan correctness failed for ") + displayName);
            }

            const auto addScanRow = [&](const char* scope, const Timing& timing,
                                        double baseline, const Comparison& comparison) {
              rows.push_back(ResultRow{
                config, uint32_t(edges.size()), 0u, backend, "vulkan", scope,
                timing, baseline / timing.medianMs, comparison, gpuInfo});
            };
            addScanRow("coverage_scan_device_timestamp", scanTimestamp,
                       scalarScanTiming.medianMs, scanComparison);
            addScanRow("coverage_scan_synchronized", scanSync,
                       scalarScanTiming.medianMs, scanComparison);
            addScanRow("coverage_scan_paint_device_timestamp", paintTimestamp,
                       scalarScanPaintTiming.medianMs, paintComparison);
            addScanRow("coverage_scan_paint_synchronized", paintSync,
                       scalarScanPaintTiming.medianMs, paintComparison);

            if (options.writeImages &&
                algorithm == gpusimd::CoverageScanAlgorithm::kSubgroup) {
              writePpm(imageDirectory / "vulkan_subgroup_scan.ppm",
                       config.width, config.height, paint.pixels);
            }
          };

          std::cout << "\nVulkan fixed-point coverage scans:\n";
          runScanAlgorithm(gpusimd::CoverageScanAlgorithm::kSerialized,
                           "gpu_scan_serialized", "serialized");
          runScanAlgorithm(gpusimd::CoverageScanAlgorithm::kSharedMemory,
                           "gpu_scan_shared", "shared");
          runScanAlgorithm(gpusimd::CoverageScanAlgorithm::kSubgroup,
                           "gpu_scan_subgroup", "subgroup");
        }
      }
#endif
    }

    if (!options.csvPath.empty()) {
      writeCsv(options.csvPath, host, options.warmup, options.iterations, rows);
      std::cout << "\nCSV written to " << std::filesystem::absolute(options.csvPath) << '\n';
    }
    if (options.writeImages)
      std::cout << "Images written under " << std::filesystem::absolute(options.outputDir) << '\n';
    return 0;
  }
  catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
