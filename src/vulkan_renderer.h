#ifndef GPUSIMD_VULKAN_RENDERER_H
#define GPUSIMD_VULKAN_RENDERER_H

#include "coverage_scan.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gpusimd {

struct VulkanDeviceInfo {
  std::string vendor;
  std::string device;
  std::string apiVersion;
  std::string driver;
  bool hardware = false;
  uint32_t subgroupSize = 0;
  std::string subgroupOperations;
  uint32_t workgroupSize = 64;
  uint32_t timestampValidBits = 0;
};

struct VulkanRunResult {
  std::vector<uint32_t> pixels;
  std::vector<double> timestampMilliseconds;
  std::vector<double> synchronizedMilliseconds;
  std::vector<double> readbackMilliseconds;
};

enum class CoverageScanAlgorithm : uint32_t {
  kSerialized = 0,
  kSharedMemory = 1,
  kSubgroup = 2
};

struct VulkanCoverageScanResult {
  std::vector<uint32_t> coverage;
  std::vector<uint32_t> pixels;
  std::vector<double> timestampMilliseconds;
  std::vector<double> synchronizedMilliseconds;
  std::vector<double> uploadMilliseconds;
  uint64_t inputBytes = 0;
};

class VulkanRenderer {
public:
  VulkanRenderer(
    uint32_t width,
    uint32_t height,
    uint32_t aaGrid,
    std::span<const Edge> edges);
  ~VulkanRenderer();

  VulkanRenderer(const VulkanRenderer&) = delete;
  VulkanRenderer& operator=(const VulkanRenderer&) = delete;

  const VulkanDeviceInfo& deviceInfo() const noexcept;
  VulkanRunResult run(uint32_t pixelsPerInvocation, uint32_t warmup, uint32_t iterations);
  VulkanCoverageScanResult runCoverageScan(
    std::span<const int32_t> deltas,
    uint32_t coverageScale,
    CoverageResolveMode resolveMode,
    CoverageScanAlgorithm algorithm,
    bool paint,
    uint32_t warmup,
    uint32_t iterations);
  VulkanCoverageScanResult runAnalyticTileScan(
    const AnalyticTileCells& tiles,
    CoverageResolveMode resolveMode,
    CoverageScanAlgorithm algorithm,
    bool paint,
    uint32_t warmup,
    uint32_t iterations);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace gpusimd

#endif
