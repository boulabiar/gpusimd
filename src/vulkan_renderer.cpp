#include "vulkan_renderer.h"

#include <vulkan/vulkan.h>

#include "vulkan_shader_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpusimd {
namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void vkFail(const std::string& message) {
  throw std::runtime_error("Vulkan: " + message);
}

void vkCheck(VkResult result, const char* operation) {
  if (result != VK_SUCCESS)
    vkFail(std::string(operation) + " failed with VkResult " + std::to_string(result));
}

std::string versionString(uint32_t version) {
  return std::to_string(VK_API_VERSION_MAJOR(version)) + '.' +
         std::to_string(VK_API_VERSION_MINOR(version)) + '.' +
         std::to_string(VK_API_VERSION_PATCH(version));
}

std::string vendorString(uint32_t vendorId) {
  const char* name = "unknown";
  switch (vendorId) {
    case 0x1002: name = "AMD"; break;
    case 0x1010: name = "Imagination"; break;
    case 0x10DE: name = "NVIDIA"; break;
    case 0x13B5: name = "ARM"; break;
    case 0x5143: name = "Qualcomm"; break;
    case 0x8086: name = "Intel"; break;
    default: break;
  }
  std::ostringstream stream;
  stream << name << " (0x" << std::hex << vendorId << ')';
  return stream.str();
}

bool isHardwareDevice(VkPhysicalDeviceType type, std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  return type != VK_PHYSICAL_DEVICE_TYPE_CPU &&
         name.find("llvmpipe") == std::string::npos &&
         name.find("lavapipe") == std::string::npos &&
         name.find("swiftshader") == std::string::npos;
}

std::string subgroupOperationString(
  VkSubgroupFeatureFlags flags,
  VkShaderStageFlags stages) {

  if ((stages & VK_SHADER_STAGE_COMPUTE_BIT) == 0)
    return "not-supported-in-compute";

  struct Entry { VkSubgroupFeatureFlagBits flag; const char* name; };
  constexpr std::array<Entry, 9> entries = {{
    {VK_SUBGROUP_FEATURE_BASIC_BIT, "basic"},
    {VK_SUBGROUP_FEATURE_VOTE_BIT, "vote"},
    {VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, "arithmetic"},
    {VK_SUBGROUP_FEATURE_BALLOT_BIT, "ballot"},
    {VK_SUBGROUP_FEATURE_SHUFFLE_BIT, "shuffle"},
    {VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT, "shuffle-relative"},
    {VK_SUBGROUP_FEATURE_CLUSTERED_BIT, "clustered"},
    {VK_SUBGROUP_FEATURE_QUAD_BIT, "quad"},
    {VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV, "partitioned-nv"}
  }};
  std::string result;
  for (const Entry& entry : entries) {
    if ((flags & entry.flag) == 0)
      continue;
    if (!result.empty()) result += '|';
    result += entry.name;
  }
  return result.empty() ? "none" : result;
}

} // namespace

struct VulkanRenderer::Impl {
  struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
  };

  struct PushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t edgeCount;
    uint32_t aaGrid;
  };

  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t aaGrid = 0;
  uint32_t edgeCount = 0;
  uint32_t pixelCount = 0;
  uint32_t queueFamilyIndex = 0;
  uint32_t maxDispatchGroups = 0;
  float timestampPeriod = 0.0f;
  VulkanDeviceInfo info;

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  std::array<VkPipeline, 2> pipelines{VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool queryPool = VK_NULL_HANDLE;
  Buffer edgeBuffer;
  Buffer pixelBuffer;
  Buffer readbackBuffer;
  void* readbackMapping = nullptr;

  ~Impl() {
    if (device != VK_NULL_HANDLE)
      vkDeviceWaitIdle(device);
    if (readbackMapping)
      vkUnmapMemory(device, readbackBuffer.memory);
    if (queryPool) vkDestroyQueryPool(device, queryPool, nullptr);
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
    for (VkPipeline pipeline : pipelines)
      if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (shaderModule) vkDestroyShaderModule(device, shaderModule, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (descriptorSetLayout) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    destroyBuffer(readbackBuffer);
    destroyBuffer(pixelBuffer);
    destroyBuffer(edgeBuffer);
    if (device) vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
  }

  void destroyBuffer(Buffer& buffer) noexcept {
    if (buffer.handle) vkDestroyBuffer(device, buffer.handle, nullptr);
    if (buffer.memory) vkFreeMemory(device, buffer.memory, nullptr);
    buffer = {};
  }

  uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags required) const {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (properties.memoryTypes[i].propertyFlags & required) == required)
        return i;
    }
    vkFail("no compatible memory type");
  }

  Buffer createBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memoryProperties) {

    Buffer buffer;
    buffer.size = size;
    const VkBufferCreateInfo createInfo{
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size, usage,
      VK_SHARING_MODE_EXCLUSIVE, 0, nullptr
    };
    vkCheck(vkCreateBuffer(device, &createInfo, nullptr, &buffer.handle), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer.handle, &requirements);
    const VkMemoryAllocateInfo allocation{
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, requirements.size,
      findMemoryType(requirements.memoryTypeBits, memoryProperties)
    };
    vkCheck(vkAllocateMemory(device, &allocation, nullptr, &buffer.memory), "vkAllocateMemory");
    vkCheck(vkBindBufferMemory(device, buffer.handle, buffer.memory, 0), "vkBindBufferMemory");
    return buffer;
  }

  void initialize(
    uint32_t imageWidth,
    uint32_t imageHeight,
    uint32_t sampleGrid,
    std::span<const Edge> edges) {

    width = imageWidth;
    height = imageHeight;
    aaGrid = sampleGrid;
    edgeCount = uint32_t(edges.size());
    const uint64_t pixels64 = uint64_t(width) * height;
    if (pixels64 > std::numeric_limits<uint32_t>::max())
      vkFail("image has more pixels than the shader's 32-bit index supports");
    pixelCount = uint32_t(pixels64);

    uint32_t loaderVersion = VK_API_VERSION_1_0;
    vkCheck(vkEnumerateInstanceVersion(&loaderVersion), "vkEnumerateInstanceVersion");
    if (loaderVersion < VK_API_VERSION_1_2)
      vkFail("Vulkan 1.2 loader is required");

    const VkApplicationInfo application{
      VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr,
      "gpusimd_vector_bench", 1, "gpusimd", 1, VK_API_VERSION_1_2
    };
    const VkInstanceCreateInfo instanceCreate{
      VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &application,
      0, nullptr, 0, nullptr
    };
    vkCheck(vkCreateInstance(&instanceCreate, nullptr, &instance), "vkCreateInstance");
    selectPhysicalDevice();
    createLogicalDevice();
    createResources(edges);
  }

  void selectPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkCheck(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices");
    if (!deviceCount)
      vkFail("no physical devices found");
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkCheck(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices");

    int bestScore = std::numeric_limits<int>::min();
    VkPhysicalDeviceProperties bestProperties{};
    VkPhysicalDeviceSubgroupProperties bestSubgroup{};
    bestSubgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceDriverProperties bestDriver{};
    bestDriver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    uint32_t bestQueue = 0;
    uint32_t bestTimestampBits = 0;

    for (VkPhysicalDevice candidate : devices) {
      VkPhysicalDeviceDriverProperties driver{};
      driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
      VkPhysicalDeviceSubgroupProperties subgroup{};
      subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
      subgroup.pNext = &driver;
      VkPhysicalDeviceProperties2 properties2{};
      properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      properties2.pNext = &subgroup;
      vkGetPhysicalDeviceProperties2(candidate, &properties2);
      if (properties2.properties.apiVersion < VK_API_VERSION_1_2 ||
          properties2.properties.limits.maxComputeWorkGroupInvocations < info.workgroupSize ||
          properties2.properties.limits.maxComputeWorkGroupSize[0] < info.workgroupSize)
        continue;

      uint32_t queueCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, nullptr);
      std::vector<VkQueueFamilyProperties> queues(queueCount);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, queues.data());
      for (uint32_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
        if ((queues[queueIndex].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0)
          continue;
        int score = 0;
        switch (properties2.properties.deviceType) {
          case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score += 500; break;
          case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score += 400; break;
          case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score += 300; break;
          case VK_PHYSICAL_DEVICE_TYPE_CPU: score += 100; break;
          default: score += 200; break;
        }
        if ((queues[queueIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) score += 20;
        if (queues[queueIndex].timestampValidBits) score += 10;
        if (score <= bestScore)
          continue;
        bestScore = score;
        physicalDevice = candidate;
        bestProperties = properties2.properties;
        bestSubgroup = subgroup;
        bestDriver = driver;
        bestQueue = queueIndex;
        bestTimestampBits = queues[queueIndex].timestampValidBits;
      }
    }
    if (physicalDevice == VK_NULL_HANDLE)
      vkFail("no compute-capable physical device found");

    queueFamilyIndex = bestQueue;
    maxDispatchGroups = bestProperties.limits.maxComputeWorkGroupCount[0];
    timestampPeriod = bestProperties.limits.timestampPeriod;
    info.vendor = vendorString(bestProperties.vendorID);
    info.device = bestProperties.deviceName;
    info.apiVersion = versionString(bestProperties.apiVersion);
    info.driver = std::string(bestDriver.driverName) + " " + bestDriver.driverInfo;
    if (info.driver == " ")
      info.driver = "driver version " + std::to_string(bestProperties.driverVersion);
    info.hardware = isHardwareDevice(bestProperties.deviceType, info.device);
    info.subgroupSize = bestSubgroup.subgroupSize;
    info.subgroupOperations = subgroupOperationString(
      bestSubgroup.supportedOperations, bestSubgroup.supportedStages);
    info.timestampValidBits = bestTimestampBits;
    if (!info.timestampValidBits)
      vkFail("selected compute queue does not support timestamps");
  }

  void createLogicalDevice() {
    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queueCreate{
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0,
      queueFamilyIndex, 1, &priority
    };
    const VkDeviceCreateInfo deviceCreate{
      VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0,
      1, &queueCreate, 0, nullptr, 0, nullptr, nullptr
    };
    vkCheck(vkCreateDevice(physicalDevice, &deviceCreate, nullptr, &device), "vkCreateDevice");
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
  }

  void createResources(std::span<const Edge> edges) {
    const VkDeviceSize edgeBytes = VkDeviceSize(edges.size_bytes());
    const VkDeviceSize pixelBytes = VkDeviceSize(pixelCount) * sizeof(uint32_t);
    edgeBuffer = createBuffer(edgeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    pixelBuffer = createBuffer(pixelBytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    readbackBuffer = createBuffer(pixelBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* edgeMapping = nullptr;
    vkCheck(vkMapMemory(device, edgeBuffer.memory, 0, edgeBytes, 0, &edgeMapping), "vkMapMemory(edges)");
    std::memcpy(edgeMapping, edges.data(), size_t(edgeBytes));
    vkUnmapMemory(device, edgeBuffer.memory);
    vkCheck(vkMapMemory(device, readbackBuffer.memory, 0, pixelBytes, 0, &readbackMapping),
            "vkMapMemory(readback)");

    constexpr std::array<VkDescriptorSetLayoutBinding, 2> bindings = {{
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    }};
    const VkDescriptorSetLayoutCreateInfo layoutCreate{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
      uint32_t(bindings.size()), bindings.data()
    };
    vkCheck(vkCreateDescriptorSetLayout(device, &layoutCreate, nullptr, &descriptorSetLayout),
            "vkCreateDescriptorSetLayout");

    const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    const VkDescriptorPoolCreateInfo poolCreate{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &poolSize
    };
    vkCheck(vkCreateDescriptorPool(device, &poolCreate, nullptr, &descriptorPool),
            "vkCreateDescriptorPool");
    const VkDescriptorSetAllocateInfo setAllocate{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 1,
      &descriptorSetLayout
    };
    vkCheck(vkAllocateDescriptorSets(device, &setAllocate, &descriptorSet),
            "vkAllocateDescriptorSets");

    const std::array<VkDescriptorBufferInfo, 2> bufferInfos = {{
      {edgeBuffer.handle, 0, edgeBytes},
      {pixelBuffer.handle, 0, pixelBytes}
    }};
    const std::array<VkWriteDescriptorSet, 2> writes = {{
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 0, 0, 1,
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfos[0], nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet, 1, 0, 1,
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfos[1], nullptr}
    }};
    vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), 0, nullptr);

    const VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    const VkPipelineLayoutCreateInfo pipelineLayoutCreate{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
      1, &descriptorSetLayout, 1, &pushRange
    };
    vkCheck(vkCreatePipelineLayout(device, &pipelineLayoutCreate, nullptr, &pipelineLayout),
            "vkCreatePipelineLayout");

    const VkShaderModuleCreateInfo shaderCreate{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
      sizeof(gpusimdVulkanShaderSpv), gpusimdVulkanShaderSpv
    };
    vkCheck(vkCreateShaderModule(device, &shaderCreate, nullptr, &shaderModule),
            "vkCreateShaderModule");
    pipelines[0] = createPipeline(1);
    pipelines[1] = createPipeline(8);

    const VkCommandPoolCreateInfo commandPoolCreate{
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex
    };
    vkCheck(vkCreateCommandPool(device, &commandPoolCreate, nullptr, &commandPool),
            "vkCreateCommandPool");
    const VkCommandBufferAllocateInfo commandAllocate{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool,
      VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1
    };
    vkCheck(vkAllocateCommandBuffers(device, &commandAllocate, &commandBuffer),
            "vkAllocateCommandBuffers");
    VkFenceCreateInfo fenceCreate{};
    fenceCreate.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCheck(vkCreateFence(device, &fenceCreate, nullptr, &fence), "vkCreateFence");
    const VkQueryPoolCreateInfo queryCreate{
      VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
      VK_QUERY_TYPE_TIMESTAMP, 2, 0
    };
    vkCheck(vkCreateQueryPool(device, &queryCreate, nullptr, &queryPool), "vkCreateQueryPool");
  }

  VkPipeline createPipeline(uint32_t pixelsPerInvocation) const {
    const VkSpecializationMapEntry mapEntry{0, 0, sizeof(uint32_t)};
    const VkSpecializationInfo specialization{
      1, &mapEntry, sizeof(pixelsPerInvocation), &pixelsPerInvocation
    };
    const VkPipelineShaderStageCreateInfo stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
      VK_SHADER_STAGE_COMPUTE_BIT, shaderModule, "main", &specialization
    };
    const VkComputePipelineCreateInfo createInfo{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
      stage, pipelineLayout, VK_NULL_HANDLE, -1
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline),
            "vkCreateComputePipelines");
    return pipeline;
  }

  VkPipeline pipelineFor(uint32_t pixelsPerInvocation) const {
    if (pixelsPerInvocation == 1) return pipelines[0];
    if (pixelsPerInvocation == 8) return pipelines[1];
    vkFail("pixels per invocation must be 1 or 8");
  }

  void record(uint32_t pixelsPerInvocation, bool copyToHost) {
    vkCheck(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
    const VkCommandBufferBeginInfo begin{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr
    };
    vkCheck(vkBeginCommandBuffer(commandBuffer, &begin), "vkBeginCommandBuffer");
    vkCmdResetQueryPool(commandBuffer, queryPool, 0, 2);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipelineFor(pixelsPerInvocation));
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    const PushConstants push{width, height, edgeCount, aaGrid};
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);

    const uint64_t invocationCount =
      (uint64_t(pixelCount) + pixelsPerInvocation - 1u) / pixelsPerInvocation;
    const uint64_t groupCount = (invocationCount + info.workgroupSize - 1u) / info.workgroupSize;
    if (groupCount > maxDispatchGroups)
      vkFail("dispatch exceeds maxComputeWorkGroupCount[0]");
    vkCmdDispatch(commandBuffer, uint32_t(groupCount), 1, 1);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);

    if (copyToHost) {
      const VkBufferMemoryBarrier barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        pixelBuffer.handle, 0, pixelBuffer.size
      };
      vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr);
      const VkBufferCopy copy{0, 0, pixelBuffer.size};
      vkCmdCopyBuffer(commandBuffer, pixelBuffer.handle, readbackBuffer.handle, 1, &copy);
    }
    vkCheck(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
  }

  double submitAndWait() {
    vkCheck(vkResetFences(device, 1, &fence), "vkResetFences");
    const VkSubmitInfo submit{
      VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr,
      0, nullptr, nullptr, 1, &commandBuffer, 0, nullptr
    };
    const auto start = Clock::now();
    vkCheck(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
    vkCheck(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
  }

  double readTimestampMilliseconds() const {
    std::array<uint64_t, 2> timestamps{};
    vkCheck(vkGetQueryPoolResults(device, queryPool, 0, 2,
      sizeof(timestamps), timestamps.data(), sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
    uint64_t ticks = timestamps[1] - timestamps[0];
    if (info.timestampValidBits < 64u) {
      const uint64_t mask = (uint64_t(1) << info.timestampValidBits) - 1u;
      ticks &= mask;
    }
    return double(ticks) * double(timestampPeriod) / 1.0e6;
  }

  VulkanRunResult run(uint32_t pixelsPerInvocation, uint32_t warmup, uint32_t iterations) {
    VulkanRunResult result;
    result.timestampMilliseconds.reserve(iterations);
    result.synchronizedMilliseconds.reserve(iterations);
    result.readbackMilliseconds.reserve(iterations);

    for (uint32_t i = 0; i < warmup; ++i) {
      record(pixelsPerInvocation, false);
      submitAndWait();
    }
    for (uint32_t i = 0; i < iterations; ++i) {
      record(pixelsPerInvocation, false);
      result.synchronizedMilliseconds.push_back(submitAndWait());
      result.timestampMilliseconds.push_back(readTimestampMilliseconds());
    }
    for (uint32_t i = 0; i < warmup; ++i) {
      record(pixelsPerInvocation, true);
      submitAndWait();
    }
    for (uint32_t i = 0; i < iterations; ++i) {
      record(pixelsPerInvocation, true);
      result.readbackMilliseconds.push_back(submitAndWait());
    }

    result.pixels.resize(pixelCount);
    std::memcpy(result.pixels.data(), readbackMapping, size_t(readbackBuffer.size));
    return result;
  }
};

VulkanRenderer::VulkanRenderer(
  uint32_t width,
  uint32_t height,
  uint32_t aaGrid,
  std::span<const Edge> edges)
  : impl_(std::make_unique<Impl>()) {
  impl_->initialize(width, height, aaGrid, edges);
}

VulkanRenderer::~VulkanRenderer() = default;

const VulkanDeviceInfo& VulkanRenderer::deviceInfo() const noexcept {
  return impl_->info;
}

VulkanRunResult VulkanRenderer::run(
  uint32_t pixelsPerInvocation,
  uint32_t warmup,
  uint32_t iterations) {
  return impl_->run(pixelsPerInvocation, warmup, iterations);
}

} // namespace gpusimd
