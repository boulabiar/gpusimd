#include "vulkan_renderer.h"

#include <vulkan/vulkan.h>

#include "vulkan_shader_spv.h"
#include "vulkan_scan_shader_spv.h"

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

  struct ScanPushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t coverageScale;
  };

  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t aaGrid = 0;
  uint32_t edgeCount = 0;
  uint32_t pixelCount = 0;
  uint32_t queueFamilyIndex = 0;
  uint32_t maxDispatchGroupsX = 0;
  uint32_t maxDispatchGroupsY = 0;
  float timestampPeriod = 0.0f;
  VkSubgroupFeatureFlags subgroupFeatures = 0;
  bool subgroupCompute = false;
  bool subgroupSizeControl = false;
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

  VkDescriptorSetLayout scanDescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool scanDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet scanDescriptorSet = VK_NULL_HANDLE;
  VkPipelineLayout scanPipelineLayout = VK_NULL_HANDLE;
  VkShaderModule scanShaderModule = VK_NULL_HANDLE;
  std::array<VkPipeline, 6> scanPipelines{
    VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
    VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
  Buffer scanDeltaBuffer;
  Buffer scanCoverageBuffer;
  Buffer scanPixelBuffer;
  Buffer scanCoverageReadback;
  Buffer scanPixelReadback;
  void* scanCoverageMapping = nullptr;
  void* scanPixelMapping = nullptr;

  ~Impl() {
    if (device != VK_NULL_HANDLE)
      vkDeviceWaitIdle(device);
    destroyScanResources();
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

  void destroyScanResources() noexcept {
    if (scanCoverageMapping) vkUnmapMemory(device, scanCoverageReadback.memory);
    if (scanPixelMapping) vkUnmapMemory(device, scanPixelReadback.memory);
    scanCoverageMapping = nullptr;
    scanPixelMapping = nullptr;
    for (VkPipeline pipeline : scanPipelines)
      if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    scanPipelines.fill(VK_NULL_HANDLE);
    if (scanShaderModule) vkDestroyShaderModule(device, scanShaderModule, nullptr);
    if (scanPipelineLayout) vkDestroyPipelineLayout(device, scanPipelineLayout, nullptr);
    if (scanDescriptorPool) vkDestroyDescriptorPool(device, scanDescriptorPool, nullptr);
    if (scanDescriptorSetLayout)
      vkDestroyDescriptorSetLayout(device, scanDescriptorSetLayout, nullptr);
    scanShaderModule = VK_NULL_HANDLE;
    scanPipelineLayout = VK_NULL_HANDLE;
    scanDescriptorPool = VK_NULL_HANDLE;
    scanDescriptorSetLayout = VK_NULL_HANDLE;
    destroyBuffer(scanPixelReadback);
    destroyBuffer(scanCoverageReadback);
    destroyBuffer(scanPixelBuffer);
    destroyBuffer(scanCoverageBuffer);
    destroyBuffer(scanDeltaBuffer);
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
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT bestSubgroupSizeControl{};
    bestSubgroupSizeControl.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
    uint32_t bestQueue = 0;
    uint32_t bestTimestampBits = 0;

    for (VkPhysicalDevice candidate : devices) {
      VkPhysicalDeviceDriverProperties driver{};
      driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
      VkPhysicalDeviceSubgroupProperties subgroup{};
      subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
      VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subgroupSizeControlProperties{};
      subgroupSizeControlProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
      subgroup.pNext = &subgroupSizeControlProperties;
      subgroupSizeControlProperties.pNext = &driver;
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
        bestSubgroup.pNext = nullptr;
        bestSubgroupSizeControl = subgroupSizeControlProperties;
        bestSubgroupSizeControl.pNext = nullptr;
        bestDriver = driver;
        bestQueue = queueIndex;
        bestTimestampBits = queues[queueIndex].timestampValidBits;
      }
    }
    if (physicalDevice == VK_NULL_HANDLE)
      vkFail("no compute-capable physical device found");

    queueFamilyIndex = bestQueue;
    maxDispatchGroupsX = bestProperties.limits.maxComputeWorkGroupCount[0];
    maxDispatchGroupsY = bestProperties.limits.maxComputeWorkGroupCount[1];
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
    subgroupFeatures = bestSubgroup.supportedOperations;
    subgroupCompute = (bestSubgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeControlFeatures{};
    subgroupSizeControlFeatures.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &subgroupSizeControlFeatures;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
    subgroupSizeControl = subgroupSizeControlFeatures.subgroupSizeControl &&
      subgroupSizeControlFeatures.computeFullSubgroups &&
      (bestSubgroupSizeControl.requiredSubgroupSizeStages &
       VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
      info.subgroupSize >= bestSubgroupSizeControl.minSubgroupSize &&
      info.subgroupSize <= bestSubgroupSizeControl.maxSubgroupSize;
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
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeFeatures{};
    subgroupSizeFeatures.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    subgroupSizeFeatures.subgroupSizeControl = subgroupSizeControl;
    subgroupSizeFeatures.computeFullSubgroups = subgroupSizeControl;
    const char* subgroupSizeExtension = VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
    const VkDeviceCreateInfo deviceCreate{
      VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      subgroupSizeControl ? &subgroupSizeFeatures : nullptr, 0,
      1, &queueCreate, 0, nullptr,
      subgroupSizeControl ? 1u : 0u,
      subgroupSizeControl ? &subgroupSizeExtension : nullptr,
      nullptr
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
    const uint32_t groupsX = uint32_t(std::min<uint64_t>(
      groupCount, maxDispatchGroupsX));
    const uint64_t groupsY64 = (groupCount + groupsX - 1u) / groupsX;
    if (groupsY64 > maxDispatchGroupsY)
      vkFail("dispatch exceeds two-dimensional compute workgroup limits");
    vkCmdDispatch(commandBuffer, groupsX, uint32_t(groupsY64), 1);
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

  void initializeScanResources() {
    if (scanDescriptorSetLayout)
      return;
    const VkDeviceSize bytes = VkDeviceSize(pixelCount) * sizeof(uint32_t);
    scanDeltaBuffer = createBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    scanCoverageBuffer = createBuffer(bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    scanPixelBuffer = createBuffer(bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    scanCoverageReadback = createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    scanPixelReadback = createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkCheck(vkMapMemory(device, scanCoverageReadback.memory, 0, bytes, 0, &scanCoverageMapping),
            "vkMapMemory(scan coverage)");
    vkCheck(vkMapMemory(device, scanPixelReadback.memory, 0, bytes, 0, &scanPixelMapping),
            "vkMapMemory(scan pixels)");

    constexpr std::array<VkDescriptorSetLayoutBinding, 3> bindings = {{
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    }};
    const VkDescriptorSetLayoutCreateInfo layoutCreate{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
      uint32_t(bindings.size()), bindings.data()
    };
    vkCheck(vkCreateDescriptorSetLayout(
      device, &layoutCreate, nullptr, &scanDescriptorSetLayout),
      "vkCreateDescriptorSetLayout(scan)");
    const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    const VkDescriptorPoolCreateInfo poolCreate{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &poolSize
    };
    vkCheck(vkCreateDescriptorPool(device, &poolCreate, nullptr, &scanDescriptorPool),
            "vkCreateDescriptorPool(scan)");
    const VkDescriptorSetAllocateInfo allocate{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
      scanDescriptorPool, 1, &scanDescriptorSetLayout
    };
    vkCheck(vkAllocateDescriptorSets(device, &allocate, &scanDescriptorSet),
            "vkAllocateDescriptorSets(scan)");
    const std::array<VkDescriptorBufferInfo, 3> bufferInfos = {{
      {scanDeltaBuffer.handle, 0, bytes},
      {scanCoverageBuffer.handle, 0, bytes},
      {scanPixelBuffer.handle, 0, bytes}
    }};
    const std::array<VkWriteDescriptorSet, 3> writes = {{
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scanDescriptorSet, 0, 0, 1,
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfos[0], nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scanDescriptorSet, 1, 0, 1,
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfos[1], nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scanDescriptorSet, 2, 0, 1,
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfos[2], nullptr}
    }};
    vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), 0, nullptr);

    const VkPushConstantRange pushRange{
      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ScanPushConstants)};
    const VkPipelineLayoutCreateInfo pipelineLayoutCreate{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
      1, &scanDescriptorSetLayout, 1, &pushRange
    };
    vkCheck(vkCreatePipelineLayout(
      device, &pipelineLayoutCreate, nullptr, &scanPipelineLayout),
      "vkCreatePipelineLayout(scan)");
    const VkShaderModuleCreateInfo shaderCreate{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
      sizeof(gpusimdVulkanScanShaderSpv), gpusimdVulkanScanShaderSpv
    };
    vkCheck(vkCreateShaderModule(device, &shaderCreate, nullptr, &scanShaderModule),
            "vkCreateShaderModule(scan)");
    for (uint32_t algorithm = 0; algorithm < 3; ++algorithm) {
      for (uint32_t paint = 0; paint < 2; ++paint)
        scanPipelines[algorithm * 2u + paint] = createScanPipeline(algorithm, paint);
    }
  }

  VkPipeline createScanPipeline(uint32_t algorithm, uint32_t paint) const {
    if (info.subgroupSize == 0u || info.subgroupSize > 64u)
      vkFail("coverage scan requires a native subgroup size from 1 to 64");
    const std::array<uint32_t, 3> values = {
      algorithm, paint, info.subgroupSize};
    constexpr std::array<VkSpecializationMapEntry, 3> entries = {{
      {0, 0, sizeof(uint32_t)},
      {1, sizeof(uint32_t), sizeof(uint32_t)},
      {2, 2u * sizeof(uint32_t), sizeof(uint32_t)}
    }};
    const VkSpecializationInfo specialization{
      uint32_t(entries.size()), entries.data(), sizeof(values), values.data()
    };
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT requiredSubgroup{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
      nullptr, info.subgroupSize
    };
    const bool requireSubgroup = algorithm ==
      uint32_t(CoverageScanAlgorithm::kSubgroup);
    const VkPipelineShaderStageCreateInfo stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      requireSubgroup ? &requiredSubgroup : nullptr,
      requireSubgroup
        ? VkPipelineShaderStageCreateFlags(
            VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT)
        : VkPipelineShaderStageCreateFlags(0),
      VK_SHADER_STAGE_COMPUTE_BIT, scanShaderModule, "main", &specialization
    };
    const VkComputePipelineCreateInfo createInfo{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
      stage, scanPipelineLayout, VK_NULL_HANDLE, -1
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCheck(vkCreateComputePipelines(
      device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline),
      "vkCreateComputePipelines(scan)");
    return pipeline;
  }

  void uploadScanDeltas(std::span<const int32_t> deltas) {
    if (deltas.size() != pixelCount)
      vkFail("coverage delta count does not match image dimensions");
    void* mapping = nullptr;
    vkCheck(vkMapMemory(device, scanDeltaBuffer.memory, 0, scanDeltaBuffer.size, 0, &mapping),
            "vkMapMemory(scan deltas)");
    std::memcpy(mapping, deltas.data(), size_t(scanDeltaBuffer.size));
    vkUnmapMemory(device, scanDeltaBuffer.memory);
  }

  void recordCoverageScan(
    CoverageScanAlgorithm algorithm,
    bool paint,
    uint32_t coverageScale,
    bool copyToHost) {

    const uint32_t algorithmIndex = uint32_t(algorithm);
    if (algorithmIndex > uint32_t(CoverageScanAlgorithm::kSubgroup))
      vkFail("unknown coverage scan algorithm");
    vkCheck(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer(scan)");
    const VkCommandBufferBeginInfo begin{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr
    };
    vkCheck(vkBeginCommandBuffer(commandBuffer, &begin), "vkBeginCommandBuffer(scan)");
    vkCmdResetQueryPool(commandBuffer, queryPool, 0, 2);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      scanPipelines[algorithmIndex * 2u + uint32_t(paint)]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
      scanPipelineLayout, 0, 1, &scanDescriptorSet, 0, nullptr);
    const ScanPushConstants push{width, height, coverageScale};
    vkCmdPushConstants(commandBuffer, scanPipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    const uint32_t groups = algorithm == CoverageScanAlgorithm::kSerialized
      ? (height + info.subgroupSize - 1u) / info.subgroupSize
      : height;
    if (groups > maxDispatchGroupsX)
      vkFail("coverage scan exceeds maxComputeWorkGroupCount[0]");
    vkCmdDispatch(commandBuffer, groups, 1, 1);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);

    if (copyToHost) {
      std::array<VkBufferMemoryBarrier, 2> barriers = {{
        {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
         scanCoverageBuffer.handle, 0, scanCoverageBuffer.size},
        {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
         scanPixelBuffer.handle, 0, scanPixelBuffer.size}
      }};
      const uint32_t barrierCount = paint ? 2u : 1u;
      vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, barrierCount, barriers.data(), 0, nullptr);
      const VkBufferCopy coverageCopy{0, 0, scanCoverageBuffer.size};
      vkCmdCopyBuffer(commandBuffer, scanCoverageBuffer.handle,
        scanCoverageReadback.handle, 1, &coverageCopy);
      if (paint) {
        const VkBufferCopy pixelCopy{0, 0, scanPixelBuffer.size};
        vkCmdCopyBuffer(commandBuffer, scanPixelBuffer.handle,
          scanPixelReadback.handle, 1, &pixelCopy);
      }
    }
    vkCheck(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer(scan)");
  }

  VulkanCoverageScanResult runCoverageScan(
    std::span<const int32_t> deltas,
    uint32_t coverageScale,
    CoverageScanAlgorithm algorithm,
    bool paint,
    uint32_t warmup,
    uint32_t iterations) {

    if (algorithm == CoverageScanAlgorithm::kSubgroup &&
        (!subgroupCompute || !subgroupSizeControl ||
         (subgroupFeatures & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) == 0 ||
         (subgroupFeatures & VK_SUBGROUP_FEATURE_BALLOT_BIT) == 0 ||
         (subgroupFeatures & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) == 0))
      vkFail("required subgroup scan/shuffle operations are unavailable in compute shaders");
    initializeScanResources();
    uploadScanDeltas(deltas);
    VulkanCoverageScanResult result;
    result.timestampMilliseconds.reserve(iterations);
    result.synchronizedMilliseconds.reserve(iterations);
    for (uint32_t i = 0; i < warmup; ++i) {
      recordCoverageScan(algorithm, paint, coverageScale, false);
      submitAndWait();
    }
    for (uint32_t i = 0; i < iterations; ++i) {
      recordCoverageScan(algorithm, paint, coverageScale, false);
      result.synchronizedMilliseconds.push_back(submitAndWait());
      result.timestampMilliseconds.push_back(readTimestampMilliseconds());
    }
    recordCoverageScan(algorithm, paint, coverageScale, true);
    submitAndWait();

    result.coverage.resize(pixelCount);
    std::memcpy(result.coverage.data(), scanCoverageMapping, size_t(scanCoverageReadback.size));
    if (paint) {
      result.pixels.resize(pixelCount);
      std::memcpy(result.pixels.data(), scanPixelMapping, size_t(scanPixelReadback.size));
    }
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

VulkanCoverageScanResult VulkanRenderer::runCoverageScan(
  std::span<const int32_t> deltas,
  uint32_t coverageScale,
  CoverageScanAlgorithm algorithm,
  bool paint,
  uint32_t warmup,
  uint32_t iterations) {
  return impl_->runCoverageScan(
    deltas, coverageScale, algorithm, paint, warmup, iterations);
}

} // namespace gpusimd
