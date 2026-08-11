cmake_minimum_required(VERSION 3.20)

project(gpusimd_vector_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)

option(GPUSIMD_ENABLE_VULKAN "Build the optional Vulkan compute backend" ON)
if(GPUSIMD_ENABLE_VULKAN)
  find_package(Vulkan QUIET)
endif()

add_executable(gpusimd_vector_bench
  src/coverage_scan.cpp
  src/coverage_scan.h
  src/main.cpp
)

target_link_libraries(gpusimd_vector_bench PRIVATE
  Threads::Threads
  OpenGL::OpenGL
  OpenGL::EGL
)

if(Vulkan_FOUND)
  target_sources(gpusimd_vector_bench PRIVATE
    src/vulkan_renderer.cpp
    src/vulkan_renderer.h
    src/vulkan_scan_shader.comp
    src/vulkan_scan_shader_spv.h
    src/vulkan_shader.comp
    src/vulkan_shader_spv.h
  )
  target_link_libraries(gpusimd_vector_bench PRIVATE Vulkan::Vulkan)
  target_compile_definitions(gpusimd_vector_bench PRIVATE GPUSIMD_HAS_VULKAN=1)
  message(STATUS "Vulkan compute backend enabled")

  find_program(GPUSIMD_GLSLANG_VALIDATOR glslangValidator)
  if(GPUSIMD_GLSLANG_VALIDATOR)
    add_custom_target(regenerate_vulkan_shader
      COMMAND ${GPUSIMD_GLSLANG_VALIDATOR}
        -V --target-env vulkan1.2
        --vn gpusimdVulkanShaderSpv
        -o ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_shader_spv.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_shader.comp
      COMMAND ${GPUSIMD_GLSLANG_VALIDATOR}
        -V --target-env vulkan1.2
        --vn gpusimdVulkanScanShaderSpv
        -o ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_scan_shader_spv.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_scan_shader.comp
      COMMENT "Regenerating embedded Vulkan SPIR-V"
      VERBATIM
    )
  endif()
elseif(GPUSIMD_ENABLE_VULKAN)
  message(STATUS "Vulkan loader/headers not found; Vulkan backend disabled")
endif()

target_compile_options(gpusimd_vector_bench PRIVATE
  -O3
  -Wall
  -Wextra
  -Wpedantic
  -mavx2
  -fno-tree-vectorize
  -ffp-contract=off
)

target_compile_definitions(gpusimd_vector_bench PRIVATE
  GPUSIMD_BUILD_TYPE="$<CONFIG>"
)

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME adaptive_cpu_correctness_smoke
    COMMAND gpusimd_vector_bench
      --width 64 --height 64 --aa 2 --iterations 1 --threads 2
      --no-gpu --no-images
  )
  add_test(
    NAME fixed_cpu_correctness_smoke
    COMMAND gpusimd_vector_bench
      --width 64 --height 64 --aa 2 --iterations 1 --threads 2
      --curve-segments 4 --no-gpu --no-images
  )
  add_test(
    NAME coverage_scan_cpu_correctness_smoke
    COMMAND gpusimd_vector_bench
      --width 70 --height 37 --aa 1 --iterations 1 --threads 2
      --curve-segments 4 --coverage-scan --no-gpu --no-images
  )
  add_test(
    NAME analytic_cells_cpu_correctness_smoke
    COMMAND gpusimd_vector_bench
      --width 70 --height 37 --aa 2 --iterations 1 --threads 2
      --curve-segments 4 --analytic-cells --no-gpu --no-images
  )
  add_test(
    NAME analytic_tiles_cpu_correctness_smoke
    COMMAND gpusimd_vector_bench
      --width 70 --height 37 --aa 2 --iterations 1 --threads 2
      --curve-segments 4 --shapes 2 --analytic-tiles --no-gpu --no-images
  )
  add_test(
    NAME csv_schema_smoke
    COMMAND ${CMAKE_COMMAND}
      -DBENCHMARK=$<TARGET_FILE:gpusimd_vector_bench>
      -DCSV_PATH=${CMAKE_CURRENT_BINARY_DIR}/test-results/smoke.csv
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate_csv.cmake
  )
endif()
