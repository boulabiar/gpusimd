cmake_minimum_required(VERSION 3.20)

project(gpusimd_vector_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)

add_executable(gpusimd_vector_bench
  src/main.cpp
)

target_link_libraries(gpusimd_vector_bench PRIVATE
  Threads::Threads
  OpenGL::OpenGL
  OpenGL::EGL
)

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
    NAME cpu_correctness_smoke
    COMMAND gpusimd_vector_bench
      --width 64 --height 64 --aa 2 --iterations 1 --threads 2
      --no-gpu --no-images
  )
  add_test(
    NAME csv_schema_smoke
    COMMAND ${CMAKE_COMMAND}
      -DBENCHMARK=$<TARGET_FILE:gpusimd_vector_bench>
      -DCSV_PATH=${CMAKE_CURRENT_BINARY_DIR}/test-results/smoke.csv
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate_csv.cmake
  )
endif()
