if(NOT DEFINED BENCHMARK OR NOT DEFINED CSV_PATH)
  message(FATAL_ERROR "BENCHMARK and CSV_PATH are required")
endif()

execute_process(
  COMMAND "${BENCHMARK}"
    --width 32 --height 24 --aa 1 --warmup 1 --iterations 1 --threads 2
    --flatness 0.25 --shapes 1 --no-gpu --no-images --csv "${CSV_PATH}"
  RESULT_VARIABLE benchmark_result
  OUTPUT_VARIABLE benchmark_output
  ERROR_VARIABLE benchmark_error
)

if(NOT benchmark_result EQUAL 0)
  message(FATAL_ERROR
    "benchmark failed (${benchmark_result})\n${benchmark_output}\n${benchmark_error}")
endif()

file(READ "${CSV_PATH}" csv)
set(expected_header
  "schema_version,timestamp_utc,hostname,os,cpu,cpu_logical_threads,compiler,build_type,api,gpu_vendor,gpu_renderer,gpu_version,gpu_hardware,subgroup_size,workgroup_size,width,height,aa_grid,flattening_mode,curve_segments,flatness_pixels,shape_count,edge_count,requested_cpu_threads,cpu_threads_used,warmup,iterations,backend,timing_scope,median_ms,minimum_ms,p90_ms,megapixels_per_second,speedup_vs_scalar,differing_pixels,max_channel_error,mean_absolute_channel_error")
string(FIND "${csv}" "${expected_header}\n" header_position)
if(NOT header_position EQUAL 0)
  message(FATAL_ERROR "CSV header does not match schema")
endif()

string(FIND "${csv}" ",adaptive,,0.250000," adaptive_position)
if(adaptive_position EQUAL -1)
  message(FATAL_ERROR "CSV does not identify adaptive 0.25-pixel flattening")
endif()

foreach(backend IN ITEMS cpu_scalar cpu_avx2 cpu_avx2_threaded)
  string(FIND "${csv}" ",${backend},render," backend_position)
  if(backend_position EQUAL -1)
    message(FATAL_ERROR "CSV is missing ${backend} row")
  endif()
endforeach()

string(REGEX MATCHALL "\n2,[^\n]+" result_rows "${csv}")
list(LENGTH result_rows result_count)
if(NOT result_count EQUAL 3)
  message(FATAL_ERROR "expected 3 CPU result rows, found ${result_count}")
endif()
