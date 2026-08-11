# GPU-distributed SIMD vector rasterization experiment

This is a deliberately small C++ experiment inspired by VectorWare's
"Rust SIMD on the GPU" article. It asks a performance question, not a language
question:

> Does treating GPU invocations as the lanes of one logical SIMD vector work
> well for a recognizable vector-graphics rasterization workload?

The program renders a filled cubic Bezier heart with an even-odd hole. Cubics
are flattened into line edges, coverage is computed with configurable NxN
supersampling, the fill uses a two-dimensional gradient, and the result is
source-over composited onto a checkerboard.

It benchmarks five execution strategies:

1. scalar C++ on one CPU thread;
2. the same logical lane program using eight AVX2 lanes on one CPU thread;
3. AVX2 across all requested CPU threads;
4. OpenGL compute with one pixel per invocation, which is the GPU-distributed
   lane mapping proposed by the article;
5. OpenGL compute with eight pixels serialized inside each invocation, as a
   control representing a CPU-like vector kept inside each GPU thread.

When `GL_KHR_shader_subgroup` is available, the program queries the hardware
subgroup size and uses that many invocations as its logical GPU SIMD width. If
the extension is unavailable, it uses groups of 32 independent invocations and
reports that the exact subgroup mapping could not be observed.

The GPU shader is embedded in the C++ executable and compiled by the OpenGL
driver. This validates the execution model and its performance, but it does not
validate VectorWare's compiler or its claim of source compatibility. A custom
C++ compiler/backend would be required to make the exact C++ SIMD source itself
target GPU subgroup lanes.

## Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/gpusimd_vector_bench
```

Useful options:

```sh
./build/gpusimd_vector_bench --width 1024 --height 1024 --aa 4 \
  --warmup 3 --iterations 7 --no-images --csv results/baseline.csv
./build/gpusimd_vector_bench --sweep-sizes 256,512,1024 \
  --sweep-threads 1,8 --sweep-curve-segments 12,48 \
  --warmup 3 --iterations 7 --no-images --csv results/scaling.csv
./build/gpusimd_vector_bench --no-gpu
./build/gpusimd_vector_bench --help
```

Sweep lists form a Cartesian product. `--sweep-sizes` uses square images;
`--width` and `--height` remain available for a single rectangular run.
`--curve-segments N` changes geometric complexity: four cubic edges and the
circular hole produce approximately `6N` flattened edges per shape.
`--shapes N` places independent, non-overlapping hearts in a grid. Start with a
small sweep because the scalar reference deliberately tests every sample
against every edge.

The program prints median, best, and p90 timings, speedups relative to scalar
CPU, synchronized GPU dispatch and dispatch-plus-readback times, and correctness
differences. A synchronized dispatch includes command submission and
`glFinish()` overhead; this deliberately conservative measurement avoids
unreliable timer-query results on some Mesa drivers. Warm-ups, shader
compilation, context creation, buffer allocation, CSV writing, and image writing
are outside timed regions.

CSV output contains host, OS, CPU, compiler, build type, OpenGL vendor/renderer,
driver version, hardware/software classification, subgroup information, full
workload configuration, timing statistics, throughput, speedup, and correctness
data. Use `--require-hardware-gpu` on a benchmark machine to reject llvmpipe,
softpipe, SwiftShader, and other known software renderers. Without that option,
software GPU rows are retained but marked `gpu_hardware=0`.

PPM images are written to `output/` unless `--no-images` is used. A multi-case
sweep puts each configuration in a separately named subdirectory.

An actual GPU is required for useful GPU conclusions. Mesa's `llvmpipe` is a
CPU software implementation of OpenGL; results that name it as the renderer
must not be interpreted as GPU performance.

## Portable validation

Run the automated CPU correctness and CSV-schema checks with:

```sh
ctest --test-dir build --output-on-failure
```

For an initial run on another computer, use the same Release build and begin
with one known workload before launching a sweep:

```sh
./build/gpusimd_vector_bench --width 1024 --height 1024 --aa 4 \
  --curve-segments 12 --shapes 1 --warmup 3 --iterations 7 \
  --threads 8 --no-images --require-hardware-gpu \
  --csv results/second-machine-1024.csv
```

Adjust `--threads` to that CPU's hardware-thread count. The CSV records both the
requested count and the number actually used.
