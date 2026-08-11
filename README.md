# GPU-distributed SIMD vector rasterization experiment

This is a deliberately small C++ experiment inspired by VectorWare's
"Rust SIMD on the GPU" article. It asks a performance question, not a language
question:

> Does treating GPU invocations as the lanes of one logical SIMD vector work
> well for a recognizable vector-graphics rasterization workload?

The measured answer is yes, conditionally. Real subgroup shuffle scans are a
good implementation of analytic coverage, but they are not universally faster
than shared memory or all-core CPU SIMD in isolation. The practical win appears
when coverage, paint, and multiple source-over draws remain GPU-resident. In the
final four-draw 2048-square case, independently measured construction, upload,
render, and final-readback medians give a 3.16x end-stage win on the Intel UHD
620 system and 2.66x on the Radeon 8060S system versus persistent all-core AVX2.
See [RESULTS.md](RESULTS.md) for the stage-by-stage qualifications.

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

The optional coverage-scan experiment adds scalar and AVX2 CPU scans plus three
Vulkan controls: one serialized invocation per row, a workgroup shared-memory
scan, and an article-style subgroup scan built from explicit `shuffleUp`
stages. The Vulkan subgroup pipeline requires a full native subgroup through
subgroup-size control; merely querying the nominal subgroup width proved
insufficient on real Intel hardware.

The CPU scan is measured both on one AVX2 thread and with independent rows
distributed across all requested CPU threads. This is the fair control for a
GPU subgroup: CPU execution combines MIMD cores with eight-lane AVX2, while the
GPU combines many 32- or 64-lane subgroups.

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
./build/gpusimd_vector_bench --flatness 0.25
./build/gpusimd_vector_bench --width 1024 --height 1024 --aa 4 \
  --warmup 3 --iterations 7 --no-images --csv results/baseline.csv
./build/gpusimd_vector_bench --sweep-sizes 256,512,1024 \
  --sweep-threads 1,8 --sweep-curve-segments 12,48 \
  --warmup 3 --iterations 7 --no-images --csv results/scaling.csv
./build/gpusimd_vector_bench --no-gpu
./build/gpusimd_vector_bench --vulkan --no-opengl
./build/gpusimd_vector_bench --coverage-scan --vulkan --no-opengl \
  --width 1024 --height 1024 --curve-segments 12 --no-images
./build/gpusimd_vector_bench --analytic-cells --vulkan --no-opengl \
  --width 1024 --height 1024 --curve-segments 12 --no-images
./build/gpusimd_vector_bench --analytic-tiles --vulkan --no-opengl \
  --width 2048 --height 2048 --curve-segments 12 --no-images
./build/gpusimd_vector_bench --help
```

Vulkan is enabled automatically when CMake finds its headers and loader. Use
`-DGPUSIMD_ENABLE_VULKAN=OFF` to verify an OpenGL/CPU-only build. `--vulkan`
runs Vulkan in addition to OpenGL; combine it with `--no-opengl` for a
Vulkan-only run. The checked-in SPIR-V lets systems build without a shader
compiler. Developers with `glslangValidator` can regenerate it after editing
the shader with:

```sh
cmake --build build --target regenerate_vulkan_shader
```

Sweep lists form a Cartesian product. `--sweep-sizes` uses square images;
`--width` and `--height` remain available for a single rectangular run.
By default, cubic curves are recursively subdivided with de Casteljau's
algorithm until their screen-space error is at most 0.25 pixels. The circular
hole derives its segment count from the same tolerance. `--flatness F` changes
that visual-quality threshold.

`--curve-segments N` explicitly switches to fixed subdivision for synthetic
complexity experiments and reproducible historical baselines: four cubic edges
and the circular hole produce approximately `6N` flattened edges per shape.
Fixed subdivision deliberately does not scale with resolution and can show
visible facets at low segment counts.

`--coverage-scan` builds signed fixed-point coverage deltas from those real
flattened edges, then benchmarks prefix scan alone and prefix scan plus paint.
`--scan-y-samples N` controls the vertical sampling used to construct this
bridge workload (default 64). The scan arithmetic itself is exact integer
arithmetic.

`--analytic-cells` replaces that sampled input with a scalar reference that
constructs Blend2D-style combined cover/area cell deltas using 8-bit subpixel
coordinates. It resolves even-odd coverage and benchmarks the same scalar,
AVX2, all-core AVX2, serialized Vulkan, shared-memory Vulkan, and subgroup
Vulkan paths. Cell construction is warmed up and recorded separately in CSV as
`analytic_cell_construction`; it is not hidden inside a scan-kernel time.
Non-zero and even-odd resolution are both implemented and covered by analytic
invariants, while the heart command currently selects even-odd to preserve its
hole. This remains a dense scanline reference: tile binning, sparse tile-local
storage, GPU cell accumulation, and multi-draw composition belong to the next
renderer milestone.

`--analytic-tiles` exercises the next intermediate representation. Edges are
binned into 16-pixel-high bands, combined cover/area cells accumulate in sparse
64x16 tiles, and only touched tiles are allocated. The program reports active
tiles, binned edge references, compact versus dense bytes, tile construction,
and dense materialization separately. Scalar and AVX2 CPU paths also scan the
compact tiles directly, carrying signed coverage through inactive tiles without
constructing dense deltas. Vulkan runs both controls: a materialized dense input
and the compact tile values plus tile lookup consumed directly by every scan
kernel. Its input buffers are sized to the actual representation, and upload
time and byte count are reported separately. All paths preserve exact analytic
coverage and paint output.

With `--analytic-tiles --shapes N` and `N > 1`, an additional renderer case
builds `N` deliberately overlapping paths with independent gradient/solid
paints. The Vulkan path packs every compact draw into one upload, issues all
subgroup scan/paint dispatches in one command buffer, source-over composites
into a device-local target, and reads back only the final image. A per-draw
submit-and-wait sequence is measured as the synchronization control. The CPU
scalar and all-core AVX2 composition references must match both Vulkan modes
exactly.

For the original point-in-path baseline, `--shapes N` places independent,
non-overlapping hearts in a grid. Start with a small sweep because that scalar
reference deliberately tests every sample against every edge.

The program prints median, best, and p90 timings, speedups relative to scalar
CPU, synchronized GPU dispatch and dispatch-plus-readback times, and correctness
differences. A synchronized OpenGL dispatch includes command submission and
`glFinish()` overhead. Vulkan additionally reports device timestamps around the
compute dispatch, independently sampled input upload, plus queue-submit/fence
and readback durations. Warm-ups, pipeline/shader compilation, context/device
creation, buffer allocation, CSV writing, and image writing are outside timed
regions.

CSV schema version 3 contains host, OS, CPU, compiler, build type, API,
vendor/renderer, driver version, hardware/software classification, subgroup
size and operation flags, timestamp precision, flattening mode and tolerance,
full workload configuration, timing statistics, throughput, speedup, and
correctness data. Use
`--require-hardware-gpu` on a benchmark machine to reject llvmpipe, softpipe,
SwiftShader, and other known software renderers. Without that option, software
GPU rows are retained but marked `gpu_hardware=0`.

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

For the reproducible subgroup experiment used in `RESULTS.md`:

```sh
./build/gpusimd_vector_bench --sweep-sizes 512,1024,2048 --aa 1 \
  --curve-segments 12 --warmup 3 --iterations 7 --threads 8 \
  --coverage-scan --vulkan --no-opengl --no-images \
  --require-hardware-gpu --csv results/coverage-scan.csv
```

For the true analytic-cell follow-up:

```sh
./build/gpusimd_vector_bench --sweep-sizes 512,1024,2048 --aa 1 \
  --curve-segments 12 --warmup 3 --iterations 7 --threads 8 \
  --analytic-cells --vulkan --no-opengl --no-images \
  --require-hardware-gpu --csv results/analytic-cells.csv
```
