# GPU SIMD vector-rendering experiment plan

## Objective

Determine whether expressing vector-rendering SIMD lanes as GPU subgroup lanes
provides a practical advantage over CPU SIMD and conventional GPU execution.
The project must distinguish three questions:

1. Does the mapping execute correctly with real subgroup operations?
2. Does a subgroup prefix scan outperform serialized and shared-memory GPU
   alternatives?
3. Does that advantage survive inside an end-to-end analytic vector renderer?

The current brute-force point-in-path renderer remains the baseline. All
benchmark and renderer implementation stays in C++/CMake; GPU programs may be
embedded or compiled as part of the C++ build without external benchmark
scripts.

## Milestone 0: preserve the baseline

Status: complete.

Deliverables:

- scalar, AVX2, threaded AVX2, distributed GPU, and packed GPU variants;
- correctness comparison and rendered images;
- initial Intel UHD Graphics 620 measurements;
- source, build files, results, and documentation staged in Git.

The baseline should not be redesigned while the measurement infrastructure is
being added. This gives the second computer a stable reference revision.

## Milestone 1: portable benchmark harness

Status: complete.

Add the following capabilities to the existing executable:

- `--warmup N` separately from measured iterations;
- `--csv FILE` with one row per backend and configuration;
- machine-readable CPU, GPU, driver, API, compiler, subgroup, and build data;
- configurable edge complexity and number of independently rendered shapes;
- a sweep mode for image sizes, CPU thread counts, and complexity levels;
- an option to disable image output during timing;
- explicit GPU-resident, synchronized-dispatch, and readback measurements;
- median, minimum, and percentile timing instead of a single aggregate;
- clear rejection of software renderers such as llvmpipe from hardware result
  tables.

Keep a single-configuration mode so correctness failures remain easy to debug.

Acceptance criteria:

- existing default output and correctness checks still work;
- CTest covers a CPU-only smoke run and CSV schema validation;
- identical commands work without source edits on both computers;
- every result row contains enough metadata to identify its machine and
  workload;
- timed regions exclude shader compilation, context creation, image writing,
  and warm-up.

## Milestone 2: unchanged cross-hardware baseline

Status: representative cross-hardware baseline complete. The larger scaling
matrix remains scheduled for Milestone 6, after the analytic renderer exists.

Run the current algorithm through the new harness before adding Vulkan or a new
rasterizer. Use at least the current Intel integrated GPU and the second
computer.

Test suites:

- resolution: 256, 512, 1024, 2048, and 4096 pixels square where practical;
- complexity: approximately 72, 288, 1152, and 4608 flattened edges;
- batching: 1, 16, 64, and 256 independent shapes;
- CPU scaling: 1 thread, physical-core count, and hardware-thread count;
- output mode: GPU-resident synchronization and dispatch plus readback.

The initial decision checkpoint used the unchanged 72-edge workload at 512 and
1024 pixels square on the Intel UHD 620 and AMD Radeon 8060S. That was enough to
show a CPU/GPU crossover and expose a real 64-lane subgroup. Running the full
Cartesian suite on the brute-force renderer now would spend substantial time on
an algorithm that Milestones 4 and 5 replace, so the remaining axes are retained
for the final comparative matrix.

Use at least three warm-ups and seven measured iterations. Store raw CSV files
under a machine-specific ignored results directory, then summarize confirmed
measurements in `RESULTS.md`.

Decision produced by this milestone:

- establish whether a stronger GPU changes only the crossover point or also
  changes the distributed-versus-packed relationship;
- determine representative small, medium, and large workloads for later
  milestones;
- avoid interpreting more pixels or curves as evidence for subgroup SIMD by
  themselves.

## Visual-quality prerequisite: adaptive curve flattening

Status: complete.

The renderer now uses recursive de Casteljau subdivision with a 0.25-pixel
screen-space error threshold by default. The circular subpath uses the same
tolerance. Fixed `--curve-segments N` subdivision remains available for the
published 72-edge baseline and controlled synthetic complexity sweeps. CSV
schema version 2 records which mode produced each result so adaptive and fixed
workloads cannot be compared accidentally.

## Milestone 3: optional Vulkan subgroup backend

Retain OpenGL as a baseline and add Vulkan as an optional CMake component.
Vulkan is preferred for the subgroup experiment because it exposes subgroup
capabilities and timestamp support explicitly.

Implementation work:

- instance, device, queue, command-pool, buffer, image, and descriptor lifetime
  wrappers;
- physical-device and queue selection with diagnostic output;
- `VkPhysicalDeviceSubgroupProperties` reporting, including subgroup size and
  supported arithmetic, shuffle, ballot, and basic operations;
- compute pipelines matching the existing distributed and packed workloads;
- timestamp-query timing with host synchronized wall-clock timing retained as
  a cross-check;
- reusable upload, GPU-resident render, and readback paths;
- graceful build-time and run-time fallback when Vulkan or the required
  subgroup feature is unavailable.

Acceptance criteria:

- Vulkan output passes the same reference-image comparison as OpenGL;
- reported subgroup width and supported operation flags come from the driver;
- GPU timestamp and synchronized wall-clock measurements are both recorded;
- software and unsupported devices are identified rather than silently used;
- OpenGL-only and CPU-only builds continue to work.

## Milestone 4: analytic coverage prefix-scan experiment

Build a focused bridge between the current renderer and Blend2D's analytic
rasterization model. Start with real vector edges and integer cover/area cell
data, but isolate the scan from edge binning so its cost and benefit are
measurable.

Pipeline:

1. Flatten the same cubic paths into edges.
2. Produce signed analytic cover and area contributions with a scalar C++
   reference implementation.
3. Store scanlines in padded tiles whose horizontal lanes map to consecutive
   pixels.
4. Propagate cover across each row and resolve cover/area into pixel alpha.
5. Paint and composite with the same output conventions as the baseline.

Implement these scan variants:

- scalar CPU reference;
- AVX2 CPU implementation;
- GPU scan serialized inside one invocation;
- GPU workgroup scan using shared memory;
- GPU subgroup inclusive scan using real subgroup arithmetic or shuffle
  operations;
- subgroup scan with explicit carry propagation across subgroup boundaries.

Use integer accumulation wherever the analytic model permits it. This separates
algorithmic errors from floating-point contraction at path boundaries.

Acceptance criteria:

- empty, solid, winding, even-odd, self-intersecting, narrow, and tile-boundary
  test paths match the scalar reference;
- no scanline seam appears at subgroup or workgroup boundaries;
- the report includes scan-only throughput and full scan-plus-paint time;
- the subgroup variant is compared with both serialized and shared-memory
  controls on the same device;
- the experiment reports a negative result honestly if subgroup operations do
  not improve the measured kernel.

This milestone is the direct test of the VectorWare article's distinctive
cross-lane execution claim.

## Milestone 5: tiled analytic vector renderer

Integrate the scan into a useful renderer rather than scaling the existing
brute-force point-in-path loop.

Stages:

1. Bin flattened edges into fixed-size tiles.
2. Accumulate tile-local cover and area cells.
3. Run horizontal subgroup scans and resolve analytic coverage.
4. Apply solid and linear-gradient paints.
5. Composite into a GPU-resident target.

Initially, perform path parsing and curve flattening on the CPU. Measure CPU
edge binning as a separate stage, then add GPU binning only if it is an observed
bottleneck. This keeps the subgroup hypothesis independently testable.

Required renderer cases:

- non-zero and even-odd fills;
- multiple paths and independent paints;
- paths crossing tile and subgroup boundaries;
- opaque and source-over composition;
- GPU-resident chaining of several draws before optional readback.

Acceptance criteria:

- image comparisons pass the analytic reference suite;
- stage timings identify flattening, binning, accumulation, scan/paint, and
  readback independently;
- no per-pixel loop over every path edge remains in the analytic renderer;
- batched draws do not require a CPU/GPU synchronization after every path;
- memory consumption and temporary-buffer sizes are reported.

## Milestone 6: final performance matrix

Measure three algorithms, not merely three APIs:

- current brute-force point-in-path baseline;
- analytic CPU scalar/AVX2 renderer;
- analytic GPU renderer with each scan strategy.

For each available computer, measure:

- small icons and large render targets;
- low and high edge complexity;
- single-path latency and batched throughput;
- one CPU thread and all available physical cores;
- GPU-resident output and output readback;
- subgroup, shared-memory, and serialized GPU scans.

Report absolute time, megapixels per second, edges processed per second,
speedup, output error, and the crossover workload. Include medians and spread,
not only the best run.

## Decision rules

- If the subgroup scan does not beat the shared-memory scan, the distributed
  mapping is not providing a material cross-lane advantage on that device.
- If scan-only wins but the complete renderer loses, profile edge binning and
  cell accumulation before changing the scan.
- If the GPU wins only when output stays resident, document the GPU-resident
  pipeline as the intended use case rather than hiding readback cost.
- If only a discrete GPU wins, classify the result by hardware class instead of
  making a universal CPU-versus-GPU claim.
- If higher edge counts help only the brute-force implementation, do not infer
  that production analytic rendering benefits in the same way.

## Completion criteria

The project is complete when it can state, with results from at least two GPU
classes:

1. whether real subgroup prefix operations accelerate analytic coverage;
2. which workload size and complexity cross over from CPU SIMD to GPU;
3. whether the win survives binning, painting, composition, and readback;
4. which part of the VectorWare execution model applies to practical vector
   graphics and which part does not.
