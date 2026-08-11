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
schema version 3 records which mode produced each result so adaptive and fixed
workloads cannot be compared accidentally.

## Milestone 3: optional Vulkan subgroup backend

Status: complete.

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

Status: complete. The initially sampled bridge remains available, and a second
mode now supplies true Blend2D-style combined cover/area cell deltas so the
same scan variants can be exercised with production-shaped analytic data.

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

Implemented bridge:

- fixed-point per-pixel coverage is derived from the real flattened heart path
  using 64 vertical samples and 256 horizontal units;
- signed deltas are scanned by scalar CPU, AVX2 CPU, serialized GPU,
  shared-memory GPU, and explicit subgroup shuffle-up GPU variants;
- an all-core AVX2 control distributes independent scanlines across the
  requested CPU threads and includes thread creation/join in its timing;
- Vulkan specializes each scan workgroup to the native subgroup width and uses
  subgroup-size control plus the full-subgroup pipeline flag;
- block carry is propagated between consecutive native-subgroup chunks of a
  scanline;
- scan-only and scan-plus-paint timings and exact integer/image comparisons are
  emitted to CSV.
- a scalar analytic-cell reference quantizes edges to 8-bit subpixels and uses
  the Blend2D combined-delta identity `cell[x] += cover*512-area`,
  `cell[x+1] += area`;
- non-zero and even-odd coverage resolution are implemented on scalar CPU,
  fully vectorized AVX2, and Vulkan scan paths;
- invariant tests cover aligned and fractional rectangles, left clipping,
  winding versus even-odd behavior, and integrated triangle area;
- analytic cell construction is warmed up and reported as its own CSV stage.

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

Result: the claim holds conditionally. The subgroup scan beats shared memory on
the Intel UHD 620 at all measured sizes and on the Radeon 8060S at 512 and 1024
pixels square. At 2048 square on the Radeon, shared memory is slightly faster.
The experiment also showed that querying a nominal subgroup size is
insufficient: the Intel driver initially selected two smaller execution
subgroups until the pipeline explicitly required one full subgroup.

Against all CPU cores, synchronized subgroup scan loses on the Intel UHD 620
but wins on the Radeon 8060S by 3.84x, 2.92x, and 2.00x at 512, 1024, and 2048
pixels square. Device-only timestamps show the kernel's larger potential, but
the synchronized comparison is the fair application-visible answer.

## Milestone 5: tiled analytic vector renderer

Status: complete. The dense scalar analytic-cell reference, exact scan
backends, fill rules, and gradient paint are complete. Edges are now binned
into 16-pixel vertical bands and accumulated into sparse 64x16 cell tiles.
Exact equivalence, compact memory, construction time, and temporary dense
materialization are measured separately. Scalar and AVX2 CPU scans now consume
compact tiles directly and propagate row carries through inactive tiles.
Vulkan now consumes the same compact tile values and lookup directly using the
serialized, shared-memory, and subgroup kernels. Exact-size input buffers and
sampled upload timings expose the benefit of eliminating dense materialization
and transfer. Multiple overlapping paths with independent paints are now packed
into one upload and source-over composited through subgroup dispatches in one
command buffer. The target stays device-local between paths, and only the final
image is optionally read back. A per-draw submit-and-wait control measures the
synchronization that batching removes.

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

Status: complete. The final matrix covers the brute-force lane experiment,
dense and compact analytic scans, subgroup/shared/serialized controls,
single-path and resident multi-draw rendering, all-core CPU controls, upload,
synchronization, and final readback on the Intel UHD 620 and Radeon 8060S
systems. Absolute medians, p90 spread, throughput, exactness, crossovers, and
the limits of the VectorWare analogy are summarized in `RESULTS.md`; raw
schema-v3 rows remain in the ignored `results/` directory.

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
