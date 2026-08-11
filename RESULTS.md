# Initial results

Measured on August 11, 2026 with:

- CPU: Intel Core i7-8550U, 4 cores / 8 hardware threads
- GPU: Intel UHD Graphics 620 (Kaby Lake GT2)
- driver: Mesa 25.2.8, OpenGL 4.6
- workload: 72 flattened vector edges, 4x4 supersampling, gradient paint,
  source-over composition
- compiler: GCC 13.3, `-O3 -mavx2`, automatic loop vectorization disabled
  so the scalar and explicit AVX2 cases remain distinct

Times are medians. GPU dispatch time includes command submission and
`glFinish()`. Readback is listed separately.

| Backend | 512x512 | 1024x1024 |
|---|---:|---:|
| CPU scalar, 1 thread | 907.397 ms | 3610.244 ms |
| CPU AVX2 x8, 1 thread | 84.586 ms | 339.633 ms |
| CPU AVX2 x8, 8 threads | 24.323 ms | 106.308 ms |
| GPU, distributed invocations | 36.531 ms | 137.455 ms |
| GPU, 8 pixels serialized per invocation | 37.710 ms | 138.293 ms |
| GPU, distributed plus readback | 37.575 ms | 136.974 ms |

## What the numbers say

- Distributing the logical vector across GPU invocations was 3.2% faster than
  keeping eight serialized pixels in each invocation at 512x512. At 1024x1024
  the difference fell to 0.6%. The proposed mapping is sound, but it did not
  create a large performance advantage over the packed control.
- The distributed GPU path was about 2.3-2.5x faster than single-thread AVX2.
- It did not beat all CPU cores: it took 1.50x as long as threaded AVX2 at
  512x512 and 1.29x as long at 1024x1024.
- GPU readback added little on this integrated GPU, which shares system memory.
- At 512x512 every output was byte-identical. At 1024x1024, both GPU variants
  differed from the CPU reference at one of 1,048,576 pixels, with a maximum
  channel error of 9. The likely cause is floating-point contraction or rounding
  at an edge intersection.

## Interpretation

This validates a limited but real part of the VectorWare idea: vector-graphics
coverage is naturally expressible as uniform control plus varying per-lane
pixel values, and the same logical lane algorithm maps cleanly to CPU SIMD and
GPU SIMT.

It does **not** yet validate the exact subgroup mapping. This Intel OpenGL driver
does not expose `GL_KHR_shader_subgroup`, so the program used workgroups of 32
independent invocations and could not observe the hardware subgroup width. It
also does not test cross-lane operations such as the prefix scan required by a
Blend2D-style analytic cell rasterizer.

The workload intentionally uses brute-force point-in-path supersampling. That
makes it a clear vector-rendering experiment, but it is more compute-heavy and
GPU-friendly than Blend2D's production scanline rasterizer. Even under those
conditions, the low-end integrated GPU lost to four-core/eight-thread AVX2.
Thus, the first result supports the programming model more strongly than it
supports a universal acceleration claim.

## Reproducible harness baseline

After adding the portable harness, the unchanged 72-edge, 4x4-AA workload was
measured again on the same hardware with three warm-ups and seven measured
iterations. These medians include the same synchronized GPU-dispatch boundary
as the initial results.

| Backend | 512x512 median (p90) | 1024x1024 median (p90) |
|---|---:|---:|
| CPU scalar, 1 thread | 966.712 ms (1016.780) | 3856.579 ms (3946.921) |
| CPU AVX2 x8, 1 thread | 90.686 ms (129.851) | 366.110 ms (387.608) |
| CPU AVX2 x8, 8 threads | 26.981 ms (29.383) | 122.375 ms (144.511) |
| GPU, distributed invocations | 39.473 ms (41.053) | 139.848 ms (147.378) |
| GPU, 8 pixels serialized per invocation | 40.963 ms (44.150) | 140.436 ms (143.703) |
| GPU, distributed plus readback | 39.654 ms (42.701) | 140.681 ms (145.168) |

The distributed GPU kernel was 3.8% faster than packed8 at 512x512 and 0.4%
faster at 1024x1024. It was 2.30-2.62x faster than one-thread AVX2, but took
1.46x and 1.14x as long as eight-thread AVX2. The smaller 1024x1024 gap makes a
cross-hardware run on a stronger GPU especially useful. Correctness matched
exactly at 512x512; at 1024x1024 the GPU again differed at one boundary pixel
with maximum channel error 9.

The machine-local raw data is stored in
`results/current-intel-baseline.csv`. Rows include host/compiler/driver data,
minimum/median/p90 timings, throughput, correctness, and the hardware-renderer
classification. The `results/` directory is intentionally ignored by Git.

## Cross-hardware result: AMD Radeon 8060S

The same source and command were run on the AMD Ryzen AI Max+ 395 / Radeon
8060S machine with:

- CPU: AMD Ryzen AI Max+ 395, 32 hardware threads;
- GPU: AMD Radeon 8060S (`radeonsi`, gfx1151);
- driver: Mesa 25.2.8, OpenGL 4.6;
- compiler: GCC 15.2.0, Release build;
- reported subgroup: 64 lanes;
- workload: 72 edges, 4x4 supersampling, three warm-ups, seven measurements.

| Backend | 512x512 median (p90) | 1024x1024 median (p90) |
|---|---:|---:|
| CPU scalar, 1 thread | 242.036 ms (243.237) | 969.859 ms (970.322) |
| CPU AVX2 x8, 1 thread | 26.043 ms (26.058) | 103.683 ms (106.642) |
| CPU AVX2 x8, 32 threads | 2.430 ms (2.461) | 9.245 ms (9.325) |
| GPU, distributed invocations | 0.710 ms (0.837) | 1.860 ms (1.881) |
| GPU, 8 pixels serialized per invocation | 0.728 ms (0.741) | 2.190 ms (2.813) |
| GPU, distributed plus readback | 0.517 ms (0.526) | 1.965 ms (2.162) |

On this machine, the distributed GPU kernel beat 32-thread AVX2 by 3.42x at
512x512 and 4.97x at 1024x1024. It beat packed8 by 2.5% and 15.1%, respectively.
Compared with the Intel UHD 620's absolute GPU times, the Radeon was about 56x
faster at 512x512 and 75x faster at 1024x1024 for this kernel.

The sub-millisecond 512x512 wall-clock samples are sensitive to scheduling and
cache state; in particular, the recorded readback median being below the
synchronized-dispatch median is measurement noise, not negative transfer cost.
The 1024x1024 relationship is more stable. Correctness was byte-identical at
512x512 and reproduced the same single boundary-pixel difference at 1024x1024.

This answers the first hardware question decisively: a substantially stronger
GPU can win even against a much stronger many-core CPU. It also gives the next
experiment suitable hardware: OpenGL reports a real 64-lane subgroup. The
current shader aligns its workgroup width to that subgroup, but does not yet use
cross-lane subgroup operations; that remains the purpose of the analytic
prefix-scan milestone.

Raw data is available as `results/amd-radeon-8060s-baseline.csv` on both
machines.

## Adaptive-flattening quality check

The initial 72-edge benchmark uses an intentionally fixed subdivision of 12
segments per cubic and 24 around the hole. This preserves a simple complexity
baseline, but its facets are visible and become larger as resolution increases.

Adaptive de Casteljau subdivision with a 0.25-pixel screen-space tolerance is
now the default for visually representative rendering. It generated 104 edges
at 256x256, 117 at 512x512, and 205 at 1024x1024 for the same heart. The edge
count grows with the transformed curve size, keeping geometric error below the
pixel-scale tolerance. CPU scalar, AVX2, and threaded AVX2 outputs remained
byte-identical in all three checks.

Historical cross-hardware numbers above remain valid fixed-complexity results;
their commands explicitly pass `--curve-segments 12`. New CSV schema version 3
records `flattening_mode`, `curve_segments`, and `flatness_pixels` to prevent
mixing fixed and adaptive workloads in later comparisons.

## Vulkan backend validation

The optional Vulkan 1.2 compute backend was validated with the fixed 72-edge,
4x4-AA scene at 256x256. It uses checked-in SPIR-V, discovers subgroup and
timestamp capabilities through the Vulkan driver, and records device timestamp,
synchronized queue submission, and readback as separate scopes.

| Device and variant | Device timestamp | Submit + fence |
|---|---:|---:|
| Intel UHD 620, distributed | 8.423 ms | 8.735 ms |
| Intel UHD 620, packed8 | 11.029 ms | 12.171 ms |
| Radeon 8060S, distributed | 0.199 ms | 0.235 ms |
| Radeon 8060S, packed8 | 0.469 ms | 0.545 ms |

The Intel driver reports a 32-lane subgroup with 36 valid timestamp bits. The
Radeon driver reports 64 lanes with 64 valid timestamp bits. Both advertise
basic, vote, arithmetic, ballot, shuffle, shuffle-relative, clustered, and quad
operations in compute shaders. Both Vulkan output variants matched the CPU
reference byte-for-byte.

At this workload, distributed invocation timestamps were 24% below packed8 on
the Intel GPU and 58% below packed8 on the Radeon GPU. These pipelines still do
not execute cross-lane instructions; capability discovery and reliable timing
are now in place for the analytic prefix-scan experiment that will use them.

Raw schema-v3 data is stored in `results/current-intel-vulkan.csv` and
`results/amd-radeon-8060s-vulkan.csv`. Sub-millisecond host timings can be noisy,
so device timestamps are the primary kernel measurement and synchronized/readback
times describe application-visible boundaries.

## Article-style subgroup coverage scan

The focused scan experiment uses the same flattened heart path to construct
signed fixed-point coverage deltas (64 vertical samples, 256 horizontal units,
scale 16,384). It then scans consecutive pixels using scalar CPU, AVX2 CPU,
serialized Vulkan, shared-memory Vulkan, and an explicit subgroup shuffle-up
prefix network. All coverage values and painted pixels below matched the scalar
reference exactly.

An important portability bug appeared during development. On the Intel driver,
a 32-thread workgroup initially contained two execution subgroups even though
the device reported a nominal subgroup size of 32. Cross-subgroup assumptions
therefore produced seams at x=32. Specializing the workgroup size was not
enough. Enabling `VK_EXT_subgroup_size_control`, requesting a 32-lane subgroup
for the pipeline, and requiring full subgroups made the article-style mapping
real and corrected the output. The Radeon pipeline analogously requires its
native 64-lane subgroup.

Scan-only device timestamps:

| Device | Backend | 512x512 | 1024x1024 | 2048x2048 |
|---|---|---:|---:|---:|
| Intel UHD 620 | CPU scalar | 0.162 ms | 0.851 ms | 3.638 ms |
| Intel UHD 620 | Vulkan shared | 0.702 ms | 2.430 ms | 8.695 ms |
| Intel UHD 620 | Vulkan subgroup shuffle | 0.411 ms | 0.675 ms | 2.711 ms |
| Radeon 8060S | CPU scalar | 0.071 ms | 0.214 ms | 0.904 ms |
| Radeon 8060S | Vulkan shared | 0.040 ms | 0.031 ms | 0.105 ms |
| Radeon 8060S | Vulkan subgroup shuffle | 0.016 ms | 0.028 ms | 0.118 ms |

Scan-plus-paint device timestamps compared with scalar CPU scan-plus-paint:

| Device | Backend | 512x512 | 1024x1024 | 2048x2048 |
|---|---|---:|---:|---:|
| Intel UHD 620 | CPU scalar | 3.882 ms | 13.768 ms | 54.714 ms |
| Intel UHD 620 | Vulkan subgroup shuffle | 0.553 ms | 0.736 ms | 2.988 ms |
| Radeon 8060S | CPU scalar | 0.918 ms | 3.014 ms | 12.141 ms |
| Radeon 8060S | Vulkan subgroup shuffle | 0.013 ms | 0.033 ms | 0.640 ms |

The result supports the article, with limits:

- on Intel, subgroup scan is slower than scalar CPU at 512 square, then crosses
  over at 1024 and reaches 1.34x at 2048; painting makes the GPU advantage much
  larger because the pixels stay GPU-resident;
- on the Radeon, subgroup scan is 4.43x, 7.76x, and 7.65x faster than scalar CPU
  across the three sizes;
- subgroup shuffle beats shared memory by 1.7-3.6x on Intel and at 512/1024 on
  Radeon, but Radeon shared memory is slightly faster at 2048 (0.105 vs 0.118
  ms), so subgroup operations are not a universal best choice;
- the sub-0.1-ms Radeon samples have visible scheduling/cache variance. Device
  timestamps are still preferable to host synchronization, but more iterations
  are warranted before interpreting small differences as architecture laws;
- the 64-sample input construction is measured separately and is not included
  in the scan kernel times. This milestone validates the cross-lane primitive,
  not yet a complete Blend2D-style analytic renderer.

Raw data is stored in `results/intel-uhd-620-coverage-scan.csv` and
`results/amd-radeon-8060s-coverage-scan.csv` (the ignored machine-local results
directory).

### All-core AVX2 control

The scan was subsequently parallelized across independent rows so the CPU
control uses all available cores as well as eight AVX2 lanes per thread. Thread
creation and joining are inside the timed CPU call. The GPU column below uses
queue submission plus fence synchronization, not the smaller device-only
timestamp.

| Device | Size | All-core AVX2 scan | Synchronized subgroup scan | GPU / CPU result |
|---|---:|---:|---:|---:|
| Intel UHD 620 / 8-thread CPU | 512 | 0.360 ms | 0.735 ms | CPU 2.04x faster |
| Intel UHD 620 / 8-thread CPU | 1024 | 0.563 ms | 0.681 ms | CPU 1.21x faster |
| Intel UHD 620 / 8-thread CPU | 2048 | 2.525 ms | 4.110 ms | CPU 1.63x faster |
| Radeon 8060S / 32-thread CPU | 512 | 0.402 ms | 0.105 ms | GPU 3.84x faster |
| Radeon 8060S / 32-thread CPU | 1024 | 0.443 ms | 0.152 ms | GPU 2.92x faster |
| Radeon 8060S / 32-thread CPU | 2048 | 0.463 ms | 0.232 ms | GPU 2.00x faster |

This answers the CPU-versus-GPU SIMD question by hardware class rather than
universally. A modern wide GPU can beat SIMD across a strong many-core CPU for
the scan itself even after submission overhead; the older integrated GPU
cannot. Scan plus paint favors the GPU more strongly because the paint work is
parallel and the intermediate coverage remains resident.

Raw controls are stored in `results/intel-uhd-620-threaded-scan.csv` and
`results/amd-radeon-8060s-threaded-scan.csv`.

## Blend2D-style analytic cells

The follow-up replaces sampled coverage with the combined analytic-cell delta
used by Blend2D's raster pipeline. Edges are quantized to 8-bit subpixels; each
edge fragment contributes `cover*512-area` to its cell and `area` to the next
cell. A signed horizontal prefix sum followed by non-zero or even-odd folding
produces coverage on a scale of 131,072. The published heart uses even-odd fill.

Aligned, fractional, clipped, doubled-winding, and diagonal-triangle invariants
pass. Scalar CPU, fully vectorized AVX2, all-core AVX2, serialized Vulkan,
shared-memory Vulkan, and subgroup Vulkan outputs all match the scalar analytic
reference exactly. Against an independent 8x8 supersampled render at 256x256,
636 of 65,536 edge pixels differed, with maximum channel error 8 and mean
absolute channel error 0.007347. The images were visually indistinguishable;
the comparison is a quality cross-check between two antialiasing methods, not
an expected bit-exact match.

Dense scalar cell-construction medians:

| CPU | 512x512 | 1024x1024 | 2048x2048 |
|---|---:|---:|---:|
| Core i7-8550U | 1.071 ms | 2.559 ms | 47.769 ms |
| Ryzen AI Max+ 395 | 0.113 ms | 0.381 ms | 6.425 ms |

The sharp 2048 increase is the most useful result from this stage. The current
reference clears and writes a dense 32-bit cell for every target pixel and
allocates temporary cut storage while splitting edges. At large sizes this
cost dominates the very fast GPU scan. It directly motivates tile-local sparse
storage and edge binning rather than further tuning the prefix network first.

The fair scan-only CPU/GPU control below compares AVX2 distributed across all
requested CPU threads with Vulkan queue submission plus fence synchronization.
The AVX2 even-odd fold is vectorized; it does not scalarize individual lanes.

| Device and CPU | Size | All-core AVX2 | Synchronized subgroup | Result |
|---|---:|---:|---:|---:|
| Intel UHD 620 / 8-thread CPU | 512 | 0.510 ms | 0.741 ms | CPU 1.45x faster |
| Intel UHD 620 / 8-thread CPU | 1024 | 0.687 ms | 2.021 ms | CPU 2.94x faster |
| Intel UHD 620 / 8-thread CPU | 2048 | 2.553 ms | 2.018 ms | GPU 1.27x faster |
| Radeon 8060S / 32-thread CPU | 512 | 0.389 ms | 0.143 ms | GPU 2.72x faster |
| Radeon 8060S / 32-thread CPU | 1024 | 0.427 ms | 0.110 ms | GPU 3.87x faster |
| Radeon 8060S / 32-thread CPU | 2048 | 0.472 ms | 0.197 ms | GPU 2.40x faster |

Small CPU scans do not always benefit from all cores because thread
creation/join is included. For example, one-thread AVX2 took 0.284 ms versus
0.510 ms with eight threads at 512 square on the Intel CPU. The Intel 1024
subgroup samples were also unusually variable (0.692 ms minimum, 8.617 ms
p90), so its 2.021 ms median should not be treated as a stable architecture
constant.

Scan plus gradient paint favors the GPU more consistently because coverage
stays resident and each pixel's paint is parallel:

| Device and CPU | Size | All-core AVX2 | Synchronized subgroup | GPU speedup |
|---|---:|---:|---:|---:|
| Intel UHD 620 / 8-thread CPU | 512 | 3.702 ms | 1.188 ms | 3.12x |
| Intel UHD 620 / 8-thread CPU | 1024 | 6.353 ms | 1.021 ms | 6.22x |
| Intel UHD 620 / 8-thread CPU | 2048 | 18.165 ms | 3.172 ms | 5.73x |
| Radeon 8060S / 32-thread CPU | 512 | 0.416 ms | 0.123 ms | 3.39x |
| Radeon 8060S / 32-thread CPU | 1024 | 0.532 ms | 0.112 ms | 4.74x |
| Radeon 8060S / 32-thread CPU | 2048 | 1.623 ms | 0.750 ms | 2.16x |

On Intel, subgroup device timestamps beat shared memory at all three sizes:
0.491 versus 0.613 ms, 1.734 versus 2.184 ms, and 1.851 versus 8.659 ms. On
Radeon they are effectively tied: shared/subgroup times were 0.010/0.012 ms,
0.031/0.028 ms, and 0.108/0.101 ms. The article-style shuffle network is thus
valuable on one architecture and merely competitive on the other.

These are still component measurements, not a completed GPU renderer. Vulkan
delta upload occurs outside the timed dispatch, and cell construction is still
on the CPU. Adding construction and scan/paint medians illustrates the current
bottleneck but would omit transfer cost. The next valid end-to-end experiment
is therefore tile-local cell accumulation feeding the resident subgroup paint,
with binning, accumulation, upload, scan/paint, and optional readback reported
as separate stages.

Raw data is stored in `results/intel-uhd-620-analytic-cells.csv` and
`results/amd-radeon-8060s-analytic-cells.csv` in the ignored machine-local
results directory.

### Sparse tile-local construction

The first Milestone 5 change bins edges into 16-pixel-high bands and
accumulates cells in sparse 64x16 tiles. It then materializes a dense delta
image only to remain compatible with the existing scan kernels. Tile output is
exactly equal to the dense analytic reference, including tests whose geometry
crosses tile boundaries.

| CPU | Size | Active tiles | Compact/dense storage | Tile build | Materialize | Dense build | Combined result |
|---|---:|---:|---:|---:|---:|---:|---:|
| Core i7-8550U | 512 | 69/256 | 27.1% | 0.337 ms | 0.072 ms | 1.071 ms | tiles 2.62x faster |
| Core i7-8550U | 1024 | 144/1024 | 14.2% | 2.626 ms | 0.297 ms | 2.559 ms | dense 1.14x faster |
| Core i7-8550U | 2048 | 288/4096 | 7.1% | 5.806 ms | 1.343 ms | 47.769 ms | tiles 6.68x faster |
| Ryzen AI Max+ 395 | 512 | 69/256 | 27.1% | 0.212 ms | 0.012 ms | 0.113 ms | dense 1.98x faster |
| Ryzen AI Max+ 395 | 1024 | 144/1024 | 14.2% | 0.448 ms | 0.039 ms | 0.381 ms | dense 1.28x faster |
| Ryzen AI Max+ 395 | 2048 | 288/4096 | 7.1% | 0.890 ms | 0.328 ms | 6.425 ms | tiles 5.28x faster |

There is a clear crossover rather than a universal tile win. For small dense
images, bin vectors, tile lookup, and compact allocation cost more than simply
clearing the dense buffer, especially on the faster CPU. At 2048 square, sparse
storage avoids clearing a dense 32 MiB 64-bit accumulator and converting every
pixel, and wins decisively on both CPUs. The compatibility materializer still
clears the 16 MiB dense output; only 1.18 MiB of compact cells is retained.

The fixed 72-edge heart produced 130, 188, and 302 binned edge references as
resolution increased. Active tiles grew approximately linearly while the full
tile grid grew quadratically, which is the expected behavior for sparse path
boundaries. The next implementation step is to scan these tiles directly with
per-row carries. That removes the 0.328-1.343 ms materialization stage and, for
the GPU path, avoids uploading a dense 16 MiB delta buffer.

Raw tile data is stored in `results/intel-i7-8550u-analytic-tiles.csv` and
`results/amd-ryzen-ai-max-395-analytic-tiles.csv`.

### Direct compact-tile CPU scan

Scalar, AVX2, and all-core AVX2 scans now consume compact tiles directly. Each
row preserves its signed carry while crossing inactive tiles, so solid
interiors are emitted without materializing zero deltas. All scan and paint
outputs match the dense reference exactly.

All-core AVX2 scan/paint medians, excluding construction:

| CPU | Size | Dense scan | Direct tile scan | Dense scan+paint | Direct tile scan+paint |
|---|---:|---:|---:|---:|---:|
| Core i7-8550U | 512 | 0.360 ms | 0.364 ms | 1.589 ms | 1.480 ms |
| Core i7-8550U | 1024 | 0.597 ms | 0.936 ms | 4.611 ms | 3.712 ms |
| Core i7-8550U | 2048 | 3.680 ms | 2.307 ms | 25.624 ms | 14.954 ms |
| Ryzen AI Max+ 395 | 512 | 0.387 ms | 0.381 ms | 0.405 ms | 0.397 ms |
| Ryzen AI Max+ 395 | 1024 | 0.414 ms | 0.398 ms | 0.534 ms | 0.548 ms |
| Ryzen AI Max+ 395 | 2048 | 0.476 ms | 0.446 ms | 1.433 ms | 1.434 ms |

The newer CPU is already fast enough that the direct scan kernel is mostly a
tie; eliminating materialization is its main benefit. On the older CPU, compact
reads improve the 2048 scan by 1.60x and scan+paint by 1.71x. The 1024 direct
scan samples on that machine were noisy, while scan+paint remained faster.

Summing independently measured construction and scan-stage medians, direct
tiles reduce 2048 construction+all-core scan from 13.198 to 9.983 ms on the
older CPU and from 1.723 to 1.332 ms on the newer CPU, about 24% in both cases.
For construction+scan+paint the reductions are 35.142 to 22.630 ms and 2.680
to 2.320 ms. These sums are diagnostic rather than a substitute for a single
end-to-end timer, but they show that removing materialization matters even when
the direct kernel itself is tied.

Raw direct-scan data is stored in
`results/intel-i7-8550u-direct-tile-scan.csv` and
`results/amd-ryzen-ai-max-395-direct-tile-scan.csv`.

### Direct compact-tile Vulkan input

The Vulkan scan kernels now read compact tile values through a tile lookup
instead of requiring a dense materialized delta image. The same serialized,
shared-memory, and subgroup implementations run against both representations,
and all scan and paint outputs match exactly. Input buffers are allocated to
the representation's actual size; dense coverage and pixel outputs remain full
image size.

| Size | Dense input | Compact input | Reduction |
|---:|---:|---:|---:|
| 512 | 1,048,576 B | 283,648 B | 3.70x |
| 1024 | 4,194,304 B | 593,920 B | 7.06x |
| 2048 | 16,777,216 B | 1,196,032 B | 14.03x |

The following diagnostic total adds independently measured stage medians. The
dense control includes materialization, upload, and synchronized subgroup
scan+paint; the compact path includes upload and synchronized subgroup
scan+paint because it does not materialize. Tile construction is common to
both and is excluded. These sums are not a substitute for a single end-to-end
timer, but they describe the current CPU-to-GPU input boundary.

| GPU | Size | Dense staged total | Compact staged total | Compact speedup |
|---|---:|---:|---:|---:|
| Intel UHD 620 | 512 | 1.649 ms | 1.381 ms | 1.19x |
| Intel UHD 620 | 1024 | 2.234 ms | 1.478 ms | 1.51x |
| Intel UHD 620 | 2048 | 8.881 ms | 3.696 ms | 2.40x |
| Radeon 8060S | 512 | 0.208 ms | 0.149 ms | 1.40x |
| Radeon 8060S | 1024 | 0.557 ms | 0.364 ms | 1.53x |
| Radeon 8060S | 2048 | 2.175 ms | 0.334 ms | 6.52x |

The two GPUs reach that result differently. On the Intel GPU, lookup
indirection is generally neutral or a small kernel cost; most of the overall
gain comes from removing dense materialization and reducing upload. Its 1024
compact scan-only timestamp was anomalously slower even though synchronized
paint remained slightly faster, so it should not be treated as a stable kernel
result. On the Radeon at 2048, compact input improves median upload by 13.3x,
device scan+paint by 5.11x, and synchronized scan+paint by 2.84x. The Radeon
1024 dense paint samples were bimodal, while its 2048 result was stable and is
the strongest evidence that sparse boundary data improves bandwidth and cache
behavior on a sufficiently capable GPU.

The remaining renderer boundary is composition rather than scan input: several
independent paths and paints need to remain GPU-resident, use source-over, and
synchronize only when the final image is requested.

Raw measurements are stored in
`results/intel-uhd-620-compact-tile-vulkan.csv` and
`results/amd-radeon-8060s-compact-tile-vulkan.csv`.

### GPU-resident multi-draw composition

The renderer now packs several compact analytic paths into one input, runs one
subgroup scan/paint dispatch per path, and source-over composites independent
gradient and solid paints into the same device-local pixel buffer. Compute
barriers connect the draws inside one command buffer; there is no CPU wait and
no readback between them. A control submits the same dispatches separately and
waits after every draw. Both modes match the scalar and all-core CPU references
exactly, including an odd 70x37 boundary case on both hardware GPUs.

At 1024 square, submission batching is useful once the queue-wait overhead is
large enough to rise above timing noise. Times below are medians; the CPU
column uses one persistent all-core worker team for the complete AVX2 scan plus
source-over batch, and both GPU columns keep the target resident until the
sequence ends.

| GPU | Draws | All-core CPU | Per-draw GPU waits | Resident GPU batch | Batch vs waits | Resident GPU vs CPU |
|---|---:|---:|---:|---:|---:|---:|
| Intel UHD 620 | 2 | 11.712 ms | 2.360 ms | 2.030 ms | 1.16x | 5.77x |
| Intel UHD 620 | 4 | 24.633 ms | 5.137 ms | 4.055 ms | 1.27x | 6.07x |
| Intel UHD 620 | 8 | 42.410 ms | 9.079 ms | 7.778 ms | 1.17x | 5.45x |
| Radeon 8060S | 2 | 0.961 ms | 0.147 ms | 0.204 ms | 0.72x | 4.70x |
| Radeon 8060S | 4 | 1.768 ms | 0.298 ms | 0.251 ms | 1.19x | 7.03x |
| Radeon 8060S | 8 | 3.112 ms | 0.575 ms | 0.464 ms | 1.24x | 6.70x |

The two-draw Radeon synchronized median is below the reliable granularity of
this comparison: the batch median regresses while dispatch-plus-readback
improves from 0.163 to 0.133 ms. At four and eight draws, batching reduces the
synchronized median by 16-21% on Radeon and 14-21% on Intel.

At 2048 square with four draws, batching provides a smaller 1.09x synchronized
gain on each GPU: 15.674 to 14.436 ms on Intel and 0.750 to 0.685 ms on Radeon.
The resident GPU stage beats the corresponding persistent all-core CPU
composition by 5.26x and 9.61x. Adding independently measured tile construction
and packed upload gives diagnostic staged totals of 21.664 ms versus 75.898 ms
on Intel and 2.359 ms versus 6.589 ms on Radeon, 3.50x and 2.79x end-stage
advantages respectively.

Device timestamps explain the result. At 1024, batching changes device work by
less than 5% on both GPUs, so the synchronized advantage is submission
amortization rather than faster subgroup arithmetic. At 2048, device work also
remains effectively unchanged. Resident composition matters most when several
draws would otherwise incur separate waits; pixel count and memory traffic
still dominate large full-frame draws.

Raw measurements are stored in
`results/intel-uhd-620-resident-draw-sweep.csv`,
`results/amd-radeon-8060s-resident-draw-sweep.csv`,
`results/intel-uhd-620-resident-2048.csv`, and
`results/amd-radeon-8060s-resident-2048.csv`.
