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
