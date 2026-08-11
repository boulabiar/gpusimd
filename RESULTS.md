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
their commands explicitly pass `--curve-segments 12`. New CSV schema version 2
records `flattening_mode`, `curve_segments`, and `flatness_pixels` to prevent
mixing fixed and adaptive workloads in later comparisons.
