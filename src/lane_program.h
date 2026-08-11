#ifndef GPUSIMD_LANE_PROGRAM_H
#define GPUSIMD_LANE_PROGRAM_H

#include <immintrin.h>

#include <array>
#include <cstdint>
#include <span>

namespace gpusimd {

struct alignas(16) Edge {
  float x0;
  float y0;
  float y1;
  float dxOverDy;
};

struct ScalarOps {
  using F = float;
  using Mask = bool;

  static F splat(float x) noexcept { return x; }
  static F add(F a, F b) noexcept { return a + b; }
  static F sub(F a, F b) noexcept { return a - b; }
  static F mul(F a, F b) noexcept { return a * b; }
  static Mask gt(F a, F b) noexcept { return a > b; }
  static Mask lt(F a, F b) noexcept { return a < b; }
  static Mask bitAnd(Mask a, Mask b) noexcept { return a && b; }
  static Mask bitXor(Mask a, Mask b) noexcept { return a != b; }
  static F select(Mask mask, F yes, F no) noexcept { return mask ? yes : no; }
};

struct Avx2Ops {
  using F = __m256;
  using Mask = __m256;

  static F splat(float x) noexcept { return _mm256_set1_ps(x); }
  static F add(F a, F b) noexcept { return _mm256_add_ps(a, b); }
  static F sub(F a, F b) noexcept { return _mm256_sub_ps(a, b); }
  static F mul(F a, F b) noexcept { return _mm256_mul_ps(a, b); }
  static Mask gt(F a, F b) noexcept { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
  static Mask lt(F a, F b) noexcept { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
  static Mask bitAnd(Mask a, Mask b) noexcept { return _mm256_and_ps(a, b); }
  static Mask bitXor(Mask a, Mask b) noexcept { return _mm256_xor_ps(a, b); }
  static F select(Mask mask, F yes, F no) noexcept { return _mm256_blendv_ps(no, yes, mask); }
};

// This is the logical SIMD program under test. With ScalarOps, F is one float.
// With Avx2Ops, F is eight floats. The GPU shader implements the same operations
// with F distributed over shader invocations instead of stored in one register.
template<class Ops>
inline typename Ops::F coverageProgram(
  typename Ops::F pixelX,
  typename Ops::F pixelY,
  std::span<const Edge> edges,
  uint32_t aaGrid) noexcept {

  using F = typename Ops::F;
  using Mask = typename Ops::Mask;

  const F zero = Ops::splat(0.0f);
  const F one = Ops::splat(1.0f);
  F hits = zero;

  for (uint32_t sampleY = 0; sampleY < aaGrid; ++sampleY) {
    const float oy = (float(sampleY) + 0.5f) / float(aaGrid);
    const F sy = Ops::add(pixelY, Ops::splat(oy));

    for (uint32_t sampleX = 0; sampleX < aaGrid; ++sampleX) {
      const float ox = (float(sampleX) + 0.5f) / float(aaGrid);
      const F sx = Ops::add(pixelX, Ops::splat(ox));
      Mask inside = Ops::gt(zero, one); // False in every lane.

      for (const Edge& edge : edges) {
        const F y0 = Ops::splat(edge.y0);
        const F y1 = Ops::splat(edge.y1);
        const Mask straddles = Ops::bitXor(Ops::gt(y0, sy), Ops::gt(y1, sy));
        const F xIntersection = Ops::add(
          Ops::splat(edge.x0),
          Ops::mul(Ops::sub(sy, y0), Ops::splat(edge.dxOverDy)));
        const Mask crosses = Ops::bitAnd(straddles, Ops::lt(sx, xIntersection));
        inside = Ops::bitXor(inside, crosses);
      }

      hits = Ops::add(hits, Ops::select(inside, one, zero));
    }
  }

  return hits;
}

inline uint32_t checkerPixel(uint32_t x, uint32_t y) noexcept {
  const uint32_t tile = ((x >> 5u) ^ (y >> 5u)) & 1u;
  const uint32_t r = tile ? 30u : 48u;
  const uint32_t g = tile ? 34u : 52u;
  const uint32_t b = tile ? 44u : 64u;
  return r | (g << 8u) | (b << 16u) | 0xFF000000u;
}

inline uint32_t paintSourcePixel(
  uint32_t x,
  uint32_t y,
  uint32_t width,
  uint32_t height,
  uint32_t paintIndex) noexcept {

  uint32_t r = 0;
  uint32_t g = 0;
  uint32_t b = 0;
  switch (paintIndex % 3u) {
    case 0:
      r = 48u + (x * 190u) / (width > 1u ? width - 1u : 1u);
      g = 32u + (y * 172u) / (height > 1u ? height - 1u : 1u);
      b = 210u;
      break;
    case 1:
      r = 238u;
      g = 82u;
      b = 66u;
      break;
    default:
      r = 42u + (y * 100u) / (height > 1u ? height - 1u : 1u);
      g = 190u;
      b = 48u + (x * 174u) / (width > 1u ? width - 1u : 1u);
      break;
  }
  return r | (g << 8u) | (b << 16u) | 0xFF000000u;
}

inline uint32_t paintSourceOverPixel(
  uint32_t destination,
  uint32_t source,
  uint32_t hits,
  uint32_t sampleCount) noexcept {

  const uint32_t alpha = (hits * 255u + sampleCount / 2u) / sampleCount;
  const uint32_t invAlpha = 255u - alpha;
  const uint32_t srcR = source & 255u;
  const uint32_t srcG = (source >> 8u) & 255u;
  const uint32_t srcB = (source >> 16u) & 255u;
  const uint32_t dstR = destination & 255u;
  const uint32_t dstG = (destination >> 8u) & 255u;
  const uint32_t dstB = (destination >> 16u) & 255u;
  const uint32_t outR = (srcR * alpha + dstR * invAlpha + 127u) / 255u;
  const uint32_t outG = (srcG * alpha + dstG * invAlpha + 127u) / 255u;
  const uint32_t outB = (srcB * alpha + dstB * invAlpha + 127u) / 255u;
  return outR | (outG << 8u) | (outB << 16u) | 0xFF000000u;
}

inline uint32_t shadePixel(
  uint32_t x,
  uint32_t y,
  uint32_t width,
  uint32_t height,
  uint32_t hits,
  uint32_t sampleCount) noexcept {

  return paintSourceOverPixel(
    checkerPixel(x, y), paintSourcePixel(x, y, width, height, 0u),
    hits, sampleCount);
}

} // namespace gpusimd

#endif
