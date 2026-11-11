// SPDX-License-Identifier: GPL-2.0-or-later
// Per-sample reference operations, shared by the C kernels and the SIMD tails.
// Internal linkage on purpose: this header is also compiled with AVX2 enabled, and inline functions with external
// linkage could be merged across translation units, letting AVX2 code run on CPUs without AVX2.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace deflicker {
namespace {

inline uint8_t correctSampleU8(uint8_t x, int16_t mult256, int16_t add, uint8_t lo, uint8_t hi) {
  const uint32_t t = (uint32_t(x) * uint16_t(mult256)) >> 8;  // _mm_mulhi_epu16(x << 8, mult256)
  int32_t r = int32_t(int16_t(uint16_t(t))) + add;              // _mm_adds_epi16
  r = r < -32768 ? -32768 : (r > 32767 ? 32767 : r);
  r = r < 0 ? 0 : (r > 255 ? 255 : r);  // _mm_packus_epi16
  r = r < lo ? lo : r;
  r = r > hi ? hi : r;
  return uint8_t(r);
}

// Same operand order and signed zero handling as _mm_min_ps(_mm_max_ps(v, lo), hi) and vminq(vmaxq(v, lo), hi).
inline float clampF(float v, float lo, float hi) {
  v = v > lo ? v : lo;
  return v < hi ? v : hi;
}

inline uint16_t correctSampleU16(uint16_t x, float mult, float add, float lo, float hi) {
  float v = float(x) * mult;
  v = v + add;
  return uint16_t(std::lrintf(clampF(v, lo, hi)));  // round half to even in the default rounding mode
}

inline float correctSampleF32(float x, float mult, float add, float lo, float hi) {
  float v = x * mult;
  v = v + add;
  return clampF(v, lo, hi);
}

// Scalar float sum of one row, starting with lane totals already accumulated by SIMD code.
inline void sumRowF32Tail(const float* p, int from, int width, const double lane[4], const double lane_sq[4], double* sum,
                          double* sqsum) {
  double s = (lane[0] + lane[1]) + (lane[2] + lane[3]);
  double q = (lane_sq[0] + lane_sq[1]) + (lane_sq[2] + lane_sq[3]);
  for (int x = from; x < width; x++) {
    const double v = p[x];
    s += v;
    q += v * v;
  }
  *sum += s;
  *sqsum += q;
}

}  // namespace
}  // namespace deflicker
