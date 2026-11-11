// SPDX-License-Identifier: GPL-2.0-or-later
// NEON kernels for ARM64 (NEON is part of the base instruction set).
#include <arm_neon.h>

#include "kernels.hpp"
#include "scalar.hpp"

namespace deflicker {
namespace {

// 16 luma bytes into 32 bit lane sums (row widths up to about 2^18 samples cannot overflow).
inline void sumBlockU8(uint8x16_t v, uint32x4_t& s, uint32x4_t& q) {
  s = vpadalq_u16(s, vpaddlq_u8(v));
  q = vpadalq_u16(q, vmull_u8(vget_low_u8(v), vget_low_u8(v)));
  q = vpadalq_u16(q, vmull_high_u8(v, v));
}

void sumU8(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  const int w16 = width & ~15;
  for (int y = 0; y < height; y++, src += stride) {
    int64_t rs = 0, rq = 0;
    int x = 0;
    while (x < w16) {
      // flush the 32 bit lanes every 4096 blocks
      const int end = (w16 - x) > 65536 ? x + 65536 : w16;
      uint32x4_t s = vdupq_n_u32(0), q = vdupq_n_u32(0);
      for (; x < end; x += 16) sumBlockU8(vld1q_u8(src + x), s, q);
      rs += int64_t(vaddlvq_u32(s));
      rq += int64_t(vaddlvq_u32(q));
    }
    for (; x < width; x++) {
      const int v = src[x];
      rs += v;
      rq += v * v;
    }
    *sum += rs;
    *sqsum += rq;
  }
}

void sumYUY2(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  const int w16 = width & ~15;
  for (int y = 0; y < height; y++, src += stride) {
    int64_t rs = 0, rq = 0;
    int x = 0;
    while (x < w16) {
      const int end = (w16 - x) > 65536 ? x + 65536 : w16;
      uint32x4_t s = vdupq_n_u32(0), q = vdupq_n_u32(0);
      for (; x < end; x += 16) sumBlockU8(vld2q_u8(src + 2 * x).val[0], s, q);
      rs += int64_t(vaddlvq_u32(s));
      rq += int64_t(vaddlvq_u32(q));
    }
    for (; x < width; x++) {
      const int v = src[2 * x];
      rs += v;
      rq += v * v;
    }
    *sum += rs;
    *sqsum += rq;
  }
}

void sumU16(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  const int w8 = width & ~7;
  for (int y = 0; y < height; y++, src += stride) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
    uint64x2_t s = vdupq_n_u64(0), q = vdupq_n_u64(0);
    for (int x = 0; x < w8; x += 8) {
      const uint16x8_t v = vld1q_u16(p + x);
      s = vpadalq_u32(s, vpaddlq_u16(v));
      q = vpadalq_u32(q, vmull_u16(vget_low_u16(v), vget_low_u16(v)));
      q = vpadalq_u32(q, vmull_high_u16(v, v));
    }
    int64_t rs = int64_t(vaddvq_u64(s)), rq = int64_t(vaddvq_u64(q));
    for (int x = w8; x < width; x++) {
      const int64_t v = p[x];
      rs += v;
      rq += v * v;
    }
    *sum += rs;
    *sqsum += rq;
  }
}

void sumF32(const uint8_t* src, ptrdiff_t stride, int width, int height, double* sum, double* sqsum) {
  const int w4 = width & ~3;
  for (int y = 0; y < height; y++, src += stride) {
    const float* p = reinterpret_cast<const float*>(src);
    float64x2_t s01 = vdupq_n_f64(0), s23 = vdupq_n_f64(0), q01 = vdupq_n_f64(0), q23 = vdupq_n_f64(0);
    for (int x = 0; x < w4; x += 4) {
      const float32x4_t v = vld1q_f32(p + x);
      const float64x2_t a = vcvt_f64_f32(vget_low_f32(v));
      const float64x2_t b = vcvt_high_f64_f32(v);
      s01 = vaddq_f64(s01, a);
      s23 = vaddq_f64(s23, b);
      q01 = vaddq_f64(q01, vmulq_f64(a, a));
      q23 = vaddq_f64(q23, vmulq_f64(b, b));
    }
    double lane[4], lane_sq[4];
    vst1q_f64(lane, s01);
    vst1q_f64(lane + 2, s23);
    vst1q_f64(lane_sq, q01);
    vst1q_f64(lane_sq + 2, q23);
    sumRowF32Tail(p, w4, width, lane, lane_sq, sum, sqsum);
  }
}

// 8 luma samples, the fixed point formula of the SSE2 code: (x * mult256) >> 8 taken as int16, saturating add,
// unsigned saturating narrow, clamp.
inline int16x8_t scaleU8(uint8x8_t x, uint16x4_t m, int16x8_t add) {
  const uint16x8_t w = vmovl_u8(x);
  const uint16x4_t lo = vshrn_n_u32(vmull_u16(vget_low_u16(w), m), 8);
  const uint16x4_t hi = vshrn_n_u32(vmull_u16(vget_high_u16(w), m), 8);
  return vqaddq_s16(vreinterpretq_s16_u16(vcombine_u16(lo, hi)), add);
}

inline uint8x16_t correctBlockU8(uint8x16_t v, uint16x4_t m, int16x8_t add, uint8x16_t lo, uint8x16_t hi) {
  const uint8x16_t r = vcombine_u8(vqmovun_s16(scaleU8(vget_low_u8(v), m, add)),
                                   vqmovun_s16(scaleU8(vget_high_u8(v), m, add)));
  return vminq_u8(vmaxq_u8(r, lo), hi);
}

void correctU8(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
               int16_t mult256, int16_t add, uint8_t lo, uint8_t hi) {
  const int w16 = width & ~15;
  const uint16x4_t vm = vdup_n_u16(uint16_t(mult256));
  const int16x8_t va = vdupq_n_s16(add);
  const uint8x16_t vlo = vdupq_n_u8(lo), vhi = vdupq_n_u8(hi);
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    for (int x = 0; x < w16; x += 16) vst1q_u8(dst + x, correctBlockU8(vld1q_u8(src + x), vm, va, vlo, vhi));
    for (int x = w16; x < width; x++) dst[x] = correctSampleU8(src[x], mult256, add, lo, hi);
  }
}

void correctYUY2(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                 int16_t mult256, int16_t add, uint8_t lo, uint8_t hi) {
  const int w16 = width & ~15;
  const uint16x4_t vm = vdup_n_u16(uint16_t(mult256));
  const int16x8_t va = vdupq_n_s16(add);
  const uint8x16_t vlo = vdupq_n_u8(lo), vhi = vdupq_n_u8(hi);
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    for (int x = 0; x < w16; x += 16) {
      uint8x16x2_t v = vld2q_u8(src + 2 * x);
      v.val[0] = correctBlockU8(v.val[0], vm, va, vlo, vhi);
      vst2q_u8(dst + 2 * x, v);
    }
    for (int x = w16; x < width; x++) {
      dst[2 * x] = correctSampleU8(src[2 * x], mult256, add, lo, hi);
      dst[2 * x + 1] = src[2 * x + 1];
    }
  }
}

inline uint32x4_t correctBlockU16(uint32x4_t v, float32x4_t m, float32x4_t a, float32x4_t lo, float32x4_t hi) {
  float32x4_t f = vmulq_f32(vcvtq_f32_u32(v), m);
  f = vaddq_f32(f, a);
  f = vminq_f32(vmaxq_f32(f, lo), hi);
  return vcvtnq_u32_f32(f);  // round half to even
}

void correctU16(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                float mult, float add, float lo, float hi) {
  const int w8 = width & ~7;
  const float32x4_t vm = vdupq_n_f32(mult), va = vdupq_n_f32(add), vlo = vdupq_n_f32(lo), vhi = vdupq_n_f32(hi);
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
    uint16_t* d = reinterpret_cast<uint16_t*>(dst);
    for (int x = 0; x < w8; x += 8) {
      const uint16x8_t v = vld1q_u16(s + x);
      const uint32x4_t a = correctBlockU16(vmovl_u16(vget_low_u16(v)), vm, va, vlo, vhi);
      const uint32x4_t b = correctBlockU16(vmovl_high_u16(v), vm, va, vlo, vhi);
      vst1q_u16(d + x, vcombine_u16(vmovn_u32(a), vmovn_u32(b)));
    }
    for (int x = w8; x < width; x++) d[x] = correctSampleU16(s[x], mult, add, lo, hi);
  }
}

void correctF32(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                float mult, float add, float lo, float hi) {
  const int w4 = width & ~3;
  const float32x4_t vm = vdupq_n_f32(mult), va = vdupq_n_f32(add), vlo = vdupq_n_f32(lo), vhi = vdupq_n_f32(hi);
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    const float* s = reinterpret_cast<const float*>(src);
    float* d = reinterpret_cast<float*>(dst);
    for (int x = 0; x < w4; x += 4) {
      float32x4_t f = vmulq_f32(vld1q_f32(s + x), vm);
      f = vaddq_f32(f, va);
      vst1q_f32(d + x, vminq_f32(vmaxq_f32(f, vlo), vhi));
    }
    for (int x = w4; x < width; x++) d[x] = correctSampleF32(s[x], mult, add, lo, hi);
  }
}

const Kernels kNEON = {sumU8, sumYUY2, sumU16, sumF32, correctU8, correctYUY2, correctU16, correctF32};

}  // namespace

const Kernels& kernelsNEON() { return kNEON; }

}  // namespace deflicker
