// SPDX-License-Identifier: GPL-2.0-or-later
// C reference kernels.
#include "kernels.hpp"
#include "scalar.hpp"

namespace deflicker {
namespace {

void sumU8(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  for (int y = 0; y < height; y++, src += stride) {
    int64_t s = 0, q = 0;
    for (int x = 0; x < width; x++) {
      const int v = src[x];
      s += v;
      q += v * v;
    }
    *sum += s;
    *sqsum += q;
  }
}

void sumYUY2(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  for (int y = 0; y < height; y++, src += stride) {
    int64_t s = 0, q = 0;
    for (int x = 0; x < width; x++) {
      const int v = src[2 * x];
      s += v;
      q += v * v;
    }
    *sum += s;
    *sqsum += q;
  }
}

void sumU16(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  for (int y = 0; y < height; y++, src += stride) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
    int64_t s = 0, q = 0;
    for (int x = 0; x < width; x++) {
      const int64_t v = p[x];
      s += v;
      q += v * v;
    }
    *sum += s;
    *sqsum += q;
  }
}

void sumF32(const uint8_t* src, ptrdiff_t stride, int width, int height, double* sum, double* sqsum) {
  const int w4 = width & ~3;
  for (int y = 0; y < height; y++, src += stride) {
    const float* p = reinterpret_cast<const float*>(src);
    double lane[4] = {0, 0, 0, 0}, lane_sq[4] = {0, 0, 0, 0};
    for (int x = 0; x < w4; x += 4) {
      for (int k = 0; k < 4; k++) {
        const double v = p[x + k];
        lane[k] += v;
        lane_sq[k] += v * v;
      }
    }
    sumRowF32Tail(p, w4, width, lane, lane_sq, sum, sqsum);
  }
}

void correctU8(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
               int16_t mult256, int16_t add, uint8_t lo, uint8_t hi) {
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride)
    for (int x = 0; x < width; x++) dst[x] = correctSampleU8(src[x], mult256, add, lo, hi);
}

void correctYUY2(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                 int16_t mult256, int16_t add, uint8_t lo, uint8_t hi) {
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride)
    for (int x = 0; x < width; x++) {
      dst[2 * x] = correctSampleU8(src[2 * x], mult256, add, lo, hi);
      dst[2 * x + 1] = src[2 * x + 1];
    }
}

void correctU16(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                float mult, float add, float lo, float hi) {
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
    uint16_t* d = reinterpret_cast<uint16_t*>(dst);
    for (int x = 0; x < width; x++) d[x] = correctSampleU16(s[x], mult, add, lo, hi);
  }
}

void correctF32(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                float mult, float add, float lo, float hi) {
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    const float* s = reinterpret_cast<const float*>(src);
    float* d = reinterpret_cast<float*>(dst);
    for (int x = 0; x < width; x++) d[x] = correctSampleF32(s[x], mult, add, lo, hi);
  }
}

const Kernels kC = {sumU8, sumYUY2, sumU16, sumF32, correctU8, correctYUY2, correctU16, correctF32};

}  // namespace

const Kernels& kernelsC() { return kC; }

}  // namespace deflicker
