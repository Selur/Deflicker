// SPDX-License-Identifier: GPL-2.0-or-later
// AVX2 kernels (compiled with AVX2 enabled, selected at runtime only).
#include <immintrin.h>

#include "kernels.hpp"
#include "scalar.hpp"

namespace deflicker {
namespace {

inline int64_t hsum64(__m256i v) {
  alignas(32) int64_t t[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(t), v);
  return (t[0] + t[1]) + (t[2] + t[3]);
}

inline void sumBlockU8(__m256i src, __m256i& sum, __m256i& sq) {
  const __m256i zero = _mm256_setzero_si256();
  sum = _mm256_add_epi64(sum, _mm256_sad_epu8(src, zero));
  const __m256i lo = _mm256_unpacklo_epi8(src, zero);
  const __m256i hi = _mm256_unpackhi_epi8(src, zero);
  const __m256i q = _mm256_add_epi32(_mm256_madd_epi16(lo, lo), _mm256_madd_epi16(hi, hi));
  sq = _mm256_add_epi64(sq, _mm256_add_epi64(_mm256_unpacklo_epi32(q, zero), _mm256_unpackhi_epi32(q, zero)));
}

void sumU8(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  const int w32 = width & ~31;
  for (int y = 0; y < height; y++, src += stride) {
    __m256i s = _mm256_setzero_si256(), q = _mm256_setzero_si256();
    for (int x = 0; x < w32; x += 32)
      sumBlockU8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + x)), s, q);
    int64_t rs = hsum64(s), rq = hsum64(q);
    for (int x = w32; x < width; x++) {
      const int v = src[x];
      rs += v;
      rq += v * v;
    }
    *sum += rs;
    *sqsum += rq;
  }
  _mm256_zeroupper();
}

void sumYUY2(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  const int w16 = width & ~15;
  const __m256i ymask = _mm256_set1_epi16(0x00FF);
  for (int y = 0; y < height; y++, src += stride) {
    __m256i s = _mm256_setzero_si256(), q = _mm256_setzero_si256();
    for (int x = 0; x < w16; x += 16)
      sumBlockU8(_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * x)), ymask), s, q);
    int64_t rs = hsum64(s), rq = hsum64(q);
    for (int x = w16; x < width; x++) {
      const int v = src[2 * x];
      rs += v;
      rq += v * v;
    }
    *sum += rs;
    *sqsum += rq;
  }
  _mm256_zeroupper();
}

void sumU16(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  const int w16 = width & ~15;
  const __m256i zero = _mm256_setzero_si256();
  for (int y = 0; y < height; y++, src += stride) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
    __m256i s = _mm256_setzero_si256(), q = _mm256_setzero_si256();
    for (int x = 0; x < w16; x += 16) {
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + x));
      const __m256i lo = _mm256_unpacklo_epi16(v, zero);
      const __m256i hi = _mm256_unpackhi_epi16(v, zero);
      const __m256i s32 = _mm256_add_epi32(lo, hi);
      s = _mm256_add_epi64(s, _mm256_add_epi64(_mm256_unpacklo_epi32(s32, zero), _mm256_unpackhi_epi32(s32, zero)));
      q = _mm256_add_epi64(q, _mm256_mul_epu32(lo, lo));
      q = _mm256_add_epi64(q, _mm256_mul_epu32(_mm256_srli_epi64(lo, 32), _mm256_srli_epi64(lo, 32)));
      q = _mm256_add_epi64(q, _mm256_mul_epu32(hi, hi));
      q = _mm256_add_epi64(q, _mm256_mul_epu32(_mm256_srli_epi64(hi, 32), _mm256_srli_epi64(hi, 32)));
    }
    int64_t rs = hsum64(s), rq = hsum64(q);
    for (int x = w16; x < width; x++) {
      const int64_t v = p[x];
      rs += v;
      rq += v * v;
    }
    *sum += rs;
    *sqsum += rq;
  }
  _mm256_zeroupper();
}

void sumF32(const uint8_t* src, ptrdiff_t stride, int width, int height, double* sum, double* sqsum) {
  const int w4 = width & ~3;
  for (int y = 0; y < height; y++, src += stride) {
    const float* p = reinterpret_cast<const float*>(src);
    __m256d s = _mm256_setzero_pd(), q = _mm256_setzero_pd();
    for (int x = 0; x < w4; x += 4) {
      const __m256d v = _mm256_cvtps_pd(_mm_loadu_ps(p + x));
      s = _mm256_add_pd(s, v);
      q = _mm256_add_pd(q, _mm256_mul_pd(v, v));
    }
    double lane[4], lane_sq[4];
    _mm256_storeu_pd(lane, s);
    _mm256_storeu_pd(lane_sq, q);
    sumRowF32Tail(p, w4, width, lane, lane_sq, sum, sqsum);
  }
  _mm256_zeroupper();
}

inline __m256i correctBlockU8(__m256i src, __m256i mult, __m256i add, __m256i lo, __m256i hi) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i ml = _mm256_mulhi_epu16(_mm256_unpacklo_epi8(zero, src), mult);
  const __m256i mh = _mm256_mulhi_epu16(_mm256_unpackhi_epi8(zero, src), mult);
  const __m256i r = _mm256_packus_epi16(_mm256_adds_epi16(ml, add), _mm256_adds_epi16(mh, add));  // per 128 bit lane
  return _mm256_min_epu8(_mm256_max_epu8(r, lo), hi);
}

void correctU8(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
               int16_t mult256, int16_t add, uint8_t lo, uint8_t hi) {
  const int w32 = width & ~31;
  const __m256i vm = _mm256_set1_epi16(mult256), va = _mm256_set1_epi16(add);
  const __m256i vlo = _mm256_set1_epi8(char(lo)), vhi = _mm256_set1_epi8(char(hi));
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    for (int x = 0; x < w32; x += 32) {
      const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + x));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + x), correctBlockU8(s, vm, va, vlo, vhi));
    }
    for (int x = w32; x < width; x++) dst[x] = correctSampleU8(src[x], mult256, add, lo, hi);
  }
  _mm256_zeroupper();
}

void correctYUY2(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                 int16_t mult256, int16_t add, uint8_t lo, uint8_t hi) {
  const int w16 = width & ~15;
  const __m256i vm = _mm256_set1_epi16(mult256), va = _mm256_set1_epi16(add);
  const __m256i vlo = _mm256_set1_epi8(char(lo)), vhi = _mm256_set1_epi8(char(hi));
  const __m256i uvmask = _mm256_set1_epi16(short(0xFF00));
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    for (int x = 0; x < w16; x += 16) {
      const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * x));
      const __m256i r = correctBlockU8(s, vm, va, vlo, vhi);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 2 * x),
                          _mm256_or_si256(_mm256_andnot_si256(uvmask, r), _mm256_and_si256(s, uvmask)));
    }
    for (int x = w16; x < width; x++) {
      dst[2 * x] = correctSampleU8(src[2 * x], mult256, add, lo, hi);
      dst[2 * x + 1] = src[2 * x + 1];
    }
  }
  _mm256_zeroupper();
}

inline __m256i correctBlockU16(__m256i v32, __m256 mult, __m256 add, __m256 lo, __m256 hi) {
  __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(v32), mult);
  f = _mm256_add_ps(f, add);
  f = _mm256_min_ps(_mm256_max_ps(f, lo), hi);
  return _mm256_cvtps_epi32(f);
}

void correctU16(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                float mult, float add, float lo, float hi) {
  const int w16 = width & ~15;
  const __m256 vm = _mm256_set1_ps(mult), va = _mm256_set1_ps(add), vlo = _mm256_set1_ps(lo),
               vhi = _mm256_set1_ps(hi);
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
    uint16_t* d = reinterpret_cast<uint16_t*>(dst);
    for (int x = 0; x < w16; x += 16) {
      const __m256i a = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + x)));
      const __m256i b = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + x + 8)));
      const __m256i r = _mm256_packus_epi32(correctBlockU16(a, vm, va, vlo, vhi), correctBlockU16(b, vm, va, vlo, vhi));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + x), _mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0)));
    }
    for (int x = w16; x < width; x++) d[x] = correctSampleU16(s[x], mult, add, lo, hi);
  }
  _mm256_zeroupper();
}

void correctF32(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                float mult, float add, float lo, float hi) {
  const int w8 = width & ~7;
  const __m256 vm = _mm256_set1_ps(mult), va = _mm256_set1_ps(add), vlo = _mm256_set1_ps(lo),
               vhi = _mm256_set1_ps(hi);
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    const float* s = reinterpret_cast<const float*>(src);
    float* d = reinterpret_cast<float*>(dst);
    for (int x = 0; x < w8; x += 8) {
      __m256 f = _mm256_mul_ps(_mm256_loadu_ps(s + x), vm);
      f = _mm256_add_ps(f, va);
      _mm256_storeu_ps(d + x, _mm256_min_ps(_mm256_max_ps(f, vlo), vhi));
    }
    for (int x = w8; x < width; x++) d[x] = correctSampleF32(s[x], mult, add, lo, hi);
  }
  _mm256_zeroupper();
}

const Kernels kAVX2 = {sumU8, sumYUY2, sumU16, sumF32, correctU8, correctYUY2, correctU16, correctF32};

}  // namespace

const Kernels& kernelsAVX2() { return kAVX2; }

}  // namespace deflicker
