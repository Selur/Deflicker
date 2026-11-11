// SPDX-License-Identifier: GPL-2.0-or-later
// SSE2 kernels. The 8 bit statistics and correction follow Deflicker 0.6 by pinterf.
#include <emmintrin.h>

#include "kernels.hpp"
#include "scalar.hpp"

namespace deflicker {
namespace {

inline int64_t hsum64(__m128i v) {
  alignas(16) int64_t t[2];
  _mm_store_si128(reinterpret_cast<__m128i*>(t), v);
  return t[0] + t[1];
}

// Sum and sum of squares of 16 bytes (luma only when masked), added to 64 bit accumulators.
inline void sumBlockU8(__m128i src, __m128i& sum, __m128i& sq) {
  const __m128i zero = _mm_setzero_si128();
  sum = _mm_add_epi64(sum, _mm_sad_epu8(src, zero));
  const __m128i lo = _mm_unpacklo_epi8(src, zero);
  const __m128i hi = _mm_unpackhi_epi8(src, zero);
  const __m128i q = _mm_add_epi32(_mm_madd_epi16(lo, lo), _mm_madd_epi16(hi, hi));  // 4 x 32 bit
  sq = _mm_add_epi64(sq, _mm_add_epi64(_mm_unpacklo_epi32(q, zero), _mm_unpackhi_epi32(q, zero)));
}

void sumU8(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  const int w16 = width & ~15;
  for (int y = 0; y < height; y++, src += stride) {
    __m128i s = _mm_setzero_si128(), q = _mm_setzero_si128();
    for (int x = 0; x < w16; x += 16) sumBlockU8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + x)), s, q);
    int64_t rs = hsum64(s), rq = hsum64(q);
    for (int x = w16; x < width; x++) {
      const int v = src[x];
      rs += v;
      rq += v * v;
    }
    *sum += rs;
    *sqsum += rq;
  }
}

void sumYUY2(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum) {
  const int w8 = width & ~7;  // 8 luma samples per 16 bytes
  const __m128i ymask = _mm_set1_epi16(0x00FF);
  for (int y = 0; y < height; y++, src += stride) {
    __m128i s = _mm_setzero_si128(), q = _mm_setzero_si128();
    for (int x = 0; x < w8; x += 8)
      sumBlockU8(_mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 2 * x)), ymask), s, q);
    int64_t rs = hsum64(s), rq = hsum64(q);
    for (int x = w8; x < width; x++) {
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
  const __m128i zero = _mm_setzero_si128();
  const __m128i mask32 = _mm_set_epi32(0, -1, 0, -1);
  for (int y = 0; y < height; y++, src += stride) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
    __m128i s = _mm_setzero_si128(), q = _mm_setzero_si128();
    for (int x = 0; x < w8; x += 8) {
      const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + x));
      const __m128i lo = _mm_unpacklo_epi16(v, zero);
      const __m128i hi = _mm_unpackhi_epi16(v, zero);
      const __m128i s32 = _mm_add_epi32(lo, hi);  // < 2^17 per lane
      s = _mm_add_epi64(s, _mm_add_epi64(_mm_and_si128(s32, mask32), _mm_srli_epi64(s32, 32)));
      q = _mm_add_epi64(q, _mm_mul_epu32(lo, lo));
      q = _mm_add_epi64(q, _mm_mul_epu32(_mm_srli_epi64(lo, 32), _mm_srli_epi64(lo, 32)));
      q = _mm_add_epi64(q, _mm_mul_epu32(hi, hi));
      q = _mm_add_epi64(q, _mm_mul_epu32(_mm_srli_epi64(hi, 32), _mm_srli_epi64(hi, 32)));
    }
    int64_t rs = hsum64(s), rq = hsum64(q);
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
    __m128d s01 = _mm_setzero_pd(), s23 = _mm_setzero_pd(), q01 = _mm_setzero_pd(), q23 = _mm_setzero_pd();
    for (int x = 0; x < w4; x += 4) {
      const __m128 v = _mm_loadu_ps(p + x);
      const __m128d a = _mm_cvtps_pd(v);
      const __m128d b = _mm_cvtps_pd(_mm_movehl_ps(v, v));
      s01 = _mm_add_pd(s01, a);
      s23 = _mm_add_pd(s23, b);
      q01 = _mm_add_pd(q01, _mm_mul_pd(a, a));
      q23 = _mm_add_pd(q23, _mm_mul_pd(b, b));
    }
    double lane[4], lane_sq[4];
    _mm_storeu_pd(lane, s01);
    _mm_storeu_pd(lane + 2, s23);
    _mm_storeu_pd(lane_sq, q01);
    _mm_storeu_pd(lane_sq + 2, q23);
    sumRowF32Tail(p, w4, width, lane, lane_sq, sum, sqsum);
  }
}

// 16 luma bytes, Deflicker 0.6 formula.
inline __m128i correctBlockU8(__m128i src, __m128i mult, __m128i add, __m128i lo, __m128i hi) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i ml = _mm_mulhi_epu16(_mm_unpacklo_epi8(zero, src), mult);
  const __m128i mh = _mm_mulhi_epu16(_mm_unpackhi_epi8(zero, src), mult);
  const __m128i r = _mm_packus_epi16(_mm_adds_epi16(ml, add), _mm_adds_epi16(mh, add));
  return _mm_min_epu8(_mm_max_epu8(r, lo), hi);
}

void correctU8(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
               int16_t mult256, int16_t add, uint8_t lo, uint8_t hi) {
  const int w16 = width & ~15;
  const __m128i vm = _mm_set1_epi16(mult256), va = _mm_set1_epi16(add);
  const __m128i vlo = _mm_set1_epi8(char(lo)), vhi = _mm_set1_epi8(char(hi));
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    for (int x = 0; x < w16; x += 16) {
      const __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + x));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + x), correctBlockU8(s, vm, va, vlo, vhi));
    }
    for (int x = w16; x < width; x++) dst[x] = correctSampleU8(src[x], mult256, add, lo, hi);
  }
}

void correctYUY2(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                 int16_t mult256, int16_t add, uint8_t lo, uint8_t hi) {
  const int w8 = width & ~7;
  const __m128i vm = _mm_set1_epi16(mult256), va = _mm_set1_epi16(add);
  const __m128i vlo = _mm_set1_epi8(char(lo)), vhi = _mm_set1_epi8(char(hi));
  const __m128i uvmask = _mm_set1_epi16(short(0xFF00));
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    for (int x = 0; x < w8; x += 8) {
      const __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 2 * x));
      const __m128i r = correctBlockU8(s, vm, va, vlo, vhi);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 2 * x),
                       _mm_or_si128(_mm_andnot_si128(uvmask, r), _mm_and_si128(s, uvmask)));
    }
    for (int x = w8; x < width; x++) {
      dst[2 * x] = correctSampleU8(src[2 * x], mult256, add, lo, hi);
      dst[2 * x + 1] = src[2 * x + 1];
    }
  }
}

inline __m128i correctBlockU16(__m128i v32, __m128 mult, __m128 add, __m128 lo, __m128 hi) {
  __m128 f = _mm_mul_ps(_mm_cvtepi32_ps(v32), mult);
  f = _mm_add_ps(f, add);
  f = _mm_min_ps(_mm_max_ps(f, lo), hi);
  return _mm_cvtps_epi32(f);  // round half to even
}

void correctU16(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                float mult, float add, float lo, float hi) {
  const int w8 = width & ~7;
  const __m128 vm = _mm_set1_ps(mult), va = _mm_set1_ps(add), vlo = _mm_set1_ps(lo), vhi = _mm_set1_ps(hi);
  const __m128i zero = _mm_setzero_si128();
  const __m128i bias32 = _mm_set1_epi32(32768);
  const __m128i bias16 = _mm_set1_epi16(short(0x8000));
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
    uint16_t* d = reinterpret_cast<uint16_t*>(dst);
    for (int x = 0; x < w8; x += 8) {
      const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + x));
      const __m128i a = correctBlockU16(_mm_unpacklo_epi16(v, zero), vm, va, vlo, vhi);
      const __m128i b = correctBlockU16(_mm_unpackhi_epi16(v, zero), vm, va, vlo, vhi);
      // results are in 0..65535: pack signed around 32768 (SSE2 has no packus_epi32)
      const __m128i r = _mm_packs_epi32(_mm_sub_epi32(a, bias32), _mm_sub_epi32(b, bias32));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(d + x), _mm_xor_si128(r, bias16));
    }
    for (int x = w8; x < width; x++) d[x] = correctSampleU16(s[x], mult, add, lo, hi);
  }
}

void correctF32(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width, int height,
                float mult, float add, float lo, float hi) {
  const int w4 = width & ~3;
  const __m128 vm = _mm_set1_ps(mult), va = _mm_set1_ps(add), vlo = _mm_set1_ps(lo), vhi = _mm_set1_ps(hi);
  for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
    const float* s = reinterpret_cast<const float*>(src);
    float* d = reinterpret_cast<float*>(dst);
    for (int x = 0; x < w4; x += 4) {
      __m128 f = _mm_mul_ps(_mm_loadu_ps(s + x), vm);
      f = _mm_add_ps(f, va);
      _mm_storeu_ps(d + x, _mm_min_ps(_mm_max_ps(f, vlo), vhi));
    }
    for (int x = w4; x < width; x++) d[x] = correctSampleF32(s[x], mult, add, lo, hi);
  }
}

const Kernels kSSE2 = {sumU8, sumYUY2, sumU16, sumF32, correctU8, correctYUY2, correctU16, correctF32};

}  // namespace

const Kernels& kernelsSSE2() { return kSSE2; }

}  // namespace deflicker
