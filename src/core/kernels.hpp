// SPDX-License-Identifier: GPL-2.0-or-later
// Pixel kernels of Deflicker: luma statistics (sum and sum of squares) and the luma correction.
// Every instruction set gives bit-identical results to the C reference.
#pragma once

#include <cstddef>
#include <cstdint>

namespace deflicker {

struct Kernels {
  // Integer formats: exact sums over width x height samples. YUY2: width counts luma samples (every second byte).
  void (*sum_u8)(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum);
  void (*sum_yuy2)(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum);
  void (*sum_u16)(const uint8_t* src, ptrdiff_t stride, int width, int height, int64_t* sum, int64_t* sqsum);
  // Float: per row four double accumulators (sample x goes to lane x % 4), reduced as (l0 + l1) + (l2 + l3),
  // remaining samples added in order, then the row total is added to the frame total.
  void (*sum_f32)(const uint8_t* src, ptrdiff_t stride, int width, int height, double* sum, double* sqsum);

  // 8 bit: the fixed point formula of Deflicker 0.6 (SSE2 semantics):
  // y = min(max(packus(adds16(int16((x * mult256) >> 8), add)), lo), hi), mult256 taken as unsigned.
  void (*correct_u8)(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width,
                     int height, int16_t mult256, int16_t add, uint8_t lo, uint8_t hi);
  // YUY2: same as correct_u8 on the luma bytes, chroma copied. width counts luma samples.
  void (*correct_yuy2)(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width,
                       int height, int16_t mult256, int16_t add, uint8_t lo, uint8_t hi);
  // 9-16 bit: y = round_half_even(min(max(float(x) * mult + add, lo), hi)).
  void (*correct_u16)(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width,
                      int height, float mult, float add, float lo, float hi);
  // 32 bit float: y = min(max(x * mult + add, lo), hi).
  void (*correct_f32)(uint8_t* dst, ptrdiff_t dst_stride, const uint8_t* src, ptrdiff_t src_stride, int width,
                      int height, float mult, float add, float lo, float hi);
};

enum class Isa { C, SSE2, AVX2, NEON };

const Kernels& kernelsC();
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#define DEFLICKER_X86 1
const Kernels& kernelsSSE2();
const Kernels& kernelsAVX2();
#elif defined(__aarch64__) || defined(_M_ARM64)
#define DEFLICKER_NEON 1
const Kernels& kernelsNEON();
#endif

// Best supported instruction set; opt < 0: automatic, 0: C, 1-2: SSE2, >= 3: AVX2 (x86), >= 1: NEON (ARM64).
// A requested set the CPU lacks falls back to the best one below it.
Isa selectIsa(int opt);
const Kernels& kernelsFor(Isa isa);
const char* isaName(Isa isa);

}  // namespace deflicker
