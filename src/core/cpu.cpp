// SPDX-License-Identifier: GPL-2.0-or-later
// Runtime instruction set selection.
#include "kernels.hpp"

#if defined(DEFLICKER_X86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace deflicker {

#if defined(DEFLICKER_X86)
namespace {

void cpuid(int leaf, int sub, unsigned r[4]) {
#if defined(_MSC_VER)
  int t[4];
  __cpuidex(t, leaf, sub);
  for (int i = 0; i < 4; i++) r[i] = unsigned(t[i]);
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

unsigned long long xgetbv0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
}

bool hasAVX2() {
  unsigned r[4];
  cpuid(0, 0, r);
  if (r[0] < 7) return false;
  cpuid(1, 0, r);
  const bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
  if (!osxsave || !avx || (xgetbv0() & 6) != 6) return false;  // OS saves XMM and YMM state
  cpuid(7, 0, r);
  return (r[1] >> 5) & 1;
}

bool hasSSE2() {
  unsigned r[4];
  cpuid(1, 0, r);
  return (r[3] >> 26) & 1;
}

}  // namespace
#endif

Isa selectIsa(int opt) {
#if defined(DEFLICKER_X86)
  static const bool avx2 = hasAVX2(), sse2 = hasSSE2();
  if (opt == 0) return Isa::C;
  if ((opt < 0 || opt >= 3) && avx2) return Isa::AVX2;
  return sse2 ? Isa::SSE2 : Isa::C;
#elif defined(DEFLICKER_NEON)
  return opt == 0 ? Isa::C : Isa::NEON;
#else
  (void)opt;
  return Isa::C;
#endif
}

const Kernels& kernelsFor(Isa isa) {
  switch (isa) {
#if defined(DEFLICKER_X86)
    case Isa::AVX2:
      return kernelsAVX2();
    case Isa::SSE2:
      return kernelsSSE2();
#elif defined(DEFLICKER_NEON)
    case Isa::NEON:
      return kernelsNEON();
#endif
    default:
      return kernelsC();
  }
}

const char* isaName(Isa isa) {
  switch (isa) {
    case Isa::SSE2:
      return "SSE2";
    case Isa::AVX2:
      return "AVX2";
    case Isa::NEON:
      return "NEON";
    default:
      return "C";
  }
}

}  // namespace deflicker
