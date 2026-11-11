// SPDX-License-Identifier: GPL-2.0-or-later
// Port of DrawDigit / DrawDigitYUY2 from info.cpp (Donald Graft, pinterf) with clipping at the frame edges.
#include "text.hpp"

#include "font.hpp"

namespace deflicker {
namespace {

template <typename T>
inline T* at(uint8_t* base, ptrdiff_t stride, int x, int y) {
  return reinterpret_cast<T*>(base + y * stride) + x;
}

template <typename T>
void drawGlyphPlanar(Frame& f, const Format& fmt, int x0, int y0, int glyph) {
  const bool fl = fmt.is_float;
  const T color = fl ? T(235 / 255.0f) : T(235 << (fmt.bits - 8));
  for (int tx = 0; tx < 10; tx++)
    for (int ty = 0; ty < 20; ty++) {
      const int x = x0 + tx, y = y0 + ty;
      if (x < 0 || y < 0 || x >= fmt.width || y >= fmt.height) continue;
      const bool pix = kFont[glyph][ty] & (1 << (15 - tx));
      T* p = at<T>(f.data[0], f.stride[0], x, y);
      if (pix)
        *p = color;
      else if (fl)
        *p = T(*p * 3.0f * 0.25f);
      else
        *p = T((int(*p) * 3) >> 2);
    }
  if (!fmt.chroma) return;
  const T mid = fl ? T(0) : T(1 << (fmt.bits - 1));
  for (int tx = 0; tx < 10; tx++)
    for (int ty = 0; ty < 20; ty++) {
      const int x = x0 + tx, y = y0 + ty;
      if (x < 0 || y < 0 || x >= fmt.width || y >= fmt.height) continue;
      const bool pix = kFont[glyph][ty] & (1 << (15 - tx));
      for (int c = 1; c <= 2; c++) {
        T* p = at<T>(f.data[c], f.stride[c], x >> fmt.ssw, y >> fmt.ssh);
        if (pix)
          *p = mid;
        else if (fl)
          *p = T((*p + mid) * 0.5f);
        else
          *p = T((int(*p) + int(mid)) >> 1);
      }
    }
}

void drawGlyphYUY2(Frame& f, const Format& fmt, int x0, int y0, int glyph) {
  for (int tx = 0; tx < 10; tx++)
    for (int ty = 0; ty < 20; ty++) {
      const int x = x0 + tx, y = y0 + ty;
      if (x < 0 || y < 0 || x >= fmt.width || y >= fmt.height) continue;
      uint8_t* dp = f.data[0] + y * f.stride[0] + 2 * x;
      // chroma bytes of the pixel pair: U, V
      uint8_t* u = (tx & 1) ? dp - 1 : dp + 1;
      uint8_t* v = (tx & 1) ? dp + 1 : dp + 3;
      if (kFont[glyph][ty] & (1 << (15 - tx))) {
        dp[0] = 250;
        *u = 128;
        *v = 128;
      } else {
        dp[0] = uint8_t((dp[0] * 3) >> 2);
        *u = uint8_t((*u + 128) >> 1);
        *v = uint8_t((*v + 128) >> 1);
      }
    }
}

}  // namespace

void drawString(Frame& dst, const Format& fmt, int col, int row, const char* s) {
  for (int i = 0; s[i]; i++) {
    int glyph = static_cast<unsigned char>(s[i]) - ' ';
    if (glyph < 0 || glyph >= kFontGlyphs) glyph = '?' - ' ';
    const int x = (col + i) * 10, y = row * 20;
    if (fmt.layout == Layout::YUY2)
      drawGlyphYUY2(dst, fmt, x, y, glyph);
    else if (fmt.is_float)
      drawGlyphPlanar<float>(dst, fmt, x, y, glyph);
    else if (fmt.bits > 8)
      drawGlyphPlanar<uint16_t>(dst, fmt, x, y, glyph);
    else
      drawGlyphPlanar<uint8_t>(dst, fmt, x, y, glyph);
  }
}

}  // namespace deflicker
