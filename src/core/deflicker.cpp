// SPDX-License-Identifier: GPL-2.0-or-later
/*
  Deflicker - mean intensity stabilizer
  (c) 2004, A.G. Balakhnin aka Fizick
  AviSynth 2.6 interface, x64, SSE2, AVX2: 2019 pinterf
  AviSynth+ / VapourSynth port, high bit depth, NEON: 2026

  This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
  License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any
  later version.
*/
#include "deflicker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "text.hpp"

#ifdef _WIN32
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringA(const char*);
#endif

namespace deflicker {
namespace {

// float -> int32 as cvttss2si does (NaN and out of range give INT32_MIN), then -> int16 by truncation, like the
// MSVC x64 build of Deflicker 0.6 converted float to short.
int16_t toShort(float f) {
  int32_t i = INT32_MIN;
  if (f >= -2147483648.0f && f < 2147483648.0f) i = static_cast<int32_t>(f);
  return static_cast<int16_t>(static_cast<uint16_t>(static_cast<uint32_t>(i)));
}

void debugOut(const char* s) {
#ifdef _WIN32
  OutputDebugStringA(s);
#else
  std::fputs(s, stderr);
#endif
}

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("Deflicker: " + msg); }

}  // namespace

Engine::Engine(const Params& p, const Format& f, int num_frames)
    : fmt_(f), num_frames_(num_frames), lag_(p.lag), range_(std::abs(p.lag)), border_(p.border), info_(p.info),
      debug_(p.debug) {
  if (p.percent < 0 || p.percent > 100) fail("percent must be from 0 to 100!");
  if (p.noise < 0) fail("noise must not be negative");
  if (p.scene < 0) fail("scene must not be negative");
  if (p.lmin < 0 || p.lmax > 255 || p.lmin > p.lmax) fail("lmin and lmax must be 0 <= lmin <= lmax <= 255");
  if (p.border < 0 || 2 * p.border >= f.width || 2 * p.border >= f.height)
    fail("border must be >= 0 and leave some of the frame");
  if (f.is_float ? f.bits != 32 : (f.bits < 8 || f.bits > 16)) fail("unsupported bit depth");
  if (f.layout == Layout::YUY2 && f.bits != 8) fail("unsupported bit depth");

  // Thresholds are given on the 8 bit scale.
  const double scale = f.is_float ? 1.0 / 255.0 : double(1 << (f.bits - 8));
  percent_ = float(p.percent);
  const float noise = float(p.noise * scale);
  varnoise_ = noise * noise;
  if (f.is_float) {
    scene_ = p.scene / 255.0;
    scene_f_ = float(scene_);
  } else {
    scene_ = double(int64_t(p.scene) << (f.bits - 8));
    scene_f_ = float(scene_);
  }
  lmin8_ = p.lmin;
  lmax8_ = p.lmax;
  if (f.is_float) {
    lo_ = p.lmin / 255.0f;
    hi_ = p.lmax / 255.0f;
  } else {
    const int shift = f.bits - 8;
    lo_ = float(p.lmin << shift);
    hi_ = float(p.lmax == 255 ? (1 << f.bits) - 1 : p.lmax << shift);  // 255 means the format maximum
  }

  isa_ = selectIsa(p.opt);
  k_ = &kernelsFor(isa_);
  capacity_ = size_t(std::max(64, 4 * (range_ + 1)));
}

void Engine::window(int n, int& first, int& last) const {
  const int lagsign = lag_ > 0 ? 1 : -1;
  int base = n - range_ * lagsign;
  base = std::min(std::max(base, 0), num_frames_ - 1);
  first = std::min(n, base);
  last = std::max(n, base);
}

Engine::Stats Engine::measure(ConstPlane p) const {
  const int b = border_;
  const int w = fmt_.width - 2 * b, h = fmt_.height - 2 * b;
  const int64_t count = int64_t(w) * h;
  Stats st;
  if (fmt_.is_float) {
    double s = 0, q = 0;
    k_->sum_f32(p.data + b * p.stride + b * 4, p.stride, w, h, &s, &q);
    st.mean = s / double(count);
    st.var = q / double(count) - st.mean * st.mean;
    return st;
  }
  int64_t s = 0, q = 0;
  if (fmt_.layout == Layout::YUY2)
    k_->sum_yuy2(p.data + b * p.stride + b * 2, p.stride, w, h, &s, &q);
  else if (fmt_.bits > 8)
    k_->sum_u16(p.data + b * p.stride + b * 2, p.stride, w, h, &s, &q);
  else
    k_->sum_u8(p.data + b * p.stride + b, p.stride, w, h, &s, &q);
  const int64_t mean = s / count;  // integer arithmetic as in 0.6
  const int64_t var = q / count - mean * mean;
  st.mean = double(mean);
  st.var = double(var);
  return st;
}

Engine::Stats Engine::stats(int k, const Fetch& fetch, int n, ConstPlane src) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(k);
    if (it != cache_.end()) return it->second;
  }
  Stats st;
  if (k == n) {
    st = measure(src);
  } else {
    const LumaRef ref = fetch(k);
    st = measure(ref.luma);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (cache_.emplace(k, st).second) {
    order_.push_back(k);
    while (order_.size() > capacity_) {
      cache_.erase(order_.front());
      order_.pop_front();
    }
  }
  return st;
}

void Engine::process(int ndest, const Fetch& fetch, ConstPlane src, Frame& dst, bool field_based) {
  const int lagsign = lag_ > 0 ? 1 : -1;
  int nbase = ndest - range_ * lagsign;  // base is some previous (lag > 0) or some next frame (lag < 0)
  nbase = std::min(std::max(nbase, 0), num_frames_ - 1);

  // Walk from the current frame towards the base frame until a bad frame or a scene change.
  std::vector<Stats> st(size_t(range_) + 1);
  auto slot = [&](int k) -> Stats& { return st[size_t(std::abs(ndest - k))]; };
  int newbase = -1;
  for (int ncur = ndest; ncur * lagsign >= nbase * lagsign; ncur -= lagsign) {
    const Stats cur = slot(ncur) = stats(ncur, fetch, ndest, src);
    if (float(cur.var) <= varnoise_) {  // too little variation (flat frame): next one starts the scene
      newbase = ncur != ndest ? ncur + lagsign : ncur;
      break;
    }
    if (ncur != ndest) {
      const Stats& prev = slot(ncur + lagsign);
      if (std::fabs(prev.mean - cur.mean) > scene_ ||
          2 * std::fabs(std::sqrt(float(prev.var)) - std::sqrt(float(cur.var))) > scene_f_) {
        newbase = ncur + lagsign;  // new scene at ncur + 1
        break;
      }
    }
  }
  if (newbase >= 0) nbase = newbase;

  // Simplified AURORA method of intensity flicker correction (P.M.B. van Roosmalen, "Restoration of Archived Film
  // and Video", 1999), applied globally to the whole frame:
  //   degradation model  z(n) = alfa * y(n) + beta(n) + eta(n)   (z measured, y true luma, eta noise)
  //   estimation         y(n) = a * z(n) + b(n)
  // propagated from the base frame to the current one, then damped by percent:
  //   y = k * (a * z + b) + (1 - k) * z,  k = percent / 100
  // Float expressions and their order are those of 0.6, which keeps 8 bit output bit-identical.
  float meansmoothed = float(slot(nbase).mean);
  double var = slot(nbase).var;
  float mult = 1, add = 0, a = 1, b = 0;
  if ((nbase + lagsign) * lagsign <= ndest * lagsign && float(var) > varnoise_) {
    for (int ncur = nbase + lagsign; ncur * lagsign <= ndest * lagsign; ncur += lagsign) {
      float var_y = a * a * float(var);  // variation of true y from the previous estimation
      const Stats& s = slot(ncur);
      const float meancur = float(s.mean);
      var = s.var;
      float alfa = std::sqrt((float(var) - varnoise_) / var_y);
      float beta = meancur - alfa * meansmoothed;
      a = std::sqrt((float(var) - varnoise_) / (float(var) * alfa));  // the article has no sqrt here: an error
      b = -beta / alfa + meancur * (1 / alfa - a);                     // corrected as well
      meansmoothed = a * meancur + b;
      // corrector step with the predicted values
      var_y = a * a * float(var);
      alfa = std::sqrt((float(var) - varnoise_) / var_y);
      beta = meancur - alfa * meansmoothed;
      a = std::sqrt((float(var) - varnoise_) / (float(var) * alfa));
      b = -beta / alfa + meancur * (1 / alfa - a);
      mult = 1 + (a - 1) * percent_ / 100;
      add = b * percent_ / 100;
      meansmoothed = meancur * mult + add;
    }
  }

  const Stats& cur = slot(ndest);
  if (debug_) {
    char buf[256];
    if (fmt_.is_float)
      std::snprintf(buf, sizeof buf,
                    "Deflicker: dest=%d base=%d mean=%.6f sq.var=%.6f mult=%5.3f add=%.6f smoothed=%.6f\n", ndest,
                    nbase, cur.mean, std::sqrt(float(cur.var)), mult, add, meansmoothed);
    else
      std::snprintf(buf, sizeof buf,
                    "Deflicker: dest=%d base=%d mean=%lld sq.var=%5.1f mult=%5.3f add=%6.1f smoothed=%5.1f\n",
                    ndest, nbase, static_cast<long long>(cur.mean), std::sqrt(float(cur.var)), mult, add,
                    meansmoothed);
    debugOut(buf);
  }

  const int w = fmt_.width, h = fmt_.height;
  if (fmt_.is_float || fmt_.bits > 8) {
    float m = mult, ad = add;
    if (!std::isfinite(m) || !std::isfinite(ad)) m = 1, ad = 0;
    if (fmt_.is_float)
      k_->correct_f32(dst.data[0], dst.stride[0], src.data, src.stride, w, h, m, ad, lo_, hi_);
    else
      k_->correct_u16(dst.data[0], dst.stride[0], src.data, src.stride, w, h, m, ad, lo_, hi_);
  } else {
    // 8 bit fixed point coefficients of 0.6
    const int16_t mult256w = toShort(mult * 256);
    const int16_t addw = toShort(add);
    const uint8_t lo = uint8_t(lmin8_), hi = uint8_t(lmax8_);
    if (fmt_.layout == Layout::YUY2)
      k_->correct_yuy2(dst.data[0], dst.stride[0], src.data, src.stride, w, h, mult256w, addw, lo, hi);
    else
      k_->correct_u8(dst.data[0], dst.stride[0], src.data, src.stride, w, h, mult256w, addw, lo, hi);
  }

  if (info_) overlay(ndest, nbase, cur, mult, add, meansmoothed, dst, field_based);
}

void Engine::overlay(int n, int base, const Stats& cur, float mult, float add, float smoothed, Frame& dst,
                     bool field_based) const {
  const int w = fmt_.width, h = fmt_.height, bd = border_;
  if (bd > 0) {
    // make the border of the measured area visible: luma mirrored inside lmin..lmax
    auto invert = [&](int x, int y) {
      uint8_t* row = dst.data[0] + y * dst.stride[0];
      if (fmt_.layout == Layout::YUY2) {
        row[2 * x] = uint8_t(lmax8_ + lmin8_ - row[2 * x]);
      } else if (fmt_.is_float) {
        float* p = reinterpret_cast<float*>(row) + x;
        *p = hi_ + lo_ - *p;
      } else if (fmt_.bits > 8) {
        uint16_t* p = reinterpret_cast<uint16_t*>(row) + x;
        *p = uint16_t(int(hi_) + int(lo_) - *p);
      } else {
        row[x] = uint8_t(lmax8_ + lmin8_ - row[x]);
      }
    };
    for (int x = bd; x < w - bd; x++) invert(x, bd);
    for (int y = bd + 1; y < h - bd - 1; y++) {
      invert(bd, y);
      invert(w - bd - 1, y);
    }
    if (h - bd - 1 > bd)
      for (int x = bd; x < w - bd; x++) invert(x, h - bd - 1);
  }

  // text at about a quarter of the width; field-based clips alternate the column per field
  int col = w / 40 - (field_based ? 7 - (n % 2) * 15 : 0);
  col = std::max(col, 0);
  const int row = std::max(h / 40 - 4, 0);
  char msg[64];
  drawString(dst, fmt_, col, row, base != n ? " DeFlicker" : " DeFlicker:NEW");
  std::snprintf(msg, sizeof msg, " frame=%7d", n);
  drawString(dst, fmt_, col, row + 1, msg);
  std::snprintf(msg, sizeof msg, " base =%7d", base);
  drawString(dst, fmt_, col, row + 2, msg);
  if (fmt_.is_float)
    std::snprintf(msg, sizeof msg, " mean   =%.4f", cur.mean);
  else
    std::snprintf(msg, sizeof msg, " mean   =%3lld", static_cast<long long>(cur.mean));
  drawString(dst, fmt_, col, row + 3, msg);
  if (fmt_.is_float)
    std::snprintf(msg, sizeof msg, " sq.var =%.4f", std::sqrt(float(cur.var)));
  else
    std::snprintf(msg, sizeof msg, " sq.var =%5.1f", std::sqrt(float(cur.var)));
  drawString(dst, fmt_, col, row + 4, msg);
  std::snprintf(msg, sizeof msg, " mult= %7.3f", mult);
  drawString(dst, fmt_, col, row + 5, msg);
  if (fmt_.is_float)
    std::snprintf(msg, sizeof msg, " add = %7.4f", add);
  else
    std::snprintf(msg, sizeof msg, " add = %7.1f", add);
  drawString(dst, fmt_, col, row + 6, msg);
  if (fmt_.is_float)
    std::snprintf(msg, sizeof msg, " m.fixed=%.4f", smoothed);
  else
    std::snprintf(msg, sizeof msg, " m.fixed=%5.1f", smoothed);
  drawString(dst, fmt_, col, row + 7, msg);
}

}  // namespace deflicker
